"""In-process PPO for variable entity sets and state-local candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jormungandr.structured import (
    DynamicActionResult,
    EntityCandidateObservation,
    EntityCandidateTransformer,
    StructuredPolicySpec,
    collate_entity_candidate_observations,
    select_dynamic_actions,
)
from jormungandr.structured_trajectory import (
    StructuredJointTrajectoryStep,
    validate_structured_joint_trajectory,
)

from .base import AlgorithmPlugin
from .common import cfg, optimizer_to
from .registry import algorithm_registry


@dataclass(frozen=True)
class StructuredTransition:
    """One on-policy transition retaining its state-local candidate identity."""

    episode_id: str
    timestep: int
    observation: EntityCandidateObservation
    candidate_id: str
    candidate_index: int
    behavior_log_probability: float
    behavior_value: float
    reward: float
    done: bool

    def __post_init__(self) -> None:
        if not str(self.episode_id).strip() or self.timestep < 0:
            raise ValueError("episode_id and a non-negative timestep are required")
        if not 0 <= self.candidate_index < len(self.observation.candidate_ids):
            raise ValueError("candidate_index is outside the local candidate set")
        if self.observation.candidate_ids[self.candidate_index] != self.candidate_id:
            raise ValueError("candidate_id does not match the selected local slot")
        if not self.observation.legal_action_mask[self.candidate_index]:
            raise ValueError("selected candidate is not legal")
        if not all(
            math.isfinite(float(value))
            for value in (
                self.behavior_log_probability,
                self.behavior_value,
                self.reward,
            )
        ):
            raise ValueError("transition numerical fields must be finite")


@dataclass(frozen=True)
class StructuredPPOUpdate:
    """Aggregate diagnostics from one structured PPO update."""

    transitions: int
    episodes: int
    epochs: int
    minibatches: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    importance_ratio_mean: float
    importance_ratio_std: float
    importance_ratio_min: float
    importance_ratio_max: float
    explained_variance: float
    advantage_mean: float
    advantage_std: float
    gradient_norm: float
    episode_return_mean: float
    episode_return_std: float
    episode_return_min: float
    episode_return_max: float
    episode_return_unique_count: int
    episode_length_mean: float
    reward_nonzero_fraction: float


@dataclass(frozen=True)
class StructuredPolicyScore:
    """Policy logits and centralized value aligned to semantic candidates."""

    candidate_ids: tuple[str, ...]
    candidate_logits: tuple[float, ...]
    value: float


def _resolve_device(value: str) -> torch.device:
    raw = str(value).strip().lower()
    if raw in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _episode_signal_statistics(
    trajectories: Sequence[Sequence[Any]],
) -> Mapping[str, float | int]:
    """Summarize the reward signal before GAE can obscure its diversity."""

    episodes = tuple(tuple(trajectory) for trajectory in trajectories if trajectory)
    if not episodes:
        raise ValueError("at least one complete trajectory is required")
    episode_returns = np.asarray(
        [sum(float(step.reward) for step in episode) for episode in episodes],
        dtype=np.float64,
    )
    episode_lengths = np.asarray(
        [len(episode) for episode in episodes], dtype=np.float64
    )
    rewards = np.asarray(
        [float(step.reward) for episode in episodes for step in episode],
        dtype=np.float64,
    )
    return {
        "episodes": len(episodes),
        "episode_return_mean": float(episode_returns.mean()),
        "episode_return_std": float(episode_returns.std()),
        "episode_return_min": float(episode_returns.min()),
        "episode_return_max": float(episode_returns.max()),
        "episode_return_unique_count": int(np.unique(episode_returns).size),
        "episode_length_mean": float(episode_lengths.mean()),
        "reward_nonzero_fraction": float(np.count_nonzero(rewards) / rewards.size),
    }


class StructuredPPOAgent:
    """PPO learner whose action slots are local to each observation."""

    def __init__(
        self,
        spec: StructuredPolicySpec,
        config: Mapping[str, Any],
        device: str = "auto",
    ) -> None:
        self.spec = spec
        self.device = _resolve_device(device)
        model_dim = max(8, int(cfg(config, "structured_model_dim", 64)))
        heads = max(1, int(cfg(config, "structured_heads", 4)))
        layers = max(1, int(cfg(config, "structured_layers", 2)))
        feedforward = max(
            model_dim,
            int(cfg(config, "structured_feedforward_dim", model_dim * 2)),
        )
        self.policy = EntityCandidateTransformer(
            global_dim=spec.global_dim,
            entity_dim=spec.entity_dim,
            candidate_dim=spec.candidate_dim,
            entity_type_count=spec.entity_type_count,
            model_dim=model_dim,
            heads=heads,
            layers=layers,
            feedforward_dim=feedforward,
            dropout=max(0.0, float(cfg(config, "structured_dropout", 0.0))),
        ).to(self.device)
        self.gamma = float(cfg(config, "gamma", 0.99))
        self.gae_lambda = float(cfg(config, "gae_lambda", 0.95))
        self.clip_ratio = max(0.0, float(cfg(config, "clip_ratio", 0.2)))
        self.entropy_coefficient = max(
            0.0, float(cfg(config, "entropy_coef", 0.01))
        )
        self.value_coefficient = max(
            0.0, float(cfg(config, "value_coef", 0.5))
        )
        self.epochs = max(1, int(cfg(config, "epochs", 4)))
        self.minibatch_size = max(
            1,
            int(cfg(config, "minibatch_size", cfg(config, "batch_size", 128))),
        )
        self.max_grad_norm = max(0.0, float(cfg(config, "max_grad", 1.0)))
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=float(cfg(config, "lr", 3e-4))
        )
        self.update_steps = 0
        self.last_metrics: Mapping[str, float] = {}

    def action_results_structured(
        self,
        observations: Sequence[EntityCandidateObservation],
        *,
        deterministic: bool,
    ) -> tuple[DynamicActionResult, ...]:
        """Score one heterogeneous batch and return semantic candidate IDs."""

        if not observations:
            return ()
        was_training = self.policy.training
        self.policy.eval()
        batch = collate_entity_candidate_observations(observations).to_torch(
            self.device
        )
        with torch.no_grad():
            output = self.policy(batch)
            results = select_dynamic_actions(
                output, batch, deterministic=deterministic
            )
        self.policy.train(was_training)
        return results

    def action_result_structured(
        self,
        observation: EntityCandidateObservation,
        *,
        deterministic: bool,
    ) -> DynamicActionResult:
        return self.action_results_structured(
            (observation,), deterministic=deterministic
        )[0]

    def score_results_structured(
        self,
        observations: Sequence[EntityCandidateObservation],
    ) -> tuple[StructuredPolicyScore, ...]:
        """Return local candidate logits without choosing a joint action."""

        if not observations:
            return ()
        was_training = self.policy.training
        self.policy.eval()
        batch = collate_entity_candidate_observations(observations).to_torch(
            self.device
        )
        with torch.no_grad():
            output = self.policy(batch)
        results = []
        for row, observation in enumerate(observations):
            count = len(observation.candidate_ids)
            logits = output.logits[row, :count].detach().cpu().tolist()
            results.append(
                StructuredPolicyScore(
                    candidate_ids=observation.candidate_ids,
                    candidate_logits=tuple(float(value) for value in logits),
                    value=float(output.values[row].detach().cpu()),
                )
            )
        self.policy.train(was_training)
        return tuple(results)

    def _targets(
        self,
        trajectories: Sequence[Sequence[StructuredTransition]],
    ) -> tuple[list[StructuredTransition], np.ndarray, np.ndarray]:
        flat: list[StructuredTransition] = []
        advantages: list[float] = []
        returns: list[float] = []
        for trajectory in trajectories:
            steps = tuple(trajectory)
            if not steps:
                continue
            if not steps[-1].done:
                raise ValueError("structured PPO currently requires complete episodes")
            episode_id = steps[0].episode_id
            if any(
                step.episode_id != episode_id or step.timestep != index
                for index, step in enumerate(steps)
            ):
                raise ValueError("trajectory identity and timesteps must be contiguous")
            episode_advantages = np.zeros(len(steps), dtype=np.float32)
            gae = 0.0
            next_value = 0.0
            for index in range(len(steps) - 1, -1, -1):
                step = steps[index]
                continuation = 0.0 if step.done else 1.0
                delta = (
                    step.reward
                    + self.gamma * continuation * next_value
                    - step.behavior_value
                )
                gae = (
                    delta
                    + self.gamma * self.gae_lambda * continuation * gae
                )
                episode_advantages[index] = gae
                next_value = step.behavior_value
            flat.extend(steps)
            advantages.extend(float(value) for value in episode_advantages)
            returns.extend(
                float(advantage + step.behavior_value)
                for advantage, step in zip(episode_advantages, steps)
            )
        if not flat:
            raise ValueError("at least one complete trajectory is required")
        return (
            flat,
            np.asarray(advantages, dtype=np.float32),
            np.asarray(returns, dtype=np.float32),
        )

    def update_structured(
        self,
        trajectories: Sequence[Sequence[StructuredTransition]],
    ) -> StructuredPPOUpdate:
        """Apply clipped PPO updates to complete variable-candidate episodes."""

        signal = _episode_signal_statistics(trajectories)
        transitions, raw_advantages, returns = self._targets(trajectories)
        advantage_mean = float(raw_advantages.mean())
        advantage_std = float(raw_advantages.std())
        advantages = (
            (raw_advantages - advantage_mean) / max(advantage_std, 1e-8)
        ).astype(np.float32)
        old_values = np.asarray(
            [step.behavior_value for step in transitions], dtype=np.float32
        )
        return_variance = float(np.var(returns))
        explained_variance = (
            0.0
            if return_variance <= 1e-12
            else 1.0 - float(np.var(returns - old_values)) / return_variance
        )

        metric_sums = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
        }
        metric_weight = 0
        minibatches = 0
        ratio_sum = 0.0
        ratio_squared_sum = 0.0
        ratio_count = 0
        ratio_min = float("inf")
        ratio_max = float("-inf")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17_000_003 + self.update_steps)
        self.policy.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(
                len(transitions), generator=generator
            ).tolist()
            for start in range(0, len(permutation), self.minibatch_size):
                indices = permutation[start : start + self.minibatch_size]
                batch = collate_entity_candidate_observations(
                    [transitions[index].observation for index in indices]
                ).to_torch(self.device)
                output = self.policy(batch)
                distribution = torch.distributions.Categorical(
                    logits=output.logits
                )
                actions = torch.as_tensor(
                    [transitions[index].candidate_index for index in indices],
                    dtype=torch.long,
                    device=self.device,
                )
                old_log_probability = torch.as_tensor(
                    [
                        transitions[index].behavior_log_probability
                        for index in indices
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                advantage = torch.as_tensor(
                    advantages[indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                target_return = torch.as_tensor(
                    returns[indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                log_probability = distribution.log_prob(actions)
                log_ratio = log_probability - old_log_probability
                ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                clipped_ratio = ratio.clamp(
                    1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                )
                policy_loss = -torch.minimum(
                    ratio * advantage, clipped_ratio * advantage
                ).mean()
                value_loss = torch.nn.functional.mse_loss(
                    output.values, target_return
                )
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm if self.max_grad_norm > 0.0 else float("inf"),
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = (
                        (torch.exp(log_ratio) - 1.0) - log_ratio
                    ).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.clip_ratio
                    ).float().mean()
                    detached_ratio = ratio.detach()
                    ratio_sum += float(detached_ratio.sum().cpu())
                    ratio_squared_sum += float(
                        detached_ratio.square().sum().cpu()
                    )
                    ratio_count += int(detached_ratio.numel())
                    ratio_min = min(
                        ratio_min, float(detached_ratio.min().cpu())
                    )
                    ratio_max = max(
                        ratio_max, float(detached_ratio.max().cpu())
                    )
                weight = len(indices)
                values = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "gradient_norm": gradient_norm,
                }
                for key, value in values.items():
                    metric_sums[key] += float(value.detach().cpu()) * weight
                metric_weight += weight
                minibatches += 1

        self.update_steps += 1
        averaged = {
            key: value / max(1, metric_weight)
            for key, value in metric_sums.items()
        }
        ratio_mean = ratio_sum / max(1, ratio_count)
        ratio_variance = max(
            0.0,
            ratio_squared_sum / max(1, ratio_count) - ratio_mean**2,
        )
        result = StructuredPPOUpdate(
            transitions=len(transitions),
            episodes=int(signal["episodes"]),
            epochs=self.epochs,
            minibatches=minibatches,
            loss=averaged["loss"],
            policy_loss=averaged["policy_loss"],
            value_loss=averaged["value_loss"],
            entropy=averaged["entropy"],
            approximate_kl=averaged["approximate_kl"],
            clip_fraction=averaged["clip_fraction"],
            importance_ratio_mean=ratio_mean,
            importance_ratio_std=math.sqrt(ratio_variance),
            importance_ratio_min=ratio_min,
            importance_ratio_max=ratio_max,
            explained_variance=explained_variance,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            gradient_norm=averaged["gradient_norm"],
            episode_return_mean=float(signal["episode_return_mean"]),
            episode_return_std=float(signal["episode_return_std"]),
            episode_return_min=float(signal["episode_return_min"]),
            episode_return_max=float(signal["episode_return_max"]),
            episode_return_unique_count=int(
                signal["episode_return_unique_count"]
            ),
            episode_length_mean=float(signal["episode_length_mean"]),
            reward_nonzero_fraction=float(signal["reward_nonzero_fraction"]),
        )
        self.last_metrics = {
            key: float(value)
            for key, value in asdict(result).items()
            if key not in {"transitions", "epochs", "minibatches"}
        }
        return result

    def _joint_targets(
        self,
        trajectories: Sequence[Sequence[StructuredJointTrajectoryStep]],
    ) -> tuple[list[StructuredJointTrajectoryStep], np.ndarray, np.ndarray]:
        flat: list[StructuredJointTrajectoryStep] = []
        advantages: list[float] = []
        returns: list[float] = []
        for raw_trajectory in trajectories:
            steps = validate_structured_joint_trajectory(raw_trajectory)
            episode_advantages = np.zeros(len(steps), dtype=np.float32)
            gae = 0.0
            next_value = 0.0
            for index in range(len(steps) - 1, -1, -1):
                step = steps[index]
                continuation = 0.0 if step.done else 1.0
                delta = (
                    step.reward
                    + self.gamma * continuation * next_value
                    - step.behavior_value
                )
                gae = (
                    delta
                    + self.gamma * self.gae_lambda * continuation * gae
                )
                episode_advantages[index] = gae
                next_value = step.behavior_value
            flat.extend(steps)
            advantages.extend(float(value) for value in episode_advantages)
            returns.extend(
                float(advantage + step.behavior_value)
                for advantage, step in zip(episode_advantages, steps)
            )
        if not flat:
            raise ValueError("at least one complete joint trajectory is required")
        return (
            flat,
            np.asarray(advantages, dtype=np.float32),
            np.asarray(returns, dtype=np.float32),
        )

    @staticmethod
    def _joint_statistics(
        logits: torch.Tensor,
        steps: Sequence[StructuredJointTrajectoryStep],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probabilities = []
        entropies = []
        for row, step in enumerate(steps):
            candidate_index = {
                candidate_id: index
                for index, candidate_id in enumerate(
                    step.observation.candidate_ids
                )
            }
            step_log_probability = torch.zeros((), device=logits.device)
            step_entropy = torch.zeros((), device=logits.device)
            for factor in step.factors:
                indices = torch.as_tensor(
                    [candidate_index[value] for value in factor.candidate_ids],
                    dtype=torch.long,
                    device=logits.device,
                )
                distribution = torch.distributions.Categorical(
                    logits=logits[row].index_select(0, indices)
                )
                selected = factor.candidate_ids.index(
                    factor.selected_candidate_id
                )
                step_log_probability = (
                    step_log_probability
                    + distribution.log_prob(
                        torch.as_tensor(selected, device=logits.device)
                    )
                )
                step_entropy = step_entropy + distribution.entropy()
            log_probabilities.append(step_log_probability)
            entropies.append(step_entropy)
        return torch.stack(log_probabilities), torch.stack(entropies)

    def update_joint_structured(
        self,
        trajectories: Sequence[Sequence[StructuredJointTrajectoryStep]],
    ) -> StructuredPPOUpdate:
        """Apply PPO to one reward-bearing record per joint environment turn."""

        signal = _episode_signal_statistics(trajectories)
        transitions, raw_advantages, returns = self._joint_targets(trajectories)
        advantage_mean = float(raw_advantages.mean())
        advantage_std = float(raw_advantages.std())
        advantages = (
            (raw_advantages - advantage_mean) / max(advantage_std, 1e-8)
        ).astype(np.float32)
        old_values = np.asarray(
            [step.behavior_value for step in transitions], dtype=np.float32
        )
        return_variance = float(np.var(returns))
        explained_variance = (
            0.0
            if return_variance <= 1e-12
            else 1.0 - float(np.var(returns - old_values)) / return_variance
        )
        metric_sums = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
        }
        metric_weight = 0
        minibatches = 0
        ratio_sum = 0.0
        ratio_squared_sum = 0.0
        ratio_count = 0
        ratio_min = float("inf")
        ratio_max = float("-inf")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(31_000_003 + self.update_steps)
        self.policy.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(
                len(transitions), generator=generator
            ).tolist()
            for start in range(0, len(permutation), self.minibatch_size):
                indices = permutation[start : start + self.minibatch_size]
                selected_steps = [transitions[index] for index in indices]
                batch = collate_entity_candidate_observations(
                    [step.observation for step in selected_steps]
                ).to_torch(self.device)
                output = self.policy(batch)
                log_probability, joint_entropy = self._joint_statistics(
                    output.logits, selected_steps
                )
                old_log_probability = torch.as_tensor(
                    [step.joint_behavior_log_probability for step in selected_steps],
                    dtype=torch.float32,
                    device=self.device,
                )
                advantage = torch.as_tensor(
                    advantages[indices], dtype=torch.float32, device=self.device
                )
                target_return = torch.as_tensor(
                    returns[indices], dtype=torch.float32, device=self.device
                )
                log_ratio = log_probability - old_log_probability
                ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                clipped_ratio = ratio.clamp(
                    1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                )
                policy_loss = -torch.minimum(
                    ratio * advantage, clipped_ratio * advantage
                ).mean()
                value_loss = torch.nn.functional.mse_loss(
                    output.values, target_return
                )
                entropy = joint_entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm if self.max_grad_norm > 0.0 else float("inf"),
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = (
                        (torch.exp(log_ratio) - 1.0) - log_ratio
                    ).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.clip_ratio
                    ).float().mean()
                    detached_ratio = ratio.detach()
                    ratio_sum += float(detached_ratio.sum().cpu())
                    ratio_squared_sum += float(
                        detached_ratio.square().sum().cpu()
                    )
                    ratio_count += int(detached_ratio.numel())
                    ratio_min = min(
                        ratio_min, float(detached_ratio.min().cpu())
                    )
                    ratio_max = max(
                        ratio_max, float(detached_ratio.max().cpu())
                    )
                weight = len(indices)
                values = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "gradient_norm": gradient_norm,
                }
                for key, metric in values.items():
                    metric_sums[key] += float(metric.detach().cpu()) * weight
                metric_weight += weight
                minibatches += 1

        self.update_steps += 1
        averaged = {
            key: value / max(1, metric_weight)
            for key, value in metric_sums.items()
        }
        ratio_mean = ratio_sum / max(1, ratio_count)
        ratio_variance = max(
            0.0,
            ratio_squared_sum / max(1, ratio_count) - ratio_mean**2,
        )
        result = StructuredPPOUpdate(
            transitions=len(transitions),
            episodes=int(signal["episodes"]),
            epochs=self.epochs,
            minibatches=minibatches,
            loss=averaged["loss"],
            policy_loss=averaged["policy_loss"],
            value_loss=averaged["value_loss"],
            entropy=averaged["entropy"],
            approximate_kl=averaged["approximate_kl"],
            clip_fraction=averaged["clip_fraction"],
            importance_ratio_mean=ratio_mean,
            importance_ratio_std=math.sqrt(ratio_variance),
            importance_ratio_min=ratio_min,
            importance_ratio_max=ratio_max,
            explained_variance=explained_variance,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            gradient_norm=averaged["gradient_norm"],
            episode_return_mean=float(signal["episode_return_mean"]),
            episode_return_std=float(signal["episode_return_std"]),
            episode_return_min=float(signal["episode_return_min"]),
            episode_return_max=float(signal["episode_return_max"]),
            episode_return_unique_count=int(
                signal["episode_return_unique_count"]
            ),
            episode_length_mean=float(signal["episode_length_mean"]),
            reward_nonzero_fraction=float(signal["reward_nonzero_fraction"]),
        )
        self.last_metrics = {
            key: float(value)
            for key, value in asdict(result).items()
            if key not in {"transitions", "epochs", "minibatches"}
        }
        return result

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "spec": asdict(self.spec),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
            "last_metrics": dict(self.last_metrics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_spec = state.get("spec")
        if isinstance(saved_spec, Mapping) and dict(saved_spec) != asdict(self.spec):
            raise ValueError("structured checkpoint feature schema does not match")
        self.policy.load_state_dict(state["policy"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            optimizer_to(self.optimizer, self.device)
        self.update_steps = int(state.get("update_steps", 0))
        self.last_metrics = {
            str(key): float(value)
            for key, value in dict(state.get("last_metrics", {})).items()
        }

    def initialize_policy_from_state(self, state: Mapping[str, Any]) -> None:
        """Initialize PPO policy weights from a schema-compatible BC state."""

        saved_spec = state.get("spec")
        if isinstance(saved_spec, Mapping) and dict(saved_spec) != asdict(self.spec):
            raise ValueError("structured initialization feature schema does not match")
        policy = state.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError("structured initialization has no policy state")
        self.policy.load_state_dict(policy)


def _build_structured(
    spec: StructuredPolicySpec,
    config: Mapping[str, Any],
    device: str,
) -> StructuredPPOAgent:
    return StructuredPPOAgent(spec, config, device)


PLUGIN = AlgorithmPlugin(
    name="structured_ppo",
    version="1.3.0",
    family="on-policy structured actor critic",
    build=None,
    build_structured=_build_structured,
    representation_modes=("entity_candidates",),
    default_export_module="policy",
    replay_mode="trajectory",
    enforce_policy_lag=True,
    description=(
        "Clipped PPO over variable entities, state-local candidates, and "
        "exact factorized joint-action trajectories."
    ),
)
algorithm_registry.register(PLUGIN)
