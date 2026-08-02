"""Model-independent masked discrete actor-critic operations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


def apply_legal_action_mask(
    logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
) -> torch.Tensor:
    """Exclude illegal action slots from a categorical policy."""

    mask = legal_action_mask.to(device=logits.device, dtype=torch.bool)
    if logits.ndim != 2:
        raise ValueError("policy logits must have shape [batch, action]")
    if mask.shape != logits.shape:
        raise ValueError(
            "legal_action_mask must have shape "
            f"{tuple(logits.shape)}, received {tuple(mask.shape)}"
        )
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every legal-action-mask row must admit an action")
    return logits.masked_fill(~mask, -torch.inf)


@dataclass(frozen=True)
class MaskedActorCriticLoss:
    """Differentiable loss terms returned to a Jörmungandr learner."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class GraphTrajectoryStep:
    """One auditable transition referencing an externally stored graph."""

    episode_id: str
    timestep: int
    state_reference: str
    action_index: int
    reward: float
    done: bool
    legal_action_mask: tuple[bool, ...]
    log_probability: float
    value: float
    policy_version: int = 0

    def __post_init__(self) -> None:
        if not self.episode_id.strip() or not self.state_reference.strip():
            raise ValueError("episode_id and state_reference are required")
        if self.timestep < 0 or self.policy_version < 0:
            raise ValueError("timestep and policy_version must be non-negative")
        if not self.legal_action_mask or not any(self.legal_action_mask):
            raise ValueError("a trajectory step must admit at least one action")
        if not 0 <= self.action_index < len(self.legal_action_mask):
            raise ValueError("trajectory action index is outside the action mask")
        if not self.legal_action_mask[self.action_index]:
            raise ValueError("trajectory action must be legal")
        if not all(
            np.isfinite(value)
            for value in (
                self.reward,
                self.log_probability,
                self.value,
            )
        ):
            raise ValueError("trajectory numerical fields must be finite")


@dataclass(frozen=True)
class GraphTrajectoryBatch:
    """Flattened GAE targets with stable graph references."""

    state_references: tuple[str, ...]
    action_indices: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    episode_ids: tuple[str, ...]
    timesteps: np.ndarray


class GraphTrajectoryBuffer:
    """Ordered graph-reference trajectories for variable-size observations."""

    def __init__(self) -> None:
        self._episodes: dict[str, list[GraphTrajectoryStep]] = {}

    def __len__(self) -> int:
        return sum(len(steps) for steps in self._episodes.values())

    def add(self, step: GraphTrajectoryStep) -> None:
        steps = self._episodes.setdefault(step.episode_id, [])
        if steps and steps[-1].done:
            raise ValueError("cannot append after a terminal trajectory step")
        if step.timestep != len(steps):
            raise ValueError(
                "trajectory timesteps must be contiguous and start at zero"
            )
        steps.append(step)

    def steps(self, episode_id: str) -> tuple[GraphTrajectoryStep, ...]:
        return tuple(self._episodes.get(episode_id, ()))

    def finish(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        bootstrap_values: dict[str, float] | None = None,
    ) -> GraphTrajectoryBatch:
        """Compute generalized-advantage targets without storing graph tensors."""

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not self._episodes:
            raise ValueError("trajectory buffer is empty")
        bootstrap = bootstrap_values or {}
        references: list[str] = []
        action_indices: list[int] = []
        returns: list[float] = []
        advantages: list[float] = []
        episode_ids: list[str] = []
        timesteps: list[int] = []
        for episode_id, steps in self._episodes.items():
            if not steps:
                continue
            if not steps[-1].done and episode_id not in bootstrap:
                raise ValueError(
                    f"non-terminal episode {episode_id} requires a bootstrap value"
                )
            next_value = 0.0 if steps[-1].done else float(bootstrap[episode_id])
            gae = 0.0
            episode_advantages = [0.0] * len(steps)
            episode_returns = [0.0] * len(steps)
            for index in range(len(steps) - 1, -1, -1):
                step = steps[index]
                continuation = 0.0 if step.done else 1.0
                delta = (
                    step.reward
                    + gamma * next_value * continuation
                    - step.value
                )
                gae = (
                    delta
                    + gamma * gae_lambda * continuation * gae
                )
                episode_advantages[index] = gae
                episode_returns[index] = gae + step.value
                next_value = step.value
            for step, target_return, advantage in zip(
                steps, episode_returns, episode_advantages
            ):
                references.append(step.state_reference)
                action_indices.append(step.action_index)
                returns.append(target_return)
                advantages.append(advantage)
                episode_ids.append(step.episode_id)
                timesteps.append(step.timestep)
        return GraphTrajectoryBatch(
            state_references=tuple(references),
            action_indices=np.asarray(action_indices, dtype=np.int64),
            returns=np.asarray(returns, dtype=np.float32),
            advantages=np.asarray(advantages, dtype=np.float32),
            episode_ids=tuple(episode_ids),
            timesteps=np.asarray(timesteps, dtype=np.int64),
        )

    def records(self) -> list[dict[str, object]]:
        """Return JSON-compatible provenance records in episode order."""

        return [
            {
                "schema_version": "jormungandr.graph_trajectory_step.v1",
                "episode_id": step.episode_id,
                "timestep": step.timestep,
                "state_reference": step.state_reference,
                "action_index": step.action_index,
                "reward": step.reward,
                "done": step.done,
                "legal_action_mask": list(step.legal_action_mask),
                "log_probability": step.log_probability,
                "value": step.value,
                "policy_version": step.policy_version,
            }
            for steps in self._episodes.values()
            for step in steps
        ]


def masked_actor_critic_loss(
    *,
    policy_logits: torch.Tensor,
    state_values: torch.Tensor,
    action_indices: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    legal_action_mask: torch.Tensor,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> MaskedActorCriticLoss:
    """Compute a common loss for any encoder that emits policy/value tensors."""

    masked_logits = apply_legal_action_mask(
        policy_logits, legal_action_mask
    )
    batch = masked_logits.shape[0]
    actions = action_indices.to(
        device=masked_logits.device, dtype=torch.long
    ).reshape(-1)
    if actions.shape != (batch,):
        raise ValueError("action_indices must contain one action per state")
    if torch.any(actions < 0) or torch.any(actions >= masked_logits.shape[1]):
        raise ValueError("action_indices contain an out-of-range slot")
    rows = torch.arange(batch, device=masked_logits.device)
    if not torch.all(
        legal_action_mask.to(
            device=masked_logits.device, dtype=torch.bool
        )[rows, actions]
    ):
        raise ValueError("the training batch contains an illegal chosen action")

    values = state_values.to(device=masked_logits.device).reshape(-1)
    advantage = advantages.to(device=masked_logits.device).reshape(-1)
    target_return = returns.to(device=masked_logits.device).reshape(-1)
    if any(
        tensor.shape != (batch,)
        for tensor in (values, advantage, target_return)
    ):
        raise ValueError(
            "state_values, advantages, and returns must align with the batch"
        )

    distribution = torch.distributions.Categorical(logits=masked_logits)
    policy_loss = -(
        distribution.log_prob(actions) * advantage.detach()
    ).mean()
    value_loss = torch.nn.functional.mse_loss(values, target_return)
    entropy = distribution.entropy().mean()
    total = (
        policy_loss
        + float(value_coefficient) * value_loss
        - float(entropy_coefficient) * entropy
    )
    return MaskedActorCriticLoss(
        total=total,
        policy=policy_loss,
        value=value_loss,
        entropy=entropy,
    )


def select_masked_actions(
    policy_logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
    *,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select legal actions from any compatible policy model."""

    masked_logits = apply_legal_action_mask(
        policy_logits, legal_action_mask
    )
    distribution = torch.distributions.Categorical(logits=masked_logits)
    if deterministic:
        actions = torch.argmax(masked_logits, dim=-1)
    else:
        actions = distribution.sample()
    return actions, distribution.log_prob(actions)
