"""Behavior cloning and advantage-weighted offline imitation agents."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .base import ActionResult, UpdateResult
from .common import (
    AuxiliaryMixin,
    MLP,
    as_tensors,
    cfg,
    clip_rewards,
    legal_mask_from_metadata,
    mask_logits,
    noisy_observations,
    optimizer_to,
)


def _continues(
    metadata: Optional[Sequence[Mapping[str, Any]]], index: int, done: bool
) -> bool:
    if done or metadata is None or index + 1 >= len(metadata):
        return False
    current = metadata[index]
    following = metadata[index + 1]
    if (
        str(current.get("actor_id", "")) != str(following.get("actor_id", ""))
        or str(current.get("episode_id", ""))
        != str(following.get("episode_id", ""))
    ):
        return False
    try:
        return int(following.get("timestep")) == int(current.get("timestep")) + 1
    except Exception:
        return False


class OfflinePolicyAgent(AuxiliaryMixin):
    def __init__(
        self,
        obs_dim: int,
        config: Mapping[str, Any],
        device: str,
        *,
        objective: str,
    ) -> None:
        self.device = torch.device(device)
        self.objective = objective
        self.action_values = tuple(float(x) for x in cfg(config, "action_values", ()))
        if not self.action_values:
            raise ValueError("offline discrete learners require action_values")
        self.num_actions = len(self.action_values)
        hidden = max(8, int(cfg(config, "hidden", 256)))
        self.policy = MLP(obs_dim, self.num_actions, hidden).to(self.device)
        self.value = MLP(obs_dim, 1, hidden).to(self.device) if objective == "marwil" else None
        self.gamma = float(cfg(config, "gamma", 0.99))
        self.value_coef = max(0.0, float(cfg(config, "value_coef", 0.5)))
        self.marwil_beta = max(0.0, float(cfg(config, "marwil_beta", 1.0)))
        self.marwil_c2_rate = min(
            1.0, max(1e-8, float(cfg(config, "marwil_c2_rate", 1e-8)))
        )
        self.marwil_c2 = max(1e-8, float(cfg(config, "marwil_c2_start", 100.0)))
        self.max_advantage_weight = max(
            1.0, float(cfg(config, "max_advantage_weight", 20.0))
        )
        self.max_grad_norm = float(cfg(config, "max_grad", 1.0))
        self.reward_clip = max(0.0, float(cfg(config, "reward_clip", 0.0)))
        self.observation_noise_std = max(
            0.0, float(cfg(config, "observation_noise_std", 0.0))
        )
        self._init_aux(obs_dim, hidden, config)
        parameters = list(self.policy.parameters())
        if self.value is not None:
            parameters.extend(self.value.parameters())
        if self.aux_head is not None:
            parameters.extend(self.aux_head.parameters())
        self.opt = torch.optim.Adam(parameters, lr=float(cfg(config, "lr", 1e-4)))
        self.last_metrics: Mapping[str, float] = {}

    def action_result(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool,
        epsilon: float = 0.0,
        action_mask: Optional[np.ndarray] = None,
    ) -> ActionResult:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        mask_t = None
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).reshape(1, -1)
        with torch.no_grad():
            raw_logits = self.policy(obs_t)
            logits = mask_logits(raw_logits, mask_t)
            distribution = torch.distributions.Categorical(logits=logits)
            policy_probs = distribution.probs[0]
            legal = torch.arange(self.num_actions, device=self.device) if mask_t is None else torch.nonzero(mask_t[0], as_tuple=False).flatten()
            if deterministic:
                action_t = logits.argmax(dim=-1)
                behavior_probs = torch.zeros_like(policy_probs)
                behavior_probs[int(action_t.item())] = 1.0
            else:
                eps = min(1.0, max(0.0, float(epsilon)))
                behavior_probs = (1.0 - eps) * policy_probs
                behavior_probs[legal] += eps / float(len(legal))
                action_t = torch.distributions.Categorical(probs=behavior_probs).sample()
            idx = int(action_t.item())
            value = None if self.value is None else float(self.value(obs_t).item())
        return ActionResult(
            action=float(self.action_values[idx]),
            action_idx=idx,
            log_probability=float(torch.log(behavior_probs[idx].clamp_min(1e-8)).item()),
            value=value,
            extras={
                "policy_logits": [float(x) for x in raw_logits[0].cpu().tolist()],
                "policy_probs": [float(x) for x in policy_probs.cpu().tolist()],
            },
        )

    def act(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> tuple[float, int]:
        result = self.action_result(
            obs,
            deterministic=deterministic,
            epsilon=epsilon,
            action_mask=action_mask,
        )
        return result.action, result.action_idx

    def inference_batch(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool,
        epsilon: float,
        action_masks: Optional[np.ndarray] = None,
    ) -> list[ActionResult]:
        return [
            self.action_result(
                row,
                deterministic=deterministic,
                epsilon=epsilon,
                action_mask=None if action_masks is None else action_masks[index],
            )
            for index, row in enumerate(np.asarray(obs, dtype=np.float32))
        ]

    def _losses(
        self,
        batch,
        *,
        training: bool,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ):
        obs, actions, rewards, next_obs, dones = as_tensors(batch, self.device)
        if training:
            obs = noisy_observations(obs, self.observation_noise_std)
        action_mask = legal_mask_from_metadata(
            metadata,
            key="action_mask",
            rows=obs.shape[0],
            actions=self.num_actions,
            device=self.device,
        )
        logits = mask_logits(self.policy(obs), action_mask)
        negative_logp = nn.functional.cross_entropy(logits, actions, reduction="none")
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        accuracy = (logits.argmax(dim=-1) == actions).float().mean()
        value_loss = torch.zeros((), device=self.device)
        weights = torch.ones_like(negative_logp)
        advantage = torch.zeros_like(negative_logp)
        if self.value is not None:
            values = self.value(obs).squeeze(-1)
            with torch.no_grad():
                next_values = self.value(next_obs).squeeze(-1)
                clipped_rewards = clip_rewards(rewards, self.reward_clip)
                targets = torch.empty_like(values)
                for index in range(values.numel() - 1, -1, -1):
                    if _continues(
                        metadata, index, bool(dones[index].item())
                    ):
                        continuation_value = targets[index + 1]
                    else:
                        continuation_value = next_values[index]
                    targets[index] = clipped_rewards[index] + self.gamma * (
                        1.0 - dones[index]
                    ) * continuation_value
                advantage = targets - values
                batch_c2 = float(advantage.pow(2).mean().cpu())
                if training:
                    self.marwil_c2 = (
                        1.0 - self.marwil_c2_rate
                    ) * self.marwil_c2 + self.marwil_c2_rate * batch_c2
                scale = max(1e-8, self.marwil_c2) ** 0.5
                weights = torch.exp(self.marwil_beta * advantage / scale).clamp(
                    max=self.max_advantage_weight
                )
            value_loss = nn.functional.huber_loss(values, targets)
        policy_loss = (weights.detach() * negative_logp).mean()
        total = policy_loss + self.value_coef * value_loss
        return total, policy_loss, value_loss, negative_logp, entropy, accuracy, advantage, weights

    def update(
        self,
        batch,
        weights,
        *,
        aux_obs: Optional[np.ndarray] = None,
        aux_targets: Optional[np.ndarray] = None,
        aux_weight: float = 0.0,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        **_: Any,
    ) -> UpdateResult:
        del weights  # Offline objectives define their own per-example weighting.
        total, policy_loss, value_loss, nll, entropy, accuracy, advantage, marwil_weights = self._losses(
            batch, training=True, metadata=metadata
        )
        auxiliary = self._aux_loss(aux_obs, aux_targets, aux_weight)
        if auxiliary is not None:
            total = total + auxiliary
        self.opt.zero_grad(set_to_none=True)
        total.backward()
        if self.max_grad_norm > 0.0:
            parameters = list(self.policy.parameters())
            if self.value is not None:
                parameters.extend(self.value.parameters())
            if self.aux_head is not None:
                parameters.extend(self.aux_head.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.opt.step()
        metrics = {
            "loss": float(total.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "policy_entropy": float(entropy.detach().cpu()),
            "action_accuracy": float(accuracy.detach().cpu()),
        }
        if self.value is not None:
            metrics.update(
                {
                    "value_loss": float(value_loss.detach().cpu()),
                    "advantage_mean": float(advantage.mean().detach().cpu()),
                    "advantage_weight_mean": float(marwil_weights.mean().detach().cpu()),
                    "advantage_c2": float(self.marwil_c2),
                }
            )
        self.last_metrics = metrics
        priorities = nll.detach().cpu().numpy()
        return UpdateResult(metrics["loss"], priorities, metrics)

    def evaluate_batch(
        self,
        batch,
        *,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        **_: Any,
    ) -> Mapping[str, float]:
        with torch.no_grad():
            total, _, value_loss, nll, entropy, accuracy, advantage, _ = self._losses(
                batch, training=False, metadata=metadata
            )
        return {
            "loss": float(total.cpu()),
            "td_abs_mean": float(advantage.abs().mean().cpu()) if self.value is not None else float(nll.mean().cpu()),
            "policy_entropy": float(entropy.cpu()),
            "action_accuracy": float(accuracy.cpu()),
            "value_loss": float(value_loss.cpu()),
            "count": float(nll.numel()),
        }

    def state_dict(self) -> Mapping[str, Any]:
        state: dict[str, Any] = {
            "policy": self.policy.state_dict(),
            "opt": self.opt.state_dict(),
            "objective": self.objective,
            "marwil_c2": self.marwil_c2,
        }
        if self.value is not None:
            state["value"] = self.value.state_dict()
        if self.aux_head is not None:
            state["aux_head"] = self.aux_head.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        if self.value is not None and "value" in state:
            self.value.load_state_dict(state["value"])
        if self.aux_head is not None and "aux_head" in state:
            self.aux_head.load_state_dict(state["aux_head"])
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])
            optimizer_to(self.opt, self.device)
        self.marwil_c2 = float(state.get("marwil_c2", self.marwil_c2))
