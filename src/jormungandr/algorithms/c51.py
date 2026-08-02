"""C51 compatibility plugin around the original public agent class."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from jormungandr.core import C51Agent

from .base import ActionResult, AlgorithmPlugin, UpdateResult
from .common import cfg, optimizer_to
from .registry import algorithm_registry


class C51PluginAgent(C51Agent):
    last_metrics: Mapping[str, float]

    def __init__(self, obs_dim: int, config: Mapping[str, Any], device: str) -> None:
        action_values = [float(x) for x in cfg(config, "action_values", ())]
        super().__init__(
            obs_dim=obs_dim,
            num_actions=len(action_values),
            action_values=action_values,
            hidden=int(cfg(config, "hidden", 256)),
            aux_hidden=int(cfg(config, "aux_hidden", 0)),
            lr=float(cfg(config, "lr", 1e-4)),
            gamma=float(cfg(config, "gamma", 0.99)),
            v_min=float(cfg(config, "v_min", -10.0)),
            v_max=float(cfg(config, "v_max", 10.0)),
            atoms=int(cfg(config, "atoms", 51)),
            target_update=int(cfg(config, "target_update", 1000)),
            max_grad_norm=float(cfg(config, "max_grad", 1.0)),
            device=device,
            aux_classes=int(cfg(config, "aux_classes", 3)) if bool(cfg(config, "aux_enabled", False)) else 0,
            aux_weight=float(cfg(config, "aux_weight", 0.0)),
            aux_class_weighting=str(cfg(config, "aux_class_weighting", "none")),
            aux_label_smoothing=float(cfg(config, "aux_label_smoothing", 0.0)),
        )
        self.reward_clip = max(0.0, float(cfg(config, "reward_clip", 0.0)))
        self.observation_noise_std = max(
            0.0, float(cfg(config, "observation_noise_std", 0.0))
        )
        self.last_metrics = {}

    def _robust_batch(self, batch, *, training: bool):
        obs, actions, rewards, next_obs, dones = batch
        obs_out = np.asarray(obs, dtype=np.float32)
        next_obs_out = np.asarray(next_obs, dtype=np.float32)
        if training and self.observation_noise_std > 0.0:
            obs_out = obs_out + np.random.normal(
                0.0, self.observation_noise_std, size=obs_out.shape
            ).astype(np.float32)
            next_obs_out = next_obs_out + np.random.normal(
                0.0, self.observation_noise_std, size=next_obs_out.shape
            ).astype(np.float32)
        rewards_out = np.asarray(rewards, dtype=np.float32)
        if self.reward_clip > 0.0:
            rewards_out = np.clip(
                rewards_out, -self.reward_clip, self.reward_clip
            )
        return obs_out, actions, rewards_out, next_obs_out, dones

    @staticmethod
    def _metadata_masks(
        metadata: Optional[Sequence[Mapping[str, Any]]], key: str
    ) -> Optional[np.ndarray]:
        if not metadata or not any(item.get(key) is not None for item in metadata):
            return None
        width = next(
            len(item.get(key))
            for item in metadata
            if item.get(key) is not None
        )
        rows = [item.get(key, [True] * width) for item in metadata]
        return np.asarray(rows, dtype=np.bool_)

    def action_result(self, obs: np.ndarray, *, deterministic: bool, epsilon: float = 0.0, action_mask=None):
        action, index = super().act(
            obs,
            epsilon=epsilon,
            deterministic=deterministic,
            action_mask=action_mask,
        )
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            q_values = self._q_values(self.q.dist(obs_t))[0]
            decision_values = q_values
            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                decision_values = q_values.masked_fill(~mask_t, -torch.inf)
            legal_count = len(self.action_values) if action_mask is None else int(np.asarray(action_mask).sum())
            eps = 0.0 if deterministic else min(1.0, max(0.0, epsilon))
            greedy = int(decision_values.argmax().item())
            probability = eps / max(1, legal_count) + (1.0 - eps if index == greedy else 0.0)
        return ActionResult(
            action=float(action),
            action_idx=int(index),
            log_probability=float(np.log(max(probability, 1e-8))),
            value=float(q_values[index].item()),
            extras={"q_values": [float(x) for x in q_values.cpu().tolist()]},
        )

    def inference_batch(self, obs: np.ndarray, *, deterministic: bool, epsilon: float, action_masks=None):
        return [
            self.action_result(
                row,
                deterministic=deterministic,
                epsilon=epsilon,
                action_mask=None if action_masks is None else action_masks[index],
            )
            for index, row in enumerate(np.asarray(obs, dtype=np.float32))
        ]

    def update(self, batch, weights, *, metadata=None, **kwargs):
        masks = self._metadata_masks(metadata, "next_action_mask")
        loss, priorities = super().update(
            self._robust_batch(batch, training=True),
            weights,
            next_action_mask=masks,
            **kwargs,
        )
        metrics = {
            "loss": float(loss),
            "td_abs_mean": float(np.mean(np.abs(priorities))) if len(priorities) else 0.0,
        }
        self.last_metrics = metrics
        return UpdateResult(float(loss), np.asarray(priorities), metrics)

    def evaluate_batch(self, batch, *, metadata=None, **kwargs):
        masks = self._metadata_masks(metadata, "next_action_mask")
        return super().evaluate_batch(
            self._robust_batch(batch, training=False),
            next_action_mask=masks,
            **kwargs,
        )

    def state_dict(self):
        state = dict(super().state_dict())
        state["update_steps"] = int(self.update_steps)
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.update_steps = int(state.get("update_steps", 0))
        optimizer_to(self.opt, self.device)


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> C51PluginAgent:
    return C51PluginAgent(obs_dim, config, device)


PLUGIN = AlgorithmPlugin(
    name="c51",
    version="1.0.0",
    family="off-policy distributional value",
    build=_build,
    default_export_module="q",
    description="Categorical distributional Q-learning retained as a built-in compatibility plugin.",
    noise_profile="A return distribution preserves more stochastic information than a scalar mean, but C51 is not by itself an adversarial-robust method.",
)
algorithm_registry.register(PLUGIN)
