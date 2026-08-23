"""Quantile-regression distributional DQN (QR-DQN).

Unlike C51, QR-DQN learns the locations of a fixed set of quantiles rather
than probabilities on a fixed value support.  The learned return distribution
can be reduced to its mean for the standard control objective, or inspected at
inference time through a lower quantile or lower-tail CVaR score.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .base import ActionResult, AlgorithmPlugin, UpdateResult
from .common import (
    AuxiliaryMixin,
    action_mask_tensor,
    cfg,
    clip_rewards,
    epsilon_greedy_batch,
    legal_mask_from_metadata,
    mask_logits,
    noisy_observations,
    optimizer_to,
)
from .registry import algorithm_registry


class QuantileQNet(nn.Module):
    """A scriptable MLP that emits ``[batch, action, quantile]`` values."""

    def __init__(self, obs_dim: int, actions: int, quantiles: int, hidden: int):
        super().__init__()
        self.actions = int(actions)
        self.quantiles = int(quantiles)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.actions * self.quantiles),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).reshape(-1, self.actions, self.quantiles)


class QuantileDQNAgent(AuxiliaryMixin):
    """Double QR-DQN with Huber quantile loss and risk-aware inference."""

    def __init__(self, obs_dim: int, config: Mapping[str, Any], device: str) -> None:
        self.device = torch.device(device)
        self.action_values = tuple(float(x) for x in cfg(config, "action_values", ()))
        if not self.action_values:
            raise ValueError("QR-DQN requires action_values")
        self.num_actions = len(self.action_values)
        self.num_quantiles = max(2, int(cfg(config, "quantiles", 51)))
        self.gamma = float(cfg(config, "gamma", 0.99))
        self.target_update = max(1, int(cfg(config, "target_update", 1000)))
        self.max_grad_norm = float(cfg(config, "max_grad", 1.0))
        self.huber_delta = max(1e-6, float(cfg(config, "huber_delta", 1.0)))
        self.reward_clip = max(0.0, float(cfg(config, "reward_clip", 0.0)))
        self.observation_noise_std = max(
            0.0, float(cfg(config, "observation_noise_std", 0.0))
        )
        self.risk_measure = str(
            cfg(config, "quantile_risk_measure", "mean")
        ).strip().lower().replace("-", "_")
        if self.risk_measure not in {"mean", "lower_quantile", "cvar"}:
            raise ValueError(
                "quantile_risk_measure must be mean, lower_quantile, or cvar"
            )
        self.risk_level = float(cfg(config, "quantile_risk_level", 0.1))
        if not 0.0 < self.risk_level <= 1.0:
            raise ValueError("quantile_risk_level must be in (0, 1]")
        hidden = max(8, int(cfg(config, "hidden", 256)))

        self.q = QuantileQNet(
            obs_dim, self.num_actions, self.num_quantiles, hidden
        ).to(self.device)
        self.target = QuantileQNet(
            obs_dim, self.num_actions, self.num_quantiles, hidden
        ).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self._init_aux(obs_dim, hidden, config)
        parameters = list(self.q.parameters())
        if self.aux_head is not None:
            parameters.extend(self.aux_head.parameters())
        self.opt = torch.optim.Adam(parameters, lr=float(cfg(config, "lr", 1e-4)))
        self.update_steps = 0
        self.last_metrics: Mapping[str, float] = {}
        self.taus = (
            (torch.arange(self.num_quantiles, dtype=torch.float32, device=self.device) + 0.5)
            / float(self.num_quantiles)
        )

    def _scores(self, quantiles: torch.Tensor) -> torch.Tensor:
        if self.risk_measure == "mean":
            return quantiles.mean(dim=-1)
        ordered = torch.sort(quantiles, dim=-1).values
        if self.risk_measure == "lower_quantile":
            index = min(
                self.num_quantiles - 1,
                max(0, int(np.ceil(self.risk_level * self.num_quantiles)) - 1),
            )
            return ordered[..., index]
        count = min(
            self.num_quantiles,
            max(1, int(np.ceil(self.risk_level * self.num_quantiles))),
        )
        return ordered[..., :count].mean(dim=-1)

    def action_result(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool = False,
        epsilon: float = 0.0,
        action_mask: Optional[np.ndarray] = None,
    ) -> ActionResult:
        masks = None if action_mask is None else np.asarray([action_mask])
        return self.inference_batch(
            np.asarray(obs, dtype=np.float32).reshape(1, -1),
            deterministic=deterministic,
            epsilon=epsilon,
            action_masks=masks,
        )[0]

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
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        legal = action_mask_tensor(
            action_masks,
            rows=obs_t.shape[0],
            actions=self.num_actions,
            device=self.device,
        )
        with torch.no_grad():
            quantiles = self.q(obs_t)
            risk_scores = self._scores(quantiles)
            indices, probabilities = epsilon_greedy_batch(
                risk_scores,
                legal,
                deterministic=deterministic,
                epsilon=epsilon,
            )
            selected_values = risk_scores.gather(
                1, indices.unsqueeze(1)
            ).squeeze(1)
            selected_probabilities = probabilities.gather(
                1, indices.unsqueeze(1)
            ).squeeze(1)
            q_mean = quantiles.mean(dim=-1)
        q_rows = q_mean.cpu().tolist()
        risk_rows = risk_scores.cpu().tolist()
        decision_rows = risk_scores.masked_fill(~legal, -1e30).cpu().tolist()
        quantile_rows = quantiles.cpu().tolist()
        index_rows = indices.cpu().tolist()
        value_rows = selected_values.cpu().tolist()
        probability_rows = selected_probabilities.cpu().tolist()
        return [
            ActionResult(
                action=float(self.action_values[int(index)]),
                action_idx=int(index),
                log_probability=float(
                    np.log(max(float(probability_rows[row]), 1e-8))
                ),
                value=float(value_rows[row]),
                extras={
                    "q_values": [float(value) for value in q_rows[row]],
                    "decision_values": [
                        float(value) for value in decision_rows[row]
                    ],
                    "risk_values": [float(value) for value in risk_rows[row]],
                    "quantiles": [
                        [float(value) for value in values]
                        for values in quantile_rows[row]
                    ],
                    "risk_measure": self.risk_measure,
                    "risk_level": float(self.risk_level),
                },
            )
            for row, index in enumerate(index_rows)
        ]

    def _target_quantiles(
        self, next_obs: torch.Tensor, next_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Standard QR-DQN optimizes expected return; risk_measure changes the
        # deployed decision rule, not the Bellman target.
        online_scores = self.q(next_obs).mean(dim=-1)
        online_scores = mask_logits(online_scores, next_mask)
        next_actions = online_scores.argmax(dim=-1)
        target_all = self.target(next_obs)
        rows = torch.arange(next_obs.shape[0], device=self.device)
        return target_all[rows, next_actions]

    def _loss_rows(
        self, chosen: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pair every predicted quantile with every target sample.
        delta = target.unsqueeze(1) - chosen.unsqueeze(2)
        absolute = delta.abs()
        huber = torch.where(
            absolute <= self.huber_delta,
            0.5 * delta.square(),
            self.huber_delta * (absolute - 0.5 * self.huber_delta),
        )
        indicator = (delta.detach() < 0.0).to(dtype=chosen.dtype)
        quantile_weight = (self.taus.reshape(1, -1, 1) - indicator).abs()
        # Mean over target samples, sum over learned quantile locations as in
        # the QR-DQN quantile-Huber objective.
        rows = (quantile_weight * huber / self.huber_delta).mean(dim=2).sum(dim=1)
        return rows, delta

    def update(
        self,
        batch,
        weights,
        *,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        aux_obs: Optional[np.ndarray] = None,
        aux_targets: Optional[np.ndarray] = None,
        aux_weight: float = 0.0,
        **_: Any,
    ) -> UpdateResult:
        from .common import as_tensors

        obs, actions, rewards, next_obs, dones = as_tensors(batch, self.device)
        weight = torch.as_tensor(
            weights, dtype=torch.float32, device=self.device
        ).reshape(-1)
        obs = noisy_observations(obs, self.observation_noise_std)
        next_obs = noisy_observations(next_obs, self.observation_noise_std)
        rewards = clip_rewards(rewards, self.reward_clip)
        next_mask = legal_mask_from_metadata(
            metadata,
            key="next_action_mask",
            rows=obs.shape[0],
            actions=self.num_actions,
            device=self.device,
        )
        rows = torch.arange(obs.shape[0], device=self.device)
        chosen = self.q(obs)[rows, actions]
        with torch.no_grad():
            target = rewards.unsqueeze(1) + self.gamma * (
                1.0 - dones.unsqueeze(1)
            ) * self._target_quantiles(next_obs, next_mask)
        loss_rows, pairwise_delta = self._loss_rows(chosen, target)
        loss = (weight * loss_rows).mean()
        auxiliary = self._aux_loss(aux_obs, aux_targets, aux_weight)
        if auxiliary is not None:
            loss = loss + auxiliary

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(self.q.parameters())
        if self.aux_head is not None:
            parameters.extend(self.aux_head.parameters())
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.opt.step()
        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

        with torch.no_grad():
            priority = pairwise_delta.abs().mean(dim=(1, 2))
            crossing = (
                chosen[:, 1:] < chosen[:, :-1]
            ).to(torch.float32).mean()
            ordered = torch.sort(chosen, dim=-1).values
            spread = ordered[:, -1] - ordered[:, 0]
        metrics = {
            "loss": float(loss.detach().cpu()),
            "quantile_loss": float(loss_rows.mean().detach().cpu()),
            "td_abs_mean": float(priority.mean().detach().cpu()),
            "q_mean": float(chosen.mean().detach().cpu()),
            "quantile_spread": float(spread.mean().detach().cpu()),
            "quantile_crossing_rate": float(crossing.detach().cpu()),
        }
        self.last_metrics = metrics
        return UpdateResult(
            loss=metrics["loss"],
            priorities=priority.detach().cpu().numpy(),
            metrics=metrics,
        )

    def evaluate_batch(
        self,
        batch,
        *,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        **_: Any,
    ) -> Mapping[str, float]:
        from .common import as_tensors

        obs, actions, rewards, next_obs, dones = as_tensors(batch, self.device)
        next_mask = legal_mask_from_metadata(
            metadata,
            key="next_action_mask",
            rows=obs.shape[0],
            actions=self.num_actions,
            device=self.device,
        )
        rows = torch.arange(obs.shape[0], device=self.device)
        with torch.no_grad():
            chosen = self.q(obs)[rows, actions]
            target = clip_rewards(rewards, self.reward_clip).unsqueeze(1) + self.gamma * (
                1.0 - dones.unsqueeze(1)
            ) * self._target_quantiles(next_obs, next_mask)
            loss_rows, delta = self._loss_rows(chosen, target)
        return {
            "loss": float(loss_rows.mean().cpu()),
            "td_abs_mean": float(delta.abs().mean().cpu()),
            "quantile_spread": float(
                (
                    torch.sort(chosen, dim=-1).values[:, -1]
                    - torch.sort(chosen, dim=-1).values[:, 0]
                ).mean().cpu()
            ),
            "count": float(obs.shape[0]),
        }

    def state_dict(self) -> Mapping[str, Any]:
        state: dict[str, Any] = {
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "update_steps": self.update_steps,
            "num_quantiles": self.num_quantiles,
            "risk_measure": self.risk_measure,
            "risk_level": self.risk_level,
        }
        if self.aux_head is not None:
            state["aux_head"] = self.aux_head.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.q.load_state_dict(state["q"])
        self.target.load_state_dict(state.get("target", state["q"]))
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])
            optimizer_to(self.opt, self.device)
        self.update_steps = int(state.get("update_steps", 0))
        if self.aux_head is not None and "aux_head" in state:
            self.aux_head.load_state_dict(state["aux_head"])


def _build(
    obs_dim: int, config: Mapping[str, Any], device: str
) -> QuantileDQNAgent:
    return QuantileDQNAgent(obs_dim, config, device)


PLUGIN = AlgorithmPlugin(
    name="qrdqn",
    version="1.0.0",
    family="off-policy distributional value",
    build=_build,
    default_export_module="q",
    aliases=("qr_dqn", "quantile_dqn"),
    description=(
        "Quantile-regression distributional DQN with mean, lower-quantile, "
        "or lower-tail CVaR action scoring."
    ),
    noise_profile=(
        "Models the return distribution directly; lower-tail scoring exposes "
        "outcome risk, but does not by itself make observations adversarially robust."
    ),
)
algorithm_registry.register(PLUGIN)
