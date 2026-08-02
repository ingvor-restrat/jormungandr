"""Modern discrete value-learning agents used by DQN, CQL, and MaxEnt plugins."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .base import ActionResult, UpdateResult
from .common import (
    AuxiliaryMixin,
    DuelingQNet,
    as_tensors,
    cfg,
    clip_rewards,
    legal_mask_from_metadata,
    mask_logits,
    noisy_observations,
    optimizer_to,
)


class DiscreteQAgent(AuxiliaryMixin):
    """Double/dueling DQN with optional conservative or soft-Q objective."""

    def __init__(
        self,
        obs_dim: int,
        config: Mapping[str, Any],
        device: str,
        *,
        objective: str,
    ) -> None:
        self.device = torch.device(device)
        self.objective = str(objective)
        self.action_values = tuple(float(x) for x in cfg(config, "action_values", ()))
        if not self.action_values:
            raise ValueError("a discrete Q learner requires action_values")
        self.num_actions = len(self.action_values)
        self.gamma = float(cfg(config, "gamma", 0.99))
        self.target_update = max(1, int(cfg(config, "target_update", 1000)))
        self.max_grad_norm = float(cfg(config, "max_grad", 1.0))
        self.huber_delta = max(1e-6, float(cfg(config, "huber_delta", 1.0)))
        self.reward_clip = max(0.0, float(cfg(config, "reward_clip", 0.0)))
        self.observation_noise_std = max(
            0.0, float(cfg(config, "observation_noise_std", 0.0))
        )
        self.temperature = max(1e-4, float(cfg(config, "temperature", 0.2)))
        self.cql_alpha = max(0.0, float(cfg(config, "cql_alpha", 1.0)))
        hidden = max(8, int(cfg(config, "hidden", 256)))

        self.q = DuelingQNet(obs_dim, self.num_actions, hidden).to(self.device)
        self.target = DuelingQNet(obs_dim, self.num_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self._init_aux(obs_dim, hidden, config)
        parameters = list(self.q.parameters())
        if self.aux_head is not None:
            parameters.extend(self.aux_head.parameters())
        self.opt = torch.optim.Adam(parameters, lr=float(cfg(config, "lr", 1e-4)))
        self.update_steps = 0
        self.last_metrics: Mapping[str, float] = {}

    def _masked_q(
        self, network: nn.Module, obs: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return mask_logits(network(obs), mask)

    def action_result(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool = False,
        epsilon: float = 0.0,
        action_mask: Optional[np.ndarray] = None,
    ) -> ActionResult:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        mask_t = None
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).reshape(1, -1)
            if mask_t.shape != (1, self.num_actions) or not bool(mask_t.any()):
                raise ValueError("action_mask must align with action_values and admit an action")
        with torch.no_grad():
            raw_q_values = self.q(obs_t)
            q_values = mask_logits(raw_q_values, mask_t)
            legal = (
                torch.arange(self.num_actions, device=self.device)
                if mask_t is None
                else torch.nonzero(mask_t[0], as_tuple=False).flatten()
            )
            if self.objective == "maxent":
                logits = q_values / self.temperature
                distribution = torch.distributions.Categorical(logits=logits)
                idx_t = logits.argmax(dim=-1) if deterministic else distribution.sample()
                logp = distribution.log_prob(idx_t)
                probs = distribution.probs[0]
            else:
                greedy = q_values.argmax(dim=-1)
                eps = 0.0 if deterministic else min(1.0, max(0.0, float(epsilon)))
                if eps > 0.0 and float(torch.rand((), device=self.device)) < eps:
                    idx_t = legal[torch.randint(len(legal), (1,), device=self.device)]
                else:
                    idx_t = greedy
                probs = torch.zeros(self.num_actions, device=self.device)
                probs[legal] = eps / float(len(legal))
                probs[int(greedy.item())] += 1.0 - eps
                logp = torch.log(probs[idx_t].clamp_min(1e-8))
            idx = int(idx_t.item())
            q_row = raw_q_values[0]
        return ActionResult(
            action=float(self.action_values[idx]),
            action_idx=idx,
            log_probability=float(logp.item()),
            value=float(q_row[idx].item()),
            extras={
                "q_values": [float(x) for x in q_row.cpu().tolist()],
                "policy_probs": [float(x) for x in probs.cpu().tolist()],
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
        rows = np.asarray(obs, dtype=np.float32)
        return [
            self.action_result(
                row,
                deterministic=deterministic,
                epsilon=epsilon,
                action_mask=None if action_masks is None else action_masks[index],
            )
            for index, row in enumerate(rows)
        ]

    def _target_values(
        self,
        next_obs: torch.Tensor,
        next_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target_q = self._masked_q(self.target, next_obs, next_mask)
        if self.objective == "maxent":
            return self.temperature * torch.logsumexp(target_q / self.temperature, dim=-1)
        online_next = self._masked_q(self.q, next_obs, next_mask)
        next_action = online_next.argmax(dim=-1)
        return target_q.gather(1, next_action.unsqueeze(1)).squeeze(1)

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
        obs, actions, rewards, next_obs, dones = as_tensors(batch, self.device)
        weight = torch.as_tensor(weights, dtype=torch.float32, device=self.device).reshape(-1)
        obs = noisy_observations(obs, self.observation_noise_std)
        next_obs = noisy_observations(next_obs, self.observation_noise_std)
        rewards = clip_rewards(rewards, self.reward_clip)
        action_mask = legal_mask_from_metadata(
            metadata,
            key="action_mask",
            rows=obs.shape[0],
            actions=self.num_actions,
            device=self.device,
        )
        next_mask = legal_mask_from_metadata(
            metadata,
            key="next_action_mask",
            rows=obs.shape[0],
            actions=self.num_actions,
            device=self.device,
        )

        q_all = self.q(obs)
        decision_q = mask_logits(q_all, action_mask)
        chosen_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = rewards + self.gamma * (1.0 - dones) * self._target_values(next_obs, next_mask)
        td_error = target - chosen_q
        td_loss = nn.functional.huber_loss(
            chosen_q,
            target,
            reduction="none",
            delta=self.huber_delta,
        )
        loss = (weight * td_loss).mean()
        conservative = torch.zeros((), device=self.device)
        if self.objective == "cql":
            conservative = (
                torch.logsumexp(decision_q, dim=-1) - chosen_q
            ).mean()
            loss = loss + self.cql_alpha * conservative

        auxiliary = self._aux_loss(aux_obs, aux_targets, aux_weight)
        if auxiliary is not None:
            loss = loss + auxiliary

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm > 0.0:
            parameters = list(self.q.parameters())
            if self.aux_head is not None:
                parameters.extend(self.aux_head.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.opt.step()

        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

        with torch.no_grad():
            policy_entropy = torch.distributions.Categorical(
                logits=decision_q / self.temperature
            ).entropy().mean()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "td_loss": float(td_loss.mean().detach().cpu()),
            "td_abs_mean": float(td_error.abs().mean().detach().cpu()),
            "q_mean": float(chosen_q.mean().detach().cpu()),
            "target_mean": float(target.mean().detach().cpu()),
        }
        if self.objective == "cql":
            metrics["cql_penalty"] = float(conservative.detach().cpu())
        if self.objective == "maxent":
            metrics["policy_entropy"] = float(policy_entropy.detach().cpu())
            metrics["temperature"] = float(self.temperature)
        self.last_metrics = metrics
        return UpdateResult(
            loss=metrics["loss"],
            priorities=td_error.detach().abs().cpu().numpy(),
            metrics=metrics,
        )

    def evaluate_batch(
        self,
        batch,
        *,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        aux_obs: Optional[np.ndarray] = None,
        aux_targets: Optional[np.ndarray] = None,
        **_: Any,
    ) -> Mapping[str, float]:
        obs, actions, rewards, next_obs, dones = as_tensors(batch, self.device)
        action_mask = legal_mask_from_metadata(
            metadata,
            key="action_mask",
            rows=obs.shape[0],
            actions=self.num_actions,
            device=self.device,
        )
        next_mask = legal_mask_from_metadata(
            metadata,
            key="next_action_mask",
            rows=obs.shape[0],
            actions=self.num_actions,
            device=self.device,
        )
        with torch.no_grad():
            q_all = self.q(obs)
            decision_q = mask_logits(q_all, action_mask)
            chosen = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            target = clip_rewards(rewards, self.reward_clip) + self.gamma * (1.0 - dones) * self._target_values(
                next_obs, next_mask
            )
            td = target - chosen
            loss = nn.functional.huber_loss(
                chosen, target, reduction="none", delta=self.huber_delta
            )
            if self.objective == "cql":
                loss = loss + self.cql_alpha * (
                    torch.logsumexp(decision_q, dim=-1) - chosen
                )
            out = {
                "loss": float(loss.mean().cpu()),
                "td_abs_mean": float(td.abs().mean().cpu()),
                "count": float(obs.shape[0]),
            }
            if self.aux_head is not None and aux_obs is not None and aux_targets is not None and len(aux_obs):
                aux_x = torch.as_tensor(aux_obs, dtype=torch.float32, device=self.device)
                aux_y = torch.as_tensor(aux_targets, dtype=torch.long, device=self.device)
                logits = self.aux_head(aux_x)
                out["aux_loss"] = float(nn.functional.cross_entropy(logits, aux_y).cpu())
                out["aux_acc"] = float((logits.argmax(-1) == aux_y).float().mean().cpu())
        return out

    def state_dict(self) -> Mapping[str, Any]:
        state: dict[str, Any] = {
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "update_steps": self.update_steps,
            "objective": self.objective,
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
