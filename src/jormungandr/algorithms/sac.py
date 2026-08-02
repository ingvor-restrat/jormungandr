"""Categorical Soft Actor-Critic learner plugin for Jörmungandr's action vocabulary."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .base import ActionResult, AlgorithmPlugin, UpdateResult
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
from .registry import algorithm_registry


class DiscreteSACAgent(AuxiliaryMixin):
    def __init__(self, obs_dim: int, config: Mapping[str, Any], device: str) -> None:
        self.device = torch.device(device)
        self.action_values = tuple(float(x) for x in cfg(config, "action_values", ()))
        if not self.action_values:
            raise ValueError("discrete SAC requires action_values")
        self.num_actions = len(self.action_values)
        hidden = max(8, int(cfg(config, "hidden", 256)))
        self.policy = MLP(obs_dim, self.num_actions, hidden).to(self.device)
        self.q1 = MLP(obs_dim, self.num_actions, hidden).to(self.device)
        self.q2 = MLP(obs_dim, self.num_actions, hidden).to(self.device)
        self.target_q1 = MLP(obs_dim, self.num_actions, hidden).to(self.device)
        self.target_q2 = MLP(obs_dim, self.num_actions, hidden).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.gamma = float(cfg(config, "gamma", 0.99))
        self.tau = min(1.0, max(1e-6, float(cfg(config, "tau", 0.005))))
        self.max_grad_norm = float(cfg(config, "max_grad", 1.0))
        self.huber_delta = max(1e-6, float(cfg(config, "huber_delta", 1.0)))
        self.reward_clip = max(0.0, float(cfg(config, "reward_clip", 0.0)))
        self.observation_noise_std = max(
            0.0, float(cfg(config, "observation_noise_std", 0.0))
        )
        initial_temperature = max(1e-5, float(cfg(config, "temperature", 0.2)))
        self.auto_entropy = bool(cfg(config, "auto_entropy", True))
        self.log_alpha = torch.tensor(
            math.log(initial_temperature),
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.auto_entropy,
        )
        target_entropy = cfg(config, "target_entropy", None)
        self.target_entropy = float(
            0.98 * math.log(self.num_actions)
            if target_entropy is None
            else target_entropy
        )
        self._init_aux(obs_dim, hidden, config)
        actor_parameters = list(self.policy.parameters())
        if self.aux_head is not None:
            actor_parameters.extend(self.aux_head.parameters())
        lr = float(cfg(config, "lr", 3e-4))
        self.actor_opt = torch.optim.Adam(actor_parameters, lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.alpha_opt = (
            torch.optim.Adam([self.log_alpha], lr=lr) if self.auto_entropy else None
        )
        self.last_metrics: Mapping[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(1e-5, 100.0)

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
            q_values = torch.minimum(self.q1(obs_t), self.q2(obs_t))[0]
        return ActionResult(
            action=float(self.action_values[idx]),
            action_idx=idx,
            log_probability=float(torch.log(behavior_probs[idx].clamp_min(1e-8)).item()),
            value=float(q_values[idx].item()),
            extras={
                "policy_logits": [float(x) for x in raw_logits[0].cpu().tolist()],
                "policy_probs": [float(x) for x in policy_probs.cpu().tolist()],
                "q_values": [float(x) for x in q_values.cpu().tolist()],
                "temperature": float(self.alpha.detach().cpu()),
            },
        )

    def act(self, obs: np.ndarray, epsilon: float = 0.0, deterministic: bool = False, action_mask=None):
        result = self.action_result(
            obs, deterministic=deterministic, epsilon=epsilon, action_mask=action_mask
        )
        return result.action, result.action_idx

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

    def _policy_stats(self, obs: torch.Tensor, mask: Optional[torch.Tensor]):
        logits = mask_logits(self.policy(obs), mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        # A masked categorical has probability zero and log probability -inf
        # on illegal slots.  Entropy and soft-value expectations contain
        # p*log(p); replace only those impossible-slot log values with zero so
        # the mathematically limiting contribution remains zero rather than
        # producing IEEE 0 * -inf = NaN.
        expectation_log_probs = torch.where(
            probs > 0.0, log_probs, torch.zeros_like(log_probs)
        )
        return logits, probs, expectation_log_probs

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for current, delayed in zip(source.parameters(), target.parameters()):
                delayed.mul_(1.0 - self.tau).add_(current, alpha=self.tau)

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
        importance = torch.as_tensor(weights, dtype=torch.float32, device=self.device).reshape(-1)
        obs = noisy_observations(obs, self.observation_noise_std)
        next_obs = noisy_observations(next_obs, self.observation_noise_std)
        rewards = clip_rewards(rewards, self.reward_clip)
        action_mask = legal_mask_from_metadata(
            metadata, key="action_mask", rows=len(actions), actions=self.num_actions, device=self.device
        )
        next_mask = legal_mask_from_metadata(
            metadata, key="next_action_mask", rows=len(actions), actions=self.num_actions, device=self.device
        )
        with torch.no_grad():
            _, next_probs, next_log_probs = self._policy_stats(next_obs, next_mask)
            next_min_q = torch.minimum(self.target_q1(next_obs), self.target_q2(next_obs))
            if next_mask is not None:
                next_min_q = next_min_q.masked_fill(~next_mask, 0.0)
            soft_value = (next_probs * (next_min_q - self.alpha.detach() * next_log_probs)).sum(-1)
            target = rewards + self.gamma * (1.0 - dones) * soft_value

        q1 = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        td1 = target - q1
        td2 = target - q2
        critic_per = nn.functional.huber_loss(q1, target, reduction="none", delta=self.huber_delta) + nn.functional.huber_loss(
            q2, target, reduction="none", delta=self.huber_delta
        )
        critic_loss = (importance * critic_per).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()), self.max_grad_norm
            )
        self.critic_opt.step()

        _, probs, log_probs = self._policy_stats(obs, action_mask)
        with torch.no_grad():
            min_q = torch.minimum(self.q1(obs), self.q2(obs))
            if action_mask is not None:
                min_q = min_q.masked_fill(~action_mask, 0.0)
        actor_per = (probs * (self.alpha.detach() * log_probs - min_q)).sum(-1)
        actor_loss = actor_per.mean()
        auxiliary = self._aux_loss(aux_obs, aux_targets, aux_weight)
        actor_total = actor_loss + (auxiliary if auxiliary is not None else 0.0)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_total.backward()
        if self.max_grad_norm > 0.0:
            parameters = list(self.policy.parameters())
            if self.aux_head is not None:
                parameters.extend(self.aux_head.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.actor_opt.step()

        entropy = -(probs.detach() * log_probs.detach()).sum(-1)
        alpha_loss_value = 0.0
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = float(alpha_loss.detach().cpu())

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        total = critic_loss.detach() + actor_total.detach()
        metrics = {
            "loss": float(total.cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": alpha_loss_value,
            "temperature": float(self.alpha.detach().cpu()),
            "policy_entropy": float(entropy.mean().cpu()),
            "td_abs_mean": float(((td1.abs() + td2.abs()) * 0.5).mean().detach().cpu()),
            "q_mean": float(((q1 + q2) * 0.5).mean().detach().cpu()),
        }
        self.last_metrics = metrics
        priorities = ((td1.abs() + td2.abs()) * 0.5).detach().cpu().numpy()
        return UpdateResult(metrics["loss"], priorities, metrics)

    def evaluate_batch(self, batch, *, metadata=None, **_: Any) -> Mapping[str, float]:
        obs, actions, rewards, next_obs, dones = as_tensors(batch, self.device)
        action_mask = legal_mask_from_metadata(
            metadata, key="action_mask", rows=len(actions), actions=self.num_actions, device=self.device
        )
        next_mask = legal_mask_from_metadata(
            metadata, key="next_action_mask", rows=len(actions), actions=self.num_actions, device=self.device
        )
        with torch.no_grad():
            _, next_probs, next_logp = self._policy_stats(next_obs, next_mask)
            next_q = torch.minimum(self.target_q1(next_obs), self.target_q2(next_obs))
            if next_mask is not None:
                next_q = next_q.masked_fill(~next_mask, 0.0)
            target = clip_rewards(rewards, self.reward_clip) + self.gamma * (1.0 - dones) * (
                next_probs * (next_q - self.alpha * next_logp)
            ).sum(-1)
            q1 = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            q2 = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            _, probs, logp = self._policy_stats(obs, action_mask)
            entropy = -(probs * logp).sum(-1)
            td = ((target - q1).abs() + (target - q2).abs()) * 0.5
        return {
            "loss": float(td.mean().cpu()),
            "td_abs_mean": float(td.mean().cpu()),
            "policy_entropy": float(entropy.mean().cpu()),
            "count": float(obs.shape[0]),
        }

    def state_dict(self) -> Mapping[str, Any]:
        state: dict[str, Any] = {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if self.alpha_opt is not None:
            state["alpha_opt"] = self.alpha_opt.state_dict()
        if self.aux_head is not None:
            state["aux_head"] = self.aux_head.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in ("policy", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(state[name])
        if self.aux_head is not None and "aux_head" in state:
            self.aux_head.load_state_dict(state["aux_head"])
        if "actor_opt" in state:
            self.actor_opt.load_state_dict(state["actor_opt"])
            optimizer_to(self.actor_opt, self.device)
        if "critic_opt" in state:
            self.critic_opt.load_state_dict(state["critic_opt"])
            optimizer_to(self.critic_opt, self.device)
        if "log_alpha" in state:
            self.log_alpha.data.copy_(torch.as_tensor(state["log_alpha"], device=self.device))
        if self.alpha_opt is not None and "alpha_opt" in state:
            self.alpha_opt.load_state_dict(state["alpha_opt"])
            optimizer_to(self.alpha_opt, self.device)


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> DiscreteSACAgent:
    return DiscreteSACAgent(obs_dim, config, device)


PLUGIN = AlgorithmPlugin(
    name="sac",
    version="1.0.0",
    family="off-policy maximum-entropy actor critic",
    build=_build,
    default_export_module="policy",
    description="Categorical SAC with twin critics, soft targets, and automatic entropy temperature.",
    noise_profile="Twin critics reduce positive value bias and entropy promotes multiple viable behaviors under some disturbances.",
)
algorithm_registry.register(PLUGIN)
