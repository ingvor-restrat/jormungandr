"""Discrete PPO, IMPALA/V-trace, and APPO/IMPACT learner implementations."""

from __future__ import annotations

import math
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


def _meta_float(
    metadata: Optional[Sequence[Mapping[str, Any]]],
    key: str,
    fallback: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    if not metadata:
        return fallback.detach(), 1.0
    values: list[float] = []
    missing = 0
    fallback_np = fallback.detach().cpu().numpy().reshape(-1)
    for index, item in enumerate(metadata):
        raw = item.get(key)
        try:
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError
        except Exception:
            value = float(fallback_np[index])
            missing += 1
        values.append(value)
    return torch.as_tensor(values, dtype=torch.float32, device=device), missing / max(1, len(values))


def _continues(
    metadata: Optional[Sequence[Mapping[str, Any]]], index: int, done: bool
) -> bool:
    if done or metadata is None or index + 1 >= len(metadata):
        return False
    current = metadata[index]
    following = metadata[index + 1]
    actor = str(current.get("actor_id", ""))
    episode = str(current.get("episode_id", ""))
    if actor != str(following.get("actor_id", "")) or episode != str(
        following.get("episode_id", "")
    ):
        return False
    try:
        return int(following.get("timestep")) == int(current.get("timestep")) + 1
    except Exception:
        return False


class DistributedPolicyAgent(AuxiliaryMixin):
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
            raise ValueError("policy-gradient service plugins require discrete action_values")
        self.num_actions = len(self.action_values)
        hidden = max(8, int(cfg(config, "hidden", 256)))
        self.policy = MLP(obs_dim, self.num_actions, hidden).to(self.device)
        self.value = MLP(obs_dim, 1, hidden).to(self.device)
        self.target_policy = None
        if objective == "appo":
            self.target_policy = MLP(obs_dim, self.num_actions, hidden).to(self.device)
            self.target_policy.load_state_dict(self.policy.state_dict())
        self.gamma = float(cfg(config, "gamma", 0.99))
        self.gae_lambda = float(cfg(config, "gae_lambda", 0.95))
        self.clip_ratio = max(0.0, float(cfg(config, "clip_ratio", 0.2)))
        self.entropy_coef = max(0.0, float(cfg(config, "entropy_coef", 0.01)))
        self.value_coef = max(0.0, float(cfg(config, "value_coef", 0.5)))
        self.epochs = max(1, int(cfg(config, "epochs", 4 if objective == "ppo" else 1)))
        self.minibatch_size = max(1, int(cfg(config, "minibatch_size", cfg(config, "batch_size", 256))))
        self.rho_clip = max(1.0, float(cfg(config, "vtrace_rho_clip", 1.0)))
        self.pg_rho_clip = max(1.0, float(cfg(config, "vtrace_pg_rho_clip", 1.0)))
        self.c_clip = max(0.0, float(cfg(config, "vtrace_c_clip", 1.0)))
        self.appo_target_worker_clip = max(
            1.0, float(cfg(config, "appo_target_worker_clip", 2.0))
        )
        self.target_update = max(1, int(cfg(config, "target_update", 1000)))
        self.huber_delta = max(1e-6, float(cfg(config, "huber_delta", 1.0)))
        self.max_grad_norm = float(cfg(config, "max_grad", 1.0))
        self.reward_clip = max(0.0, float(cfg(config, "reward_clip", 0.0)))
        self.observation_noise_std = max(
            0.0, float(cfg(config, "observation_noise_std", 0.0))
        )
        self._init_aux(obs_dim, hidden, config)
        parameters = list(self.policy.parameters()) + list(self.value.parameters())
        if self.aux_head is not None:
            parameters.extend(self.aux_head.parameters())
        self.opt = torch.optim.Adam(parameters, lr=float(cfg(config, "lr", 1e-4)))
        self.update_steps = 0
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
            value = float(self.value(obs_t).item())
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

    def _gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = rewards + self.gamma * (1.0 - dones) * next_values - values
        advantages = torch.zeros_like(delta)
        following = torch.zeros((), device=self.device)
        for index in range(delta.numel() - 1, -1, -1):
            continuation = 1.0 if _continues(metadata, index, bool(dones[index].item())) else 0.0
            following = delta[index] + self.gamma * self.gae_lambda * continuation * following
            advantages[index] = following
        return advantages, advantages + values

    def _vtrace(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        logp: torch.Tensor,
        behavior_logp: torch.Tensor,
        metadata: Optional[Sequence[Mapping[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_ratio = torch.exp((logp.detach() - behavior_logp).clamp(-20.0, 20.0))
        rho = raw_ratio.clamp(max=self.rho_clip)
        trace_c = raw_ratio.clamp(max=self.c_clip)
        delta = rho * (rewards + self.gamma * (1.0 - dones) * next_values - values)
        targets = torch.empty_like(values)
        for index in range(values.numel() - 1, -1, -1):
            if _continues(metadata, index, bool(dones[index].item())):
                correction = self.gamma * trace_c[index] * (targets[index + 1] - next_values[index])
            else:
                correction = torch.zeros((), device=self.device)
            targets[index] = values[index] + delta[index] + correction
        pg_advantage = torch.empty_like(values)
        for index in range(values.numel()):
            next_target = (
                targets[index + 1]
                if _continues(metadata, index, bool(dones[index].item()))
                else next_values[index]
            )
            pg_advantage[index] = raw_ratio[index].clamp(max=self.pg_rho_clip) * (
                rewards[index]
                + self.gamma * (1.0 - dones[index]) * next_target
                - values[index]
            )
        return targets.detach(), pg_advantage.detach(), raw_ratio

    def _targets(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        metadata: Optional[Sequence[Mapping[str, Any]]],
    ):
        with torch.no_grad():
            logits = mask_logits(self.policy(obs), action_mask)
            distribution = torch.distributions.Categorical(logits=logits)
            current_logp = distribution.log_prob(actions)
            values = self.value(obs).squeeze(-1)
            next_values = self.value(next_obs).squeeze(-1)
            behavior_logp, fallback_rate = _meta_float(
                metadata, "behavior_logp", current_logp, self.device
            )
            behavior_value, value_fallback_rate = _meta_float(
                metadata, "behavior_value", values, self.device
            )
            if self.objective == "ppo":
                behavior_next_value = next_values.clone()
                for index in range(values.numel() - 1):
                    if _continues(
                        metadata, index, bool(dones[index].item())
                    ):
                        behavior_next_value[index] = behavior_value[index + 1]
                advantage, returns = self._gae(
                    rewards,
                    dones,
                    behavior_value,
                    behavior_next_value,
                    metadata,
                )
                ratios = torch.ones_like(advantage)
            else:
                returns, advantage, ratios = self._vtrace(
                    rewards,
                    dones,
                    values,
                    next_values,
                    current_logp,
                    behavior_logp,
                    metadata,
                )
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        return (
            behavior_logp,
            returns,
            advantage,
            fallback_rate,
            value_fallback_rate,
            ratios,
        )

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
        del weights
        obs, actions, rewards, next_obs, dones = as_tensors(batch, self.device)
        obs = noisy_observations(obs, self.observation_noise_std)
        rewards = clip_rewards(rewards, self.reward_clip)
        action_mask = legal_mask_from_metadata(
            metadata,
            key="action_mask",
            rows=obs.shape[0],
            actions=self.num_actions,
            device=self.device,
        )
        (
            old_logp,
            returns,
            advantage,
            fallback_rate,
            value_fallback_rate,
            initial_ratios,
        ) = self._targets(obs, actions, rewards, next_obs, dones, action_mask, metadata)
        target_logp = None
        if self.target_policy is not None:
            with torch.no_grad():
                target_logits = mask_logits(self.target_policy(obs), action_mask)
                target_logp = torch.distributions.Categorical(logits=target_logits).log_prob(actions)

        indices = np.arange(obs.shape[0])
        totals: list[float] = []
        policies: list[float] = []
        values_out: list[float] = []
        entropies: list[float] = []
        clip_fractions: list[float] = []
        approx_kls: list[float] = []
        last_per_example = torch.zeros(obs.shape[0], device=self.device)
        aux_scale = 1.0 / float(max(1, self.epochs * math.ceil(len(indices) / self.minibatch_size)))

        for _epoch in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                mb_np = indices[start : start + self.minibatch_size]
                mb = torch.as_tensor(mb_np, dtype=torch.long, device=self.device)
                logits = mask_logits(
                    self.policy(obs[mb]), None if action_mask is None else action_mask[mb]
                )
                distribution = torch.distributions.Categorical(logits=logits)
                logp = distribution.log_prob(actions[mb])
                entropy = distribution.entropy().mean()
                current_value = self.value(obs[mb]).squeeze(-1)
                ratio = torch.exp((logp - old_logp[mb]).clamp(-20.0, 20.0))
                if self.objective == "appo" and target_logp is not None:
                    log_beta = math.log(self.appo_target_worker_clip)
                    denominator = torch.maximum(target_logp[mb], old_logp[mb] + log_beta)
                    ratio = torch.exp((logp - denominator).clamp(-20.0, 20.0))
                clipped_ratio = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                if self.objective == "impala":
                    policy_loss_per = -(logp * advantage[mb])
                else:
                    policy_loss_per = -torch.minimum(
                        ratio * advantage[mb], clipped_ratio * advantage[mb]
                    )
                value_loss_per = nn.functional.huber_loss(
                    current_value,
                    returns[mb],
                    reduction="none",
                    delta=self.huber_delta,
                )
                loss = (
                    policy_loss_per.mean()
                    + self.value_coef * value_loss_per.mean()
                    - self.entropy_coef * entropy
                )
                # Recompute the optional graph for every optimizer step; a
                # single loss tensor cannot be backpropagated repeatedly.
                aux_term = self._aux_loss(aux_obs, aux_targets, aux_weight)
                if aux_term is not None:
                    loss = loss + aux_scale * aux_term
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0.0:
                    parameters = list(self.policy.parameters()) + list(self.value.parameters())
                    if self.aux_head is not None:
                        parameters.extend(self.aux_head.parameters())
                    torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
                self.opt.step()
                last_per_example[mb] = value_loss_per.detach() + policy_loss_per.detach().abs()
                totals.append(float(loss.detach().cpu()))
                policies.append(float(policy_loss_per.mean().detach().cpu()))
                values_out.append(float(value_loss_per.mean().detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                clip_fractions.append(float((ratio.ne(clipped_ratio)).float().mean().detach().cpu()))
                approx_kls.append(float((old_logp[mb] - logp).mean().detach().cpu()))

        self.update_steps += 1
        if self.target_policy is not None and self.update_steps % self.target_update == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
        metrics = {
            "loss": float(np.mean(totals)),
            "policy_loss": float(np.mean(policies)),
            "value_loss": float(np.mean(values_out)),
            "policy_entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "approx_kl": float(np.mean(approx_kls)),
            "behavior_logp_fallback_rate": float(fallback_rate),
            "behavior_value_fallback_rate": float(value_fallback_rate),
            "importance_ratio_mean": float(initial_ratios.mean().cpu()),
        }
        self.last_metrics = metrics
        return UpdateResult(metrics["loss"], last_per_example.cpu().numpy(), metrics)

    def evaluate_batch(
        self,
        batch,
        *,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        **_: Any,
    ) -> Mapping[str, float]:
        obs, actions, rewards, next_obs, dones = as_tensors(batch, self.device)
        rewards = clip_rewards(rewards, self.reward_clip)
        action_mask = legal_mask_from_metadata(
            metadata,
            key="action_mask",
            rows=obs.shape[0],
            actions=self.num_actions,
            device=self.device,
        )
        (
            old_logp,
            returns,
            advantage,
            fallback_rate,
            value_fallback_rate,
            _,
        ) = self._targets(obs, actions, rewards, next_obs, dones, action_mask, metadata)
        with torch.no_grad():
            logits = mask_logits(self.policy(obs), action_mask)
            distribution = torch.distributions.Categorical(logits=logits)
            logp = distribution.log_prob(actions)
            values = self.value(obs).squeeze(-1)
            policy_loss = -(logp * advantage).mean()
            value_loss = nn.functional.huber_loss(values, returns, delta=self.huber_delta)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * distribution.entropy().mean()
        return {
            "loss": float(loss.cpu()),
            "td_abs_mean": float((returns - values).abs().mean().cpu()),
            "policy_entropy": float(distribution.entropy().mean().cpu()),
            "behavior_logp_fallback_rate": float(fallback_rate),
            "behavior_value_fallback_rate": float(value_fallback_rate),
            "count": float(obs.shape[0]),
        }

    def state_dict(self) -> Mapping[str, Any]:
        state: dict[str, Any] = {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "opt": self.opt.state_dict(),
            "update_steps": self.update_steps,
            "objective": self.objective,
        }
        if self.target_policy is not None:
            state["target_policy"] = self.target_policy.state_dict()
        if self.aux_head is not None:
            state["aux_head"] = self.aux_head.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        if self.target_policy is not None:
            self.target_policy.load_state_dict(state.get("target_policy", state["policy"]))
        if self.aux_head is not None and "aux_head" in state:
            self.aux_head.load_state_dict(state["aux_head"])
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])
            optimizer_to(self.opt, self.device)
        self.update_steps = int(state.get("update_steps", 0))
