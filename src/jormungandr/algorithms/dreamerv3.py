"""A compact DreamerV3 plugin for vector observations and discrete actions.

This implementation keeps the defining service-relevant pieces: a learned
latent world model, symlog-scaled reward/value prediction, imagined rollouts,
an entropy-regularized actor, and a slow value target.  It intentionally does
not pretend that Jörmungandr's fixed-vector transition protocol is a pixel
benchmark implementation of the full recurrent categorical model.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

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
    mask_logits,
    noisy_observations,
    optimizer_to,
    symexp,
    symlog,
)
from .registry import algorithm_registry


class DreamerV3Agent(AuxiliaryMixin):
    def __init__(self, obs_dim: int, config: Mapping[str, Any], device: str) -> None:
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.action_values = tuple(float(x) for x in cfg(config, "action_values", ()))
        if not self.action_values:
            raise ValueError("DreamerV3 discrete mode requires action_values")
        self.num_actions = len(self.action_values)
        hidden = max(16, int(cfg(config, "hidden", 256)))
        latent = max(8, int(cfg(config, "dreamer_latent", min(128, hidden))))
        self.encoder = MLP(obs_dim, latent, hidden).to(self.device)
        self.decoder = MLP(latent, obs_dim, hidden).to(self.device)
        self.dynamics = MLP(latent + self.num_actions, latent, hidden).to(self.device)
        self.reward_head = MLP(latent, 1, hidden).to(self.device)
        self.continue_head = MLP(latent, 1, hidden).to(self.device)
        self.actor = MLP(latent, self.num_actions, hidden).to(self.device)
        self.value = MLP(latent, 1, hidden).to(self.device)
        self.target_value = MLP(latent, 1, hidden).to(self.device)
        self.target_value.load_state_dict(self.value.state_dict())
        # Exporters and inference consumers expect a single policy module.
        self.policy = nn.Sequential(self.encoder, self.actor)
        self.gamma = float(cfg(config, "gamma", 0.99))
        self.imagination_horizon = max(1, int(cfg(config, "imagination_horizon", 15)))
        self.return_lambda = min(1.0, max(0.0, float(cfg(config, "dreamer_lambda", 0.95))))
        self.entropy_coef = max(0.0, float(cfg(config, "entropy_coef", 3e-4)))
        self.target_tau = min(1.0, max(1e-6, float(cfg(config, "tau", 0.02))))
        self.max_grad_norm = float(cfg(config, "max_grad", 100.0))
        self.reward_clip = max(0.0, float(cfg(config, "reward_clip", 0.0)))
        self.observation_noise_std = max(
            0.0, float(cfg(config, "observation_noise_std", 0.0))
        )
        self._init_aux(obs_dim, hidden, config)
        lr = float(cfg(config, "lr", 1e-4))
        world_parameters = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_head.parameters())
            + list(self.continue_head.parameters())
        )
        if self.aux_head is not None:
            world_parameters.extend(self.aux_head.parameters())
        self.world_opt = torch.optim.Adam(world_parameters, lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)
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
            latent = self.encoder(obs_t)
            raw_logits = self.actor(latent)
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
            value = symexp(self.value(latent)).item()
        return ActionResult(
            action=float(self.action_values[idx]),
            action_idx=idx,
            log_probability=float(torch.log(behavior_probs[idx].clamp_min(1e-8)).item()),
            value=float(value),
            extras={
                "policy_logits": [float(x) for x in raw_logits[0].cpu().tolist()],
                "policy_probs": [float(x) for x in policy_probs.cpu().tolist()],
                "latent_norm": float(latent.norm(dim=-1).item()),
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

    def _world_losses(self, batch, *, training: bool):
        obs, actions, rewards, next_obs, dones = as_tensors(batch, self.device)
        if training:
            obs = noisy_observations(obs, self.observation_noise_std)
            next_obs = noisy_observations(next_obs, self.observation_noise_std)
        rewards = clip_rewards(rewards, self.reward_clip)
        latent = self.encoder(obs)
        with torch.no_grad():
            next_latent_target = self.encoder(next_obs)
        one_hot = nn.functional.one_hot(actions, self.num_actions).float()
        predicted_next = self.dynamics(torch.cat((latent, one_hot), dim=-1))
        reconstructed = self.decoder(latent)
        reward_prediction = self.reward_head(predicted_next).squeeze(-1)
        continue_logits = self.continue_head(predicted_next).squeeze(-1)
        representation_loss = nn.functional.smooth_l1_loss(predicted_next, next_latent_target)
        reconstruction_loss = nn.functional.smooth_l1_loss(reconstructed, obs)
        reward_loss_per = nn.functional.smooth_l1_loss(
            reward_prediction, symlog(rewards), reduction="none"
        )
        continue_loss = nn.functional.binary_cross_entropy_with_logits(
            continue_logits, 1.0 - dones
        )
        total = representation_loss + reconstruction_loss + reward_loss_per.mean() + continue_loss
        return (
            total,
            representation_loss,
            reconstruction_loss,
            reward_loss_per,
            continue_loss,
            predicted_next,
        )

    def _imagine(self, starts: torch.Tensor):
        latent = starts
        latents: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        continuations: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for _ in range(self.imagination_horizon):
            logits = self.actor(latent)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropies.append(-(probs * log_probs).sum(-1))
            latent = self.dynamics(torch.cat((latent, probs), dim=-1))
            latents.append(latent)
            rewards.append(symexp(self.reward_head(latent).squeeze(-1)))
            continuations.append(torch.sigmoid(self.continue_head(latent).squeeze(-1)))
        with torch.no_grad():
            target_values = [symexp(self.target_value(item).squeeze(-1)) for item in latents]
        bootstrap = target_values[-1]
        returns: list[torch.Tensor] = [bootstrap] * len(latents)
        following = bootstrap
        for index in range(len(latents) - 1, -1, -1):
            mixed = (
                (1.0 - self.return_lambda) * target_values[index]
                + self.return_lambda * following
            )
            following = rewards[index] + self.gamma * continuations[index] * mixed
            returns[index] = following
        return latents, rewards, continuations, entropies, returns

    def _clip(self, parameters) -> None:
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(list(parameters), self.max_grad_norm)

    def update(
        self,
        batch,
        weights,
        *,
        aux_obs: Optional[np.ndarray] = None,
        aux_targets: Optional[np.ndarray] = None,
        aux_weight: float = 0.0,
        **_: Any,
    ) -> UpdateResult:
        del weights
        (
            world_loss,
            representation_loss,
            reconstruction_loss,
            reward_loss_per,
            continue_loss,
            _predicted_next,
        ) = self._world_losses(batch, training=True)
        auxiliary = self._aux_loss(aux_obs, aux_targets, aux_weight)
        if auxiliary is not None:
            world_loss = world_loss + auxiliary
        self.world_opt.zero_grad(set_to_none=True)
        world_loss.backward()
        world_parameters = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_head.parameters())
            + list(self.continue_head.parameters())
        )
        if self.aux_head is not None:
            world_parameters.extend(self.aux_head.parameters())
        self._clip(world_parameters)
        self.world_opt.step()

        obs = torch.as_tensor(batch[0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            starts = self.encoder(obs)
        latents, imagined_rewards, continuations, entropies, returns = self._imagine(starts)
        discounts: list[torch.Tensor] = []
        discount = torch.ones(starts.shape[0], device=self.device)
        for continuation in continuations:
            discounts.append(discount)
            discount = discount * self.gamma * continuation.detach()
        # Returns are fixed targets, but action probabilities influence imagined
        # rewards and continuation through the differentiable expected action.
        actor_terms = [
            weight * (target_return + self.entropy_coef * entropy)
            for weight, target_return, entropy in zip(discounts, returns, entropies)
        ]
        actor_loss = -torch.stack(actor_terms).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip(self.actor.parameters())
        self.actor_opt.step()

        value_predictions = torch.stack(
            [self.value(latent.detach()).squeeze(-1) for latent in latents]
        )
        value_targets = symlog(torch.stack(returns).detach())
        value_loss = nn.functional.smooth_l1_loss(value_predictions, value_targets)
        self.value_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self._clip(self.value.parameters())
        self.value_opt.step()
        with torch.no_grad():
            for current, delayed in zip(self.value.parameters(), self.target_value.parameters()):
                delayed.mul_(1.0 - self.target_tau).add_(current, alpha=self.target_tau)

        total = world_loss.detach() + actor_loss.detach() + value_loss.detach()
        metrics = {
            "loss": float(total.cpu()),
            "world_model_loss": float(world_loss.detach().cpu()),
            "representation_loss": float(representation_loss.detach().cpu()),
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "reward_loss": float(reward_loss_per.mean().detach().cpu()),
            "continuation_loss": float(continue_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "policy_entropy": float(torch.stack(entropies).mean().detach().cpu()),
            "imagined_reward_mean": float(torch.stack(imagined_rewards).mean().detach().cpu()),
        }
        self.last_metrics = metrics
        priorities = reward_loss_per.detach().cpu().numpy() + float(representation_loss.detach().cpu())
        return UpdateResult(metrics["loss"], priorities, metrics)

    def evaluate_batch(self, batch, **_: Any) -> Mapping[str, float]:
        with torch.no_grad():
            total, representation, reconstruction, reward_per, continuation, _, = self._world_losses(
                batch, training=False
            )
        return {
            "loss": float(total.cpu()),
            "td_abs_mean": float(reward_per.mean().cpu()),
            "world_model_loss": float(total.cpu()),
            "representation_loss": float(representation.cpu()),
            "reconstruction_loss": float(reconstruction.cpu()),
            "continuation_loss": float(continuation.cpu()),
            "count": float(len(batch[0])),
        }

    def state_dict(self) -> Mapping[str, Any]:
        state: dict[str, Any] = {
            name: getattr(self, name).state_dict()
            for name in (
                "encoder",
                "decoder",
                "dynamics",
                "reward_head",
                "continue_head",
                "actor",
                "value",
                "target_value",
            )
        }
        state.update(
            {
                "world_opt": self.world_opt.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "value_opt": self.value_opt.state_dict(),
            }
        )
        if self.aux_head is not None:
            state["aux_head"] = self.aux_head.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in (
            "encoder",
            "decoder",
            "dynamics",
            "reward_head",
            "continue_head",
            "actor",
            "value",
            "target_value",
        ):
            getattr(self, name).load_state_dict(state[name])
        if self.aux_head is not None and "aux_head" in state:
            self.aux_head.load_state_dict(state["aux_head"])
        for name in ("world_opt", "actor_opt", "value_opt"):
            if name in state:
                optimizer = getattr(self, name)
                optimizer.load_state_dict(state[name])
                optimizer_to(optimizer, self.device)


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> DreamerV3Agent:
    return DreamerV3Agent(obs_dim, config, device)


PLUGIN = AlgorithmPlugin(
    name="dreamerv3",
    version="1.0.0-vector",
    family="model-based latent imagination",
    build=_build,
    default_export_module="policy",
    replay_mode="trajectory",
    aliases=("dreamer_v3",),
    description="Vector-observation DreamerV3 profile with latent dynamics, symlog scaling, and imagined actor-critic updates.",
    noise_profile="World-model reconstruction, slow targets, and symlog scaling improve numerical stability; model bias remains a distinct risk.",
)
algorithm_registry.register(PLUGIN)
