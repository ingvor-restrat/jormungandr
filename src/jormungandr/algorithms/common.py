"""Small neural-network and numerical helpers used by built-in plugins."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn


def cfg(config: Mapping[str, Any], key: str, default: Any) -> Any:
    plugin_cfg = config.get("plugin_config")
    if isinstance(plugin_cfg, Mapping) and key in plugin_cfg:
        return plugin_cfg[key]
    return config.get(key, default)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, *, layers: int = 2):
        super().__init__()
        modules: list[nn.Module] = []
        width = int(hidden)
        previous = int(in_dim)
        for _ in range(max(1, int(layers))):
            modules.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        modules.append(nn.Linear(previous, int(out_dim)))
        self.net = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, actions: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(obs)
        value = self.value(latent)
        advantage = self.advantage(latent)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def as_tensors(batch: tuple[np.ndarray, ...], device: torch.device):
    obs, actions, rewards, next_obs, dones = batch
    return (
        torch.as_tensor(obs, dtype=torch.float32, device=device),
        torch.as_tensor(actions, dtype=torch.long, device=device).reshape(-1),
        torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1),
        torch.as_tensor(next_obs, dtype=torch.float32, device=device),
        torch.as_tensor(dones, dtype=torch.float32, device=device).reshape(-1),
    )


def legal_mask_from_metadata(
    metadata: Optional[Sequence[Mapping[str, Any]]],
    *,
    key: str,
    rows: int,
    actions: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not metadata:
        return None
    parsed: list[list[bool]] = []
    any_present = False
    for item in metadata:
        raw = item.get(key)
        if raw is None:
            parsed.append([True] * actions)
            continue
        any_present = True
        mask = [bool(x) for x in raw]
        if len(mask) != actions or not any(mask):
            raise ValueError(f"{key} must contain {actions} flags and admit an action")
        parsed.append(mask)
    if not any_present:
        return None
    if len(parsed) != rows:
        raise ValueError("metadata does not align with the sampled batch")
    return torch.as_tensor(parsed, dtype=torch.bool, device=device)


def mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits
    if mask.shape != logits.shape or not torch.all(mask.any(dim=-1)):
        raise ValueError("legal action mask must align with logits and admit an action")
    return logits.masked_fill(~mask, -torch.inf)


def clip_rewards(rewards: torch.Tensor, limit: float) -> torch.Tensor:
    return rewards.clamp(-limit, limit) if limit > 0.0 else rewards


def noisy_observations(obs: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        return obs
    return obs + torch.randn_like(obs) * float(std)


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.expm1(torch.abs(value).clamp(max=20.0))


class AuxiliaryMixin:
    """Shared optional supervised head and metrics."""

    aux_head: Optional[nn.Module]
    aux_classes: int
    last_aux_loss: Optional[float]
    last_aux_acc: Optional[float]

    def _init_aux(self, obs_dim: int, hidden: int, config: Mapping[str, Any]) -> None:
        enabled = bool(cfg(config, "aux_enabled", False))
        self.aux_classes = int(cfg(config, "aux_classes", 3)) if enabled else 0
        aux_hidden = int(cfg(config, "aux_hidden", 0)) or int(hidden)
        self.aux_head = MLP(obs_dim, self.aux_classes, aux_hidden) if self.aux_classes else None
        if self.aux_head is not None:
            self.aux_head.to(self.device)
        self.last_aux_loss = None
        self.last_aux_acc = None

    def _aux_loss(
        self,
        aux_obs: Optional[np.ndarray],
        aux_targets: Optional[np.ndarray],
        weight: float,
    ) -> Optional[torch.Tensor]:
        self.last_aux_loss = None
        self.last_aux_acc = None
        if self.aux_head is None or aux_obs is None or aux_targets is None or len(aux_obs) == 0:
            return None
        obs = torch.as_tensor(aux_obs, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(aux_targets, dtype=torch.long, device=self.device).reshape(-1)
        logits = self.aux_head(obs)
        loss = nn.functional.cross_entropy(logits, target)
        with torch.no_grad():
            self.last_aux_loss = float(loss.detach().cpu())
            self.last_aux_acc = float((logits.argmax(dim=-1) == target).float().mean().cpu())
        return float(max(0.0, weight)) * loss


def action_values_tensor(values: Sequence[float], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(list(values), dtype=torch.float32, device=device)

