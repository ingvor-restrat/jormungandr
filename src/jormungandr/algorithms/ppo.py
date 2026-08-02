"""PPO learner plugin."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AlgorithmPlugin
from .policy_gradient import DistributedPolicyAgent
from .registry import algorithm_registry


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> DistributedPolicyAgent:
    return DistributedPolicyAgent(obs_dim, config, device, objective="ppo")


PLUGIN = AlgorithmPlugin(
    name="ppo",
    version="1.0.0",
    family="on-policy actor critic",
    build=_build,
    default_export_module="policy",
    replay_mode="trajectory",
    enforce_policy_lag=True,
    runtime_defaults={"epochs": 4, "max_policy_lag": 0},
    description="Clipped-surrogate discrete PPO with GAE and multi-epoch minibatches.",
    noise_profile="Clipping limits destructive policy updates; PPO does not itself guarantee robustness to observation noise.",
)
algorithm_registry.register(PLUGIN)
