"""IMPALA learner plugin."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AlgorithmPlugin
from .policy_gradient import DistributedPolicyAgent
from .registry import algorithm_registry


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> DistributedPolicyAgent:
    return DistributedPolicyAgent(obs_dim, config, device, objective="impala")


PLUGIN = AlgorithmPlugin(
    name="impala",
    version="1.0.0",
    family="asynchronous actor critic",
    build=_build,
    default_export_module="policy",
    replay_mode="trajectory",
    enforce_policy_lag=True,
    description="Decoupled actor-learner updates with clipped V-trace correction.",
    noise_profile="V-trace handles behavior-policy lag, which is distinct from environment or measurement noise.",
)
algorithm_registry.register(PLUGIN)
