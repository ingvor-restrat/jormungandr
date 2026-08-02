"""APPO/IMPACT learner plugin."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AlgorithmPlugin
from .policy_gradient import DistributedPolicyAgent
from .registry import algorithm_registry


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> DistributedPolicyAgent:
    return DistributedPolicyAgent(obs_dim, config, device, objective="appo")


PLUGIN = AlgorithmPlugin(
    name="appo",
    version="1.0.0",
    family="asynchronous proximal actor critic",
    build=_build,
    default_export_module="policy",
    replay_mode="trajectory",
    enforce_policy_lag=True,
    aliases=("impact",),
    description="IMPACT-style APPO with V-trace, target-policy clipping, and bounded replay reuse.",
    noise_profile="Target-policy and importance clipping stabilize asynchronous policy lag; they do not denoise observations.",
)
algorithm_registry.register(PLUGIN)
