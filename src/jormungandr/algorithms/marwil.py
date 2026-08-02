"""MARWIL learner plugin."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AlgorithmPlugin
from .offline import OfflinePolicyAgent
from .registry import algorithm_registry


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> OfflinePolicyAgent:
    return OfflinePolicyAgent(obs_dim, config, device, objective="marwil")


PLUGIN = AlgorithmPlugin(
    name="marwil",
    version="1.0.0",
    family="offline advantage-weighted imitation",
    build=_build,
    default_export_module="policy",
    replay_mode="trajectory",
    description="Moving-average normalized, exponentially advantage-weighted imitation.",
    noise_profile="Moving advantage normalization and weight clipping temper scale noise; inaccurate rewards can still bias imitation weights.",
)
algorithm_registry.register(PLUGIN)
