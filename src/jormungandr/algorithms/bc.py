"""Behavior-cloning learner plugin."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AlgorithmPlugin
from .offline import OfflinePolicyAgent
from .registry import algorithm_registry


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> OfflinePolicyAgent:
    return OfflinePolicyAgent(obs_dim, config, device, objective="bc")


PLUGIN = AlgorithmPlugin(
    name="bc",
    version="1.0.0",
    family="offline imitation",
    build=_build,
    default_export_module="policy",
    aliases=("behavior_cloning",),
    description="Categorical behavior cloning from demonstration actions.",
    noise_profile="Supervised imitation is stable to reward noise because it ignores rewards, but remains vulnerable to noisy labels and covariate shift.",
)
algorithm_registry.register(PLUGIN)

