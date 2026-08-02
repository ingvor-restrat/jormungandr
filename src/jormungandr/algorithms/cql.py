"""Discrete Conservative Q-Learning plugin."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AlgorithmPlugin
from .registry import algorithm_registry
from .value_based import DiscreteQAgent


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> DiscreteQAgent:
    return DiscreteQAgent(obs_dim, config, device, objective="cql")


PLUGIN = AlgorithmPlugin(
    name="cql",
    version="1.0.0",
    family="offline value",
    build=_build,
    default_export_module="q",
    description="Discrete CQL(H) regularization on top of Double DQN.",
    noise_profile="Conservatism addresses offline action-distribution shift; it is not a general sensor-noise guarantee.",
)
algorithm_registry.register(PLUGIN)

