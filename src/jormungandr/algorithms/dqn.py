"""DQN learner plugin."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AlgorithmPlugin
from .registry import algorithm_registry
from .value_based import DiscreteQAgent


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> DiscreteQAgent:
    return DiscreteQAgent(obs_dim, config, device, objective="dqn")


PLUGIN = AlgorithmPlugin(
    name="dqn",
    version="1.0.0",
    family="off-policy value",
    build=_build,
    default_export_module="q",
    description="Double dueling DQN with target network and Huber TD loss.",
    noise_profile="Huber loss and Double-DQN targets reduce sensitivity to reward outliers and value overestimation.",
)
algorithm_registry.register(PLUGIN)

