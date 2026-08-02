"""Entropy-regularized soft Q-learning plugin."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AlgorithmPlugin
from .registry import algorithm_registry
from .value_based import DiscreteQAgent


def _build(obs_dim: int, config: Mapping[str, Any], device: str) -> DiscreteQAgent:
    return DiscreteQAgent(obs_dim, config, device, objective="maxent")


PLUGIN = AlgorithmPlugin(
    name="maxent",
    version="1.0.0",
    family="off-policy maximum entropy",
    build=_build,
    default_export_module="q",
    aliases=("maximum_entropy", "maximum_entropy_rl", "soft_q"),
    description="Discrete soft Q-learning with a Boltzmann policy and entropy-regularized Bellman backup.",
    noise_profile="Targets the MaxEnt objective linked to specified reward/dynamics perturbation sets; temperature controls the tradeoff.",
)
algorithm_registry.register(PLUGIN)
