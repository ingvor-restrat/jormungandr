"""Stable contracts shared by Jörmungandr learner plugins.

The vector service exchanges NumPy arrays and plain mappings at this boundary;
structured plugins additionally consume the generic entity/candidate schema.
A built-in plugin may use PyTorch, while a future native plugin can implement
the same contracts through a Python extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from jormungandr.structured import (
        DynamicActionResult,
        EntityCandidateObservation,
        StructuredPolicySpec,
    )


TransitionBatch = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


@dataclass(frozen=True)
class ActionResult:
    """One fixed-profile action plus async behavior-policy data."""

    action: float
    action_idx: int
    log_probability: Optional[float] = None
    value: Optional[float] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateResult:
    """Normalized result returned by every learner plugin update."""

    loss: float
    priorities: np.ndarray
    metrics: Mapping[str, float] = field(default_factory=dict)


class LearnerAgent(Protocol):
    """Representation-neutral runtime portion of an algorithm implementation."""

    device: torch.device
    last_metrics: Mapping[str, float]

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


BuildAgent = Callable[[int, Mapping[str, Any], str], LearnerAgent]


class StructuredLearnerAgent(LearnerAgent, Protocol):
    """Optional learner interface for variable entities and local candidates."""

    def action_result_structured(
        self,
        observation: "EntityCandidateObservation",
        *,
        deterministic: bool,
    ) -> "DynamicActionResult": ...


BuildStructuredAgent = Callable[
    ["StructuredPolicySpec", Mapping[str, Any], str],
    StructuredLearnerAgent,
]


@dataclass(frozen=True)
class AlgorithmPlugin:
    """Description and factory for one independently replaceable learner."""

    name: str
    version: str
    family: str
    build: Optional[BuildAgent]
    default_export_module: str
    replay_mode: str = "transition"
    enforce_policy_lag: bool = False
    aliases: tuple[str, ...] = ()
    description: str = ""
    noise_profile: str = ""
    backend: str = "python-torch"
    runtime_defaults: Mapping[str, Any] = field(default_factory=dict)
    representation_modes: tuple[str, ...] = ("vector_discrete",)
    build_structured: Optional[BuildStructuredAgent] = None

    @property
    def checkpoint_id(self) -> str:
        return f"{self.name}@{self.version}"

    def supports_representation(self, mode: str) -> bool:
        return str(mode) in self.representation_modes


def normalize_update_result(value: Any) -> UpdateResult:
    """Accept the modern result and the legacy C51 ``(loss, td_error)`` pair."""

    if isinstance(value, UpdateResult):
        priorities = np.asarray(value.priorities, dtype=np.float32).reshape(-1)
        return UpdateResult(
            loss=float(value.loss),
            priorities=priorities,
            metrics={str(k): float(v) for k, v in value.metrics.items()},
        )
    if isinstance(value, tuple) and len(value) == 2:
        loss, priorities = value
        return UpdateResult(
            loss=float(loss),
            priorities=np.asarray(priorities, dtype=np.float32).reshape(-1),
            metrics={"loss": float(loss)},
        )
    raise TypeError("learner update must return UpdateResult or (loss, priorities)")
