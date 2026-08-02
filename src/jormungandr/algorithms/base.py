"""Stable contracts shared by Jörmungandr learner plugins.

The service deliberately exchanges NumPy arrays and plain mappings at this
boundary.  A built-in plugin may use PyTorch, while a future native plugin can
implement the same contract through a Python extension without changing the
wire protocol or checkpoint envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import numpy as np
import torch


TransitionBatch = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


@dataclass(frozen=True)
class ActionResult:
    """One action plus the behavior-policy data needed by async learners."""

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
    """Runtime-facing portion of an algorithm implementation."""

    device: torch.device
    action_values: Sequence[float]
    last_metrics: Mapping[str, float]

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


BuildAgent = Callable[[int, Mapping[str, Any], str], LearnerAgent]


@dataclass(frozen=True)
class AlgorithmPlugin:
    """Description and factory for one independently replaceable learner."""

    name: str
    version: str
    family: str
    build: BuildAgent
    default_export_module: str
    replay_mode: str = "transition"
    enforce_policy_lag: bool = False
    aliases: tuple[str, ...] = ()
    description: str = ""
    noise_profile: str = ""
    backend: str = "python-torch"
    runtime_defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_id(self) -> str:
        return f"{self.name}@{self.version}"


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
