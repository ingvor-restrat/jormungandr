"""Prioritized replay for variable entity/candidate transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from jormungandr.structured import EntityCandidateObservation


@dataclass(frozen=True)
class StructuredReplayTransition:
    """One off-policy transition with actor-owned semantic action identity."""

    observation: EntityCandidateObservation
    candidate_id: str
    reward: float
    next_observation: EntityCandidateObservation
    done: bool
    actor_id: str
    episode_id: str
    timestep: int
    policy_version: int
    split: str = "train"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id is required")
        try:
            candidate_index = self.observation.candidate_ids.index(candidate_id)
        except ValueError as exc:
            raise ValueError(
                "candidate_id is not present in the recorded observation"
            ) from exc
        if not bool(self.observation.legal_action_mask[candidate_index]):
            raise ValueError("candidate_id was illegal in the recorded observation")
        if not str(self.actor_id).strip() or not str(self.episode_id).strip():
            raise ValueError("actor_id and episode_id are required")
        if self.timestep < 0 or self.policy_version < 0:
            raise ValueError("timestep and policy_version must be non-negative")
        if str(self.split) not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        observed = (
            self.observation.global_dim,
            self.observation.entity_dim,
            self.observation.candidate_dim,
        )
        following = (
            self.next_observation.global_dim,
            self.next_observation.entity_dim,
            self.next_observation.candidate_dim,
        )
        if observed != following:
            raise ValueError(
                "observation and next_observation feature dimensions must match"
            )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "actor_id", str(self.actor_id))
        object.__setattr__(self, "episode_id", str(self.episode_id))
        object.__setattr__(self, "split", str(self.split))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def candidate_index(self) -> int:
        return self.observation.candidate_ids.index(self.candidate_id)


class StructuredPrioritizedReplayBuffer:
    """Bounded ring replay without flattening variable observations."""

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.pos = 0
        self.full = False
        self.items: list[StructuredReplayTransition | None] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(
        self,
        transition: StructuredReplayTransition,
        *,
        priority: float | None = None,
    ) -> int:
        if not isinstance(transition, StructuredReplayTransition):
            raise TypeError("transition must be StructuredReplayTransition")
        index = self.pos
        self.items[index] = transition
        resolved_priority = self.max_priority
        if priority is not None:
            candidate = float(priority)
            if math.isfinite(candidate) and candidate > 0.0:
                resolved_priority = candidate
        self.priorities[index] = max(1e-6, float(resolved_priority))
        self.max_priority = max(self.max_priority, float(self.priorities[index]))
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True
        return index

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[tuple[StructuredReplayTransition, ...], np.ndarray, np.ndarray]:
        size = len(self)
        if size == 0:
            raise ValueError("replay buffer is empty")
        count = max(1, min(int(batch_size), size))
        priorities = self.priorities[:size].astype(np.float64)
        probabilities = priorities ** self.alpha
        total = float(probabilities.sum())
        if not math.isfinite(total) or total <= 0.0:
            probabilities.fill(1.0 / size)
        else:
            probabilities /= total
        indices = np.random.choice(size, count, replace=True, p=probabilities)
        weights = (size * probabilities[indices]) ** (-max(0.0, float(beta)))
        weights /= max(float(weights.max()), 1e-12)
        sampled = tuple(self.items[int(index)] for index in indices)
        if any(item is None for item in sampled):
            raise RuntimeError("replay sampled an uninitialized slot")
        return (
            tuple(item for item in sampled if item is not None),
            indices.astype(np.int64),
            weights.astype(np.float32),
        )

    def update_priorities(
        self,
        indices: Sequence[int] | np.ndarray,
        priorities: Sequence[float] | np.ndarray,
    ) -> None:
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        priority_array = np.abs(
            np.asarray(priorities, dtype=np.float32).reshape(-1)
        )
        if index_array.shape != priority_array.shape:
            raise ValueError("indices and priorities must be aligned")
        if index_array.size == 0:
            return
        if np.any(index_array < 0) or np.any(index_array >= len(self)):
            raise IndexError("replay priority index is out of bounds")
        priority_array = np.maximum(priority_array, np.float32(1e-6))
        self.priorities[index_array] = priority_array
        self.max_priority = max(self.max_priority, float(priority_array.max()))
