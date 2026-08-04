"""Bounded on-policy storage for complete structured trajectories."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from jormungandr.structured_trajectory import (
    StructuredJointTrajectoryStep,
    validate_structured_joint_trajectory,
)


@dataclass(frozen=True)
class StructuredTrajectoryAddResult:
    steps_added: int
    trajectories_added: int
    steps_evicted: int
    trajectories_evicted: int


class StructuredTrajectoryBuffer:
    """FIFO episode buffer bounded by environment steps, not factor count."""

    def __init__(self, capacity_steps: int) -> None:
        if int(capacity_steps) <= 0:
            raise ValueError("trajectory capacity must be positive")
        self.capacity_steps = int(capacity_steps)
        self._trajectories: deque[
            tuple[StructuredJointTrajectoryStep, ...]
        ] = deque()
        self._step_count = 0
        self._episode_keys: set[tuple[str, str, str]] = set()

    def __len__(self) -> int:
        return len(self._trajectories)

    @property
    def step_count(self) -> int:
        return self._step_count

    @staticmethod
    def _key(
        trajectory: Sequence[StructuredJointTrajectoryStep],
    ) -> tuple[str, str, str]:
        first = trajectory[0]
        return first.actor_id, first.episode_id, first.split

    def add(
        self,
        trajectory: Sequence[StructuredJointTrajectoryStep],
    ) -> StructuredTrajectoryAddResult:
        steps = validate_structured_joint_trajectory(trajectory)
        if len(steps) > self.capacity_steps:
            raise ValueError("one trajectory exceeds the trajectory capacity")
        key = self._key(steps)
        if key in self._episode_keys:
            raise ValueError(
                "duplicate actor/episode/split trajectory was already ingested"
            )
        evicted_steps = 0
        evicted_trajectories = 0
        while self._step_count + len(steps) > self.capacity_steps:
            removed = self._trajectories.popleft()
            self._episode_keys.remove(self._key(removed))
            self._step_count -= len(removed)
            evicted_steps += len(removed)
            evicted_trajectories += 1
        self._trajectories.append(steps)
        self._episode_keys.add(key)
        self._step_count += len(steps)
        return StructuredTrajectoryAddResult(
            steps_added=len(steps),
            trajectories_added=1,
            steps_evicted=evicted_steps,
            trajectories_evicted=evicted_trajectories,
        )

    def pop_at_least(
        self,
        minimum_steps: int,
        *,
        maximum_steps: int | None = None,
    ) -> tuple[tuple[StructuredJointTrajectoryStep, ...], ...]:
        minimum = max(1, int(minimum_steps))
        maximum = (
            None if maximum_steps is None else max(minimum, int(maximum_steps))
        )
        if self._step_count < minimum:
            return ()
        selected = []
        selected_steps = 0
        while self._trajectories and selected_steps < minimum:
            candidate = self._trajectories[0]
            if (
                maximum is not None
                and selected
                and selected_steps >= minimum
                and selected_steps + len(candidate) > maximum
            ):
                break
            selected.append(self._trajectories.popleft())
            self._episode_keys.remove(self._key(candidate))
            selected_steps += len(candidate)
            self._step_count -= len(candidate)
        return tuple(selected)

    def snapshot(
        self,
    ) -> tuple[tuple[StructuredJointTrajectoryStep, ...], ...]:
        return tuple(self._trajectories)
