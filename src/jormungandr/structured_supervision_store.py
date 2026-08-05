"""Bounded split-local storage for structured supervision examples."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Sequence

import numpy as np

from jormungandr.structured_supervision import StructuredSupervisionExample


class StructuredSupervisionBuffer:
    def __init__(self, capacity: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("supervision capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[StructuredSupervisionExample] = deque()
        self._keys: set[tuple[str, str, int, str, str]] = set()

    @staticmethod
    def _key(item: StructuredSupervisionExample):
        return (
            item.actor_id,
            item.episode_id,
            item.timestep,
            item.factor_id,
            item.split,
        )

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: StructuredSupervisionExample) -> bool:
        key = self._key(item)
        if key in self._keys:
            raise ValueError("duplicate structured supervision example")
        evicted = False
        if len(self._items) >= self.capacity:
            removed = self._items.popleft()
            self._keys.remove(self._key(removed))
            evicted = True
        self._items.append(item)
        self._keys.add(key)
        return evicted

    def sample(
        self,
        count: int,
        *,
        rng: np.random.Generator,
        strategy: str = "uniform",
    ) -> tuple[StructuredSupervisionExample, ...]:
        """Draw one optimization batch.

        ``uniform`` preserves the historical behavior: draw records uniformly
        and leave their declared loss weights intact.  ``sample_weight`` draws
        with replacement in proportion to those weights and returns unit-loss
        copies.  For any per-record loss ``L_i`` this makes the batch mean an
        unbiased estimator of ``sum(w_i L_i) / sum(w_i)`` without applying the
        weight twice.  It also places rare, highly weighted strata into batches
        more consistently than uniform sampling followed by loss weighting.
        """

        if not self._items:
            raise ValueError("cannot sample an empty supervision buffer")
        size = max(1, int(count))
        items = tuple(self._items)
        mode = str(strategy).strip().lower()
        if mode == "uniform":
            indices = rng.choice(
                len(items), size=size, replace=len(items) < size
            )
            return tuple(
                items[int(index)] for index in np.asarray(indices).reshape(-1)
            )
        if mode != "sample_weight":
            raise ValueError(
                "supervision sampling strategy must be uniform or sample_weight"
            )
        weights = np.asarray(
            [item.sample_weight for item in items], dtype=np.float64
        )
        probabilities = weights / float(weights.sum())
        indices = rng.choice(
            len(items), size=size, replace=True, p=probabilities
        )
        return tuple(
            replace(items[int(index)], sample_weight=1.0)
            for index in np.asarray(indices).reshape(-1)
        )

    def snapshot(self) -> tuple[StructuredSupervisionExample, ...]:
        return tuple(self._items)
