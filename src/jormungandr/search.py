"""Bounded branch search with pluggable frontier pruning.

The search layer is deliberately policy- and environment-neutral.  A caller
supplies cheap utility estimates and embeddings for generated children; a
frontier pruner then decides which nodes deserve the next expensive expansion.
The built-in QUBO pruner reuses the same auditable binary utility/diversity
solver used by replay selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from jormungandr.selectors import QUBORolloutSelector, RolloutCandidate


FRONTIER_PRUNER_ENTRY_POINT_GROUP = "jormungandr.frontier_pruners"


@dataclass(frozen=True)
class SearchNode:
    """One generated branch that can be retained or discarded.

    ``utility`` must be a cheap estimate available before the expensive branch
    evaluation.  ``embedding`` contains state/path features used to detect
    redundant candidates.  The opaque payload remains owned by the caller.
    """

    key: str
    utility: float
    embedding: np.ndarray
    payload: Any = None


@dataclass(frozen=True)
class FrontierSelection:
    selected: tuple[SearchNode, ...]
    selected_indices: np.ndarray
    decisions: np.ndarray
    metrics: Mapping[str, float] = field(default_factory=dict)
    audit: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeamSearchLevel:
    depth: int
    parent_count: int
    candidate_count: int
    selected_count: int
    selector_time_ms: float
    candidate_keys: tuple[str, ...]
    selected_keys: tuple[str, ...]
    decisions: tuple[int, ...]
    selector_metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BeamSearchResult:
    pruner_name: str
    frontier: tuple[SearchNode, ...]
    levels: tuple[BeamSearchLevel, ...]
    expanded_nodes: int
    generated_candidates: int
    retained_candidates: int
    pruned_candidates: int
    selector_time_ms: float
    search_time_ms: float


class FrontierPruner(Protocol):
    name: str

    def select(
        self, candidates: Sequence[SearchNode], select_count: int
    ) -> FrontierSelection: ...


def _validate_selection_size(candidate_count: int, select_count: int) -> int:
    count = int(select_count)
    if not 0 < count <= int(candidate_count):
        raise ValueError("select_count must be in [1, number of candidates]")
    return count


class UtilityFrontierPruner:
    """The ordinary beam-search baseline: retain the largest utilities."""

    name = "utility"

    def select(
        self, candidates: Sequence[SearchNode], select_count: int
    ) -> FrontierSelection:
        started = time.perf_counter()
        count = _validate_selection_size(len(candidates), select_count)

        def finite_utility(index: int) -> float:
            value = float(candidates[index].utility)
            return value if np.isfinite(value) else -np.inf

        ranked = sorted(
            range(len(candidates)),
            key=lambda index: (-finite_utility(index), candidates[index].key, index),
        )
        chosen = np.asarray(sorted(ranked[:count]), dtype=np.int64)
        decisions = np.zeros(len(candidates), dtype=np.int8)
        decisions[chosen] = 1
        elapsed = (time.perf_counter() - started) * 1000.0
        return FrontierSelection(
            selected=tuple(candidates[int(index)] for index in chosen.tolist()),
            selected_indices=chosen,
            decisions=decisions,
            metrics={
                "selector_candidate_count": float(len(candidates)),
                "selector_selected_count": float(count),
                "selector_time_ms": elapsed,
            },
            audit={
                "candidate_keys": [candidate.key for candidate in candidates],
                "decisions": [int(value) for value in decisions.tolist()],
            },
        )


class QUBOFrontierPruner:
    """Retain a high-utility, non-redundant branch subset through QUBO."""

    name = "qubo"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(config or {})
        self.solver = QUBORolloutSelector(
            utility_weight=float(values.get("qubo_utility_weight", 1.0)),
            diversity_weight=float(values.get("qubo_diversity_weight", 0.35)),
            cardinality_penalty=float(
                values.get("qubo_cardinality_penalty", 4.0)
            ),
            local_search_passes=int(values.get("qubo_local_search_passes", 8)),
        )

    def select(
        self, candidates: Sequence[SearchNode], select_count: int
    ) -> FrontierSelection:
        started = time.perf_counter()
        count = _validate_selection_size(len(candidates), select_count)
        result = self.solver.select(
            [
                RolloutCandidate(
                    key=candidate.key,
                    utility=float(candidate.utility),
                    embedding=np.asarray(candidate.embedding, dtype=np.float64),
                    payload=candidate,
                )
                for candidate in candidates
            ],
            count,
        )
        chosen = result.selected_indices
        elapsed = (time.perf_counter() - started) * 1000.0
        return FrontierSelection(
            selected=tuple(candidates[int(index)] for index in chosen.tolist()),
            selected_indices=chosen,
            decisions=result.decisions,
            metrics={
                "selector_candidate_count": float(len(candidates)),
                "selector_selected_count": float(count),
                "selector_qubo_energy": float(result.energy),
                "selector_utility_sum": float(result.utility_sum),
                "selector_redundancy": float(result.redundancy),
                "selector_solve_time_ms": float(result.solve_time_ms),
                "selector_time_ms": elapsed,
                "selector_qubo_matrix_bytes": float(result.qubo.nbytes),
            },
            audit={
                "candidate_keys": [candidate.key for candidate in candidates],
                "decisions": [int(value) for value in result.decisions.tolist()],
            },
        )


def build_frontier_pruner(
    name: str, config: Mapping[str, Any] | None = None
) -> FrontierPruner:
    """Construct a built-in or entry-point frontier-pruner plugin."""

    normalized = str(name or "utility").strip().lower().replace("-", "_")
    values = dict(config or {})
    if normalized in {"", "utility", "top_k", "beam"}:
        return UtilityFrontierPruner()
    if normalized == "qubo":
        return QUBOFrontierPruner(values)
    try:
        discovered = importlib_metadata.entry_points()
        entries = (
            discovered.select(
                group=FRONTIER_PRUNER_ENTRY_POINT_GROUP, name=normalized
            )
            if hasattr(discovered, "select")
            else [
                item
                for item in discovered.get(FRONTIER_PRUNER_ENTRY_POINT_GROUP, ())
                if item.name == normalized
            ]
        )
        for entry in entries:
            factory = entry.load()
            pruner = factory(values)
            if hasattr(pruner, "select"):
                return pruner
    except Exception as exc:
        raise ValueError(f"could not load frontier pruner {name}: {exc}") from exc
    raise ValueError(f"unsupported frontier pruner: {name}")


def bounded_beam_search(
    root: SearchNode,
    expand: Callable[[SearchNode], Sequence[SearchNode]],
    *,
    beam_width: int,
    max_depth: int,
    pruner: FrontierPruner,
) -> BeamSearchResult:
    """Expand and prune a branch frontier while accounting for all work.

    The function does not evaluate terminal payloads.  Consequently callers
    can measure the expensive evaluation of the returned frontier separately
    from cheap child generation and binary selection overhead.
    """

    width = int(beam_width)
    depth_limit = int(max_depth)
    if width < 1:
        raise ValueError("beam_width must be positive")
    if depth_limit < 1:
        raise ValueError("max_depth must be positive")

    started = time.perf_counter()
    frontier = (root,)
    levels: list[BeamSearchLevel] = []
    expanded_nodes = 0
    generated_candidates = 0
    retained_candidates = 0
    selector_time_ms = 0.0

    for depth in range(1, depth_limit + 1):
        parents = frontier
        expanded_nodes += len(parents)
        candidates = tuple(
            child for parent in parents for child in tuple(expand(parent))
        )
        if not candidates:
            break
        keys = [candidate.key for candidate in candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("frontier candidate keys must be unique")

        generated_candidates += len(candidates)
        count = min(width, len(candidates))
        selection = pruner.select(candidates, count)
        if len(selection.selected) != count:
            raise ValueError(
                f"frontier pruner returned {len(selection.selected)} nodes; expected {count}"
            )
        decisions = np.asarray(selection.decisions, dtype=np.int8).reshape(-1)
        if decisions.size != len(candidates) or int(decisions.sum()) != count:
            raise ValueError(
                "frontier pruner decisions must align with candidates and select exactly "
                f"{count} nodes"
            )
        frontier = selection.selected
        retained_candidates += len(frontier)
        level_selector_ms = float(selection.metrics.get("selector_time_ms", 0.0))
        selector_time_ms += level_selector_ms
        levels.append(
            BeamSearchLevel(
                depth=depth,
                parent_count=len(parents),
                candidate_count=len(candidates),
                selected_count=len(frontier),
                selector_time_ms=level_selector_ms,
                candidate_keys=tuple(keys),
                selected_keys=tuple(candidate.key for candidate in frontier),
                decisions=tuple(int(value) for value in decisions.tolist()),
                selector_metrics=dict(selection.metrics),
            )
        )

    return BeamSearchResult(
        pruner_name=str(pruner.name),
        frontier=frontier,
        levels=tuple(levels),
        expanded_nodes=expanded_nodes,
        generated_candidates=generated_candidates,
        retained_candidates=retained_candidates,
        pruned_candidates=generated_candidates - retained_candidates,
        selector_time_ms=selector_time_ms,
        search_time_ms=(time.perf_counter() - started) * 1000.0,
    )
