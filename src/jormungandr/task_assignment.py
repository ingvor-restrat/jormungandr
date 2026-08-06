"""Domain-neutral, task-first worker assignment.

The solver treats the task set as primary: a worker may receive at most one
task, while each task has a caller-declared positive capacity that defaults to
one.  It first maximizes the number of assignments and then maximizes supplied
assignment utility.  A
``None`` utility means that the caller has no ranking for that eligible edge;
seeded randomized edge order resolves such ties without broadcasting the same
task to every worker.

Alternatively, callers with a broad optional pool may maximize total declared
utility while treating an unassigned worker or task as zero utility. This
mode never adds a zero- or negative-improvement augmenting path.

This module deliberately does not invent tasks, distances, deadlines, or
values. Those remain application-owned inputs. The implementation is an exact
minimum-cost flow solver over the declared bipartite graph and has no SciPy
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Sequence


TASK_ASSIGNMENT_SCHEMA = "jormungandr.task_assignment.v1"
TASK_ASSIGNMENT_OBJECTIVES = (
    "cardinality_then_utility",
    "maximize_total_utility",
)


@dataclass(frozen=True)
class TaskAssignmentCandidate:
    """One eligible worker/task edge and its optional ranking utility."""

    worker_id: str
    task_id: str
    utility: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.worker_id.strip() or not self.task_id.strip():
            raise ValueError("worker_id and task_id cannot be empty")
        if self.utility is not None and not math.isfinite(float(self.utility)):
            raise ValueError("assignment utility must be finite when supplied")


@dataclass(frozen=True)
class TaskAssignmentChoice:
    worker_id: str
    task_id: str
    utility: float
    ranked: bool
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TaskAssignmentResult:
    choices: tuple[TaskAssignmentChoice, ...]
    unassigned_workers: tuple[str, ...]
    unassigned_tasks: tuple[str, ...]
    total_utility: float
    candidate_count: int
    tie_break_seed: int
    maximum_assignments: int
    objective_mode: str
    task_capacities: Mapping[str, int] = field(default_factory=dict)
    remaining_task_capacities: Mapping[str, int] = field(default_factory=dict)

    @property
    def cardinality(self) -> int:
        return len(self.choices)

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "schema": TASK_ASSIGNMENT_SCHEMA,
            "method": "exact_bipartite_min_cost_max_flow",
            "objective": (
                {
                    "mode": self.objective_mode,
                    "primary": "maximize_assignment_cardinality",
                    "secondary": "maximize_total_declared_utility",
                    "unranked_utility": 0.0,
                }
                if self.objective_mode == "cardinality_then_utility"
                else {
                    "mode": self.objective_mode,
                    "primary": "maximize_total_declared_utility",
                    "unassigned_utility": 0.0,
                    "zero_improvement_assignment": "omit",
                    "unranked_utility": 0.0,
                }
            ),
            "tie_break": {
                "method": "seeded_random_edge_order",
                "seed": int(self.tie_break_seed),
            },
            "cardinality": self.cardinality,
            "maximum_assignments": int(self.maximum_assignments),
            "candidate_count": int(self.candidate_count),
            "total_utility": float(self.total_utility),
            "task_capacities": {
                str(key): int(value)
                for key, value in self.task_capacities.items()
            },
            "remaining_task_capacities": {
                str(key): int(value)
                for key, value in self.remaining_task_capacities.items()
            },
            "choices": [
                {
                    "worker_id": choice.worker_id,
                    "task_id": choice.task_id,
                    "utility": float(choice.utility),
                    "ranked": bool(choice.ranked),
                    "metadata": choice.metadata,
                }
                for choice in self.choices
            ],
            "unassigned_workers": list(self.unassigned_workers),
            "unassigned_tasks": list(self.unassigned_tasks),
        }


@dataclass
class _ResidualEdge:
    target: int
    reverse: int
    capacity: int
    cost: float


def _add_residual_edge(
    graph: list[list[_ResidualEdge]],
    source: int,
    target: int,
    cost: float,
    *,
    capacity: int = 1,
) -> int:
    if int(capacity) <= 0:
        raise ValueError("residual edge capacity must be positive")
    forward_index = len(graph[source])
    reverse_index = len(graph[target])
    graph[source].append(
        _ResidualEdge(
            target=target,
            reverse=reverse_index,
            capacity=int(capacity),
            cost=float(cost),
        )
    )
    graph[target].append(
        _ResidualEdge(
            target=source,
            reverse=forward_index,
            capacity=0,
            cost=-float(cost),
        )
    )
    return forward_index


def solve_task_assignment(
    workers: Sequence[str],
    tasks: Sequence[str],
    candidates: Sequence[TaskAssignmentCandidate],
    *,
    tie_break_seed: int = 0,
    maximum_assignments: int | None = None,
    objective_mode: str = "cardinality_then_utility",
    task_capacities: Mapping[str, int] | None = None,
) -> TaskAssignmentResult:
    """Solve the declared task-first assignment exactly.

    Each task capacity defaults to one. A larger declared capacity permits
    several distinct workers to choose the same logical task without copying
    that task into artificial slots at the application boundary.

    In ``cardinality_then_utility`` mode, assignments are assumed worth
    attempting: cardinality is maximized even when an edge has negative utility. In
    ``maximize_total_utility`` mode, leaving a worker and task unassigned is a
    zero-utility alternative, so only a strictly improving augmenting path is
    committed. Both modes use exact residual-path optimization.
    """

    worker_ids = tuple(str(value) for value in workers)
    task_ids = tuple(str(value) for value in tasks)
    if any(not value.strip() for value in worker_ids):
        raise ValueError("worker ids cannot be empty")
    if any(not value.strip() for value in task_ids):
        raise ValueError("task ids cannot be empty")
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError("worker ids must be unique")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task ids must be unique")
    declared_capacities = {
        str(key): int(value)
        for key, value in dict(task_capacities or {}).items()
    }
    unknown_capacity_tasks = set(declared_capacities).difference(task_ids)
    if unknown_capacity_tasks:
        raise ValueError(
            "task capacities refer to unknown tasks: "
            + ", ".join(sorted(unknown_capacity_tasks))
        )
    if any(value <= 0 for value in declared_capacities.values()):
        raise ValueError("task capacities must be positive")
    capacities = {
        task_id: int(declared_capacities.get(task_id, 1))
        for task_id in task_ids
    }
    mode = str(objective_mode)
    if mode not in TASK_ASSIGNMENT_OBJECTIVES:
        raise ValueError(
            "objective_mode must be cardinality_then_utility or "
            "maximize_total_utility"
        )

    worker_index = {value: index for index, value in enumerate(worker_ids)}
    task_index = {value: index for index, value in enumerate(task_ids)}
    records = tuple(candidates)
    pairs: set[tuple[str, str]] = set()
    for candidate in records:
        if candidate.worker_id not in worker_index:
            raise ValueError(
                f"assignment candidate refers to unknown worker "
                f"{candidate.worker_id!r}"
            )
        if candidate.task_id not in task_index:
            raise ValueError(
                f"assignment candidate refers to unknown task "
                f"{candidate.task_id!r}"
            )
        pair = (candidate.worker_id, candidate.task_id)
        if pair in pairs:
            raise ValueError(f"duplicate assignment candidate {pair!r}")
        pairs.add(pair)

    natural_limit = min(len(worker_ids), sum(capacities.values()))
    limit = natural_limit if maximum_assignments is None else int(maximum_assignments)
    if not 0 <= limit <= natural_limit:
        raise ValueError(
            "maximum_assignments must be between zero and min(workers, tasks)"
        )
    if not worker_ids or not task_ids or not records or limit == 0:
        return TaskAssignmentResult(
            choices=(),
            unassigned_workers=worker_ids,
            unassigned_tasks=task_ids,
            total_utility=0.0,
            candidate_count=len(records),
            tie_break_seed=int(tie_break_seed),
            maximum_assignments=limit,
            objective_mode=mode,
            task_capacities=capacities,
            remaining_task_capacities=capacities,
        )

    source = 0
    worker_offset = 1
    task_offset = worker_offset + len(worker_ids)
    sink = task_offset + len(task_ids)
    graph: list[list[_ResidualEdge]] = [[] for _ in range(sink + 1)]
    for index in range(len(worker_ids)):
        _add_residual_edge(graph, source, worker_offset + index, 0.0)
    for index in range(len(task_ids)):
        _add_residual_edge(
            graph,
            task_offset + index,
            sink,
            0.0,
            capacity=capacities[task_ids[index]],
        )

    rng = random.Random(int(tie_break_seed))
    randomized_records = list(enumerate(records))
    rng.shuffle(randomized_records)
    candidate_edges: list[
        tuple[int, int, TaskAssignmentCandidate]
    ] = []
    for _, candidate in randomized_records:
        node = worker_offset + worker_index[candidate.worker_id]
        edge_index = _add_residual_edge(
            graph,
            node,
            task_offset + task_index[candidate.task_id],
            -float(candidate.utility or 0.0),
        )
        candidate_edges.append((node, edge_index, candidate))

    node_order = list(range(len(graph)))
    # Source first gives one-pass propagation on the common acyclic case while
    # retaining seeded ordering among equal residual alternatives.
    middle = node_order[1:]
    rng.shuffle(middle)
    node_order = [source, *middle]
    flow = 0
    tolerance = 1e-12
    while flow < limit:
        distance = [float("inf")] * len(graph)
        predecessor: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0.0
        for _ in range(len(graph) - 1):
            changed = False
            for node in node_order:
                if not math.isfinite(distance[node]):
                    continue
                for edge_index, edge in enumerate(graph[node]):
                    if edge.capacity <= 0:
                        continue
                    proposed = distance[node] + edge.cost
                    if proposed < distance[edge.target] - tolerance:
                        distance[edge.target] = proposed
                        predecessor[edge.target] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if predecessor[sink] is None:
            break
        if mode == "maximize_total_utility" and distance[sink] >= -tolerance:
            break

        cursor = sink
        while cursor != source:
            previous = predecessor[cursor]
            if previous is None:  # pragma: no cover - guarded by sink path
                raise RuntimeError("assignment residual path is incomplete")
            node, edge_index = previous
            edge = graph[node][edge_index]
            edge.capacity -= 1
            graph[cursor][edge.reverse].capacity += 1
            cursor = node
        flow += 1

    selected: list[TaskAssignmentCandidate] = []
    for node, edge_index, candidate in candidate_edges:
        if graph[node][edge_index].capacity == 0:
            selected.append(candidate)
    selected.sort(
        key=lambda candidate: (
            task_index[candidate.task_id],
            worker_index[candidate.worker_id],
        )
    )
    choices = tuple(
        TaskAssignmentChoice(
            worker_id=candidate.worker_id,
            task_id=candidate.task_id,
            utility=float(candidate.utility or 0.0),
            ranked=candidate.utility is not None,
            metadata=candidate.metadata,
        )
        for candidate in selected
    )
    assigned_workers = {choice.worker_id for choice in choices}
    assigned_task_counts = {
        task_id: sum(choice.task_id == task_id for choice in choices)
        for task_id in task_ids
    }
    return TaskAssignmentResult(
        choices=choices,
        unassigned_workers=tuple(
            worker for worker in worker_ids if worker not in assigned_workers
        ),
        unassigned_tasks=tuple(
            task for task in task_ids if assigned_task_counts[task] == 0
        ),
        total_utility=float(sum(choice.utility for choice in choices)),
        candidate_count=len(records),
        tie_break_seed=int(tie_break_seed),
        maximum_assignments=limit,
        objective_mode=mode,
        task_capacities=capacities,
        remaining_task_capacities={
            task_id: capacities[task_id] - assigned_task_counts[task_id]
            for task_id in task_ids
        },
    )


__all__ = [
    "TASK_ASSIGNMENT_OBJECTIVES",
    "TASK_ASSIGNMENT_SCHEMA",
    "TaskAssignmentCandidate",
    "TaskAssignmentChoice",
    "TaskAssignmentResult",
    "solve_task_assignment",
]
