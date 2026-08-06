"""Exact worker/task assignment with caller-owned shared resource limits.

The ordinary task-assignment solver is a dependency-free min-cost flow for
worker and task capacities.  This optional solver adds arbitrary non-negative
linear resource usage to each eligible edge.  The resulting binary program is
application-neutral but requires SciPy's HiGHS MILP backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .task_assignment import TASK_ASSIGNMENT_OBJECTIVES


CONSTRAINED_TASK_ASSIGNMENT_SCHEMA = "jormungandr.constrained_task_assignment.v2"


@dataclass(frozen=True)
class ConstrainedTaskAssignmentCandidate:
    worker_id: str
    task_id: str
    utility: float
    resources: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] | None = None
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        if not self.worker_id.strip() or not self.task_id.strip():
            raise ValueError("worker_id and task_id cannot be empty")
        if self.candidate_id is not None and not str(self.candidate_id).strip():
            raise ValueError("candidate_id cannot be empty when supplied")
        if not math.isfinite(float(self.utility)):
            raise ValueError("assignment utility must be finite")
        resources = {str(key): float(value) for key, value in self.resources.items()}
        if any(
            not key or not math.isfinite(value) or value < 0.0
            for key, value in resources.items()
        ):
            raise ValueError(
                "resource names must be nonempty and usages finite/non-negative"
            )
        object.__setattr__(self, "resources", resources)
        if self.candidate_id is not None:
            object.__setattr__(self, "candidate_id", str(self.candidate_id).strip())

    @property
    def resolved_candidate_id(self) -> str:
        """Return a stable edge identity, preserving the legacy pair default."""

        return (
            str(self.candidate_id)
            if self.candidate_id is not None
            else f"{self.worker_id}|{self.task_id}"
        )


@dataclass(frozen=True)
class ConstrainedTaskAssignmentChoice:
    """One selected alternative within a capacity-constrained task slot."""

    worker_id: str
    task_id: str
    candidate_id: str
    utility: float
    ranked: bool
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ConstrainedTaskAssignmentResult:
    choices: tuple[ConstrainedTaskAssignmentChoice, ...]
    unassigned_workers: tuple[str, ...]
    unassigned_tasks: tuple[str, ...]
    total_utility: float
    candidate_count: int
    tie_break_seed: int
    maximum_assignments: int
    objective_mode: str
    task_capacities: Mapping[str, int]
    remaining_task_capacities: Mapping[str, int]
    resource_capacities: Mapping[str, float]
    resource_usage: Mapping[str, float]
    solver_status: str

    @property
    def cardinality(self) -> int:
        return len(self.choices)

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(choice.candidate_id for choice in self.choices)

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "schema": CONSTRAINED_TASK_ASSIGNMENT_SCHEMA,
            "method": "scipy_highs_binary_milp",
            "objective_mode": self.objective_mode,
            "cardinality": self.cardinality,
            "maximum_assignments": self.maximum_assignments,
            "candidate_count": self.candidate_count,
            "total_utility": self.total_utility,
            "tie_break_seed": self.tie_break_seed,
            "task_capacities": dict(self.task_capacities),
            "remaining_task_capacities": dict(self.remaining_task_capacities),
            "resource_capacities": dict(self.resource_capacities),
            "resource_usage": dict(self.resource_usage),
            "solver_status": self.solver_status,
            "choices": [
                {
                    "worker_id": choice.worker_id,
                    "task_id": choice.task_id,
                    "candidate_id": choice.candidate_id,
                    "utility": choice.utility,
                    "ranked": choice.ranked,
                    "metadata": choice.metadata,
                }
                for choice in self.choices
            ],
            "unassigned_workers": list(self.unassigned_workers),
            "unassigned_tasks": list(self.unassigned_tasks),
        }


def solve_resource_constrained_task_assignment(
    workers: Sequence[str],
    tasks: Sequence[str],
    candidates: Sequence[ConstrainedTaskAssignmentCandidate],
    *,
    task_capacities: Mapping[str, int] | None = None,
    resource_capacities: Mapping[str, float] | None = None,
    maximum_assignments: int | None = None,
    objective_mode: str = "maximize_total_utility",
    tie_break_seed: int = 0,
) -> ConstrainedTaskAssignmentResult:
    """Solve binary assignment under task and additive resource capacities."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "resource-constrained assignment requires the 'optimization' extra "
            "(SciPy >= 1.11)"
        ) from exc

    worker_ids = tuple(str(value) for value in workers)
    task_ids = tuple(str(value) for value in tasks)
    if any(not value for value in worker_ids + task_ids):
        raise ValueError("worker and task identifiers cannot be empty")
    if len(set(worker_ids)) != len(worker_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("worker and task identifiers must be unique")
    worker_set = set(worker_ids)
    task_set = set(task_ids)
    capacities = {
        task_id: int(dict(task_capacities or {}).get(task_id, 1))
        for task_id in task_ids
    }
    unknown_tasks = set(dict(task_capacities or {})).difference(task_set)
    if unknown_tasks:
        raise ValueError("task capacities refer to unknown tasks")
    if any(value <= 0 for value in capacities.values()):
        raise ValueError("task capacities must be positive")
    resources = {
        str(key): float(value)
        for key, value in dict(resource_capacities or {}).items()
    }
    if any(
        not key or not math.isfinite(value) or value < 0.0
        for key, value in resources.items()
    ):
        raise ValueError(
            "resource names must be nonempty and capacities finite/non-negative"
        )
    mode = str(objective_mode)
    if mode not in TASK_ASSIGNMENT_OBJECTIVES:
        raise ValueError("unknown constrained assignment objective mode")

    records = tuple(candidates)
    candidate_ids: set[str] = set()
    for candidate in records:
        if candidate.worker_id not in worker_set or candidate.task_id not in task_set:
            raise ValueError("assignment candidate refers to an unknown worker or task")
        if not set(candidate.resources) <= set(resources):
            raise ValueError("assignment candidate uses an undeclared resource")
        candidate_id = candidate.resolved_candidate_id
        if candidate_id in candidate_ids:
            raise ValueError("duplicate constrained assignment candidate_id")
        candidate_ids.add(candidate_id)

    natural_limit = min(len(worker_ids), sum(capacities.values()))
    limit = natural_limit if maximum_assignments is None else int(maximum_assignments)
    if not 0 <= limit <= natural_limit:
        raise ValueError("maximum_assignments is outside the natural assignment limit")
    active = tuple(
        candidate
        for candidate in records
        if mode == "cardinality_then_utility" or candidate.utility > 1e-12
    )
    if not active or limit == 0:
        return ConstrainedTaskAssignmentResult(
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
            resource_capacities=resources,
            resource_usage={key: 0.0 for key in resources},
            solver_status="empty_optimum",
        )

    order = list(range(len(active)))
    random.Random(int(tie_break_seed)).shuffle(order)
    variables = tuple(active[index] for index in order)
    row_names = (
        tuple(("worker", value) for value in worker_ids)
        + tuple(("task", value) for value in task_ids)
        + tuple(("resource", value) for value in sorted(resources))
        + (("assignment_limit", "all"),)
    )
    row_index = {value: index for index, value in enumerate(row_names)}
    lower = np.full(len(row_names), -np.inf, dtype=np.float64)
    upper = np.zeros(len(row_names), dtype=np.float64)
    for worker_id in worker_ids:
        upper[row_index[("worker", worker_id)]] = 1.0
    for task_id in task_ids:
        upper[row_index[("task", task_id)]] = float(capacities[task_id])
    for resource, capacity in resources.items():
        upper[row_index[("resource", resource)]] = capacity
    upper[row_index[("assignment_limit", "all")]] = float(limit)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for column, candidate in enumerate(variables):
        for row in (
            row_index[("worker", candidate.worker_id)],
            row_index[("task", candidate.task_id)],
            row_index[("assignment_limit", "all")],
        ):
            rows.append(row)
            columns.append(column)
            values.append(1.0)
        for resource, usage in candidate.resources.items():
            if usage:
                rows.append(row_index[("resource", resource)])
                columns.append(column)
                values.append(float(usage))
    matrix = csr_matrix(
        (values, (rows, columns)), shape=(len(row_names), len(variables))
    )
    utilities = np.asarray(
        [float(candidate.utility) for candidate in variables], dtype=np.float64
    )
    if mode == "cardinality_then_utility":
        # Across any two feasible subsets, the absolute utility difference is
        # at most twice the sum of absolute edge utilities. One extra selected
        # edge therefore dominates every possible secondary-utility change.
        cardinality_scale = 2.0 * float(np.abs(utilities).sum()) + 1.0
        objective = -(cardinality_scale + utilities)
    else:
        objective = -utilities
    result = milp(
        c=objective,
        integrality=np.ones(len(variables), dtype=np.int8),
        bounds=Bounds(
            np.zeros(len(variables), dtype=np.float64),
            np.ones(len(variables), dtype=np.float64),
        ),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True},
    )
    if not bool(result.success) or result.x is None:
        raise RuntimeError(
            f"resource-constrained assignment failed: status={result.status} "
            f"message={result.message}"
        )
    selected = tuple(
        candidate
        for candidate, value in zip(variables, result.x, strict=True)
        if float(value) >= 0.5
    )
    selected = tuple(
        sorted(
            selected,
            key=lambda value: (
                value.task_id,
                value.worker_id,
                value.resolved_candidate_id,
            ),
        )
    )
    choices = tuple(
        ConstrainedTaskAssignmentChoice(
            worker_id=candidate.worker_id,
            task_id=candidate.task_id,
            candidate_id=candidate.resolved_candidate_id,
            utility=float(candidate.utility),
            ranked=True,
            metadata=candidate.metadata,
        )
        for candidate in selected
    )
    assigned_workers = {choice.worker_id for choice in choices}
    task_usage = {task_id: 0 for task_id in task_ids}
    resource_usage = {resource: 0.0 for resource in resources}
    for candidate in selected:
        task_usage[candidate.task_id] += 1
        for resource, usage in candidate.resources.items():
            resource_usage[resource] += float(usage)
    return ConstrainedTaskAssignmentResult(
        choices=choices,
        unassigned_workers=tuple(
            worker_id for worker_id in worker_ids if worker_id not in assigned_workers
        ),
        unassigned_tasks=tuple(
            task_id for task_id in task_ids if task_usage[task_id] == 0
        ),
        total_utility=float(sum(choice.utility for choice in choices)),
        candidate_count=len(records),
        tie_break_seed=int(tie_break_seed),
        maximum_assignments=limit,
        objective_mode=mode,
        task_capacities=capacities,
        remaining_task_capacities={
            task_id: capacities[task_id] - task_usage[task_id]
            for task_id in task_ids
        },
        resource_capacities=resources,
        resource_usage=dict(sorted(resource_usage.items())),
        solver_status="optimal",
    )


__all__ = [
    "CONSTRAINED_TASK_ASSIGNMENT_SCHEMA",
    "ConstrainedTaskAssignmentCandidate",
    "ConstrainedTaskAssignmentChoice",
    "ConstrainedTaskAssignmentResult",
    "solve_resource_constrained_task_assignment",
]
