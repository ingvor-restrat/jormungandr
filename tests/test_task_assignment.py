import pytest

from jormungandr.task_assignment import (
    TASK_ASSIGNMENT_SCHEMA,
    TaskAssignmentCandidate,
    solve_task_assignment,
)


def test_global_assignment_beats_worker_first_greedy_conflict() -> None:
    result = solve_task_assignment(
        ("worker-1", "worker-2"),
        ("task-1", "task-2"),
        (
            TaskAssignmentCandidate("worker-1", "task-1", 10.0),
            TaskAssignmentCandidate("worker-1", "task-2", 9.0),
            TaskAssignmentCandidate("worker-2", "task-1", 8.0),
            TaskAssignmentCandidate("worker-2", "task-2", 0.0),
        ),
    )

    assert {(item.worker_id, item.task_id) for item in result.choices} == {
        ("worker-1", "task-2"),
        ("worker-2", "task-1"),
    }
    assert result.total_utility == 17.0
    assert result.cardinality == 2


def test_task_is_never_broadcast_to_duplicate_workers() -> None:
    result = solve_task_assignment(
        ("a", "b", "c"),
        ("water-once",),
        tuple(
            TaskAssignmentCandidate(worker, "water-once", None)
            for worker in ("a", "b", "c")
        ),
        tie_break_seed=17,
    )

    assert result.cardinality == 1
    assert result.choices[0].task_id == "water-once"
    assert result.choices[0].ranked is False
    assert len(result.unassigned_workers) == 2


def test_declared_capacity_shares_one_logical_task_without_slot_copies() -> None:
    result = solve_task_assignment(
        ("a", "b", "c"),
        ("shared-pickup",),
        (
            TaskAssignmentCandidate("a", "shared-pickup", 10.0),
            TaskAssignmentCandidate("b", "shared-pickup", 8.0),
            TaskAssignmentCandidate("c", "shared-pickup", -1.0),
        ),
        task_capacities={"shared-pickup": 2},
        objective_mode="maximize_total_utility",
    )

    assert {(item.worker_id, item.task_id) for item in result.choices} == {
        ("a", "shared-pickup"),
        ("b", "shared-pickup"),
    }
    assert result.cardinality == 2
    assert result.task_capacities == {"shared-pickup": 2}
    assert result.remaining_task_capacities == {"shared-pickup": 0}
    assert result.to_payload()["task_capacities"] == {"shared-pickup": 2}


def test_unranked_tie_break_is_seeded_and_can_select_different_helpers() -> None:
    def selected(seed: int) -> str:
        result = solve_task_assignment(
            ("a", "b", "c"),
            ("task",),
            tuple(
                TaskAssignmentCandidate(worker, "task")
                for worker in ("a", "b", "c")
            ),
            tie_break_seed=seed,
        )
        return result.choices[0].worker_id

    assert selected(9) == selected(9)
    assert len({selected(seed) for seed in range(20)}) > 1


def test_cardinality_precedes_utility_for_caller_declared_tasks() -> None:
    result = solve_task_assignment(
        ("a", "b"),
        ("required-1", "required-2"),
        (
            TaskAssignmentCandidate("a", "required-1", 5.0),
            TaskAssignmentCandidate("b", "required-2", -100.0),
        ),
    )

    assert result.cardinality == 2
    assert result.total_utility == -95.0


def test_total_utility_mode_leaves_nonpositive_optional_work_unassigned() -> None:
    result = solve_task_assignment(
        ("a", "b", "c"),
        ("valuable", "harmful", "unranked"),
        (
            TaskAssignmentCandidate("a", "valuable", 7.0),
            TaskAssignmentCandidate("b", "harmful", -1.0),
            TaskAssignmentCandidate("c", "unranked", None),
        ),
        objective_mode="maximize_total_utility",
    )

    assert [(item.worker_id, item.task_id) for item in result.choices] == [
        ("a", "valuable")
    ]
    assert result.total_utility == 7.0
    assert result.unassigned_workers == ("b", "c")
    assert result.to_payload()["objective"]["unassigned_utility"] == 0.0


def test_total_utility_mode_can_reassign_through_residual_path() -> None:
    result = solve_task_assignment(
        ("a", "b"),
        ("one", "two"),
        (
            TaskAssignmentCandidate("a", "one", 10.0),
            TaskAssignmentCandidate("a", "two", 9.0),
            TaskAssignmentCandidate("b", "one", 8.0),
            TaskAssignmentCandidate("b", "two", -100.0),
        ),
        objective_mode="maximize_total_utility",
    )

    assert {(item.worker_id, item.task_id) for item in result.choices} == {
        ("a", "two"),
        ("b", "one"),
    }
    assert result.total_utility == 17.0


def test_maximum_assignment_budget_and_payload_contract() -> None:
    result = solve_task_assignment(
        ("a", "b"),
        ("one", "two"),
        (
            TaskAssignmentCandidate("a", "one", 1.0),
            TaskAssignmentCandidate("b", "two", 2.0),
        ),
        maximum_assignments=1,
    )

    assert result.cardinality == 1
    assert result.choices[0].task_id == "two"
    payload = result.to_payload()
    assert payload["schema"] == TASK_ASSIGNMENT_SCHEMA
    assert payload["objective"]["primary"] == "maximize_assignment_cardinality"


def test_assignment_rejects_unknown_and_duplicate_edges() -> None:
    with pytest.raises(ValueError, match="unknown worker"):
        solve_task_assignment(
            ("a",),
            ("task",),
            (TaskAssignmentCandidate("missing", "task", 1.0),),
        )
    with pytest.raises(ValueError, match="duplicate assignment"):
        solve_task_assignment(
            ("a",),
            ("task",),
            (
                TaskAssignmentCandidate("a", "task", 1.0),
                TaskAssignmentCandidate("a", "task", 2.0),
            ),
        )
    with pytest.raises(ValueError, match="objective_mode"):
        solve_task_assignment(
            ("a",),
            ("task",),
            (TaskAssignmentCandidate("a", "task", 1.0),),
            objective_mode="unknown",
        )
    with pytest.raises(ValueError, match="unknown tasks"):
        solve_task_assignment(
            ("a",),
            ("task",),
            (TaskAssignmentCandidate("a", "task", 1.0),),
            task_capacities={"missing": 2},
        )
    with pytest.raises(ValueError, match="capacities must be positive"):
        solve_task_assignment(
            ("a",),
            ("task",),
            (TaskAssignmentCandidate("a", "task", 1.0),),
            task_capacities={"task": 0},
        )
