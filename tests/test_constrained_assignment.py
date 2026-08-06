import pytest

from jormungandr.constrained_assignment import (
    ConstrainedTaskAssignmentCandidate,
    solve_resource_constrained_task_assignment,
)


def test_shared_resource_capacity_couples_different_tasks() -> None:
    result = solve_resource_constrained_task_assignment(
        ("w0", "w1"),
        ("tile0", "tile1"),
        (
            ConstrainedTaskAssignmentCandidate(
                "w0", "tile0", 9.0, resources={"seed:wheat": 1}
            ),
            ConstrainedTaskAssignmentCandidate(
                "w1", "tile1", 8.0, resources={"seed:wheat": 1}
            ),
        ),
        resource_capacities={"seed:wheat": 1},
        tie_break_seed=3,
    )

    assert result.cardinality == 1
    assert result.total_utility == 9.0
    assert result.resource_usage == {"seed:wheat": 1.0}


def test_task_and_resource_capacities_are_enforced_together() -> None:
    result = solve_resource_constrained_task_assignment(
        ("w0", "w1", "w2"),
        ("shared",),
        tuple(
            ConstrainedTaskAssignmentCandidate(
                worker, "shared", utility, resources={"stock": 1}
            )
            for worker, utility in (("w0", 3.0), ("w1", 2.0), ("w2", 1.0))
        ),
        task_capacities={"shared": 2},
        resource_capacities={"stock": 3},
    )

    assert result.cardinality == 2
    assert {choice.worker_id for choice in result.choices} == {"w0", "w1"}
    assert result.remaining_task_capacities == {"shared": 0}


def test_optional_objective_omits_nonpositive_edges() -> None:
    result = solve_resource_constrained_task_assignment(
        ("worker",),
        ("task",),
        (ConstrainedTaskAssignmentCandidate("worker", "task", -1.0),),
    )

    assert result.choices == ()
    assert result.total_utility == 0.0


def test_cardinality_objective_dominates_negative_secondary_utility() -> None:
    result = solve_resource_constrained_task_assignment(
        ("w0", "w1"),
        ("t0", "t1"),
        (
            ConstrainedTaskAssignmentCandidate("w0", "t0", -100.0),
            ConstrainedTaskAssignmentCandidate("w1", "t1", -200.0),
        ),
        objective_mode="cardinality_then_utility",
    )

    assert result.cardinality == 2
    assert result.total_utility == -300.0


def test_constrained_assignment_rejects_undeclared_resources() -> None:
    with pytest.raises(ValueError, match="undeclared"):
        solve_resource_constrained_task_assignment(
            ("worker",),
            ("task",),
            (
                ConstrainedTaskAssignmentCandidate(
                    "worker", "task", 1.0, resources={"seed": 1}
                ),
            ),
        )


def test_distinct_alternatives_survive_a_binding_shared_resource_cap() -> None:
    result = solve_resource_constrained_task_assignment(
        ("w0", "w1"),
        ("tile0", "tile1"),
        (
            ConstrainedTaskAssignmentCandidate(
                "w0",
                "tile0",
                10.0,
                resources={"build": 1},
                candidate_id="w0|tile0|build",
            ),
            ConstrainedTaskAssignmentCandidate(
                "w0",
                "tile0",
                9.0,
                resources={"seed": 1},
                candidate_id="w0|tile0|plant",
            ),
            ConstrainedTaskAssignmentCandidate(
                "w1",
                "tile1",
                10.0,
                resources={"build": 1},
                candidate_id="w1|tile1|build",
            ),
            ConstrainedTaskAssignmentCandidate(
                "w1",
                "tile1",
                9.0,
                resources={"seed": 1},
                candidate_id="w1|tile1|plant",
            ),
        ),
        resource_capacities={"build": 1, "seed": 1},
        tie_break_seed=5,
    )

    assert result.cardinality == 2
    assert result.total_utility == 19.0
    assert len(result.selected_candidate_ids) == 2
    assert sum(value.endswith("|build") for value in result.selected_candidate_ids) == 1
    assert sum(value.endswith("|plant") for value in result.selected_candidate_ids) == 1


def test_candidate_ids_must_be_unique() -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        solve_resource_constrained_task_assignment(
            ("w0", "w1"),
            ("t0", "t1"),
            (
                ConstrainedTaskAssignmentCandidate(
                    "w0", "t0", 1.0, candidate_id="same"
                ),
                ConstrainedTaskAssignmentCandidate(
                    "w1", "t1", 1.0, candidate_id="same"
                ),
            ),
        )
