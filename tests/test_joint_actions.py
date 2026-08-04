from jormungandr.joint_actions import (
    JointActionChoice,
    JointActionFactor,
    compose_joint_action,
)


def test_joint_composer_maximizes_utility_under_shared_capacity() -> None:
    factors = (
        JointActionFactor(
            "worker:0",
            (
                JointActionChoice("pass", 0.0),
                JointActionChoice("plant", 5.0, resources={"seed": 1}),
            ),
        ),
        JointActionFactor(
            "worker:1",
            (
                JointActionChoice("pass", 0.0),
                JointActionChoice("plant", 4.0, resources={"seed": 1}),
            ),
        ),
    )

    result = compose_joint_action(
        factors, resource_capacities={"seed": 1}, beam_width=8
    )

    assert [choice.key for choice in result.choices] == ["plant", "pass"]
    assert result.utility == 5.0
    assert result.resource_usage == {"seed": 1.0}
    assert result.audit.exhaustive_combinations == 4
    assert result.audit.infeasible_candidates >= 1


def test_joint_composer_supports_domain_owned_non_linear_constraints() -> None:
    factors = tuple(
        JointActionFactor(
            f"market:{index}",
            (
                JointActionChoice("pass", 0.0),
                JointActionChoice("hire", 3.0),
            ),
        )
        for index in range(3)
    )

    def affordable(choices, _complete):
        # Marginal hire costs are 1, 1, 2 and the caller owns $2.
        costs = (1, 1, 2)
        hires = sum(choice.key == "hire" for choice in choices)
        return sum(costs[:hires]) <= 2

    result = compose_joint_action(factors, feasible=affordable, beam_width=8)

    assert sum(choice.key == "hire" for choice in result.choices) == 2
    assert result.audit.factor_count == 3


def test_joint_composer_is_deterministic_on_ties() -> None:
    factors = (
        JointActionFactor(
            "x",
            (JointActionChoice("b", 1.0), JointActionChoice("a", 1.0)),
        ),
    )

    first = compose_joint_action(factors)
    second = compose_joint_action(factors)

    assert first.audit.selected_choice_keys == second.audit.selected_choice_keys
    assert first.choices[0].key == "a"


def test_joint_composer_can_recover_an_explicit_fallback_pruned_by_a_narrow_beam() -> None:
    factors = (
        JointActionFactor(
            "first",
            (JointActionChoice("pass", 0.0), JointActionChoice("trap", 10.0)),
        ),
        JointActionFactor(
            "second",
            (JointActionChoice("pass", 0.0),),
        ),
    )

    result = compose_joint_action(
        factors,
        beam_width=1,
        feasible=lambda choices, complete: not (
            complete and choices[0].key == "trap"
        ),
        fallback_choice_keys={"first": "pass", "second": "pass"},
    )

    assert [choice.key for choice in result.choices] == ["pass", "pass"]
