from __future__ import annotations

import json
import math

import numpy as np
import pytest

from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_trajectory import (
    StructuredActionFactor,
    StructuredFactorChoice,
    StructuredJointTrajectoryStep,
    sample_structured_joint_action,
    structured_joint_step_from_payload,
    structured_joint_step_to_payload,
    structured_joint_trajectory_from_sequence_payload,
    structured_joint_trajectory_to_sequence_payload,
    validate_structured_joint_trajectory,
)


def _observation(timestep: int, factor_count: int) -> EntityCandidateObservation:
    candidates = tuple(
        candidate
        for factor in range(factor_count)
        for candidate in (f"factor:{factor}:pass", f"factor:{factor}:act")
    )
    return EntityCandidateObservation(
        global_features=np.asarray([timestep, 1.0], dtype=np.float32),
        entity_features=np.asarray([[timestep, 0.0]], dtype=np.float32),
        entity_type_ids=np.asarray([0], dtype=np.int64),
        entity_ids=("entity:0",),
        candidate_features=np.arange(
            len(candidates) * 3, dtype=np.float32
        ).reshape(len(candidates), 3),
        candidate_ids=candidates,
        legal_action_mask=np.ones(len(candidates), dtype=np.bool_),
    )


def _step(
    timestep: int,
    factor_count: int,
    *,
    done: bool,
    reward: float = 0.0,
    next_factor_count: int | None = None,
) -> StructuredJointTrajectoryStep:
    logp = -math.log(2.0)
    factors = tuple(
        StructuredFactorChoice(
            factor_id=f"factor:{factor}",
            candidate_ids=(
                f"factor:{factor}:pass",
                f"factor:{factor}:act",
            ),
            selected_candidate_id=f"factor:{factor}:pass",
            behavior_log_probability=logp,
        )
        for factor in range(factor_count)
    )
    return StructuredJointTrajectoryStep(
        actor_id="actor:0",
        episode_id="episode:0",
        timestep=timestep,
        policy_version=7,
        observation=_observation(timestep, factor_count),
        factors=factors,
        joint_behavior_log_probability=factor_count * logp,
        behavior_value=0.25,
        reward=reward,
        next_observation=_observation(
            timestep + 1,
            factor_count if next_factor_count is None else next_factor_count,
        ),
        terminated=done,
    )


def test_one_environment_turn_is_one_step_independent_of_factor_count() -> None:
    one_factor = [
        _step(index, 1, done=index == 3, reward=1.0 if index == 3 else 0.0)
        for index in range(4)
    ]
    five_factors = [
        _step(index, 5, done=index == 3, reward=1.0 if index == 3 else 0.0)
        for index in range(4)
    ]

    assert len(validate_structured_joint_trajectory(one_factor)) == 4
    assert len(validate_structured_joint_trajectory(five_factors)) == 4
    assert sum(step.reward for step in one_factor) == 1.0
    assert sum(step.reward for step in five_factors) == 1.0


def test_disappearing_factor_does_not_terminate_the_joint_trajectory() -> None:
    first = _step(0, 3, done=False, next_factor_count=1)
    second = _step(1, 1, done=True, reward=-1.0)

    trajectory = validate_structured_joint_trajectory((first, second))

    assert not trajectory[0].done
    assert trajectory[1].done
    assert len(trajectory) == 2


def test_joint_log_probability_must_equal_conditional_sum() -> None:
    step = _step(0, 2, done=True)
    payload = structured_joint_step_to_payload(step)
    payload["joint_behavior_log_probability"] = 0.0

    with pytest.raises(ValueError, match="conditional sum"):
        structured_joint_step_from_payload(payload)


def test_joint_step_payload_round_trip_preserves_semantic_choices() -> None:
    step = _step(0, 3, done=True, reward=1.0)

    restored = structured_joint_step_from_payload(
        structured_joint_step_to_payload(step)
    )

    assert restored.selected_candidate_ids == step.selected_candidate_ids
    assert restored.joint_behavior_log_probability == pytest.approx(
        step.joint_behavior_log_probability
    )
    assert restored.reward == 1.0


def test_compact_sequence_round_trip_stores_observation_chain_once() -> None:
    trajectory = tuple(
        _step(index, 3, done=index == 3, reward=float(index == 3))
        for index in range(4)
    )
    sequence = structured_joint_trajectory_to_sequence_payload(trajectory)
    restored = structured_joint_trajectory_from_sequence_payload(sequence)
    legacy = [structured_joint_step_to_payload(step) for step in trajectory]

    assert len(sequence["observations"]) == len(trajectory) + 1
    assert len(sequence["steps"]) == len(trajectory)
    assert restored[-1].reward == 1.0
    assert restored[1].observation is restored[0].next_observation
    assert len(json.dumps(sequence, separators=(",", ":"))) < len(
        json.dumps(legacy, separators=(",", ":"))
    )


def test_joint_trajectory_rejects_out_of_order_or_inconsistent_steps() -> None:
    first = _step(0, 1, done=False)
    late = _step(2, 1, done=True)

    with pytest.raises(ValueError, match="contiguous"):
        validate_structured_joint_trajectory((first, late))


def test_factor_candidates_must_partition_observation_candidates() -> None:
    step = _step(0, 2, done=True)
    payload = structured_joint_step_to_payload(step)
    payload["factors"][0]["candidate_ids"] = ["factor:0:pass"]

    with pytest.raises(ValueError, match="partition"):
        structured_joint_step_from_payload(payload)


def test_joint_sampler_records_exact_conditional_probabilities_and_masks() -> None:
    observation = _observation(0, 2)
    factors = (
        StructuredActionFactor(
            "factor:0", ("factor:0:pass", "factor:0:act")
        ),
        StructuredActionFactor(
            "factor:1", ("factor:1:pass", "factor:1:act")
        ),
    )

    def restrict_second(index, selected, mask):
        if index == 1 and selected == ("factor:0:act",):
            mask[3] = False
        return mask

    result = sample_structured_joint_action(
        observation,
        factors,
        np.asarray([0.0, 10.0, 0.0, 10.0]),
        behavior_value=0.75,
        deterministic=True,
        legal_mask_update=restrict_second,
    )

    assert result.selected_candidate_ids == (
        "factor:0:act",
        "factor:1:pass",
    )
    assert result.observation.legal_action_mask.tolist() == [True, True, True, False]
    assert result.factors[1].behavior_log_probability == 0.0
    assert result.factors[1].conditional_candidate_ids == ("factor:1:pass",)
    assert result.joint_log_probability == pytest.approx(
        result.factors[0].behavior_log_probability
        + result.factors[1].behavior_log_probability
    )


def test_joint_sampler_conditions_later_preferences_on_selected_prefix() -> None:
    observation = _observation(0, 2)
    factors = (
        StructuredActionFactor(
            "factor:0", ("factor:0:pass", "factor:0:act")
        ),
        StructuredActionFactor(
            "factor:1", ("factor:1:pass", "factor:1:act")
        ),
    )
    keys = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
    )
    values = np.asarray(
        [[2.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )

    after_pass = sample_structured_joint_action(
        observation,
        factors,
        np.asarray([3.0, 0.0, 0.0, 0.0]),
        behavior_value=0.0,
        deterministic=True,
        candidate_prefix_keys=keys,
        candidate_prefix_values=values,
    )
    after_act = sample_structured_joint_action(
        observation,
        factors,
        np.asarray([0.0, 3.0, 0.0, 0.0]),
        behavior_value=0.0,
        deterministic=True,
        candidate_prefix_keys=keys,
        candidate_prefix_values=values,
    )

    assert after_pass.selected_candidate_ids == (
        "factor:0:pass",
        "factor:1:pass",
    )
    assert after_act.selected_candidate_ids == (
        "factor:0:act",
        "factor:1:act",
    )
    assert np.array_equal(
        after_pass.observation.legal_action_mask,
        after_act.observation.legal_action_mask,
    )
    assert all(
        factor.metadata["preference_conditioning"] == "low_rank_additive_v1"
        for factor in after_pass.factors
    )


def test_joint_sampler_rejects_post_hoc_projection_of_prior_factor() -> None:
    observation = _observation(0, 2)
    factors = (
        StructuredActionFactor(
            "factor:0", ("factor:0:pass", "factor:0:act")
        ),
        StructuredActionFactor(
            "factor:1", ("factor:1:pass", "factor:1:act")
        ),
    )

    def rewrite_prior(index, selected, mask):
        if index == 1:
            mask[0] = False
        return mask

    with pytest.raises(ValueError, match="prior-factor"):
        sample_structured_joint_action(
            observation,
            factors,
            np.zeros(4),
            behavior_value=0.0,
            deterministic=True,
            legal_mask_update=rewrite_prior,
        )
