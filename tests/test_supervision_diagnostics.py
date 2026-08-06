from dataclasses import replace

import numpy as np

from jormungandr import (
    structured_supervision_deterministic_ceiling as public_ceiling,
)
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_supervision import StructuredSupervisionExample
from jormungandr.supervision_diagnostics import (
    structured_supervision_deterministic_ceiling,
    structured_supervision_model_input_fingerprint,
    structured_supervision_stratified_subset,
    structured_supervision_time_dependence,
)


def _observation(
    *,
    state: float = 0.0,
    entity_ids: tuple[str, ...] = ("worker",),
    candidate_ids: tuple[str, ...] = ("context", "left", "right"),
) -> EntityCandidateObservation:
    return EntityCandidateObservation(
        global_features=np.asarray([state, 1.0], dtype=np.float32),
        entity_features=np.asarray([[2.0, 3.0]], dtype=np.float32),
        entity_type_ids=np.asarray([0], dtype=np.int64),
        entity_ids=entity_ids,
        candidate_features=np.asarray(
            [[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=np.float32
        ),
        candidate_ids=candidate_ids,
        legal_action_mask=np.ones(3, dtype=np.bool_),
    )


def _example(
    target: str,
    *,
    observation: EntityCandidateObservation | None = None,
    prefix: tuple[str, ...] = (),
    weight: float = 1.0,
    target_group: str | None = None,
) -> StructuredSupervisionExample:
    observation = observation or _observation()
    return StructuredSupervisionExample(
        actor_id="teacher",
        episode_id=f"episode:{target}:{prefix}",
        timestep=0,
        observation=observation,
        factor_id="choice",
        candidate_ids=observation.candidate_ids[1:],
        target_candidate_id=target,
        selected_prefix_candidate_ids=prefix,
        factor_group="choice",
        target_group=target_group or target,
        sample_weight=weight,
    )


def test_deterministic_ceiling_finds_conflicting_identical_model_inputs() -> None:
    report = structured_supervision_deterministic_ceiling(
        (_example("left", weight=1.0), _example("right", weight=3.0))
    )

    assert public_ceiling is structured_supervision_deterministic_ceiling
    assert report["overall"]["model_inputs"] == 1
    assert report["overall"]["conflicting_model_inputs"] == 1
    assert report["overall"]["examples_on_conflicting_inputs"] == 2
    assert report["overall"]["raw_accuracy_ceiling"] == 0.5
    assert report["overall"]["weighted_accuracy_ceiling"] == 0.75
    assert report["overall"]["target_macro_accuracy_ceiling"] == 0.5


def test_model_input_fingerprint_excludes_audit_ids_but_includes_state() -> None:
    first = _example("left")
    renamed_observation = _observation(
        entity_ids=("renamed-worker",),
        candidate_ids=("renamed-context", "renamed-left", "renamed-right"),
    )
    renamed = _example(
        "renamed-right",
        observation=renamed_observation,
        target_group="right",
    )
    changed_state = replace(first, observation=_observation(state=1.0))

    assert structured_supervision_model_input_fingerprint(first) == (
        structured_supervision_model_input_fingerprint(renamed)
    )
    assert structured_supervision_model_input_fingerprint(first) != (
        structured_supervision_model_input_fingerprint(changed_state)
    )
    report = structured_supervision_deterministic_ceiling((first, renamed))
    assert report["overall"]["raw_accuracy_ceiling"] == 0.5


def test_selected_prefix_separates_otherwise_opposite_labels() -> None:
    observation = _observation()
    examples = (
        _example("left", observation=observation),
        _example("right", observation=observation, prefix=("context",)),
    )

    report = structured_supervision_deterministic_ceiling(examples)

    assert report["overall"]["model_inputs"] == 2
    assert report["overall"]["conflicting_model_inputs"] == 0
    assert report["overall"]["raw_accuracy_ceiling"] == 1.0
    assert report["overall"]["target_macro_accuracy_ceiling"] == 1.0


def test_time_dependence_distinguishes_open_loop_compatibility_from_response() -> None:
    first = replace(
        _example("left", observation=_observation(state=0.0)),
        episode_id="episode-a",
    )
    same_at_other_state = replace(
        _example("left", observation=_observation(state=1.0)),
        episode_id="episode-b",
    )
    open_loop = structured_supervision_time_dependence(
        (first, same_at_other_state)
    )
    assert open_loop["overall"] == {
        "examples": 2,
        "time_factor_cells": 1,
        "paired_time_factor_cells": 1,
        "paired_examples": 2,
        "state_varying_time_factor_cells": 1,
        "state_varying_examples": 2,
        "responsive_time_factor_cells": 0,
        "constant_target_state_varying_cells": 1,
        "time_only_target_accuracy_on_paired_cells": 1.0,
        "time_only_target_accuracy_on_state_varying_cells": 1.0,
        "state_response_cell_rate": 0.0,
        "classification": "open_loop_compatible_no_state_response_observed",
    }

    responds = replace(same_at_other_state, target_candidate_id="right", target_group="right")
    responsive = structured_supervision_time_dependence((first, responds))
    assert responsive["overall"]["classification"] == "state_response_observed"
    assert responsive["overall"]["responsive_time_factor_cells"] == 1
    assert responsive["overall"]["time_only_target_accuracy_on_state_varying_cells"] == 0.5


def test_stratified_subset_is_capped_seeded_and_input_order_independent() -> None:
    examples = tuple(
        replace(
            _example(
                "left" if index < 5 else "right",
                observation=_observation(state=float(index)),
            ),
            episode_id=f"episode-{index}",
            target_group="left" if index < 5 else "right",
        )
        for index in range(9)
    )

    first, receipt = structured_supervision_stratified_subset(
        examples, per_group=2, seed=19
    )
    reordered, reordered_receipt = structured_supervision_stratified_subset(
        tuple(reversed(examples)), per_group=2, seed=19
    )
    other_seed, other_receipt = structured_supervision_stratified_subset(
        examples, per_group=2, seed=23
    )

    assert len(first) == 4
    assert receipt["eligible_group_counts"] == {"left": 5, "right": 4}
    assert receipt["selected_group_counts"] == {"left": 2, "right": 2}
    assert receipt == reordered_receipt
    assert [item.episode_id for item in first] == [
        item.episode_id for item in reordered
    ]
    assert receipt["selected_identity_sha256"] != other_receipt[
        "selected_identity_sha256"
    ]
    assert [item.episode_id for item in first] != [
        item.episode_id for item in other_seed
    ]


def test_stratified_subset_rejects_invalid_contract() -> None:
    example = _example("left")
    for kwargs, message in (
        ({"per_group": 0, "seed": 1}, "per_group"),
        ({"per_group": 1, "seed": 1, "group_by": "metadata"}, "group_by"),
    ):
        try:
            structured_supervision_stratified_subset((example,), **kwargs)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid subset contract was accepted")
