import numpy as np
import pytest
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.structured import EntityCandidateObservation, StructuredPolicySpec
from jormungandr.structured_supervision import (
    StructuredSupervisionExample,
    StructuredSupervisionFrame,
    StructuredSupervisionLabel,
    apply_structured_supervision_frame_balance_weights,
    structured_supervision_frame_balance_weights,
    structured_supervision_balance_weights_from_counts,
    structured_supervision_frame_from_payload,
    structured_supervision_frame_to_payload,
)


SPEC = StructuredPolicySpec(2, 3, 2, 2)


def _observation() -> EntityCandidateObservation:
    return EntityCandidateObservation(
        global_features=np.asarray([0.0, 1.0], dtype=np.float32),
        entity_features=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        entity_type_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=("worker:0", "worker:1"),
        candidate_features=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        candidate_ids=("w0:pass", "w0:act", "w1:pass", "w1:act"),
        legal_action_mask=np.ones(4, dtype=np.bool_),
        candidate_entity_indices=np.asarray([[0], [0], [1], [1]]),
    )


def _frame(split: str = "train") -> StructuredSupervisionFrame:
    return StructuredSupervisionFrame(
        actor_id="teacher",
        episode_id="episode:1",
        timestep=3,
        observation=_observation(),
        labels=(
            StructuredSupervisionLabel(
                factor_id="worker:0",
                candidate_ids=("w0:pass", "w0:act"),
                target_candidate_id="w0:act",
                factor_group="worker",
                target_group="act",
                sample_weight=2.0,
            ),
            StructuredSupervisionLabel(
                factor_id="worker:1",
                candidate_ids=("w1:pass", "w1:act"),
                target_candidate_id="w1:pass",
                factor_group="worker",
                target_group="pass",
            ),
        ),
        split=split,
        source_group="fixture",
    )


def test_shared_supervision_frame_round_trips_one_observation_and_many_labels() -> None:
    source = _frame()

    restored = structured_supervision_frame_from_payload(
        structured_supervision_frame_to_payload(source), spec=SPEC
    )

    assert restored.actor_id == source.actor_id
    assert restored.label_count == 2
    assert restored.labels == source.labels
    assert restored.observation.candidate_ids == source.observation.candidate_ids
    assert np.array_equal(
        restored.observation.candidate_features,
        source.observation.candidate_features,
    )


def test_shared_supervision_frame_rejects_cross_factor_or_illegal_candidates() -> None:
    with pytest.raises(ValueError, match="absent"):
        StructuredSupervisionFrame(
            actor_id="teacher",
            episode_id="episode:1",
            timestep=0,
            observation=_observation(),
            labels=(
                StructuredSupervisionLabel(
                    factor_id="worker:0",
                    candidate_ids=("w0:pass", "missing"),
                    target_candidate_id="missing",
                ),
            ),
        )

    observation = _observation()
    masked = EntityCandidateObservation(
        global_features=observation.global_features,
        entity_features=observation.entity_features,
        entity_type_ids=observation.entity_type_ids,
        entity_ids=observation.entity_ids,
        candidate_features=observation.candidate_features,
        candidate_ids=observation.candidate_ids,
        legal_action_mask=np.asarray([True, False, True, True]),
        candidate_entity_indices=observation.candidate_entity_indices,
    )
    with pytest.raises(ValueError, match="illegal"):
        StructuredSupervisionFrame(
            actor_id="teacher",
            episode_id="episode:1",
            timestep=0,
            observation=masked,
            labels=(_frame().labels[0],),
        )


def test_shared_frame_update_matches_expanded_examples_without_candidate_coupling() -> None:
    config = {
        "structured_model_dim": 16,
        "structured_heads": 4,
        "structured_layers": 1,
        "structured_feedforward_dim": 32,
        "structured_dropout": 0.0,
        "structured_candidate_attention_layers": 0,
        "lr": 1e-2,
        "max_grad": 5.0,
    }
    frame = _frame()
    examples = tuple(
        StructuredSupervisionExample(
            actor_id=frame.actor_id,
            episode_id=frame.episode_id,
            timestep=frame.timestep,
            observation=frame.observation,
            factor_id=label.factor_id,
            candidate_ids=label.candidate_ids,
            target_candidate_id=label.target_candidate_id,
            selected_prefix_candidate_ids=label.selected_prefix_candidate_ids,
            split=frame.split,
            source_group=frame.source_group,
            factor_group=label.factor_group,
            target_group=label.target_group,
            balance_group=label.balance_group,
            sample_weight=label.sample_weight,
            metadata=label.metadata,
        )
        for label in frame.labels
    )
    torch.manual_seed(173)
    framed_agent = algorithm_registry.get("structured_bc").build_structured(
        SPEC, config, "cpu"
    )
    torch.manual_seed(173)
    expanded_agent = algorithm_registry.get("structured_bc").build_structured(
        SPEC, config, "cpu"
    )

    framed = framed_agent.update_structured_supervision_frames((frame,))
    expanded = expanded_agent.update_structured_supervision(examples)

    assert framed.examples == expanded.examples == 2
    assert framed.loss == pytest.approx(expanded.loss, abs=1e-7)
    assert framed.accuracy == expanded.accuracy
    assert framed.metrics["frames"] == 1.0
    assert framed.metrics["labels_per_frame"] == 2.0
    # The two objectives are algebraically identical.  Duplicate transformer
    # rows and an index-selected shared row accumulate a few near-zero
    # gradients in a different floating-point order; Adam can amplify those
    # otherwise irrelevant cancellations on its first step.
    assert all(
        torch.allclose(left, right, atol=2e-4, rtol=1e-5)
        for left, right in zip(
            framed_agent.policy.parameters(),
            expanded_agent.policy.parameters(),
            strict=True,
        )
    )


def test_shared_frame_balancing_counts_labels_without_expanding_observations() -> None:
    source = _frame()
    second = StructuredSupervisionFrame(
        actor_id=source.actor_id,
        episode_id="episode:2",
        timestep=4,
        observation=source.observation,
        labels=(source.labels[0],),
    )

    weights = structured_supervision_frame_balance_weights(
        (source, second), exponent=1.0
    )
    weighted = apply_structured_supervision_frame_balance_weights(
        (source, second), weights
    )
    act_weights = [
        label.sample_weight
        for frame in weighted
        for label in frame.labels
        if label.balance_group == "act"
    ]
    pass_weights = [
        label.sample_weight
        for frame in weighted
        for label in frame.labels
        if label.balance_group == "pass"
    ]

    assert weights["pass"] == pytest.approx(2.0 * weights["act"])
    assert weights == structured_supervision_balance_weights_from_counts(
        {"act": 2, "pass": 1}, exponent=1.0
    )
    assert sum(act_weights) == pytest.approx(sum(pass_weights))
    assert weighted[0].observation is source.observation
