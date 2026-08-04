import numpy as np
import pytest
import torch

from jormungandr.structured import (
    EntityCandidateObservation,
    EntityCandidateTransformer,
    StructuredPolicySpec,
    collate_entity_candidate_observations,
    entity_candidate_observation_from_payload,
    entity_candidate_observation_to_payload,
    select_dynamic_actions,
)


def _observation(
    *,
    entity_count: int,
    candidate_ids: tuple[str, ...],
    legal: tuple[bool, ...],
) -> EntityCandidateObservation:
    return EntityCandidateObservation(
        global_features=np.asarray([0.25, -0.5], dtype=np.float32),
        entity_features=np.arange(entity_count * 3, dtype=np.float32).reshape(
            entity_count, 3
        ),
        entity_type_ids=np.arange(entity_count, dtype=np.int64) % 3,
        entity_ids=tuple(f"entity-{index}" for index in range(entity_count)),
        candidate_features=np.arange(
            len(candidate_ids) * 4, dtype=np.float32
        ).reshape(len(candidate_ids), 4),
        candidate_ids=candidate_ids,
        legal_action_mask=np.asarray(legal),
    )


def test_collation_preserves_local_candidate_identity_and_masks_padding() -> None:
    first = _observation(
        entity_count=2,
        candidate_ids=("move:north", "water", "pass"),
        legal=(True, False, True),
    )
    second = _observation(
        entity_count=1,
        candidate_ids=("harvest",),
        legal=(True,),
    )

    batch = collate_entity_candidate_observations((first, second))

    assert batch.entity_features.shape == (2, 2, 3)
    assert batch.candidate_features.shape == (2, 3, 4)
    assert batch.entity_mask.tolist() == [[True, True], [True, False]]
    assert batch.candidate_mask.tolist() == [
        [True, True, True],
        [True, False, False],
    ]
    assert batch.legal_action_mask.tolist() == [
        [True, False, True],
        [True, False, False],
    ]
    assert batch.candidate_ids[0][0] == "move:north"
    assert batch.candidate_ids[1][0] == "harvest"


def test_transformer_scores_different_candidate_counts_and_never_selects_padding() -> None:
    torch.manual_seed(7)
    observations = (
        _observation(
            entity_count=3,
            candidate_ids=("move", "invalid", "pass"),
            legal=(True, False, True),
        ),
        _observation(
            entity_count=1,
            candidate_ids=("harvest",),
            legal=(True,),
        ),
    )
    batch = collate_entity_candidate_observations(observations).to_torch()
    model = EntityCandidateTransformer(
        global_dim=2,
        entity_dim=3,
        candidate_dim=4,
        entity_type_count=3,
        model_dim=16,
        heads=4,
        layers=1,
        feedforward_dim=32,
    )

    output = model(batch)
    selected = select_dynamic_actions(output, batch, deterministic=True)
    loss = output.values.square().mean() - torch.log_softmax(
        output.logits, dim=-1
    )[0, 0]
    loss.backward()

    assert output.logits.shape == (2, 3)
    assert output.values.shape == (2,)
    assert torch.isneginf(output.logits[0, 1])
    assert torch.isneginf(output.logits[1, 1:]).all()
    assert selected[0].candidate_id in {"move", "pass"}
    assert selected[1].candidate_id == "harvest"
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_observation_rejects_duplicate_or_entirely_illegal_candidates() -> None:
    with pytest.raises(ValueError, match="candidate_ids must be unique"):
        _observation(
            entity_count=1,
            candidate_ids=("pass", "pass"),
            legal=(True, True),
        )
    with pytest.raises(ValueError, match="admit at least one candidate"):
        _observation(
            entity_count=1,
            candidate_ids=("pass",),
            legal=(False,),
        )


def test_collation_rejects_mixed_feature_schemas() -> None:
    first = _observation(
        entity_count=1,
        candidate_ids=("pass",),
        legal=(True,),
    )
    second = EntityCandidateObservation(
        global_features=np.asarray([1.0, 2.0, 3.0]),
        entity_features=np.zeros((1, 3), dtype=np.float32),
        entity_type_ids=np.zeros(1, dtype=np.int64),
        entity_ids=("entity",),
        candidate_features=np.zeros((1, 4), dtype=np.float32),
        candidate_ids=("pass",),
        legal_action_mask=np.ones(1, dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="share feature dimensions"):
        collate_entity_candidate_observations((first, second))


def test_structured_wire_codec_round_trips_variable_identity() -> None:
    observation = _observation(
        entity_count=3,
        candidate_ids=("move:north", "pass"),
        legal=(True, True),
    )

    payload = entity_candidate_observation_to_payload(observation)
    restored = entity_candidate_observation_from_payload(
        payload,
        spec=StructuredPolicySpec(2, 3, 4, 3),
    )

    assert restored.entity_ids == observation.entity_ids
    assert restored.candidate_ids == observation.candidate_ids
    assert np.array_equal(restored.entity_features, observation.entity_features)
    assert np.array_equal(
        restored.candidate_features, observation.candidate_features
    )

    with pytest.raises(ValueError, match="dimensions"):
        entity_candidate_observation_from_payload(
            payload,
            spec=StructuredPolicySpec(3, 3, 4, 3),
        )


def test_candidates_can_reference_multiple_entities() -> None:
    first = EntityCandidateObservation(
        global_features=np.asarray([0.25, -0.5], dtype=np.float32),
        entity_features=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        entity_type_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=("unit", "destination"),
        candidate_features=np.zeros((2, 4), dtype=np.float32),
        candidate_ids=("move", "pass"),
        legal_action_mask=np.ones(2, dtype=np.bool_),
        candidate_entity_indices=np.asarray([[0, 1], [0, -1]]),
    )
    second = _observation(
        entity_count=1,
        candidate_ids=("pass",),
        legal=(True,),
    )

    payload = entity_candidate_observation_to_payload(first)
    restored = entity_candidate_observation_from_payload(payload)
    batch = collate_entity_candidate_observations((restored, second))
    model = EntityCandidateTransformer(
        global_dim=2,
        entity_dim=3,
        candidate_dim=4,
        entity_type_count=3,
        model_dim=16,
        heads=4,
        layers=1,
        feedforward_dim=32,
    )
    output = model(batch.to_torch())

    assert restored.candidate_entity_indices.tolist() == [[0, 1], [0, -1]]
    assert batch.candidate_entity_indices.shape == (2, 2, 2)
    assert output.logits.shape == (2, 2)
    assert torch.isfinite(output.logits[0]).all()
