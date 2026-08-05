from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from jormungandr import STRUCTURED_SUPERVISION_SCHEMA
from jormungandr import StructuredSupervisionExample as PublicSupervisionExample
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_supervision import StructuredSupervisionExample
from jormungandr.structured_supervision import (
    apply_structured_supervision_balance_weights,
    structured_supervision_from_payload,
    structured_supervision_balance_weights,
    structured_supervision_to_payload,
)
from jormungandr.structured_supervision_store import StructuredSupervisionBuffer


def _example(index: int, *, sample_weight: float = 1.0):
    observation = EntityCandidateObservation(
        global_features=np.asarray([index], dtype=np.float32),
        entity_features=np.asarray([[index]], dtype=np.float32),
        entity_type_ids=np.asarray([0]),
        entity_ids=("entity",),
        candidate_features=np.asarray([[0.0], [1.0]], dtype=np.float32),
        candidate_ids=("pass", "act"),
        legal_action_mask=np.ones(2, dtype=np.bool_),
    )
    return StructuredSupervisionExample(
        actor_id="actor",
        episode_id=f"episode:{index}",
        timestep=0,
        observation=observation,
        factor_id="factor",
        candidate_ids=observation.candidate_ids,
        target_candidate_id="act",
        sample_weight=sample_weight,
    )


def test_supervision_store_is_bounded_samples_and_rejects_duplicates() -> None:
    store = StructuredSupervisionBuffer(2)
    store.add(_example(0))
    try:
        store.add(_example(0))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate supervision was accepted")
    store.add(_example(1))
    assert store.add(_example(2)) is True
    sampled = store.sample(4, rng=np.random.default_rng(3))
    assert len(sampled) == 4
    assert {item.episode_id for item in store.snapshot()} == {
        "episode:1",
        "episode:2",
    }


def test_supervision_contract_exports_and_rejects_illegal_factor_candidates() -> None:
    example = _example(0)
    observation = replace(
        example.observation,
        legal_action_mask=np.asarray([True, False], dtype=np.bool_),
    )

    assert PublicSupervisionExample is StructuredSupervisionExample
    assert STRUCTURED_SUPERVISION_SCHEMA.endswith(".v1")
    with pytest.raises(ValueError, match="factor's legal candidates"):
        replace(example, observation=observation)


def test_supervision_target_group_survives_wire_round_trip() -> None:
    example = replace(
        _example(3),
        target_group="rare productive action",
        balance_group="rare conditional decision",
    )
    restored = structured_supervision_from_payload(
        structured_supervision_to_payload(example)
    )

    assert restored.target_group == "rare productive action"
    assert restored.balance_group == "rare conditional decision"


def test_legacy_supervision_payload_defaults_balance_group_to_target() -> None:
    payload = structured_supervision_to_payload(
        replace(_example(4), target_group="semantic target")
    )
    del payload["balance_group"]

    restored = structured_supervision_from_payload(payload)

    assert restored.balance_group == "semantic target"


def test_sample_weight_sampling_preserves_objective_without_double_weighting() -> None:
    store = StructuredSupervisionBuffer(2)
    store.add(_example(0, sample_weight=1.0))
    store.add(_example(1, sample_weight=9.0))

    sampled = store.sample(
        20_000,
        rng=np.random.default_rng(17),
        strategy="sample_weight",
    )
    heavy_rate = np.mean([item.episode_id == "episode:1" for item in sampled])

    assert heavy_rate == pytest.approx(0.9, abs=0.01)
    assert {item.sample_weight for item in sampled} == {1.0}


def test_supervision_store_rejects_unknown_sampling_strategy() -> None:
    store = StructuredSupervisionBuffer(1)
    store.add(_example(0))

    with pytest.raises(ValueError, match="uniform or sample_weight"):
        store.sample(1, rng=np.random.default_rng(19), strategy="mystery")


def test_balance_weights_are_training_only_mean_one_and_group_aware() -> None:
    common = [
        replace(_example(index), balance_group="common") for index in range(9)
    ]
    rare = [replace(_example(9), balance_group="rare")]

    weights = structured_supervision_balance_weights(
        (*common, *rare), exponent=0.5
    )
    weighted = apply_structured_supervision_balance_weights(
        (*common, *rare), weights
    )

    assert weights["rare"] > weights["common"]
    assert np.mean([item.sample_weight for item in weighted]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unseen balance groups"):
        apply_structured_supervision_balance_weights(
            (replace(_example(10), balance_group="unseen"),), weights
        )
