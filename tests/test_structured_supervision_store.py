from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from jormungandr import STRUCTURED_SUPERVISION_SCHEMA
from jormungandr import StructuredSupervisionExample as PublicSupervisionExample
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_supervision import StructuredSupervisionExample
from jormungandr.structured_supervision import (
    structured_supervision_from_payload,
    structured_supervision_to_payload,
)
from jormungandr.structured_supervision_store import StructuredSupervisionBuffer


def _example(index: int):
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
    example = replace(_example(3), target_group="rare productive action")
    restored = structured_supervision_from_payload(
        structured_supervision_to_payload(example)
    )

    assert restored.target_group == "rare productive action"
