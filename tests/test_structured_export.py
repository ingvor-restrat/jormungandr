from __future__ import annotations

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.structured import EntityCandidateObservation, StructuredPolicySpec
from jormungandr.structured_export import (
    STRUCTURED_BUNDLE_SCHEMA,
    export_structured_policy_bundle,
    load_structured_policy_bundle,
)


def test_structured_export_preserves_semantic_candidate_selection(tmp_path) -> None:
    torch.manual_seed(109)
    spec = StructuredPolicySpec(2, 3, 4, 2)
    config = {
        "structured_model_dim": 16,
        "structured_heads": 4,
        "structured_layers": 1,
        "structured_feedforward_dim": 32,
    }
    agent = algorithm_registry.get("structured_ppo").build_structured(
        spec, config, "cpu"
    )
    observation = EntityCandidateObservation(
        global_features=np.asarray([0.1, 1.0], dtype=np.float32),
        entity_features=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        entity_type_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=("worker", "job"),
        candidate_features=np.asarray(
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        candidate_ids=("action:work", "action:pass"),
        legal_action_mask=np.ones(2, dtype=np.bool_),
    )
    expected = agent.action_result_structured(
        observation, deterministic=True
    ).candidate_id

    exported = export_structured_policy_bundle(
        tmp_path / "bundle",
        agent=agent,
        spec=spec,
        algorithm="structured_ppo",
        agent_config=config,
    )
    restored, manifest = load_structured_policy_bundle(
        exported["output_dir"]
    )
    observed = restored.action_result_structured(
        observation, deterministic=True
    ).candidate_id

    assert manifest["schema"] == STRUCTURED_BUNDLE_SCHEMA
    assert observed == expected
    assert manifest["identity_contract"]["candidates"].startswith("semantic IDs")
