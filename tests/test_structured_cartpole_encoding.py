from __future__ import annotations

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.structured import (
    collate_entity_candidate_observations,
    entity_candidate_observation_from_payload,
    entity_candidate_observation_to_payload,
)

from examples.benchmark_structured_cartpole_ppo import (
    CANDIDATE_IDS,
    SPEC,
    encode_cartpole,
)


def test_cartpole_semantics_survive_permutation_wire_and_inference() -> None:
    torch.manual_seed(101)
    agent = algorithm_registry.get("structured_ppo").build_structured(
        SPEC,
        {
            "structured_model_dim": 16,
            "structured_heads": 4,
            "structured_layers": 1,
            "structured_feedforward_dim": 32,
        },
        "cpu",
    )
    state = np.asarray([0.1, -0.2, 0.03, 0.7], dtype=np.float32)
    canonical = encode_cartpole(state)
    permuted = encode_cartpole(
        state, entity_order=(3, 1, 0, 2), candidate_order=(1, 0)
    )
    restored = entity_candidate_observation_from_payload(
        entity_candidate_observation_to_payload(permuted), spec=SPEC
    )

    def scores(observation):
        batch = collate_entity_candidate_observations((observation,)).to_torch()
        with torch.no_grad():
            logits = agent.policy(batch).logits[0]
        return {
            identifier: float(logits[index])
            for index, identifier in enumerate(observation.candidate_ids)
        }

    left = scores(canonical)
    right = scores(restored)

    assert restored.candidate_ids == ("action:right", "action:left")
    assert all(abs(left[key] - right[key]) <= 1e-6 for key in CANDIDATE_IDS)
