from __future__ import annotations

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.structured import EntityCandidateObservation, StructuredPolicySpec
from jormungandr.structured_replay import (
    StructuredPrioritizedReplayBuffer,
    StructuredReplayTransition,
)


def _observation(step: int, candidate_count: int) -> EntityCandidateObservation:
    return EntityCandidateObservation(
        global_features=np.asarray([step / 5.0, 1.0], dtype=np.float32),
        entity_features=np.asarray(
            [[step, 0.0, 1.0], [0.0, 1.0, step]], dtype=np.float32
        ),
        entity_type_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=(f"unit:{step}", f"asset:{step}"),
        candidate_features=np.arange(
            candidate_count * 4, dtype=np.float32
        ).reshape(candidate_count, 4),
        candidate_ids=tuple(
            f"candidate:{step}:{index}" for index in range(candidate_count)
        ),
        legal_action_mask=np.ones(candidate_count, dtype=np.bool_),
    )


def test_structured_dqn_updates_prioritized_variable_candidate_replay() -> None:
    torch.manual_seed(41)
    np.random.seed(41)
    plugin = algorithm_registry.get("structured_dqn")
    assert plugin.replay_mode == "transition"
    assert plugin.build is None
    assert plugin.build_structured is not None
    agent = plugin.build_structured(
        StructuredPolicySpec(2, 3, 4, 2),
        {
            "structured_model_dim": 16,
            "structured_heads": 4,
            "structured_layers": 1,
            "structured_feedforward_dim": 32,
            "target_update": 2,
            "lr": 1e-3,
        },
        "cpu",
    )
    replay = StructuredPrioritizedReplayBuffer(capacity=16, alpha=0.6)
    for timestep, candidate_count in enumerate((2, 4, 3, 5, 2, 3)):
        observation = _observation(timestep, candidate_count)
        selected = agent.action_result_structured(
            observation, deterministic=False, epsilon=0.25
        )
        replay.add(
            StructuredReplayTransition(
                observation=observation,
                candidate_id=selected.candidate_id,
                reward=1.0 if timestep == 5 else -0.1,
                next_observation=_observation(timestep + 1, candidate_count + 1),
                done=timestep == 5,
                actor_id="actor-a",
                episode_id="episode-a",
                timestep=timestep,
                policy_version=0,
            )
        )
    transitions, indices, weights = replay.sample(4, beta=0.4)
    before = [parameter.detach().clone() for parameter in agent.q.parameters()]

    result = agent.update_structured(transitions, weights)
    replay.update_priorities(indices, result.priorities)

    assert np.isfinite(result.loss)
    assert result.priorities.shape == (4,)
    assert all(value > 0.0 for value in result.priorities)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.q.parameters())
    )


def test_structured_dqn_applies_large_margin_loss_to_demonstrations() -> None:
    torch.manual_seed(43)
    plugin = algorithm_registry.get("structured_dqn")
    agent = plugin.build_structured(
        StructuredPolicySpec(2, 3, 4, 2),
        {
            "structured_model_dim": 16,
            "structured_heads": 4,
            "structured_layers": 1,
            "structured_feedforward_dim": 32,
            "demonstration_margin": 1.0,
            "demonstration_weight": 2.0,
        },
        "cpu",
    )
    observation = _observation(0, 3)
    transition = StructuredReplayTransition(
        observation=observation,
        candidate_id=observation.candidate_ids[0],
        reward=0.0,
        next_observation=_observation(1, 3),
        done=False,
        actor_id="expert-0",
        episode_id="demonstration-0",
        timestep=0,
        policy_version=0,
        metadata={"demonstration": True},
    )

    result = agent.update_structured((transition,))

    assert result.metrics["demonstration_fraction"] == 1.0
    assert result.metrics["demonstration_loss"] > 0.0
    assert result.loss >= result.metrics["td_loss"]
