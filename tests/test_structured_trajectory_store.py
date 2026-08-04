from __future__ import annotations

import math

import numpy as np

from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_trajectory import (
    StructuredFactorChoice,
    StructuredJointTrajectoryStep,
)
from jormungandr.structured_trajectory_store import StructuredTrajectoryBuffer


def _observation(timestep: int):
    return EntityCandidateObservation(
        global_features=np.asarray([timestep], dtype=np.float32),
        entity_features=np.asarray([[timestep]], dtype=np.float32),
        entity_type_ids=np.asarray([0], dtype=np.int64),
        entity_ids=("entity:0",),
        candidate_features=np.asarray([[0.0], [1.0]], dtype=np.float32),
        candidate_ids=("pass", "act"),
        legal_action_mask=np.ones(2, dtype=np.bool_),
    )


def _trajectory(episode: int, length: int):
    steps = []
    for timestep in range(length):
        factor = StructuredFactorChoice(
            factor_id="factor:0",
            candidate_ids=("pass", "act"),
            selected_candidate_id="pass",
            behavior_log_probability=-math.log(2.0),
        )
        steps.append(
            StructuredJointTrajectoryStep(
                actor_id="actor:0",
                episode_id=f"episode:{episode}",
                timestep=timestep,
                policy_version=0,
                observation=_observation(timestep),
                factors=(factor,),
                joint_behavior_log_probability=-math.log(2.0),
                behavior_value=0.0,
                reward=1.0 if timestep == length - 1 else 0.0,
                next_observation=_observation(timestep + 1),
                terminated=timestep == length - 1,
            )
        )
    return tuple(steps)


def test_trajectory_store_batches_whole_episodes_by_environment_steps() -> None:
    store = StructuredTrajectoryBuffer(10)
    store.add(_trajectory(0, 2))
    store.add(_trajectory(1, 3))

    batch = store.pop_at_least(4)

    assert [len(trajectory) for trajectory in batch] == [2, 3]
    assert store.step_count == 0


def test_trajectory_store_rejects_duplicate_episode_identity() -> None:
    store = StructuredTrajectoryBuffer(10)
    store.add(_trajectory(0, 2))

    try:
        store.add(_trajectory(0, 2))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate trajectory was accepted")


def test_trajectory_store_evicts_only_complete_oldest_episodes() -> None:
    store = StructuredTrajectoryBuffer(4)
    store.add(_trajectory(0, 2))
    result = store.add(_trajectory(1, 3))

    assert result.steps_evicted == 2
    assert result.trajectories_evicted == 1
    assert store.step_count == 3
    assert len(store) == 1
