from __future__ import annotations

from copy import deepcopy
import os
import time

import numpy as np
import pytest

from jormungandr.actors import ProcessActorPool
from jormungandr.client import JormungandrClient, JormungandrClientError
from jormungandr.service import JormungandrHttpServer, JormungandrRuntime
from jormungandr.structured import (
    EntityCandidateObservation,
    entity_candidate_observation_to_payload,
)
from jormungandr.structured_trajectory import (
    StructuredActionFactor,
    StructuredJointTrajectoryStep,
    sample_structured_joint_action,
    structured_joint_step_from_payload,
    structured_joint_step_to_payload,
    structured_joint_trajectory_to_sequence_payload,
)


REPRESENTATION = {
    "global_dim": 2,
    "entity_dim": 3,
    "candidate_dim": 4,
    "entity_type_count": 2,
}


def _observation(step: int, factor_count: int) -> EntityCandidateObservation:
    candidate_ids = tuple(
        candidate_id
        for factor in range(factor_count)
        for candidate_id in (
            f"factor:{factor}:pass",
            f"factor:{factor}:act",
        )
    )
    features = []
    for factor in range(factor_count):
        features.extend(
            ([factor / 4.0, 1.0, 0.0, 0.0], [factor / 4.0, 0.0, 1.0, 0.0])
        )
    return EntityCandidateObservation(
        global_features=np.asarray([step / 10.0, 1.0], dtype=np.float32),
        entity_features=np.asarray(
            [[step, 0.0, 1.0], [0.0, 1.0, step]], dtype=np.float32
        ),
        entity_type_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=("worker", "resource"),
        candidate_features=np.asarray(features, dtype=np.float32),
        candidate_ids=candidate_ids,
        legal_action_mask=np.ones(len(candidate_ids), dtype=np.bool_),
    )


def _factors(observation: EntityCandidateObservation, factor_count: int):
    return tuple(
        StructuredActionFactor(
            f"factor:{factor}",
            observation.candidate_ids[factor * 2 : factor * 2 + 2],
        )
        for factor in range(factor_count)
    )


def _make_step(
    client: JormungandrClient,
    model_id: str,
    *,
    actor: int,
    episode: int,
    factor_count: int,
    split: str = "train",
):
    observation = _observation(episode, factor_count)
    started = time.perf_counter()
    scored = client.score_structured(
        model_id, (entity_candidate_observation_to_payload(observation),)
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    item = scored["items"][0]
    sampled = sample_structured_joint_action(
        observation,
        _factors(observation, factor_count),
        item["candidate_logits"],
        behavior_value=item["behavior_value"],
        deterministic=False,
        rng=np.random.default_rng(10_000 * actor + episode),
        candidate_prefix_keys=item.get("candidate_prefix_keys"),
        candidate_prefix_values=item.get("candidate_prefix_values"),
    )
    selected_acts = sum(
        candidate_id.endswith(":act")
        for candidate_id in sampled.selected_candidate_ids
    )
    step = StructuredJointTrajectoryStep(
        actor_id=f"actor:{actor}",
        episode_id=f"episode:{actor}:{episode}",
        timestep=0,
        policy_version=int(scored["policy_version"]),
        observation=sampled.observation,
        factors=sampled.factors,
        joint_behavior_log_probability=sampled.joint_log_probability,
        behavior_value=sampled.behavior_value,
        reward=1.0 if selected_acts else -1.0,
        next_observation=_observation(episode + 1, factor_count),
        terminated=True,
        split=split,
        metadata={"actor_latency_ms": latency_ms},
    )
    return structured_joint_step_to_payload(step)


def _actor_job(job):
    base_url, model_id, actor = job
    client = JormungandrClient(
        base_url, timeout=30.0, compress_threshold_bytes=1
    )
    time.sleep(0.05)
    additions = []
    for episode in range(2):
        payload = _make_step(
            client,
            model_id,
            actor=actor,
            episode=episode,
            factor_count=actor + 1,
        )
        additions.append(
            client.add_structured_trajectories(model_id, ((payload,),))
        )
    return {
        "pid": os.getpid(),
        "actor": actor,
        "steps": sum(item["added_steps"] for item in additions),
    }


def test_multiprocess_joint_trajectory_service_updates_checkpoints_and_metrics(
    tmp_path,
) -> None:
    runtime = JormungandrRuntime(
        checkpoint_root=str(tmp_path), tensorboard_root=str(tmp_path)
    )
    server = JormungandrHttpServer("127.0.0.1", 0, runtime=runtime)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    client = JormungandrClient(base_url, timeout=30.0)
    learner = {
        "enabled": True,
        "algo": "structured_ppo",
        "device": "cpu",
        "tick_interval_s": 0.01,
        "updates_per_tick": 1,
        "min_trajectory_steps": 4,
        "max_trajectory_batch_steps": 8,
        "max_policy_lag": 0,
        "checkpoint_every": 0,
        "structured_model_dim": 16,
        "structured_heads": 4,
        "structured_layers": 1,
        "structured_feedforward_dim": 32,
        "structured_prefix_dim": 4,
        "epochs": 1,
        "minibatch_size": 4,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "lr": 1e-3,
    }
    model_id = "joint-trajectory-http-test"
    try:
        created = client.create_structured_model(
            {
                "model_id": model_id,
                "representation": REPRESENTATION,
                "replay": {"capacity": 64, "alpha": 0.0},
                "validation": {"capacity": 16},
                "learner": learner,
                "tensorboard": {
                    "enabled": True,
                    "logdir": "joint-events",
                },
            }
        )["model"]
        assert created["algorithm"]["replay_mode"] == "trajectory"
        assert created["policy_conditioning"] == {
            "mode": "low_rank_additive_v1",
            "prefix_dim": 4,
        }

        with ProcessActorPool(_actor_job, workers=2) as pool:
            results = pool.map(
                ((base_url, model_id, 0), (base_url, model_id, 1))
            )
        assert len({item["pid"] for item in results}) == 2
        assert sum(item["steps"] for item in results) == 4

        deadline = time.time() + 10.0
        model = client.get_structured_model(model_id)["model"]
        while model["updates"] < 1 and time.time() < deadline:
            time.sleep(0.02)
            model = client.get_structured_model(model_id)["model"]
        assert model["updates"] == 1
        assert model["policy_version"] == 1
        assert model["trajectories"]["trained_steps"] == 4
        assert model["experience"]["train_items"] == 4
        assert model["actor_latency_ms"]["count"] == 4
        assert model["last_error"] == ""

        metrics = client.get_structured_metrics(model_id)
        point = metrics["history"][-1]
        required = {
            "policy_loss",
            "value_loss",
            "entropy",
            "approximate_kl",
            "explained_variance",
            "gradient_norm",
            "actor_latency_ms_mean",
            "policy_lag_mean",
            "policy_lag_max",
        }
        assert required.issubset(point["metrics"])
        assert all(np.isfinite(point["metrics"][key]) for key in required)

        old_payload = _make_step(
            client,
            model_id,
            actor=7,
            episode=0,
            factor_count=1,
        )
        old_payload["policy_version"] = 0
        with pytest.raises(JormungandrClientError, match="stale trajectory"):
            client.add_structured_trajectories(
                model_id, ((old_payload,),)
            )

        validation_payload = deepcopy(old_payload)
        validation_payload["split"] = "validation"
        validation_payload["episode_id"] = "validation:episode"
        added_validation = client.add_structured_trajectories(
            model_id, ((validation_payload,),)
        )
        assert added_validation["added_by_split"] == {
            "train": 0,
            "validation": 1,
        }

        compact_payload = _make_step(
            client,
            model_id,
            actor=77,
            episode=0,
            factor_count=2,
            split="validation",
        )
        compact_sequence = structured_joint_trajectory_to_sequence_payload(
            (structured_joint_step_from_payload(compact_payload),)
        )
        added_compact = client.add_structured_trajectory_sequences(
            model_id, (compact_sequence,)
        )
        assert added_compact["added_by_split"] == {
            "train": 0,
            "validation": 1,
        }

        current_payload = _make_step(
            client,
            model_id,
            actor=8,
            episode=0,
            factor_count=1,
        )
        client.add_structured_trajectories(model_id, ((current_payload,),))
        with pytest.raises(JormungandrClientError, match="duplicate"):
            client.add_structured_trajectories(model_id, ((current_payload,),))

        out_of_order = deepcopy(current_payload)
        out_of_order["actor_id"] = "actor:malformed"
        out_of_order["episode_id"] = "episode:malformed"
        out_of_order["timestep"] = 2
        with pytest.raises(JormungandrClientError, match="contiguous"):
            client.add_structured_trajectories(model_id, ((out_of_order,),))

        probe = entity_candidate_observation_to_payload(_observation(9, 2))
        source_scores = client.score_structured(model_id, (probe,))["items"][0]
        checkpoint = client.checkpoint_structured_model(model_id)["checkpoint"]
        frozen_learner = dict(learner)
        frozen_learner["enabled"] = False
        frozen_id = "joint-frozen"
        client.create_structured_model(
            {
                "model_id": frozen_id,
                "representation": REPRESENTATION,
                "replay": {"capacity": 1, "alpha": 0.0},
                "validation": {"capacity": 1},
                "learner": frozen_learner,
                "checkpoint_path": checkpoint,
                "tensorboard": {"enabled": False},
            }
        )
        restored_scores = client.score_structured(frozen_id, (probe,))["items"][0]
        assert restored_scores["candidate_ids"] == source_scores["candidate_ids"]
        assert np.allclose(
            restored_scores["candidate_logits"],
            source_scores["candidate_logits"],
        )
        assert restored_scores["behavior_value"] == pytest.approx(
            source_scores["behavior_value"]
        )
        with pytest.raises(JormungandrClientError, match="frozen"):
            client.add_structured_trajectories(
                frozen_id, ((current_payload,),)
            )
        assert list((tmp_path / "joint-events" / model_id).glob("events.out.tfevents.*"))
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        runtime.close_all()
        server.server_close()
