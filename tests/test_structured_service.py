from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

import numpy as np
import pytest

from jormungandr.service import JormungandrHttpServer, JormungandrRuntime
from jormungandr.structured import (
    EntityCandidateObservation,
    entity_candidate_observation_to_payload,
)


def _observation(step: int, candidate_count: int = 3) -> dict:
    observation = EntityCandidateObservation(
        global_features=np.asarray([step / 10.0, 1.0], dtype=np.float32),
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
        metadata={"step": step},
    )
    return entity_candidate_observation_to_payload(observation)


def _post(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10.0) as response:
        return json.load(response)


def _get(url: str) -> dict:
    with urlopen(url, timeout=10.0) as response:
        return json.load(response)


def test_v2_http_actors_share_structured_inference_replay_and_learner(
    tmp_path,
) -> None:
    runtime = JormungandrRuntime(
        checkpoint_root=str(tmp_path), tensorboard_root=str(tmp_path)
    )
    server = JormungandrHttpServer("127.0.0.1", 0, runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        created = _post(
            f"{base_url}/v2/models",
            {
                "model_id": "structured-http-test",
                "representation": {
                    "global_dim": 2,
                    "entity_dim": 3,
                    "candidate_dim": 4,
                    "entity_type_count": 2,
                },
                "replay": {"capacity": 32, "alpha": 0.6},
                "validation": {"capacity": 8},
                "learner": {
                    "enabled": True,
                    "algo": "structured_dqn",
                    "device": "cpu",
                    "batch_size": 2,
                    "min_replay": 2,
                    "tick_interval_s": 0.01,
                    "checkpoint_every": 0,
                    "structured_model_dim": 16,
                    "structured_heads": 4,
                    "structured_layers": 1,
                    "structured_feedforward_dim": 32,
                },
                "tensorboard": {
                    "enabled": True,
                    "logdir": "structured-test-events",
                },
            },
        )
        assert created["ok"] is True
        assert created["model"]["representation"]["mode"] == "entity_candidates"

        inference = _post(
            f"{base_url}/v2/models/structured-http-test/policy/infer",
            {
                "observation": _observation(0),
                "deterministic": False,
                "epsilon": 0.2,
            },
        )
        assert inference["ok"] is True
        assert inference["items"][0]["candidate_id"].startswith("candidate:0:")
        assert len(inference["items"][0]["candidate_values"]) == 3

        policy_version = inference["policy_version"]
        added = _post(
            f"{base_url}/v2/models/structured-http-test/experience/add",
            {
                "schema": "jormungandr.structured_experience.v1",
                "items": [
                    {
                        "split": "train",
                        "actor_id": "actor-0",
                        "episode_id": "episode-0",
                        "timestep": 0,
                        "policy_version": policy_version,
                        "observation": _observation(0),
                        "candidate_id": inference["items"][0]["candidate_id"],
                        "reward": -0.1,
                        "next_observation": _observation(1, 4),
                        "done": False,
                    },
                    {
                        "split": "train",
                        "actor_id": "actor-1",
                        "episode_id": "episode-1",
                        "timestep": 0,
                        "policy_version": policy_version,
                        "observation": _observation(2),
                        "candidate_id": "candidate:2:0",
                        "reward": 1.0,
                        "next_observation": _observation(3, 2),
                        "done": True,
                    },
                ],
            },
        )
        assert added["added_by_split"] == {"train": 2, "validation": 0}

        deadline = time.time() + 10.0
        model = _get(f"{base_url}/v2/models/structured-http-test")["model"]
        while model["updates"] < 1 and time.time() < deadline:
            time.sleep(0.02)
            model = _get(
                f"{base_url}/v2/models/structured-http-test"
            )["model"]
        assert model["updates"] >= 1
        assert model["policy_version"] >= 1
        assert model["replay"]["size"] == 2
        assert model["last_error"] == ""

        logged = _post(
            f"{base_url}/v2/models/structured-http-test/metrics",
            {
                "step": 2,
                "metrics": {
                    "evaluation/mean_return": 1.5,
                    "ignored/non_finite": float("nan"),
                    "ignored/text": "not-a-number",
                },
            },
        )
        assert logged["logged"] == 1
        metrics = _get(
            f"{base_url}/v2/models/structured-http-test/metrics"
        )
        assert metrics["schema"] == "jormungandr.structured_metrics.v1"
        assert metrics["updates"] >= 1
        assert len(metrics["history"]) >= 1
        assert metrics["history"][-1]["update"] == metrics["updates"]
        assert metrics["performance_history"][-1] == {
            "step": 2,
            "ts": metrics["performance_history"][-1]["ts"],
            "metrics": {"evaluation/mean_return": 1.5},
        }
        assert metrics["latest_performance_metrics"] == {
            "evaluation/mean_return": 1.5
        }
        event_dir = tmp_path / "structured-test-events" / "structured-http-test"
        assert any(event_dir.glob("events.out.tfevents.*"))

        checkpoint = _post(
            f"{base_url}/v2/models/structured-http-test/policy/checkpoint",
            {},
        )
        assert checkpoint["ok"] is True
        assert Path(checkpoint["checkpoint"]).is_file()
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        runtime.close_all()
        server.server_close()


def test_structured_checkpoint_can_be_loaded_as_frozen_api_model(tmp_path) -> None:
    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path))
    representation = {
        "global_dim": 2,
        "entity_dim": 3,
        "candidate_dim": 4,
        "entity_type_count": 2,
    }
    learner = {
        "enabled": True,
        "algo": "structured_dqn",
        "device": "cpu",
        "batch_size": 2,
        "min_replay": 2,
        "checkpoint_every": 0,
        "structured_model_dim": 16,
        "structured_heads": 4,
        "structured_layers": 1,
        "structured_feedforward_dim": 32,
    }
    try:
        runtime.create_structured_model(
            representation=representation,
            learner=learner,
            model_id="source",
        )
        source = runtime.structured_policy_infer(
            "source",
            observations=(_observation(7),),
            deterministic=True,
            epsilon=0.0,
        )
        checkpoint = runtime.force_structured_checkpoint("source")["checkpoint"]

        frozen_learner = dict(learner)
        frozen_learner["enabled"] = False
        frozen = runtime.create_structured_model(
            representation=representation,
            learner=frozen_learner,
            model_id="ancestor-0",
            checkpoint_path=checkpoint,
        )
        restored = runtime.structured_policy_infer(
            "ancestor-0",
            observations=(_observation(7),),
            deterministic=True,
            epsilon=0.0,
        )

        assert frozen["trainable"] is False
        assert frozen["checkpoint_source"] == str(Path(checkpoint).resolve())
        assert restored["items"][0]["candidate_id"] == source["items"][0]["candidate_id"]
        assert np.allclose(
            restored["items"][0]["candidate_values"],
            source["items"][0]["candidate_values"],
        )
        with pytest.raises(ValueError, match="frozen"):
            runtime.structured_experience_add("ancestor-0", [{}])
    finally:
        runtime.close_all()


def test_structured_service_model_seed_is_reproducible_and_isolated(tmp_path) -> None:
    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path))
    representation = {
        "global_dim": 2,
        "entity_dim": 3,
        "candidate_dim": 4,
        "entity_type_count": 2,
    }
    base = {
        "enabled": False,
        "algo": "structured_bc",
        "device": "cpu",
        "structured_model_dim": 16,
        "structured_heads": 4,
        "structured_layers": 1,
        "structured_feedforward_dim": 32,
    }
    try:
        for model_id, seed in (("same-a", 701), ("other", 702), ("same-b", 701)):
            runtime.create_structured_model(
                representation=representation,
                learner={**base, "seed": seed},
                model_id=model_id,
            )
        observation = _observation(13)
        scores = {
            model_id: runtime.structured_policy_score(
                model_id, observations=(observation,)
            )["items"][0]["candidate_logits"]
            for model_id in ("same-a", "other", "same-b")
        }

        assert np.array_equal(scores["same-a"], scores["same-b"])
        assert not np.array_equal(scores["same-a"], scores["other"])
    finally:
        runtime.close_all()


def test_structured_service_enforces_exact_supervision_update_budget(tmp_path) -> None:
    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path))
    representation = {
        "global_dim": 2,
        "entity_dim": 3,
        "candidate_dim": 4,
        "entity_type_count": 2,
    }
    learner = {
        "enabled": True,
        "algo": "structured_bc",
        "device": "cpu",
        "seed": 709,
        "max_updates": 3,
        "batch_size": 2,
        "min_replay": 2,
        "replay_ratio": 100.0,
        "tick_interval_s": 0.001,
        "checkpoint_every": 0,
        "structured_model_dim": 16,
        "structured_heads": 4,
        "structured_layers": 1,
        "structured_feedforward_dim": 32,
    }
    try:
        runtime.create_structured_model(
            representation=representation,
            learner=learner,
            model_id="budgeted-bc",
            capacity=8,
            validation_capacity=8,
        )
        examples = []
        for index in range(2):
            observation = _observation(index)
            examples.append(
                {
                    "schema": "jormungandr.structured_supervision.v1",
                    "actor_id": "teacher",
                    "episode_id": f"episode-{index}",
                    "timestep": 0,
                    "split": "train",
                    "observation": observation,
                    "factor_id": "factor",
                    "candidate_ids": observation["candidates"]["ids"],
                    "target_candidate_id": observation["candidates"]["ids"][0],
                    "source_group": "test",
                    "factor_group": "test",
                    "target_group": "test",
                    "sample_weight": 1.0,
                }
            )
        runtime.structured_supervision_add("budgeted-bc", examples)
        deadline = time.time() + 5.0
        model = runtime.get_structured_model("budgeted-bc")
        while model["updates"] < 3 and time.time() < deadline:
            time.sleep(0.01)
            model = runtime.get_structured_model("budgeted-bc")
        assert model["updates"] == 3
        time.sleep(0.05)
        assert runtime.get_structured_model("budgeted-bc")["updates"] == 3
    finally:
        runtime.close_all()
