import json
import threading
import time
from urllib.request import Request, urlopen

import pytest

from jormungandr.service import (
    JormungandrHttpServer,
    JormungandrRuntime,
    JormungandrServiceError,
)


def experience(
    *,
    split: str,
    timestep: int,
    action: float,
    aux_label: int | None = None,
) -> dict:
    item = {
        "split": split,
        "actor_id": "actor-a",
        "episode_id": "episode-a",
        "timestep": timestep,
        "policy_version": 0,
        "obs": [float(timestep), 0.0],
        "action": action,
        "reward": float(timestep % 2),
        "next_obs": [float(timestep + 1), 0.0],
        "done": False,
    }
    if aux_label is not None:
        item["aux"] = {"kind": "direction", "label": aux_label}
    return item


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5.0) as response:
        return json.load(response)


def test_interleaved_experience_is_split_without_validation_leakage() -> None:
    runtime = JormungandrRuntime()
    runtime.create_model(
        obs_dim=2,
        model_id="split-test",
        capacity=8,
        validation_capacity=8,
        tensorboard_enabled=False,
        learner={
            "enabled": True,
            "device": "cpu",
            "hidden": 8,
            "batch_size": 4,
            "min_replay": 8,
            "checkpoint_every": 0,
        },
    )
    try:
        result = runtime.experience_add(
            "split-test",
            [
                experience(split="train", timestep=0, action=-1.0),
                experience(split="validation", timestep=1, action=0.0),
                experience(split="train", timestep=2, action=1.0),
                experience(split="val", timestep=3, action=-1.0),
            ],
        )
        rec = runtime.get_model_record("split-test")

        assert result["added_by_split"] == {"train": 2, "validation": 2}
        assert len(rec.replay) == 2
        assert len(rec.validation) == 2
        assert rec.obs_normalizer.count == 4
        assert rec.replay.action[:2, 0].tolist() == [0.0, 2.0]
        assert rec.validation.action[:2, 0].tolist() == [1.0, 0.0]

        delayed = runtime.experience_aux_update(
            "split-test",
            [
                {
                    "split": "validation",
                    "actor_id": "actor-a",
                    "episode_id": "episode-a",
                    "timestep": 3,
                    "aux": {"kind": "direction", "label": 2},
                }
            ],
        )
        assert delayed["matched"] == 1
        assert rec.validation_aux_by_idx[1]["label"] == 2
    finally:
        runtime.close_all()


def test_public_experience_contract_requires_actor_identity() -> None:
    runtime = JormungandrRuntime()
    runtime.create_model(
        obs_dim=2,
        model_id="identity-test",
        tensorboard_enabled=False,
    )
    try:
        item = experience(split="train", timestep=0, action=0.0)
        del item["actor_id"]
        with pytest.raises(JormungandrServiceError, match="actor_id"):
            runtime.experience_add("identity-test", [item])
    finally:
        runtime.close_all()


def test_learner_evaluates_interleaved_validation_samples() -> None:
    runtime = JormungandrRuntime()
    runtime.create_model(
        obs_dim=2,
        model_id="validation-test",
        capacity=32,
        validation_capacity=16,
        tensorboard_enabled=False,
        learner={
            "enabled": True,
            "device": "cpu",
            "hidden": 8,
            "batch_size": 2,
            "min_replay": 2,
            "tick_interval_s": 0.005,
            "validation_every": 1,
            "validation_batch_size": 2,
            "min_validation": 2,
            "checkpoint_every": 0,
            "aux_enabled": True,
            "aux_weight": 0.1,
            "aux_classes": 3,
            "aux_kind": "direction",
        },
    )
    try:
        runtime.experience_add(
            "validation-test",
            [
                experience(split="train", timestep=0, action=-1.0, aux_label=0),
                experience(split="validation", timestep=1, action=0.0, aux_label=1),
                experience(split="train", timestep=2, action=1.0, aux_label=2),
                experience(split="validation", timestep=3, action=-1.0, aux_label=0),
            ],
        )

        deadline = time.time() + 5.0
        learner = runtime.get_model("validation-test")["learner"]
        while learner["validation_runs"] < 1 and time.time() < deadline:
            time.sleep(0.01)
            learner = runtime.get_model("validation-test")["learner"]

        assert learner["updates"] >= 1
        assert learner["validation_runs"] >= 1
        assert learner["validation_policy_version"] >= 1
        assert learner["last_validation_count"] == 2
        assert learner["last_validation_ts"] > 0
    finally:
        runtime.close_all()


def test_http_experience_endpoint_routes_both_splits() -> None:
    runtime = JormungandrRuntime()
    server = JormungandrHttpServer("127.0.0.1", 0, runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        created = post_json(
            f"{base_url}/v1/models",
            {
                "model_id": "http-test",
                "obs_dim": 2,
                "replay": {"capacity": 8},
                "validation": {"capacity": 8},
                "tensorboard": {"enabled": False},
                "learner": {"enabled": False},
            },
        )
        assert created["ok"] is True

        result = post_json(
            f"{base_url}/v1/models/http-test/experience/add",
            {
                "schema": "jormungandr.experience.v1",
                "items": [
                    experience(split="train", timestep=0, action=-1.0),
                    experience(split="validation", timestep=1, action=0.0),
                ]
            },
        )
        assert result["ok"] is True
        assert result["added_by_split"] == {"train": 1, "validation": 1}
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        runtime.close_all()
        server.server_close()


def test_internal_comparison_retains_actor_performance_metrics() -> None:
    runtime = JormungandrRuntime()
    runtime.create_model(
        obs_dim=2,
        model_id="comparison-test",
        tensorboard_enabled=False,
        learner={"enabled": True, "device": "cpu", "hidden": 8},
    )
    try:
        runtime.log_metrics(
            "comparison-test",
            7,
            {
                "sampler/validation/reward_mean": 1.25,
                "domain/downside_cvar": -0.5,
            },
        )

        comparison = runtime.compare_models()
        row = comparison["models"][0]
        history = runtime.get_training_metrics("comparison-test")

        assert row["performance_metrics"]["validation/sampler/reward_mean"] == 1.25
        assert row["performance_metrics"]["domain/downside_cvar"] == -0.5
        assert (
            "validation/sampler/reward_mean"
            in comparison["performance_metric_names"]
        )
        assert history["performance_history"][-1]["step"] == 7
    finally:
        runtime.close_all()
