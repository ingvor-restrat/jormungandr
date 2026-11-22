from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from jormungandr.client import JormungandrClient
from jormungandr.generated_sbe import (
    SCHEMA_FORMAT,
    SCHEMA_ID,
    SCHEMA_SHA256,
    decode_policy_infer_request,
    decode_policy_infer_response,
    encode_policy_infer_request,
    encode_policy_infer_response,
)
from jormungandr.service import JormungandrHttpServer, JormungandrRuntime


def test_canonical_schema_digest_matches_generated_codec() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "jormungandr_inference_schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == SCHEMA_SHA256
    assert schema["format"] == SCHEMA_FORMAT
    assert schema["sbe"]["schema_id"] == SCHEMA_ID


def test_sbe_request_and_response_round_trip() -> None:
    request = decode_policy_infer_request(
        encode_policy_infer_request(
            [0.25, -0.5],
            deterministic=False,
            epsilon=0.125,
            request_id=71,
        )
    )
    assert request["request_id"] == 71
    assert request["deterministic"] is False
    assert request["epsilon"] == pytest.approx(0.125)
    assert np.allclose(request["obs"], [0.25, -0.5])

    response = decode_policy_infer_response(
        encode_policy_infer_response(
            71,
            {
                "action_idx": 2,
                "action": 1.0,
                "policy_version": 9,
                "updates": 8,
                "inference_replica_id": 1,
                "q_values": [1.0, 2.0, 3.0],
                "decision_values": [0.1, 0.2, 0.3],
                "aux_probs": [0.4, 0.6],
            },
        )
    )
    assert response["request_id"] == 71
    assert response["policy_version"] == 9
    assert response["q_values"] == pytest.approx([1.0, 2.0, 3.0])
    assert response["decision_values"] == pytest.approx([0.1, 0.2, 0.3])
    assert response["aux_probs"] == pytest.approx([0.4, 0.6])


def test_sbe_endpoint_uses_the_fixed_vector_inference_pool() -> None:
    runtime = JormungandrRuntime(
        inference_devices=["cpu"],
        inference_replicas=1,
        inference_auto_tune=False,
    )
    runtime.create_model(
        obs_dim=2,
        model_id="sbe-policy",
        tensorboard_enabled=False,
        learner={
            "enabled": True,
            "device": "cpu",
            "hidden": 8,
            "min_replay": 1000,
            "checkpoint_every": 0,
            "aux_enabled": True,
            "aux_classes": 2,
        },
    )
    server = JormungandrHttpServer("127.0.0.1", 0, runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = JormungandrClient(
        f"http://127.0.0.1:{server.server_address[1]}"
    )
    try:
        result = client.infer_policy_sbe(
            "sbe-policy", [0.0, 0.5], request_id=99
        )
        assert result["request_id"] == 99
        assert result["action_idx"] in {0, 1, 2}
        assert len(result["q_values"]) == 3
        assert len(result["aux_probs"]) == 2
        assert result["inference_replica_id"] == 0
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        runtime.close_all()
        server.server_close()
