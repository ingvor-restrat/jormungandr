import json
from pathlib import Path
import threading
import time

import torch

from jormungandr.export import export_inference_bundle, inspect_checkpoint
from jormungandr.service import JormungandrRuntime


def test_checkpoint_and_inference_bundle_use_jormungandr_formats(tmp_path) -> None:
    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path / "checkpoints"))
    runtime.create_model(
        obs_dim=2,
        model_id="artifact-test",
        tensorboard_enabled=False,
        metadata={"feature_keys": ["signal", "inventory"]},
        learner={
            "enabled": True,
            "device": "cpu",
            "hidden": 8,
            "checkpoint_every": 0,
            "checkpoint_dir": "models",
            "aux_enabled": True,
            "aux_classes": 3,
        },
    )
    try:
        checkpoint_result = runtime.force_policy_checkpoint("artifact-test")
        checkpoint_path = checkpoint_result["checkpoint"]
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        assert payload["format"] == "jormungandr.checkpoint.v1"
        assert payload["experience_schema"] == "jormungandr.experience.v1"

        inspected = inspect_checkpoint(checkpoint_path)
        assert inspected["spec"]["model_id"] == "artifact-test"
        assert inspected["spec"]["feature_keys"] == ["signal", "inventory"]

        exported = export_inference_bundle(
            checkpoint_path,
            str(tmp_path / "bundle"),
            module="heads",
        )
        manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
        header = (tmp_path / "bundle" / "jormungandr_model_spec.hpp").read_text()

        assert exported["ok"] is True
        assert manifest["format"] == "jormungandr.inference_bundle.v1"
        assert manifest["model"]["aux_classes"] == 3
        assert "namespace jormungandr_bundle" in header
    finally:
        runtime.close_all()


def test_checkpoint_onnx_export_is_async_and_reports_status(
    tmp_path, monkeypatch
) -> None:
    started = threading.Event()
    release = threading.Event()

    def fake_export(checkpoint, out_path, module="auto"):
        started.set()
        assert release.wait(timeout=5.0)
        Path(out_path).write_bytes(b"onnx-test")
        return {
            "ok": True,
            "onnx": str(Path(out_path).resolve()),
            "module": module,
            "parity": {"passed": True},
        }

    monkeypatch.setattr(
        "jormungandr.export.export_onnx_from_checkpoint", fake_export
    )
    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path))
    runtime.create_model(
        obs_dim=2,
        model_id="async-onnx",
        tensorboard_enabled=False,
        learner={
            "enabled": True,
            "device": "cpu",
            "hidden": 8,
            "min_replay": 1000,
            "checkpoint_every": 0,
            "checkpoint_export_onnx": True,
            "checkpoint_export_module": "heads",
        },
    )
    try:
        checkpoint = runtime.force_policy_checkpoint("async-onnx")
        assert started.wait(timeout=2.0)
        assert checkpoint["onnx_export"]["status"] == "pending"
        release.set()
        deadline = time.time() + 5.0
        status = runtime.get_model("async-onnx")["learner"]["onnx_export"]
        while status.get("status") == "pending" and time.time() < deadline:
            time.sleep(0.01)
            status = runtime.get_model("async-onnx")["learner"]["onnx_export"]
        assert status["status"] == "ready"
        assert status["parity"] == {"passed": True}
        assert Path(status["onnx"]).read_bytes() == b"onnx-test"
    finally:
        release.set()
        runtime.close_all()
