import json

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
