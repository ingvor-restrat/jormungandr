from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jormungandr.structured import StructuredPolicySpec
from jormungandr.structured_checkpoint import (
    STRUCTURED_CHECKPOINT_SCHEMA,
    save_structured_checkpoint,
    structured_checkpoint_payload,
)


SPEC = StructuredPolicySpec(
    global_dim=2,
    entity_dim=3,
    candidate_dim=4,
    entity_type_count=2,
)


def test_structured_checkpoint_payload_preserves_portable_agent_state() -> None:
    weight = torch.asarray([1.0, 2.0])
    payload = structured_checkpoint_payload(
        model_id="external-bc",
        representation=SPEC,
        plugin_name="structured_bc",
        plugin_version="1.3.0",
        agent_state={"policy": {"weight": weight}},
        config={"agent_config": {"structured_model_dim": 16}},
        updates=7,
        policy_version=9,
        metadata={"source": "external-compatible-state"},
    )

    assert payload["schema"] == STRUCTURED_CHECKPOINT_SCHEMA
    assert payload["representation"] == {
        "global_dim": 2,
        "entity_dim": 3,
        "candidate_dim": 4,
        "entity_type_count": 2,
    }
    assert payload["plugin"] == {
        "name": "structured_bc",
        "version": "1.3.0",
    }
    assert payload["updates"] == 7
    assert payload["policy_version"] == 9
    assert torch.equal(payload["agent"]["policy"]["weight"], weight)


def test_save_structured_checkpoint_is_loadable(tmp_path: Path) -> None:
    path = save_structured_checkpoint(
        tmp_path / "nested" / "model.pt",
        model_id="frozen",
        representation=SPEC,
        plugin_name="structured_bc",
        plugin_version="1.3.0",
        agent_state={"policy": {"weight": torch.asarray([3.0])}},
    )

    assert path == (tmp_path / "nested" / "model.pt").resolve()
    assert not path.with_suffix(".pt.tmp").exists()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["schema"] == STRUCTURED_CHECKPOINT_SCHEMA
    assert loaded["model_id"] == "frozen"


@pytest.mark.parametrize(
    "override, message",
    [
        ({"model_id": ""}, "model_id"),
        ({"plugin_name": ""}, "plugin"),
        ({"agent_state": {}}, "agent state"),
        ({"updates": -1}, "counters"),
    ],
)
def test_structured_checkpoint_rejects_incomplete_contract(
    override: dict, message: str
) -> None:
    values = {
        "model_id": "valid",
        "representation": SPEC,
        "plugin_name": "structured_bc",
        "plugin_version": "1.3.0",
        "agent_state": {"policy": {"weight": torch.asarray([1.0])}},
    }
    values.update(override)
    with pytest.raises(ValueError, match=message):
        structured_checkpoint_payload(**values)
