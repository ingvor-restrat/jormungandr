"""Portable checkpoints for structured entity/candidate policies.

The checkpoint contains only Jormungandr concepts: a representation, an
algorithm plugin, its agent state, and provenance.  A business-case package
can therefore wrap an older or externally trained compatible agent state and
serve it through the same frozen-model API used by Jormungandr training runs.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from jormungandr.structured import StructuredPolicySpec


STRUCTURED_CHECKPOINT_SCHEMA = "jormungandr.structured_checkpoint.v1"


def structured_checkpoint_payload(
    *,
    model_id: str,
    representation: StructuredPolicySpec | Mapping[str, Any],
    plugin_name: str,
    plugin_version: str,
    agent_state: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    updates: int = 0,
    policy_version: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard payload accepted by the structured service.

    This function deliberately does not interpret the agent state.  The
    selected algorithm plugin remains responsible for validating it when the
    checkpoint is loaded.
    """

    identifier = str(model_id).strip()
    algorithm = str(plugin_name).strip()
    version = str(plugin_version).strip()
    if not identifier:
        raise ValueError("structured checkpoint model_id cannot be empty")
    if not algorithm or not version:
        raise ValueError("structured checkpoint plugin name and version are required")
    if not isinstance(agent_state, Mapping) or not agent_state:
        raise ValueError("structured checkpoint agent state must be a nonempty object")
    if int(updates) < 0 or int(policy_version) < 0:
        raise ValueError("structured checkpoint counters cannot be negative")
    raw_representation = (
        asdict(representation)
        if isinstance(representation, StructuredPolicySpec)
        else dict(representation)
    )
    spec = StructuredPolicySpec(
        global_dim=int(raw_representation.get("global_dim", 0)),
        entity_dim=int(raw_representation.get("entity_dim", 0)),
        candidate_dim=int(raw_representation.get("candidate_dim", 0)),
        entity_type_count=int(raw_representation.get("entity_type_count", 0)),
    )
    return {
        "schema": STRUCTURED_CHECKPOINT_SCHEMA,
        "model_id": identifier,
        "representation": asdict(spec),
        "plugin": {"name": algorithm, "version": version},
        "config": dict(config or {}),
        "updates": int(updates),
        "policy_version": int(policy_version),
        "metadata": dict(metadata or {}),
        "agent": dict(agent_state),
    }


def save_structured_checkpoint(
    path: str | Path,
    **payload_arguments: Any,
) -> Path:
    """Write one standard checkpoint atomically and return its absolute path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = structured_checkpoint_payload(**payload_arguments)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


__all__ = [
    "STRUCTURED_CHECKPOINT_SCHEMA",
    "save_structured_checkpoint",
    "structured_checkpoint_payload",
]
