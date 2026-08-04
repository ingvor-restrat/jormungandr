"""Versioned deployment bundles for entity/candidate policy plugins."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from jormungandr.algorithms import algorithm_registry, canonical_algorithm_name
from jormungandr.structured import StructuredPolicySpec


STRUCTURED_BUNDLE_SCHEMA = "jormungandr.structured_inference_bundle.v1"


def export_structured_policy_bundle(
    output_dir: str | Path,
    *,
    agent: Any,
    spec: StructuredPolicySpec,
    algorithm: str,
    agent_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write a portable structured-policy state and validated JSON manifest."""

    plugin = algorithm_registry.get(canonical_algorithm_name(algorithm))
    if plugin.build_structured is None:
        raise ValueError(f"algorithm {plugin.name!r} has no structured policy")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    state = dict(agent.state_dict())
    # Deployment does not need optimizer moments; load_state_dict accepts their
    # absence while retaining the exact policy parameters and feature schema.
    state.pop("optimizer", None)
    policy_path = destination / "structured_policy.pt"
    torch.save(state, policy_path)
    digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    manifest = {
        "schema": STRUCTURED_BUNDLE_SCHEMA,
        "algorithm": {
            "name": plugin.name,
            "version": plugin.version,
        },
        "representation": asdict(spec),
        "agent_config": dict(agent_config),
        "artifact": {
            "path": policy_path.name,
            "sha256": digest,
        },
        "identity_contract": {
            "entities": "semantic IDs remain aligned to entity rows",
            "candidates": "semantic IDs remain aligned to local candidate rows",
            "selection": "the caller resolves returned local rows by candidate ID",
        },
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **manifest,
        "output_dir": str(destination),
        "manifest_path": str(manifest_path),
        "policy_path": str(policy_path),
    }


def load_structured_policy_bundle(
    output_dir: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Any, Mapping[str, Any]]:
    """Validate and restore a structured deployment bundle."""

    source = Path(output_dir).expanduser().resolve()
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != STRUCTURED_BUNDLE_SCHEMA:
        raise ValueError("unsupported structured inference bundle schema")
    algorithm = manifest.get("algorithm")
    representation = manifest.get("representation")
    agent_config = manifest.get("agent_config")
    artifact = manifest.get("artifact")
    if not all(
        isinstance(value, Mapping)
        for value in (algorithm, representation, agent_config, artifact)
    ):
        raise ValueError("structured inference bundle manifest is incomplete")
    plugin = algorithm_registry.get(
        canonical_algorithm_name(str(algorithm.get("name", "")))
    )
    if str(algorithm.get("version", "")) != plugin.version:
        raise ValueError("structured inference bundle plugin version does not match")
    if plugin.build_structured is None:
        raise ValueError(f"algorithm {plugin.name!r} has no structured builder")
    spec = StructuredPolicySpec(**{
        key: int(representation[key])
        for key in (
            "global_dim",
            "entity_dim",
            "candidate_dim",
            "entity_type_count",
        )
    })
    policy_path = source / str(artifact.get("path", ""))
    digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if digest != str(artifact.get("sha256", "")):
        raise ValueError("structured inference bundle artifact hash does not match")
    agent = plugin.build_structured(spec, dict(agent_config), device)
    state = torch.load(policy_path, map_location=device, weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError("structured inference bundle policy state is invalid")
    agent.load_state_dict(state)
    return agent, manifest
