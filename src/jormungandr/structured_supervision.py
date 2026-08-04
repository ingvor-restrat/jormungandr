"""Reward-free supervision records for state-local structured actions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from jormungandr.structured import (
    EntityCandidateObservation,
    StructuredPolicySpec,
    entity_candidate_observation_from_payload,
    entity_candidate_observation_to_payload,
)


STRUCTURED_SUPERVISION_SCHEMA = "jormungandr.structured_supervision.v1"


@dataclass(frozen=True)
class StructuredSupervisionExample:
    """One weighted semantic label for one state-local action factor."""

    actor_id: str
    episode_id: str
    timestep: int
    observation: EntityCandidateObservation
    factor_id: str
    candidate_ids: tuple[str, ...]
    target_candidate_id: str
    split: str = "train"
    source_group: str = "default"
    factor_group: str = "default"
    target_group: str = "default"
    sample_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actor_id = str(self.actor_id).strip()
        episode_id = str(self.episode_id).strip()
        factor_id = str(self.factor_id).strip()
        if not actor_id or not episode_id or not factor_id or self.timestep < 0:
            raise ValueError(
                "actor, episode, factor, and non-negative timestep are required"
            )
        candidates = tuple(str(value).strip() for value in self.candidate_ids)
        if not candidates or any(not value for value in candidates):
            raise ValueError("supervision candidates must be non-empty")
        if len(set(candidates)) != len(candidates):
            raise ValueError("supervision candidate IDs must be unique")
        observation_candidates = set(self.observation.candidate_ids)
        if not set(candidates).issubset(observation_candidates):
            raise ValueError("supervision candidates are absent from observation")
        illegal = [
            candidate_id
            for candidate_id in candidates
            if not bool(
                self.observation.legal_action_mask[
                    self.observation.candidate_ids.index(candidate_id)
                ]
            )
        ]
        if illegal:
            raise ValueError(
                "supervision candidates must be the factor's legal candidates"
            )
        target = str(self.target_candidate_id).strip()
        if target not in candidates:
            raise ValueError("supervision target is absent from its factor")
        split = "validation" if str(self.split) == "val" else str(self.split)
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        source_group = str(self.source_group).strip()
        factor_group = str(self.factor_group).strip()
        target_group = str(self.target_group).strip()
        weight = float(self.sample_weight)
        if not source_group or not factor_group or not target_group:
            raise ValueError("source, factor, and target groups are required")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("sample weight must be finite and positive")
        object.__setattr__(self, "actor_id", actor_id)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "target_candidate_id", target)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "source_group", source_group)
        object.__setattr__(self, "factor_group", factor_group)
        object.__setattr__(self, "target_group", target_group)
        object.__setattr__(self, "sample_weight", weight)
        object.__setattr__(self, "metadata", dict(self.metadata))


def structured_supervision_to_payload(
    example: StructuredSupervisionExample,
) -> dict[str, Any]:
    return {
        "schema": STRUCTURED_SUPERVISION_SCHEMA,
        "actor_id": example.actor_id,
        "episode_id": example.episode_id,
        "timestep": example.timestep,
        "split": example.split,
        "observation": entity_candidate_observation_to_payload(
            example.observation
        ),
        "factor_id": example.factor_id,
        "candidate_ids": list(example.candidate_ids),
        "target_candidate_id": example.target_candidate_id,
        "source_group": example.source_group,
        "factor_group": example.factor_group,
        "target_group": example.target_group,
        "sample_weight": example.sample_weight,
        "metadata": dict(example.metadata),
    }


def structured_supervision_from_payload(
    payload: Mapping[str, Any],
    *,
    spec: StructuredPolicySpec | None = None,
) -> StructuredSupervisionExample:
    if not isinstance(payload, Mapping):
        raise ValueError("structured supervision example must be an object")
    if payload.get("schema") != STRUCTURED_SUPERVISION_SCHEMA:
        raise ValueError(
            f"structured supervision schema must be {STRUCTURED_SUPERVISION_SCHEMA!r}"
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("supervision metadata must be an object")
    return StructuredSupervisionExample(
        actor_id=str(payload.get("actor_id", "")),
        episode_id=str(payload.get("episode_id", "")),
        timestep=int(payload.get("timestep", -1)),
        observation=entity_candidate_observation_from_payload(
            payload.get("observation", {}), spec=spec
        ),
        factor_id=str(payload.get("factor_id", "")),
        candidate_ids=tuple(payload.get("candidate_ids", ())),
        target_candidate_id=str(payload.get("target_candidate_id", "")),
        split=str(payload.get("split", "train")),
        source_group=str(payload.get("source_group", "default")),
        factor_group=str(payload.get("factor_group", "default")),
        target_group=str(payload.get("target_group", "default")),
        sample_weight=float(payload.get("sample_weight", 1.0)),
        metadata=metadata,
    )
