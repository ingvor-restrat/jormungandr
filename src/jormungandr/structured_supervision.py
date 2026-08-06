"""Reward-free supervision records for state-local structured actions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping, Sequence

from jormungandr.structured import (
    EntityCandidateObservation,
    StructuredPolicySpec,
    entity_candidate_observation_from_payload,
    entity_candidate_observation_to_payload,
)


STRUCTURED_SUPERVISION_SCHEMA = "jormungandr.structured_supervision.v1"
STRUCTURED_SUPERVISION_FRAME_SCHEMA = (
    "jormungandr.structured_supervision_frame.v1"
)


@dataclass(frozen=True)
class StructuredSupervisionExample:
    """One weighted semantic label for one state-local action factor.

    ``target_group`` is the reporting class.  ``balance_group`` is an
    optional, finer training stratum such as a target plus its conditional
    choice set and autoregressive prefix.  Keeping the two names separate
    prevents a coarse metric class from hiding a rare decision boundary.
    Legacy payloads omit ``balance_group`` and therefore balance by
    ``target_group`` exactly as before.
    """

    actor_id: str
    episode_id: str
    timestep: int
    observation: EntityCandidateObservation
    factor_id: str
    candidate_ids: tuple[str, ...]
    target_candidate_id: str
    selected_prefix_candidate_ids: tuple[str, ...] = ()
    split: str = "train"
    source_group: str = "default"
    factor_group: str = "default"
    target_group: str = "default"
    balance_group: str = ""
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
        selected_prefix = tuple(
            str(value).strip()
            for value in self.selected_prefix_candidate_ids
        )
        if any(not value for value in selected_prefix) or (
            len(set(selected_prefix)) != len(selected_prefix)
        ):
            raise ValueError(
                "selected supervision prefix entries must be non-empty and unique"
            )
        if not set(selected_prefix).issubset(observation_candidates):
            raise ValueError("selected supervision prefix is absent from observation")
        if set(selected_prefix).intersection(candidates):
            raise ValueError("selected supervision prefix must precede its factor")
        illegal_prefix = [
            candidate_id
            for candidate_id in selected_prefix
            if not bool(
                self.observation.legal_action_mask[
                    self.observation.candidate_ids.index(candidate_id)
                ]
            )
        ]
        if illegal_prefix:
            raise ValueError("selected supervision prefix must remain legal")
        split = "validation" if str(self.split) == "val" else str(self.split)
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        source_group = str(self.source_group).strip()
        factor_group = str(self.factor_group).strip()
        target_group = str(self.target_group).strip()
        balance_group = str(self.balance_group).strip() or target_group
        weight = float(self.sample_weight)
        if not source_group or not factor_group or not target_group or not balance_group:
            raise ValueError(
                "source, factor, target, and balance groups are required"
            )
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("sample weight must be finite and positive")
        object.__setattr__(self, "actor_id", actor_id)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "target_candidate_id", target)
        object.__setattr__(
            self, "selected_prefix_candidate_ids", selected_prefix
        )
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "source_group", source_group)
        object.__setattr__(self, "factor_group", factor_group)
        object.__setattr__(self, "target_group", target_group)
        object.__setattr__(self, "balance_group", balance_group)
        object.__setattr__(self, "sample_weight", weight)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class StructuredSupervisionLabel:
    """One factor label within a shared-observation supervision frame."""

    factor_id: str
    candidate_ids: tuple[str, ...]
    target_candidate_id: str
    selected_prefix_candidate_ids: tuple[str, ...] = ()
    factor_group: str = "default"
    target_group: str = "default"
    balance_group: str = ""
    sample_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        factor = str(self.factor_id).strip()
        candidates = tuple(str(value).strip() for value in self.candidate_ids)
        target = str(self.target_candidate_id).strip()
        prefix = tuple(
            str(value).strip() for value in self.selected_prefix_candidate_ids
        )
        if not factor or not candidates or any(not value for value in candidates):
            raise ValueError("frame label factor and candidates are required")
        if len(set(candidates)) != len(candidates):
            raise ValueError("frame label candidate IDs must be unique")
        if target not in candidates:
            raise ValueError("frame label target must be one of its candidates")
        if any(not value for value in prefix) or len(set(prefix)) != len(prefix):
            raise ValueError("frame label prefix IDs must be nonempty and unique")
        if set(prefix).intersection(candidates):
            raise ValueError("frame label prefix must precede its factor")
        factor_group = str(self.factor_group).strip()
        target_group = str(self.target_group).strip()
        balance_group = str(self.balance_group).strip() or target_group
        if not factor_group or not target_group or not balance_group:
            raise ValueError("frame label factor, target, and balance groups are required")
        weight = float(self.sample_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("frame label sample weight must be finite and positive")
        object.__setattr__(self, "factor_id", factor)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "target_candidate_id", target)
        object.__setattr__(self, "selected_prefix_candidate_ids", prefix)
        object.__setattr__(self, "factor_group", factor_group)
        object.__setattr__(self, "target_group", target_group)
        object.__setattr__(self, "balance_group", balance_group)
        object.__setattr__(self, "sample_weight", weight)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class StructuredSupervisionFrame:
    """Several factor labels sharing one structured model input.

    This is a storage and compute normalization only. Each label retains its
    own conditional candidate set, prefix, reporting groups, and loss weight.
    """

    actor_id: str
    episode_id: str
    timestep: int
    observation: EntityCandidateObservation
    labels: tuple[StructuredSupervisionLabel, ...]
    split: str = "train"
    source_group: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actor = str(self.actor_id).strip()
        episode = str(self.episode_id).strip()
        labels = tuple(self.labels)
        if not actor or not episode or int(self.timestep) < 0:
            raise ValueError("frame actor, episode, and non-negative timestep are required")
        if not labels:
            raise ValueError("a supervision frame requires at least one label")
        if len({value.factor_id for value in labels}) != len(labels):
            raise ValueError("supervision frame factor IDs must be unique")
        observation_candidates = set(self.observation.candidate_ids)
        legal_candidates = {
            candidate_id
            for candidate_id, legal in zip(
                self.observation.candidate_ids,
                self.observation.legal_action_mask,
                strict=True,
            )
            if bool(legal)
        }
        for label in labels:
            if not set(label.candidate_ids).issubset(observation_candidates):
                raise ValueError("frame label candidates are absent from observation")
            if not set(label.candidate_ids).issubset(legal_candidates):
                raise ValueError("frame label contains an illegal candidate")
            if not set(label.selected_prefix_candidate_ids).issubset(
                observation_candidates
            ):
                raise ValueError("frame label prefix is absent from observation")
            if not set(label.selected_prefix_candidate_ids).issubset(
                legal_candidates
            ):
                raise ValueError("frame label prefix contains an illegal candidate")
        split = "validation" if str(self.split) == "val" else str(self.split).strip()
        if split not in {"train", "validation"}:
            raise ValueError("frame split must be train or validation")
        source_group = str(self.source_group).strip()
        if not source_group:
            raise ValueError("frame source group is required")
        object.__setattr__(self, "actor_id", actor)
        object.__setattr__(self, "episode_id", episode)
        object.__setattr__(self, "timestep", int(self.timestep))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "source_group", source_group)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def label_count(self) -> int:
        return len(self.labels)


def structured_supervision_examples_from_frame(
    frame: StructuredSupervisionFrame,
) -> tuple[StructuredSupervisionExample, ...]:
    """Expand a shared-observation frame into its semantic label records.

    The returned examples all retain the same observation object.  This is a
    reporting and compatibility view; callers can still score the observation
    once and reuse that score for every label in the frame.
    """

    return tuple(
        StructuredSupervisionExample(
            actor_id=frame.actor_id,
            episode_id=frame.episode_id,
            timestep=frame.timestep,
            observation=frame.observation,
            factor_id=label.factor_id,
            candidate_ids=label.candidate_ids,
            target_candidate_id=label.target_candidate_id,
            selected_prefix_candidate_ids=label.selected_prefix_candidate_ids,
            split=frame.split,
            source_group=frame.source_group,
            factor_group=label.factor_group,
            target_group=label.target_group,
            balance_group=label.balance_group,
            sample_weight=label.sample_weight,
            metadata={**dict(frame.metadata), **dict(label.metadata)},
        )
        for label in frame.labels
    )


def structured_supervision_balance_weights(
    examples: Sequence[StructuredSupervisionExample],
    *,
    exponent: float = 0.5,
) -> Mapping[str, float]:
    """Return mean-one inverse-frequency weights for declared balance groups.

    The caller owns the semantics of ``balance_group``.  Jormungandr only
    implements the generic objective ``count(group) ** -exponent`` and derives
    weights from the supplied training split.  ``exponent=0`` is unweighted;
    ``exponent=1`` gives every group equal total mass.
    """

    items = tuple(examples)
    if not items:
        raise ValueError("at least one supervision example is required")
    counts = Counter(item.balance_group for item in items)
    return structured_supervision_balance_weights_from_counts(
        counts, exponent=exponent
    )


def structured_supervision_balance_weights_from_counts(
    counts: Mapping[str, int],
    *,
    exponent: float = 0.5,
) -> Mapping[str, float]:
    """Return mean-one inverse-frequency weights from frozen class counts."""

    normalized = {str(group).strip(): int(count) for group, count in counts.items()}
    power = float(exponent)
    if not normalized or any(not group for group in normalized):
        raise ValueError("supervision balance counts require named groups")
    if any(count <= 0 for count in normalized.values()):
        raise ValueError("supervision balance counts must be positive")
    if not 0.0 <= power <= 1.0:
        raise ValueError("supervision balance exponent must be in [0, 1]")
    total = sum(normalized.values())
    raw = {
        group: (total / count) ** power
        for group, count in normalized.items()
    }
    normalization = sum(
        raw[group] * count for group, count in normalized.items()
    ) / total
    return {
        group: float(value / normalization)
        for group, value in sorted(raw.items())
    }


def apply_structured_supervision_balance_weights(
    examples: Sequence[StructuredSupervisionExample],
    weights: Mapping[str, float],
) -> tuple[StructuredSupervisionExample, ...]:
    """Apply training-derived balance weights to one split.

    Validation may reuse training weights but cannot introduce a new balance
    group silently.  Weight validity is enforced by the supervision record.
    """

    items = tuple(examples)
    missing = {item.balance_group for item in items}.difference(weights)
    if missing:
        raise ValueError(
            "supervision contains unseen balance groups: "
            + ", ".join(sorted(missing))
        )
    return tuple(
        replace(item, sample_weight=float(weights[item.balance_group]))
        for item in items
    )


def structured_supervision_frame_balance_weights(
    frames: Sequence[StructuredSupervisionFrame],
    *,
    exponent: float = 0.5,
) -> Mapping[str, float]:
    """Return mean-one inverse-frequency weights across frame labels.

    Frames are deliberately retained as the unit of observation storage and
    model execution.  Only their lightweight labels are counted here.
    """

    items = tuple(frames)
    if not items:
        raise ValueError("at least one supervision frame is required")
    counts = Counter(
        label.balance_group for frame in items for label in frame.labels
    )
    return structured_supervision_balance_weights_from_counts(
        counts, exponent=exponent
    )


def apply_structured_supervision_frame_balance_weights(
    frames: Sequence[StructuredSupervisionFrame],
    weights: Mapping[str, float],
) -> tuple[StructuredSupervisionFrame, ...]:
    """Apply training-derived class weights while preserving shared frames."""

    items = tuple(frames)
    missing = {
        label.balance_group
        for frame in items
        for label in frame.labels
        if label.balance_group not in weights
    }
    if missing:
        raise ValueError(
            "supervision contains unseen balance groups: "
            + ", ".join(sorted(missing))
        )
    return tuple(
        replace(
            frame,
            labels=tuple(
                replace(
                    label,
                    sample_weight=float(weights[label.balance_group]),
                )
                for label in frame.labels
            ),
        )
        for frame in items
    )


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
        "selected_prefix_candidate_ids": list(
            example.selected_prefix_candidate_ids
        ),
        "source_group": example.source_group,
        "factor_group": example.factor_group,
        "target_group": example.target_group,
        "balance_group": example.balance_group,
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
        selected_prefix_candidate_ids=tuple(
            payload.get("selected_prefix_candidate_ids", ())
        ),
        split=str(payload.get("split", "train")),
        source_group=str(payload.get("source_group", "default")),
        factor_group=str(payload.get("factor_group", "default")),
        target_group=str(payload.get("target_group", "default")),
        balance_group=str(payload.get("balance_group", "")),
        sample_weight=float(payload.get("sample_weight", 1.0)),
        metadata=metadata,
    )


def structured_supervision_frame_to_payload(
    frame: StructuredSupervisionFrame,
) -> dict[str, Any]:
    """Encode a shared-observation frame as one JSON-compatible object."""

    return {
        "schema": STRUCTURED_SUPERVISION_FRAME_SCHEMA,
        "actor_id": frame.actor_id,
        "episode_id": frame.episode_id,
        "timestep": frame.timestep,
        "split": frame.split,
        "source_group": frame.source_group,
        "observation": entity_candidate_observation_to_payload(frame.observation),
        "labels": [
            {
                "factor_id": label.factor_id,
                "candidate_ids": list(label.candidate_ids),
                "target_candidate_id": label.target_candidate_id,
                "selected_prefix_candidate_ids": list(
                    label.selected_prefix_candidate_ids
                ),
                "factor_group": label.factor_group,
                "target_group": label.target_group,
                "balance_group": label.balance_group,
                "sample_weight": label.sample_weight,
                "metadata": dict(label.metadata),
            }
            for label in frame.labels
        ],
        "metadata": dict(frame.metadata),
    }


def structured_supervision_frame_from_payload(
    payload: Mapping[str, Any],
    *,
    spec: StructuredPolicySpec | None = None,
) -> StructuredSupervisionFrame:
    """Validate and decode one shared-observation supervision frame."""

    if not isinstance(payload, Mapping):
        raise ValueError("structured supervision frame must be an object")
    if payload.get("schema") != STRUCTURED_SUPERVISION_FRAME_SCHEMA:
        raise ValueError(
            "structured supervision frame schema must be "
            f"{STRUCTURED_SUPERVISION_FRAME_SCHEMA!r}"
        )
    raw_labels = payload.get("labels", ())
    if not isinstance(raw_labels, Sequence) or isinstance(
        raw_labels, (str, bytes, bytearray)
    ):
        raise ValueError("structured supervision frame labels must be a sequence")
    labels = []
    for raw in raw_labels:
        if not isinstance(raw, Mapping):
            raise ValueError("structured supervision frame label must be an object")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("structured supervision frame label metadata must be an object")
        labels.append(
            StructuredSupervisionLabel(
                factor_id=str(raw.get("factor_id", "")),
                candidate_ids=tuple(raw.get("candidate_ids", ())),
                target_candidate_id=str(raw.get("target_candidate_id", "")),
                selected_prefix_candidate_ids=tuple(
                    raw.get("selected_prefix_candidate_ids", ())
                ),
                factor_group=str(raw.get("factor_group", "default")),
                target_group=str(raw.get("target_group", "default")),
                balance_group=str(raw.get("balance_group", "")),
                sample_weight=float(raw.get("sample_weight", 1.0)),
                metadata=metadata,
            )
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("structured supervision frame metadata must be an object")
    return StructuredSupervisionFrame(
        actor_id=str(payload.get("actor_id", "")),
        episode_id=str(payload.get("episode_id", "")),
        timestep=int(payload.get("timestep", -1)),
        observation=entity_candidate_observation_from_payload(
            payload.get("observation", {}), spec=spec
        ),
        labels=tuple(labels),
        split=str(payload.get("split", "train")),
        source_group=str(payload.get("source_group", "default")),
        metadata=metadata,
    )
