"""Ranked exposure/feedback records and explicit preference supervision.

No environment rewards are inferred. User choice becomes pairwise preference
only when a caller explicitly requests the chosen-over-displayed assumption.
"""

from dataclasses import dataclass, field
import json
import math
from numbers import Integral
from typing import Mapping

import torch
from torch.nn import functional as F

from jormungandr.structured import (
    EntityCandidateObservation,
    entity_candidate_observation_from_payload,
    entity_candidate_observation_to_payload,
)

EXPOSURE_SCHEMA = "jormungandr.ranked_exposure.v1"
FEEDBACK_SCHEMA = "jormungandr.preference_feedback.v1"


def _id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("identifiers must be nonempty strings")
    return value


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class RankedExposure:
    exposure_id: str
    actor_id: str
    episode_id: str
    timestep: int
    policy_version: int
    shown_at_ns: int
    observation: EntityCandidateObservation
    displayed_ids: tuple[str, ...]
    displayed_scores: tuple[float, ...]
    split: str = "train"
    display_probability: float | None = None

    def __post_init__(self):
        for value in (self.exposure_id, self.actor_id, self.episode_id):
            _id(value)
        for name in ("timestep", "policy_version", "shown_at_ns"):
            _nonnegative_integer(getattr(self, name), name)
        ids = tuple(_id(v) for v in self.displayed_ids)
        scores = tuple(float(v) for v in self.displayed_scores)
        if not ids or len(set(ids)) != len(ids) or len(scores) != len(ids):
            raise ValueError("displayed IDs must be unique and align with scores")
        if not all(math.isfinite(v) for v in scores):
            raise ValueError("displayed scores must be finite")
        legal = {
            k
            for k, allowed in zip(
                self.observation.candidate_ids, self.observation.legal_action_mask
            )
            if allowed
        }
        if not set(ids).issubset(legal):
            raise ValueError("displayed candidates must be observation-legal")
        if self.split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if self.display_probability is not None and (
            not math.isfinite(self.display_probability)
            or not 0 < self.display_probability <= 1
        ):
            raise ValueError("display_probability must be in (0, 1]")
        object.__setattr__(self, "displayed_ids", ids)
        object.__setattr__(self, "displayed_scores", scores)


@dataclass(frozen=True)
class PairwisePreference:
    first_id: str
    second_id: str
    target: float  # 1: prefer first; 0: prefer second; .5: explicit tie.

    def __post_init__(self):
        _id(self.first_id)
        _id(self.second_id)
        if self.first_id == self.second_id or self.target not in (0.0, 0.5, 1.0):
            raise ValueError(
                "comparison requires distinct candidates and target 0, .5 or 1"
            )


@dataclass(frozen=True)
class PreferenceFeedback:
    feedback_id: str
    exposure_id: str
    received_at_ns: int
    selected_id: str | None = None
    abstained: bool = False
    ratings: Mapping[str, float] = field(default_factory=dict)
    comparisons: tuple[PairwisePreference, ...] = ()

    def __post_init__(self):
        _id(self.feedback_id)
        _id(self.exposure_id)
        _nonnegative_integer(self.received_at_ns, "received_at_ns")
        if not isinstance(self.abstained, bool):
            raise ValueError("abstained must be boolean")
        if self.selected_id is not None:
            _id(self.selected_id)
        if self.selected_id is not None and self.abstained:
            raise ValueError("selection and abstention are mutually exclusive")
        ratings = {_id(k): float(v) for k, v in self.ratings.items()}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in ratings.values()):
            raise ValueError("ratings must be normalized to [0, 1]")
        comparisons = tuple(self.comparisons)
        if any(not isinstance(p, PairwisePreference) for p in comparisons):
            raise ValueError("comparisons must be PairwisePreference records")
        pairs = [frozenset((p.first_id, p.second_id)) for p in comparisons]
        if len(pairs) != len(set(pairs)):
            raise ValueError("duplicate or conflicting comparison pairs")
        if (
            self.selected_id is None
            and not self.abstained
            and not ratings
            and not comparisons
        ):
            raise ValueError(
                "feedback must contain a choice, abstention, rating or comparison"
            )
        object.__setattr__(self, "ratings", ratings)
        object.__setattr__(self, "comparisons", comparisons)


def validate_feedback(exposure, feedback):
    if exposure.exposure_id != feedback.exposure_id:
        raise ValueError("feedback does not match exposure")
    if feedback.received_at_ns < exposure.shown_at_ns:
        raise ValueError("feedback receipt precedes exposure")
    referenced = set(feedback.ratings)
    if feedback.selected_id is not None:
        referenced.add(feedback.selected_id)
    for pair in feedback.comparisons:
        referenced.update((pair.first_id, pair.second_id))
    if not referenced.issubset(exposure.displayed_ids):
        raise ValueError("feedback references a candidate not displayed")


def exposure_to_payload(exposure):
    return {
        "schema": EXPOSURE_SCHEMA,
        **{
            k: getattr(exposure, k)
            for k in (
                "exposure_id",
                "actor_id",
                "episode_id",
                "timestep",
                "policy_version",
                "shown_at_ns",
                "split",
                "display_probability",
            )
        },
        "observation": entity_candidate_observation_to_payload(exposure.observation),
        "displayed_ids": list(exposure.displayed_ids),
        "displayed_scores": list(exposure.displayed_scores),
    }


def exposure_from_payload(payload):
    values = dict(payload)
    if values.pop("schema", None) != EXPOSURE_SCHEMA:
        raise ValueError("unsupported ranked exposure schema")
    values["observation"] = entity_candidate_observation_from_payload(
        values["observation"]
    )
    return RankedExposure(**values)


def feedback_to_payload(feedback):
    return {
        "schema": FEEDBACK_SCHEMA,
        **{
            k: getattr(feedback, k)
            for k in (
                "feedback_id",
                "exposure_id",
                "received_at_ns",
                "selected_id",
                "abstained",
            )
        },
        "ratings": dict(feedback.ratings),
        "comparisons": [
            {"first_id": p.first_id, "second_id": p.second_id, "target": p.target}
            for p in feedback.comparisons
        ],
    }


def feedback_from_payload(payload):
    values = dict(payload)
    if values.pop("schema", None) != FEEDBACK_SCHEMA:
        raise ValueError("unsupported preference feedback schema")
    values["comparisons"] = tuple(
        PairwisePreference(**p) for p in values.get("comparisons", ())
    )
    return PreferenceFeedback(**values)


class PreferenceLedger:
    """Bounded, in-memory, idempotent join; externally synchronize concurrent use.

    Stores serialized snapshots. No eviction: full capacity fails explicitly,
    preserving duplicate protection. Persistence/authentication are caller-owned.
    IDs must be unique within this ledger; feedback can arrive before exposure.
    """

    def __init__(self, capacity=10000):
        _nonnegative_integer(capacity, "capacity")
        if capacity == 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._exposures = {}
        self._feedback = {}
        self._feedback_by_exposure = {}

    @staticmethod
    def _canonical(payload):
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def _admit(self, store, key, value):
        if key in store:
            if store[key] != value:
                raise ValueError("conflicting duplicate identifier")
            return False
        if len(store) >= self.capacity:
            raise ValueError("ledger capacity reached; persist/rotate explicitly")
        return True

    def add_exposure(self, exposure):
        value = self._canonical(exposure_to_payload(exposure))
        exposure = exposure_from_payload(json.loads(value))
        if not self._admit(self._exposures, exposure.exposure_id, value):
            return False
        for feedback_id in self._feedback_by_exposure.get(exposure.exposure_id, ()):
            feedback = feedback_from_payload(json.loads(self._feedback[feedback_id]))
            validate_feedback(exposure, feedback)
        self._exposures[exposure.exposure_id] = value
        return True

    def add_feedback(self, feedback):
        value = self._canonical(feedback_to_payload(feedback))
        feedback = feedback_from_payload(json.loads(value))
        if not self._admit(self._feedback, feedback.feedback_id, value):
            return False
        if feedback.exposure_id in self._exposures:
            exposure = exposure_from_payload(
                json.loads(self._exposures[feedback.exposure_id])
            )
            validate_feedback(exposure, feedback)
        self._feedback[feedback.feedback_id] = value
        self._feedback_by_exposure.setdefault(feedback.exposure_id, []).append(
            feedback.feedback_id
        )
        return True

    def resolved(self, *, split):
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        for serialized in self._feedback.values():
            feedback = feedback_from_payload(json.loads(serialized))
            raw = self._exposures.get(feedback.exposure_id)
            if raw is not None:
                exposure = exposure_from_payload(json.loads(raw))
                if exposure.split == split:
                    yield exposure, feedback

    @property
    def pending_count(self):
        return sum(
            json.loads(v)["exposure_id"] not in self._exposures
            for v in self._feedback.values()
        )


@dataclass(frozen=True)
class PreferenceExample:
    exposure: RankedExposure
    feedback_id: str
    comparison: PairwisePreference
    source: str  # explicit or inferred_selection

    def __post_init__(self):
        _id(self.feedback_id)
        if self.source not in {"explicit", "inferred_selection"}:
            raise ValueError("unknown preference source")
        if not {self.comparison.first_id, self.comparison.second_id}.issubset(
            self.exposure.displayed_ids
        ):
            raise ValueError("comparison candidates were not displayed")


def preference_examples(exposure, feedback, *, infer_selection=False):
    """Default: explicit comparisons only. Ratings/abstention are preserved, not converted.

    Opt-in selection inference assumes chosen > every other displayed candidate.
    It is weak preference supervision, never a correctness label. Explicit pairs
    take precedence over inferred pairs for the same unordered candidate pair.
    """
    validate_feedback(exposure, feedback)
    rows = [
        PreferenceExample(exposure, feedback.feedback_id, p, "explicit")
        for p in feedback.comparisons
    ]
    seen = {frozenset((p.first_id, p.second_id)) for p in feedback.comparisons}
    if infer_selection and feedback.selected_id is not None:
        for other in exposure.displayed_ids:
            if (
                other != feedback.selected_id
                and frozenset((other, feedback.selected_id)) not in seen
            ):
                rows.append(
                    PreferenceExample(
                        exposure,
                        feedback.feedback_id,
                        PairwisePreference(feedback.selected_id, other, 1.0),
                        "inferred_selection",
                    )
                )
    return tuple(rows)


def pairwise_preference_loss(first, second, targets, *, weights=None):
    """Mean logistic loss on score differences; soft .5 targets encode ties.

    Weights multiply rows without renormalization, preserving microbatch means.
    Scores are relative utilities, not independent correctness probabilities.
    """
    if (
        first.ndim != 1
        or first.numel() == 0
        or second.shape != first.shape
        or targets.shape != first.shape
    ):
        raise ValueError("scores and targets must be aligned nonempty vectors")
    if any(
        t.device != first.device
        or not t.is_floating_point()
        or not torch.isfinite(t).all()
        for t in (first, second, targets)
    ):
        raise ValueError(
            "scores and targets must be finite floating tensors on one device"
        )
    if ((targets < 0) | (targets > 1)).any():
        raise ValueError("targets must be in [0, 1]")
    losses = F.binary_cross_entropy_with_logits(
        first - second, targets, reduction="none"
    )
    if weights is not None:
        if (
            weights.shape != first.shape
            or weights.device != first.device
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("weights must be aligned, finite and nonnegative")
        losses = losses * weights
    return losses.mean()


def preference_batch_loss(candidate_logits, examples, *, weights=None):
    """Training-only loss over scores aligned with each original observation catalog."""
    if (
        not examples
        or candidate_logits.ndim != 2
        or len(candidate_logits) != len(examples)
    ):
        raise ValueError("logits must align with a nonempty example batch")
    if any(e.exposure.split != "train" for e in examples):
        raise ValueError("only training preferences may update a model")
    if any(
        candidate_logits.shape[1] < len(e.exposure.observation.candidate_ids)
        for e in examples
    ):
        raise ValueError("logits do not cover observation candidates")
    device = candidate_logits.device
    rows = torch.arange(len(examples), device=device)
    first = torch.tensor(
        [
            e.exposure.observation.candidate_ids.index(e.comparison.first_id)
            for e in examples
        ],
        device=device,
    )
    second = torch.tensor(
        [
            e.exposure.observation.candidate_ids.index(e.comparison.second_id)
            for e in examples
        ],
        device=device,
    )
    targets = candidate_logits.new_tensor([e.comparison.target for e in examples])
    return pairwise_preference_loss(
        candidate_logits[rows, first],
        candidate_logits[rows, second],
        targets,
        weights=weights,
    )


def ranked_shortlist(observation, candidate_scores, *, k):
    """Sort legal candidates by descending score, breaking ties by semantic ID.

    Returned scores are not acceptance probabilities. Presentation/exploration
    policies can subsequently reorder this list; record the actual shown order.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    scores = tuple(float(v) for v in candidate_scores)
    if len(scores) != len(observation.candidate_ids):
        raise ValueError("scores must align with observation candidates")
    candidates = [
        (key, score)
        for key, score, legal in zip(
            observation.candidate_ids, scores, observation.legal_action_mask
        )
        if legal
    ]
    if not all(math.isfinite(score) for _, score in candidates):
        raise ValueError("legal candidate scores must be finite")
    return tuple(sorted(candidates, key=lambda item: (-item[1], item[0]))[:k])
