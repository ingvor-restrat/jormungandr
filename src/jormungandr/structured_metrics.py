"""Exact grouped metrics for structured policy supervision."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .structured_supervision import (
    StructuredSupervisionExample,
    StructuredSupervisionFrame,
    structured_supervision_examples_from_frame,
)
from .structured_trajectory import apply_candidate_prefix_numpy


STRUCTURED_SUPERVISION_POLICY_METRICS_SCHEMA = (
    "jormungandr.structured_supervision_policy_metrics.v1"
)
STRUCTURED_SUPERVISION_FRAME_POLICY_METRICS_SCHEMA = (
    "jormungandr.structured_supervision_frame_policy_metrics.v1"
)


def _score_field(score: Any, name: str, default: Any = None) -> Any:
    if isinstance(score, Mapping):
        return score.get(name, default)
    return getattr(score, name, default)


@dataclass
class _MetricCell:
    count: int = 0
    correct: int = 0
    nll_sum: float = 0.0
    entropy_sum: float = 0.0
    weight_sum: float = 0.0
    weighted_correct_sum: float = 0.0
    weighted_nll_sum: float = 0.0
    weighted_entropy_sum: float = 0.0

    def add(
        self,
        *,
        correct: bool,
        nll: float,
        entropy: float,
        weight: float,
    ) -> None:
        self.count += 1
        self.correct += int(correct)
        self.nll_sum += float(nll)
        self.entropy_sum += float(entropy)
        self.weight_sum += float(weight)
        self.weighted_correct_sum += float(weight) * int(correct)
        self.weighted_nll_sum += float(weight) * float(nll)
        self.weighted_entropy_sum += float(weight) * float(entropy)

    def summary(self) -> Mapping[str, float | int]:
        return {
            "count": self.count,
            "correct": self.correct,
            "accuracy": self.correct / max(1, self.count),
            "nll": self.nll_sum / max(1, self.count),
            "entropy": self.entropy_sum / max(1, self.count),
            "weighted_accuracy": self.weighted_correct_sum
            / max(1e-12, self.weight_sum),
            "weighted_nll": self.weighted_nll_sum
            / max(1e-12, self.weight_sum),
            "weighted_entropy": self.weighted_entropy_sum
            / max(1e-12, self.weight_sum),
            "weight_sum": self.weight_sum,
        }


class StructuredSupervisionMetricsAccumulator:
    """Accumulate prefix-exact metrics without retaining policy scores."""

    def __init__(self) -> None:
        self._cells: dict[tuple[str, str], _MetricCell] = {
            ("overall", "all"): _MetricCell()
        }

    @property
    def examples(self) -> int:
        return self._cells[("overall", "all")].count

    def add(self, example: StructuredSupervisionExample, score: Any) -> bool:
        candidate_ids = tuple(_score_field(score, "candidate_ids", ()))
        if candidate_ids != example.observation.candidate_ids:
            raise ValueError("structured metric candidate IDs changed")
        logits = tuple(float(value) for value in _score_field(score, "candidate_logits", ()))
        if len(logits) != len(candidate_ids):
            raise ValueError("structured metric candidate logits changed cardinality")
        candidate_index = {
            candidate_id: index
            for index, candidate_id in enumerate(candidate_ids)
        }
        prefix_keys = _score_field(score, "candidate_prefix_keys")
        prefix_values = _score_field(score, "candidate_prefix_values")
        conditioned = apply_candidate_prefix_numpy(
            logits,
            [
                candidate_index[candidate_id]
                for candidate_id in example.selected_prefix_candidate_ids
            ],
            candidate_prefix_keys=(prefix_keys if prefix_keys else None),
            candidate_prefix_values=(prefix_values if prefix_values else None),
        )
        factor_logits = np.asarray(
            [conditioned[candidate_index[value]] for value in example.candidate_ids],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(factor_logits)):
            raise ValueError("structured metric factor logits must be finite")
        shifted = factor_logits - float(np.max(factor_logits))
        probabilities = np.exp(shifted)
        probabilities /= float(probabilities.sum())
        target_index = example.candidate_ids.index(example.target_candidate_id)
        prediction = int(np.argmax(factor_logits))
        correct = prediction == target_index
        nll = -math.log(float(probabilities[target_index]))
        entropy = -float(
            np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
        )
        for key in (
            ("overall", "all"),
            ("source", example.source_group),
            ("factor", example.factor_group),
            ("target", example.target_group),
        ):
            self._cells.setdefault(key, _MetricCell()).add(
                correct=correct,
                nll=nll,
                entropy=entropy,
                weight=example.sample_weight,
            )
        return bool(correct)

    def summary(self) -> Mapping[str, Any]:
        if not self.examples:
            raise ValueError("at least one structured metric example is required")
        grouped = {
            group_type: {
                group: cell.summary()
                for (kind, group), cell in sorted(self._cells.items())
                if kind == group_type
            }
            for group_type in ("source", "factor", "target")
        }
        target_accuracies = [
            float(value["accuracy"]) for value in grouped["target"].values()
        ]
        return {
            "schema": STRUCTURED_SUPERVISION_POLICY_METRICS_SCHEMA,
            "examples": self.examples,
            "overall": self._cells[("overall", "all")].summary(),
            "groups": grouped,
            "target_macro_accuracy": float(np.mean(target_accuracies)),
        }


class StructuredSupervisionFrameMetricsAccumulator:
    """Report label and whole-frame accuracy from one score per observation."""

    def __init__(self) -> None:
        self._labels = StructuredSupervisionMetricsAccumulator()
        self._frames = 0
        self._exact_frames = 0
        self._within_frame_accuracy_sum = 0.0

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def examples(self) -> int:
        return self._labels.examples

    def add(self, frame: StructuredSupervisionFrame, score: Any) -> None:
        examples = structured_supervision_examples_from_frame(frame)
        correct = tuple(self._labels.add(example, score) for example in examples)
        agreement = sum(correct) / len(correct)
        self._frames += 1
        self._exact_frames += int(all(correct))
        self._within_frame_accuracy_sum += agreement

    def summary(self) -> Mapping[str, Any]:
        if not self._frames:
            raise ValueError("at least one structured supervision frame is required")
        return {
            "schema": STRUCTURED_SUPERVISION_FRAME_POLICY_METRICS_SCHEMA,
            "frames": self._frames,
            "examples": self.examples,
            "mean_within_frame_accuracy": (
                self._within_frame_accuracy_sum / self._frames
            ),
            "exact_frame_accuracy": self._exact_frames / self._frames,
            "labels": self._labels.summary(),
        }


def structured_supervision_policy_metrics(
    examples: Sequence[StructuredSupervisionExample],
    scores: Sequence[Any],
) -> Mapping[str, Any]:
    """Compute exact raw, weighted, and semantic-group policy metrics."""

    items = tuple(examples)
    scored = tuple(scores)
    if len(items) != len(scored):
        raise ValueError("structured metric examples and scores must align")
    accumulator = StructuredSupervisionMetricsAccumulator()
    for example, score in zip(items, scored, strict=True):
        accumulator.add(example, score)
    return accumulator.summary()


def structured_supervision_frame_policy_metrics(
    frames: Sequence[StructuredSupervisionFrame],
    scores: Sequence[Any],
) -> Mapping[str, Any]:
    """Compute grouped metrics while scoring each shared observation once."""

    items = tuple(frames)
    scored = tuple(scores)
    if len(items) != len(scored):
        raise ValueError("structured metric frames and scores must align")
    accumulator = StructuredSupervisionFrameMetricsAccumulator()
    for frame, score in zip(items, scored, strict=True):
        accumulator.add(frame, score)
    return accumulator.summary()


__all__ = [
    "STRUCTURED_SUPERVISION_FRAME_POLICY_METRICS_SCHEMA",
    "STRUCTURED_SUPERVISION_POLICY_METRICS_SCHEMA",
    "StructuredSupervisionFrameMetricsAccumulator",
    "StructuredSupervisionMetricsAccumulator",
    "structured_supervision_frame_policy_metrics",
    "structured_supervision_policy_metrics",
]
