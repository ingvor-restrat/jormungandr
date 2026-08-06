"""Streaming metrics for ranked candidates and selected sets.

The caller owns candidate meaning, scores, and reference labels. This module
only measures whether reference identifiers occur near the top of a declared
ranking and whether a predicted set agrees with a reference set. It therefore
applies to task assignment, retrieval, tool selection, and other structured
policies without introducing application-specific semantics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any, Mapping, Sequence


RANKING_METRICS_SCHEMA = "jormungandr.ranking_metrics.v1"
SELECTION_SET_METRICS_SCHEMA = "jormungandr.selection_set_metrics.v1"
SELECTION_MULTISET_METRICS_SCHEMA = "jormungandr.selection_multiset_metrics.v2"


def _identifiers(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"{name} identifiers cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} identifiers must be unique")
    return result


class RankingMetricsAccumulator:
    """Accumulate deterministic per-reference ranks without retaining scores.

    Equal scores are ordered by identifier. This makes the metric exactly
    reproducible and avoids accidental credit for an unspecified tie break.
    """

    def __init__(self, *, top_ks: Sequence[int] = (1, 3, 5, 10)) -> None:
        values = tuple(sorted({int(value) for value in top_ks}))
        if not values or any(value <= 0 for value in values):
            raise ValueError("ranking top-k cutoffs must be positive")
        self.top_ks = values
        self.queries = 0
        self.candidates = 0
        self.reference_items = 0
        self.supported_reference_items = 0
        self.missing_reference_items = 0
        self.reciprocal_rank_sum = 0.0
        self.rank_sum = 0.0
        self.normalized_rank_sum = 0.0
        self.top_k_hits = {value: 0 for value in values}

    def add(
        self,
        candidate_scores: Mapping[str, float],
        reference_ids: Sequence[str],
    ) -> None:
        scores = {str(key): float(value) for key, value in candidate_scores.items()}
        if any(not key for key in scores):
            raise ValueError("ranking candidate identifiers cannot be empty")
        if any(not math.isfinite(value) for value in scores.values()):
            raise ValueError("ranking candidate scores must be finite")
        references = _identifiers(reference_ids, name="ranking reference")
        if not references:
            raise ValueError("ranking query requires at least one reference")

        ordered = tuple(
            key
            for key, _ in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )
        )
        rank_by_id = {key: index + 1 for index, key in enumerate(ordered)}
        self.queries += 1
        self.candidates += len(ordered)
        self.reference_items += len(references)
        for reference in references:
            rank = rank_by_id.get(reference)
            if rank is None:
                self.missing_reference_items += 1
                continue
            self.supported_reference_items += 1
            self.rank_sum += rank
            self.reciprocal_rank_sum += 1.0 / rank
            self.normalized_rank_sum += (
                0.0 if len(ordered) <= 1 else (rank - 1) / (len(ordered) - 1)
            )
            for cutoff in self.top_ks:
                self.top_k_hits[cutoff] += int(rank <= cutoff)

    def summary(self) -> Mapping[str, Any]:
        if not self.queries:
            raise ValueError("at least one ranking query is required")
        supported = max(1, self.supported_reference_items)
        total = max(1, self.reference_items)
        return {
            "schema": RANKING_METRICS_SCHEMA,
            "queries": self.queries,
            "candidates": self.candidates,
            "mean_candidates_per_query": self.candidates / self.queries,
            "reference_items": self.reference_items,
            "supported_reference_items": self.supported_reference_items,
            "missing_reference_items": self.missing_reference_items,
            "support_rate": self.supported_reference_items / total,
            "mean_reciprocal_rank": self.reciprocal_rank_sum / total,
            "mean_supported_rank": self.rank_sum / supported,
            "mean_supported_normalized_rank": self.normalized_rank_sum / supported,
            "top_k_recall": {
                str(cutoff): self.top_k_hits[cutoff] / total
                for cutoff in self.top_ks
            },
        }


@dataclass
class SelectionSetMetricsAccumulator:
    """Accumulate micro and per-query agreement for identifier sets."""

    queries: int = 0
    reference_items: int = 0
    predicted_items: int = 0
    true_positive_items: int = 0
    exact_queries: int = 0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    f1_sum: float = 0.0

    def add(
        self,
        reference_ids: Sequence[str],
        predicted_ids: Sequence[str],
    ) -> None:
        reference = set(_identifiers(reference_ids, name="selection reference"))
        predicted = set(_identifiers(predicted_ids, name="selection prediction"))
        overlap = len(reference.intersection(predicted))
        precision = overlap / max(1, len(predicted))
        recall = overlap / max(1, len(reference))
        f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        self.queries += 1
        self.reference_items += len(reference)
        self.predicted_items += len(predicted)
        self.true_positive_items += overlap
        self.exact_queries += int(reference == predicted)
        self.precision_sum += precision
        self.recall_sum += recall
        self.f1_sum += f1

    def summary(self) -> Mapping[str, Any]:
        if not self.queries:
            raise ValueError("at least one selection-set query is required")
        precision = self.true_positive_items / max(1, self.predicted_items)
        recall = self.true_positive_items / max(1, self.reference_items)
        micro_f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        return {
            "schema": SELECTION_SET_METRICS_SCHEMA,
            "queries": self.queries,
            "reference_items": self.reference_items,
            "predicted_items": self.predicted_items,
            "true_positive_items": self.true_positive_items,
            "exact_query_rate": self.exact_queries / self.queries,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": micro_f1,
            "macro_precision": self.precision_sum / self.queries,
            "macro_recall": self.recall_sum / self.queries,
            "macro_f1": self.f1_sum / self.queries,
            "mean_reference_cardinality": self.reference_items / self.queries,
            "mean_predicted_cardinality": self.predicted_items / self.queries,
        }


@dataclass
class SelectionMultisetMetricsAccumulator:
    """Accumulate agreement while retaining repeated identifier counts."""

    queries: int = 0
    reference_items: int = 0
    predicted_items: int = 0
    true_positive_items: int = 0
    exact_queries: int = 0
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    f1_sum: float = 0.0
    reference_by_identifier: Counter[str] = field(default_factory=Counter)
    predicted_by_identifier: Counter[str] = field(default_factory=Counter)
    true_positive_by_identifier: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        reference_ids: Sequence[str],
        predicted_ids: Sequence[str],
    ) -> None:
        reference_values = tuple(str(value) for value in reference_ids)
        predicted_values = tuple(str(value) for value in predicted_ids)
        if any(not value for value in reference_values + predicted_values):
            raise ValueError("selection multiset identifiers cannot be empty")
        reference = Counter(reference_values)
        predicted = Counter(predicted_values)
        overlap = sum(
            min(count, predicted.get(identifier, 0))
            for identifier, count in reference.items()
        )
        precision = overlap / max(1, len(predicted_values))
        recall = overlap / max(1, len(reference_values))
        f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        self.queries += 1
        self.reference_items += len(reference_values)
        self.predicted_items += len(predicted_values)
        self.true_positive_items += overlap
        self.exact_queries += int(reference == predicted)
        self.precision_sum += precision
        self.recall_sum += recall
        self.f1_sum += f1
        self.reference_by_identifier.update(reference)
        self.predicted_by_identifier.update(predicted)
        self.true_positive_by_identifier.update(
            {
                identifier: min(count, predicted.get(identifier, 0))
                for identifier, count in reference.items()
            }
        )

    def summary(self) -> Mapping[str, Any]:
        if not self.queries:
            raise ValueError("at least one selection-multiset query is required")
        precision = self.true_positive_items / max(1, self.predicted_items)
        recall = self.true_positive_items / max(1, self.reference_items)
        micro_f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        identifiers = sorted(
            set(self.reference_by_identifier) | set(self.predicted_by_identifier)
        )
        per_identifier = {}
        for identifier in identifiers:
            reference_count = self.reference_by_identifier[identifier]
            predicted_count = self.predicted_by_identifier[identifier]
            true_positive_count = self.true_positive_by_identifier[identifier]
            identifier_precision = true_positive_count / max(1, predicted_count)
            identifier_recall = true_positive_count / max(1, reference_count)
            identifier_f1 = (
                0.0
                if identifier_precision + identifier_recall == 0.0
                else 2.0
                * identifier_precision
                * identifier_recall
                / (identifier_precision + identifier_recall)
            )
            per_identifier[identifier] = {
                "reference_items": reference_count,
                "predicted_items": predicted_count,
                "true_positive_items": true_positive_count,
                "precision": identifier_precision,
                "recall": identifier_recall,
                "f1": identifier_f1,
            }
        return {
            "schema": SELECTION_MULTISET_METRICS_SCHEMA,
            "queries": self.queries,
            "reference_items": self.reference_items,
            "predicted_items": self.predicted_items,
            "true_positive_items": self.true_positive_items,
            "exact_query_rate": self.exact_queries / self.queries,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": micro_f1,
            "macro_precision": self.precision_sum / self.queries,
            "macro_recall": self.recall_sum / self.queries,
            "macro_f1": self.f1_sum / self.queries,
            "mean_reference_cardinality": self.reference_items / self.queries,
            "mean_predicted_cardinality": self.predicted_items / self.queries,
            "per_identifier": per_identifier,
        }


__all__ = [
    "RANKING_METRICS_SCHEMA",
    "SELECTION_SET_METRICS_SCHEMA",
    "SELECTION_MULTISET_METRICS_SCHEMA",
    "RankingMetricsAccumulator",
    "SelectionMultisetMetricsAccumulator",
    "SelectionSetMetricsAccumulator",
]
