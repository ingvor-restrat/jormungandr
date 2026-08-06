"""Domain-neutral pairwise outcome summaries and local skill estimates.

The module deliberately distinguishes three quantities that are often
confused in competitive-control experiments:

* an environment value or margin;
* a win/tie/loss score in ``[0, 1]``; and
* a fitted skill rating inferred from a panel of pairwise outcomes.

The Bradley--Terry fit treats a tie as half a win for each participant and
uses an explicit L2 prior.  The result is a reproducible *local panel rating*,
not an estimate of any external ladder's hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PAIRWISE_SUMMARY_SCHEMA = "jormungandr.pairwise_summary.v1"


@dataclass(frozen=True)
class PairwiseOutcome:
    """One observed result, oriented from ``left`` to ``right``.

    ``left_score`` must be zero for a loss, one half for a tie, or one for a
    win.  Fractional scores other than one half are accepted so callers can
    aggregate probabilistic or multi-round comparisons without changing the
    fitting code.  ``margin`` is diagnostic only and never affects skill.
    """

    left: str
    right: str
    left_score: float
    margin: float | None = None
    weight: float = 1.0
    context: str = ""

    def __post_init__(self) -> None:
        if not self.left.strip() or not self.right.strip():
            raise ValueError("pairwise participant names cannot be empty")
        if self.left == self.right:
            raise ValueError("a pairwise outcome requires distinct participants")
        if not math.isfinite(float(self.left_score)) or not (
            0.0 <= float(self.left_score) <= 1.0
        ):
            raise ValueError("left_score must be finite and in [0, 1]")
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("weight must be finite and positive")
        if self.margin is not None and not math.isfinite(float(self.margin)):
            raise ValueError("margin must be finite when supplied")


def outcome_from_values(
    left: str,
    right: str,
    left_value: float,
    right_value: float,
    *,
    weight: float = 1.0,
    context: str = "",
) -> PairwiseOutcome:
    """Create a W/L/T outcome while retaining the oriented value margin."""

    left_number = float(left_value)
    right_number = float(right_value)
    if not math.isfinite(left_number) or not math.isfinite(right_number):
        raise ValueError("pairwise values must be finite")
    score = 1.0 if left_number > right_number else 0.5 if left_number == right_number else 0.0
    return PairwiseOutcome(
        left=left,
        right=right,
        left_score=score,
        margin=left_number - right_number,
        weight=weight,
        context=context,
    )


def pairwise_outcome_from_payload(payload: Mapping[str, Any]) -> PairwiseOutcome:
    """Parse and validate a JSON-compatible pairwise outcome."""

    return PairwiseOutcome(
        left=str(payload["left"]),
        right=str(payload["right"]),
        left_score=float(payload["left_score"]),
        margin=(
            None if payload.get("margin") is None else float(payload["margin"])
        ),
        weight=float(payload.get("weight", 1.0)),
        context=str(payload.get("context", "")),
    )


def _coerce_outcomes(
    outcomes: Iterable[PairwiseOutcome | Mapping[str, Any]],
) -> tuple[PairwiseOutcome, ...]:
    result = tuple(
        item
        if isinstance(item, PairwiseOutcome)
        else pairwise_outcome_from_payload(item)
        for item in outcomes
    )
    if not result:
        raise ValueError("at least one pairwise outcome is required")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def fit_bradley_terry(
    outcomes: Iterable[PairwiseOutcome | Mapping[str, Any]],
    *,
    l2_prior: float = 1.0,
    rating_center: float = 1_000.0,
    rating_scale: float = 400.0 / math.log(10.0),
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> Mapping[str, Any]:
    """Fit regularized Bradley--Terry strengths with deterministic Newton steps.

    The L2 prior makes separated and disconnected finite panels identifiable.
    Ratings are centered after every step.  ``rating_scale`` defaults to the
    conventional transformation where a 400-point difference corresponds to
    10:1 odds, but the returned metadata makes the convention explicit.
    """

    records = _coerce_outcomes(outcomes)
    if not math.isfinite(float(l2_prior)) or float(l2_prior) <= 0.0:
        raise ValueError("l2_prior must be finite and positive")
    if not math.isfinite(float(rating_center)):
        raise ValueError("rating_center must be finite")
    if not math.isfinite(float(rating_scale)) or float(rating_scale) <= 0.0:
        raise ValueError("rating_scale must be finite and positive")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")

    participants = tuple(
        sorted({record.left for record in records} | {record.right for record in records})
    )
    index = {name: offset for offset, name in enumerate(participants)}
    strengths = np.zeros(len(participants), dtype=np.float64)
    converged = False
    iterations = 0

    for iteration in range(1, int(max_iterations) + 1):
        gradient = -float(l2_prior) * strengths
        information = np.eye(len(participants), dtype=np.float64) * float(l2_prior)
        for record in records:
            left_index = index[record.left]
            right_index = index[record.right]
            probability = _sigmoid(
                float(strengths[left_index] - strengths[right_index])
            )
            residual = float(record.weight) * (
                float(record.left_score) - probability
            )
            gradient[left_index] += residual
            gradient[right_index] -= residual
            curvature = float(record.weight) * probability * (1.0 - probability)
            information[left_index, left_index] += curvature
            information[right_index, right_index] += curvature
            information[left_index, right_index] -= curvature
            information[right_index, left_index] -= curvature

        delta = np.linalg.solve(information, gradient)
        strengths += delta
        strengths -= float(np.mean(strengths))
        iterations = iteration
        if float(np.max(np.abs(delta))) <= float(tolerance):
            converged = True
            break

    objective = -0.5 * float(l2_prior) * float(np.dot(strengths, strengths))
    for record in records:
        probability = _sigmoid(
            float(strengths[index[record.left]] - strengths[index[record.right]])
        )
        probability = min(max(probability, 1e-15), 1.0 - 1e-15)
        objective += float(record.weight) * (
            float(record.left_score) * math.log(probability)
            + (1.0 - float(record.left_score)) * math.log(1.0 - probability)
        )

    rows = [
        {
            "participant": name,
            "strength_logit": float(strengths[index[name]]),
            "rating": float(
                float(rating_center) + float(rating_scale) * strengths[index[name]]
            ),
        }
        for name in participants
    ]
    rows.sort(key=lambda row: (-float(row["rating"]), str(row["participant"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "method": "regularized_bradley_terry_fractional_ties",
        "participants": rows,
        "converged": converged,
        "iterations": iterations,
        "objective": float(objective),
        "l2_prior": float(l2_prior),
        "rating_center": float(rating_center),
        "rating_scale": float(rating_scale),
        "games": len(records),
        "total_weight": float(sum(record.weight for record in records)),
        "external_ladder_estimate": False,
    }


def _result_label(score: float) -> str:
    if math.isclose(score, 1.0):
        return "win"
    if math.isclose(score, 0.5):
        return "tie"
    if math.isclose(score, 0.0):
        return "loss"
    return "fractional"


def summarize_pairwise_outcomes(
    outcomes: Iterable[PairwiseOutcome | Mapping[str, Any]],
    *,
    l2_prior: float = 1.0,
    rating_center: float = 1_000.0,
) -> Mapping[str, Any]:
    """Return participant, unordered-pair, and local-rating summaries."""

    records = _coerce_outcomes(outcomes)
    names = tuple(
        sorted({record.left for record in records} | {record.right for record in records})
    )

    participant_rows: list[dict[str, Any]] = []
    for name in names:
        scores: list[float] = []
        margins: list[float] = []
        weights: list[float] = []
        labels: list[str] = []
        for record in records:
            if record.left == name:
                score = float(record.left_score)
                margin = record.margin
            elif record.right == name:
                score = 1.0 - float(record.left_score)
                margin = None if record.margin is None else -float(record.margin)
            else:
                continue
            scores.append(score)
            weights.append(float(record.weight))
            labels.append(_result_label(score))
            if margin is not None:
                margins.append(float(margin))
        total_weight = float(sum(weights))
        weighted_score = float(
            sum(score * weight for score, weight in zip(scores, weights))
        )
        participant_rows.append(
            {
                "participant": name,
                "games": len(scores),
                "total_weight": total_weight,
                "wins": labels.count("win"),
                "ties": labels.count("tie"),
                "losses": labels.count("loss"),
                "fractional": labels.count("fractional"),
                "win_score": weighted_score / total_weight,
                "mean_margin": (
                    None if not margins else float(np.mean(margins))
                ),
                "median_margin": (
                    None if not margins else float(np.median(margins))
                ),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for left_offset, first in enumerate(names):
        for second in names[left_offset + 1 :]:
            oriented: list[PairwiseOutcome] = []
            for record in records:
                if record.left == first and record.right == second:
                    oriented.append(record)
                elif record.left == second and record.right == first:
                    oriented.append(
                        PairwiseOutcome(
                            left=first,
                            right=second,
                            left_score=1.0 - float(record.left_score),
                            margin=(
                                None
                                if record.margin is None
                                else -float(record.margin)
                            ),
                            weight=float(record.weight),
                            context=record.context,
                        )
                    )
            if not oriented:
                continue
            total_weight = float(sum(record.weight for record in oriented))
            scores = [float(record.left_score) for record in oriented]
            margins = [
                float(record.margin)
                for record in oriented
                if record.margin is not None
            ]
            labels = [_result_label(score) for score in scores]
            pair_rows.append(
                {
                    "first": first,
                    "second": second,
                    "games": len(oriented),
                    "total_weight": total_weight,
                    "first_wins": labels.count("win"),
                    "ties": labels.count("tie"),
                    "second_wins": labels.count("loss"),
                    "fractional": labels.count("fractional"),
                    "first_win_score": float(
                        sum(
                            float(record.left_score) * float(record.weight)
                            for record in oriented
                        )
                        / total_weight
                    ),
                    "first_mean_margin": (
                        None if not margins else float(np.mean(margins))
                    ),
                    "first_median_margin": (
                        None if not margins else float(np.median(margins))
                    ),
                }
            )

    skill = fit_bradley_terry(
        records,
        l2_prior=l2_prior,
        rating_center=rating_center,
    )
    rating_by_name = {
        str(row["participant"]): float(row["rating"])
        for row in skill["participants"]
    }
    participant_rows.sort(
        key=lambda row: (
            -rating_by_name[str(row["participant"])],
            str(row["participant"]),
        )
    )
    return {
        "schema": PAIRWISE_SUMMARY_SCHEMA,
        "games": len(records),
        "participants": participant_rows,
        "pairs": pair_rows,
        "skill": skill,
        "quantity_contract": {
            "environment_margin_is_diagnostic_only": True,
            "win_score_drives_pairwise_skill": True,
            "skill_is_local_panel_rating": True,
            "skill_is_external_ladder_rating": False,
        },
    }

