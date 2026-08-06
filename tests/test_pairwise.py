import math

import pytest

from jormungandr.pairwise import (
    PairwiseOutcome,
    fit_bradley_terry,
    outcome_from_values,
    summarize_pairwise_outcomes,
)


def test_outcome_from_values_keeps_margin_separate_from_win_score() -> None:
    narrow = outcome_from_values("a", "b", 11.0, 10.0)
    wide = outcome_from_values("a", "b", 10_000.0, 0.0)

    assert narrow.left_score == wide.left_score == 1.0
    assert narrow.margin == 1.0
    assert wide.margin == 10_000.0


def test_pairwise_outcome_rejects_self_play_and_invalid_scores() -> None:
    with pytest.raises(ValueError, match="distinct"):
        PairwiseOutcome("same", "same", 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PairwiseOutcome("a", "b", 1.1)


def test_bradley_terry_orders_a_consistent_three_agent_panel() -> None:
    outcomes = [
        PairwiseOutcome("alpha", "beta", 1.0, context="seed-1"),
        PairwiseOutcome("beta", "alpha", 0.0, context="seed-2"),
        PairwiseOutcome("alpha", "gamma", 1.0),
        PairwiseOutcome("beta", "gamma", 1.0),
    ]

    fit = fit_bradley_terry(outcomes)
    rows = fit["participants"]

    assert fit["converged"] is True
    assert [row["participant"] for row in rows] == ["alpha", "beta", "gamma"]
    assert rows[0]["rating"] > rows[1]["rating"] > rows[2]["rating"]
    assert math.isclose(
        sum(float(row["rating"]) for row in rows) / len(rows),
        1_000.0,
        abs_tol=1e-9,
    )
    assert fit["external_ladder_estimate"] is False


def test_tie_only_panel_is_centered_and_equal() -> None:
    fit = fit_bradley_terry(
        [
            PairwiseOutcome("left", "right", 0.5),
            PairwiseOutcome("right", "left", 0.5),
        ]
    )

    assert [row["rating"] for row in fit["participants"]] == [1_000.0, 1_000.0]


def test_summary_orients_reversed_games_and_reports_quantity_contract() -> None:
    summary = summarize_pairwise_outcomes(
        [
            outcome_from_values("a", "b", 20.0, 10.0),
            outcome_from_values("b", "a", 5.0, 5.0),
        ]
    )

    pair = summary["pairs"][0]
    assert pair == {
        "first": "a",
        "second": "b",
        "games": 2,
        "total_weight": 2.0,
        "first_wins": 1,
        "ties": 1,
        "second_wins": 0,
        "fractional": 0,
        "first_win_score": 0.75,
        "first_mean_margin": 5.0,
        "first_median_margin": 5.0,
    }
    assert summary["quantity_contract"] == {
        "environment_margin_is_diagnostic_only": True,
        "win_score_drives_pairwise_skill": True,
        "skill_is_local_panel_rating": True,
        "skill_is_external_ladder_rating": False,
    }


def test_fractional_scores_and_weights_affect_fit_without_becoming_counts() -> None:
    summary = summarize_pairwise_outcomes(
        [
            PairwiseOutcome("a", "b", 0.75, weight=2.0),
            PairwiseOutcome("a", "b", 0.25, weight=1.0),
        ]
    )

    row = next(item for item in summary["participants"] if item["participant"] == "a")
    assert row["fractional"] == 2
    assert row["wins"] == row["ties"] == row["losses"] == 0
    assert row["win_score"] == pytest.approx(7.0 / 12.0)
