"""Controls that make representation and training-coverage comparisons interpretable."""

import json

import numpy as np
import pytest

from examples.tool_gate_comparison import (
    BROAD_TEXT,
    BROAD_DESC,
    FINAL_TEXT,
    FINAL_DESC,
    HOLDOUT_TEXT,
    HOLDOUT_DESC,
    TRAIN_TEXT,
    TRAIN_DESC,
    Case,
    choose_recipe,
    digest_cases,
    evaluation_cases,
    metrics,
    observation,
    parser,
    run,
    training_cases,
)


def test_reserved_wording_is_disjoint_from_training_and_development():
    for training, development, final in (
        (BROAD_TEXT, HOLDOUT_TEXT, FINAL_TEXT),
        (BROAD_DESC, HOLDOUT_DESC, FINAL_DESC),
    ):
        assert not set(sum(final, ())) & set(sum(training, ()) + sum(development, ()))
    assert set(sum(TRAIN_TEXT[:2], ())) <= set(sum(BROAD_TEXT, ()))
    assert set(sum(TRAIN_DESC, ())) <= set(sum(BROAD_DESC, ()))


@pytest.mark.parametrize("seed", [2111, 2129, 2131])
def test_coverage_preserves_catalog_labels_and_exposure_budget(seed):
    narrow = training_cases(seed, "narrow", 1088)
    broad = training_cases(seed, "broad", 1088)
    assert len(narrow) == len(broad) == 544 * 2
    assert [c.labels for c in narrow] == [c.labels for c in broad]
    assert all(sum(c.labels) == 2 for c in narrow + broad)
    assert digest_cases(narrow) == digest_cases(training_cases(seed, "narrow", 1088))
    assert digest_cases(narrow) != digest_cases(broad)


def test_shared_ranking_metric_separates_ordering_from_threshold_failure():
    case = Case("request", ("a", "b", "c", "d"), (False, False, True, True))
    scores = np.array([[-4.0, -3.0, -2.0, -1.0]])
    ranking = metrics([case], scores, "ranking")
    binary = metrics([case], scores, "binary")
    assert ranking["primary_exact_set"] == binary["primary_exact_set"] == 1
    assert binary["native"]["positive_recall"] == 0
    assert binary["native"]["exact_set"] == 0
    assert binary["native"]["none_fit_accuracy"] is None
    assert binary["native"]["by_suitable_count"]["0"] is None
    json.dumps(binary, allow_nan=False)


def test_selection_uses_only_primary_development_metric_with_declared_ties():
    records = [
        dict(representation=r, coverage=c, evaluation=dict(primary_exact_set=0.5))
        for r in ("binary", "ranking")
        for c in ("broad", "narrow")
    ]
    assert choose_recipe(records) == dict(representation="ranking", coverage="narrow")
    records[0]["evaluation"]["primary_exact_set"] = 0.75
    assert choose_recipe(records) == dict(representation="binary", coverage="broad")


def test_primary_and_native_cases_are_explicit_and_history_free():
    for stage in ("development", "confirmation"):
        cases = evaluation_cases(stage)
        assert set(sum(c.labels) for c in cases) == set(range(5))
        assert any(sum(c.labels) == 2 for c in cases)
        obs = observation(cases[0], "train")
        assert not np.any(obs.entity_features) and not np.any(obs.candidate_features)
        assert obs.metadata["llm"]["history"] == []
        assert obs.metadata["llm"]["split"] == "train"
    assert digest_cases(evaluation_cases("development")) != digest_cases(
        evaluation_cases("confirmation")
    )


def test_confirmation_rejects_changed_suite_before_creating_run(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            dict(
                representation="ranking",
                coverage="narrow",
                updates=544,
                reserved_confirmation_digest="changed",
            )
        )
    )
    args = parser().parse_args(
        [
            "--output",
            str(tmp_path / "output"),
            "--numeric-root",
            str(tmp_path),
            "--stage",
            "confirmation",
            "--selection",
            str(selection),
            "--seeds",
            "2129",
        ]
    )
    with pytest.raises(ValueError, match="suite changed"):
        run(args)
    assert not (tmp_path / "output").exists()


def test_comparison_rejects_mismatched_initialization_or_data():
    from examples.tool_gate_comparison import validate_controls

    record = dict(
        seed=11,
        coverage="narrow",
        initial_digest="weights",
        training_catalog_digest="catalogs",
        numeric_artifact=dict(sha256="numeric"),
    )
    validate_controls([record, dict(record)])
    for field, value in [
        ("initial_digest", "different"),
        ("training_catalog_digest", "different"),
    ]:
        with pytest.raises(ValueError):
            validate_controls([record, {**record, field: value}])
    with pytest.raises(ValueError, match="numeric actor"):
        validate_controls(
            [record, {**record, "numeric_artifact": dict(sha256="different")}]
        )
