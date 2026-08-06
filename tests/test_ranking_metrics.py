import pytest

from jormungandr.ranking_metrics import (
    RankingMetricsAccumulator,
    SelectionMultisetMetricsAccumulator,
    SelectionSetMetricsAccumulator,
)


def test_ranking_metrics_report_top_k_mrr_and_missing_references() -> None:
    metrics = RankingMetricsAccumulator(top_ks=(1, 2, 3))
    metrics.add({"b": 2.0, "a": 2.0, "c": 1.0}, ("b",))
    metrics.add({"x": 0.0, "y": -1.0}, ("absent",))

    report = metrics.summary()

    # Equal scores use identifier order, so b is deterministically second.
    assert report["queries"] == 2
    assert report["reference_items"] == 2
    assert report["supported_reference_items"] == 1
    assert report["support_rate"] == 0.5
    assert report["mean_reciprocal_rank"] == 0.25
    assert report["mean_supported_rank"] == 2.0
    assert report["mean_supported_normalized_rank"] == 0.5
    assert report["top_k_recall"] == {"1": 0.0, "2": 0.5, "3": 0.5}


def test_selection_set_metrics_report_micro_macro_and_exact_agreement() -> None:
    metrics = SelectionSetMetricsAccumulator()
    metrics.add(("a", "b"), ("b", "c"))
    metrics.add((), ())

    report = metrics.summary()

    assert report["queries"] == 2
    assert report["reference_items"] == 2
    assert report["predicted_items"] == 2
    assert report["true_positive_items"] == 1
    assert report["micro_precision"] == 0.5
    assert report["micro_recall"] == 0.5
    assert report["micro_f1"] == 0.5
    assert report["exact_query_rate"] == 0.5
    assert report["macro_precision"] == 0.25
    assert report["macro_recall"] == 0.25
    assert report["macro_f1"] == 0.25


def test_ranking_and_selection_metrics_reject_ambiguous_inputs() -> None:
    ranking = RankingMetricsAccumulator()
    with pytest.raises(ValueError, match="finite"):
        ranking.add({"bad": float("nan")}, ("bad",))
    with pytest.raises(ValueError, match="unique"):
        ranking.add({"a": 1.0}, ("a", "a"))

    selection = SelectionSetMetricsAccumulator()
    with pytest.raises(ValueError, match="unique"):
        selection.add(("a", "a"), ())


def test_selection_multiset_metrics_preserve_repeated_work() -> None:
    metrics = SelectionMultisetMetricsAccumulator()
    metrics.add(("pickup", "pickup", "water"), ("pickup", "water", "water"))

    report = metrics.summary()

    assert report["reference_items"] == 3
    assert report["predicted_items"] == 3
    assert report["true_positive_items"] == 2
    assert report["micro_precision"] == pytest.approx(2 / 3)
    assert report["micro_recall"] == pytest.approx(2 / 3)
    assert report["micro_f1"] == pytest.approx(2 / 3)
    assert report["exact_query_rate"] == 0.0
    assert report["per_identifier"]["pickup"] == {
        "reference_items": 2,
        "predicted_items": 1,
        "true_positive_items": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
    }
    assert report["per_identifier"]["water"]["precision"] == 0.5
    assert report["per_identifier"]["water"]["recall"] == 1.0
