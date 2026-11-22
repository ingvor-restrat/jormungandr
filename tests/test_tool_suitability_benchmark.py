"""Contracts for independent semantic gating and frozen composition."""

import io
from types import SimpleNamespace

import numpy as np
import pytest

from examples.tool_suitability_benchmark import (
    TRAIN_TEXT,
    HOLDOUT_TEXT,
    TRAIN_DESC,
    HOLDOUT_DESC,
    catalog_decision,
    compose,
    make_world,
    pair_observation,
    parser_accepts,
    score_pairs,
    training_data,
)


def test_holdout_wording_disjoint_and_training_has_positive_negative_pairs():
    assert not set(sum(TRAIN_TEXT, ())) & set(sum(HOLDOUT_TEXT, ()))
    assert not set(sum(TRAIN_DESC, ())) & set(sum(HOLDOUT_DESC, ()))
    data = training_data()
    assert {tuple(y) for _, y in data} == {(0.0, 1.0), (1.0, 0.0)}
    assert all(obs.metadata["llm"]["split"] == "train" for obs, _ in data)
    assert all(obs.metadata["llm"]["history"] == [] for obs, _ in data)


def test_zero_to_four_eligible_and_identity_equivariance():
    ids = ("a", "b", "c", "d")
    for count in range(5):
        scores = np.array([0.9] * count + [0.1] * (4 - count))
        result = catalog_decision(ids, scores)
        assert result["eligible_ids"] == ids[:count]
        assert result["action"] == ("select" if count else "abstain")
        reverse = catalog_decision(ids[::-1], scores[::-1])
        assert set(reverse["eligible_ids"]) == set(ids[:count])
        renamed = catalog_decision(("e", "f", "g", "h"), scores)
        assert renamed["eligible_ids"] == ("e", "f", "g", "h")[:count]


@pytest.mark.parametrize(
    "ids,scores",
    [(("a", "a"), [0.1, 0.9]), (("a",), []), (("a",), [np.nan]), (("a",), [2.0])],
)
def test_invalid_gate_inputs_rejected(ids, scores):
    with pytest.raises(ValueError):
        catalog_decision(ids, scores)


def test_pair_scoring_deduplicates_without_leaking_catalog_identity():
    class Scorer:
        def score_results_structured(self, observations):
            assert len(observations) == 2
            assert all(o.candidate_ids == ("no", "yes") for o in observations)
            return [
                SimpleNamespace(candidate_logits=np.array([0.0, 2.0])),
                SimpleNamespace(candidate_logits=np.array([2.0, 0.0])),
            ]

    pairs = [
        ("translate", "translator"),
        ("summarize", "translator"),
        ("translate", "translator"),
    ]
    logits, scores = score_pairs(Scorer(), pairs)
    assert scores[0] == scores[2] > 0.5 > scores[1]
    np.testing.assert_array_equal(logits[0], logits[2])


def test_abstention_does_not_call_numeric_policy_or_fabricate_feedback():
    class NeverCalled:
        def action_result_structured(self, *args, **kwargs):
            raise AssertionError("empty gate must not call actor")

    trace = io.StringIO()
    result = compose(
        NeverCalled(), make_world(11, 2, 0, 16), np.zeros((16, 4)), 11, trace
    )
    assert result["abstention_rate"] == 1
    assert result["mean_reward"] == 0
    assert result["compatibility"] == 0
    assert len(trace.getvalue().splitlines()) == 16


def test_parser_exclusions_and_unsupported_request():
    description = "Translates text into another language."
    assert parser_accepts("Do not summarize; translate the text.", description)
    assert not parser_accepts("Do not translate; summarize the text.", description)
    assert not parser_accepts("Check for spelling mistakes.", description)


def test_perfect_gate_matches_oracle_with_identical_random_draws():
    from examples.adaptation_benchmark import SPEC, algorithm_registry
    from examples.language_tool_benchmark import evaluate_policy

    agent = algorithm_registry.get("structured_dqn").build_structured(
        SPEC,
        dict(structured_model_dim=16, structured_heads=2, structured_layers=1),
        "cpu",
    )
    world = make_world(11, 2, 0, 16)
    scores = (world.capabilities[None, :] == world.requests[:, None]).astype(float)
    gated = compose(agent, world, scores, 11, io.StringIO())
    oracle = evaluate_policy(
        agent, world, np.zeros((16, 4)), "oracle_numeric", 11, 8, io.StringIO()
    )
    for metric in (
        "mean_reward",
        "compatibility",
        "optimal_action_fraction",
        "mean_expected_regret",
    ):
        assert gated[metric] == oracle[metric]
