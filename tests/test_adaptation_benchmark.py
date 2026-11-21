"""Tests for the generic example's causal protocol and measurements."""

from argparse import Namespace
from dataclasses import replace
import json

import numpy as np
import pytest

from examples.adaptation_benchmark import (
    make_scenario,
    observation,
    rolling_probabilities,
    rollout,
    run,
    trial_metrics,
)
from jormungandr.llm_context import AgentContext, Experience


def test_scenarios_are_reproducible_disjoint_and_pairable():
    a = make_scenario(11, 2, 0, 32)
    b = make_scenario(11, 2, 0, 32)
    train = make_scenario(11, 1, 0, 32)
    assert np.array_equal(a.successes, b.successes)
    assert np.array_equal(a.probabilities, b.probabilities)
    assert not np.array_equal(a.successes, train.successes)
    assert a.identifier != train.identifier
    assert len(a.changes) == 2
    assert len(train.changes) == 1
    assert make_scenario(11, 2, 1, 32).family == "drift"
    assert a.reward(3, 0) == b.reward(3, 0)
    with pytest.raises(ValueError):
        a.successes[0, 0] = 0


def test_hidden_future_is_not_in_observation_and_no_history_is_a_real_ablation():
    scenario = make_scenario(23, 2, 1, 32)
    changed = replace(
        scenario,
        probabilities=1 - scenario.probabilities,
        successes=1 - scenario.successes,
    )
    context = AgentContext("a", "trial", "validation", max_events=4)
    a, b = observation(scenario, context), observation(changed, context)
    assert a.metadata == b.metadata
    assert np.array_equal(a.entity_features, b.entity_features)
    assert np.array_equal(a.candidate_features, b.candidate_features)
    context = context.append(
        Experience("previous", 0, "request", "route-a", "success", 0.9, 0)
    )
    raw = observation(scenario, context)
    numeric = observation(scenario, context, numeric_history=True)
    assert not raw.entity_features.any()
    assert numeric.entity_features[0, 0] == pytest.approx(0.9)
    assert len(raw.metadata["llm"]["history"]) == 1
    reset = observation(
        scenario, AgentContext("a", "trial", "validation", max_events=0)
    )
    assert not reset.entity_features.any()
    assert not reset.metadata["llm"]["history"]


def test_rolling_baseline_uses_only_observed_rewards_and_beats_random_in_slow_switches():
    empty = AgentContext("a", "trial", "validation", max_events=8)
    assert np.array_equal(rolling_probabilities(empty), [0.5, 0.5])
    one = empty.append(Experience("e", 0, "o", "route-a", "failure", -0.1, 0))
    assert np.array_equal(rolling_probabilities(one), [0.0, 1.0])
    random, rolling = [], []
    for seed in range(8):
        scenario = make_scenario(seed, 1, 0, 64)
        random.append(
            rollout(scenario, method="random", history=8, seed=seed)[1][
                "mean_expected_regret"
            ]
        )
        rolling.append(
            rollout(scenario, method="rolling", history=8, seed=seed)[1][
                "mean_expected_regret"
            ]
        )
    assert np.mean(rolling) < np.mean(random) - 0.1


def test_regret_oracle_and_censored_recovery():
    scenario = make_scenario(11, 2, 0, 32)
    best = scenario.utilities.argmax(axis=1)
    p = np.full((32, 2), 0.5)
    oracle = trial_metrics(scenario, best, np.ones(32), p, np.zeros(32))
    worst = trial_metrics(scenario, 1 - best, np.zeros(32), p, np.zeros(32))
    assert oracle["mean_expected_regret"] == 0
    assert oracle["recovery_censored_fraction"] == 0
    assert all(r["delay"] == 4 for r in oracle["recovery"])
    assert worst["recovery_censored_fraction"] == 1
    assert all(r["delay"] is None for r in worst["recovery"])


def test_numeric_benchmark_runs_and_refuses_to_overwrite(tmp_path):
    args = Namespace(
        output=str(tmp_path / "benchmark"),
        seeds="11,23",
        rounds=1,
        train_trials=1,
        eval_trials=2,
        steps=16,
        history=4,
        cpu_threads=1,
        no_llm=True,
        model=None,
        revision=None,
        device="cpu",
        dtype="float32",
        max_tokens=1024,
        learning_rate=0.0001,
    )
    result = run(args)
    assert result["training_steps_per_method_per_seed"] == 16
    assert set(result["results"]["by_family"]) == {"all", "abrupt", "drift"}
    assert set(result["results"]["by_family"]["all"]) == {
        "random",
        "rolling",
        "numeric_untrained",
        "numeric_trained",
    }
    first = json.loads((tmp_path / "benchmark" / "result-11.json").read_text())
    assert len(first["learning"]["numeric"]) == 1
    assert len(first["evaluation"]["numeric_trained"]) == 2
    with pytest.raises(FileExistsError):
        run(args)
