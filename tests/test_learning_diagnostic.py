"""Causal labels and independent controls for the learning diagnostic example."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.adaptation_benchmark import make_scenario, observation
from examples.learning_diagnostic import (
    IndependentQ,
    QView,
    parser,
    run,
    teacher_data,
)
from jormungandr.llm_context import AgentContext
from jormungandr.structured_replay import StructuredReplayTransition


def test_teacher_labels_can_be_reconstructed_from_visible_summaries():
    data = teacher_data([make_scenario(41, 1, 0, 32)], 41, 4, "train")
    for obs, target in data:
        features = obs.entity_features
        untried = np.flatnonzero(features[:, 1] == 0)
        if len(untried):
            expected = np.zeros(2)
            expected[untried] = 1 / len(untried)
        else:
            expected = np.full(2, 0.05)
            best = np.flatnonzero(np.isclose(features[:, 0], features[:, 0].max()))
            expected[best] += 0.9 / len(best)
        np.testing.assert_allclose(target, expected)
        assert obs.metadata["llm"]["history"] == []


def test_independent_reward_learner_uses_the_cue_and_epsilon_distribution():
    torch.manual_seed(7)
    agent = IndependentQ()
    base = observation(make_scenario(41, 1, 0, 16), AgentContext("a", "t", "train"))
    samples = []
    for cue in (0, 1):
        features = np.zeros((2, 3))
        features[cue, 0] = 1
        obs = replace(
            base, entity_features=features, candidate_features=np.zeros((2, 3))
        )
        for action in (0, 1):
            samples.append(
                StructuredReplayTransition(
                    obs,
                    obs.candidate_ids[action],
                    float(action == cue),
                    obs,
                    True,
                    "a",
                    f"{cue}-{action}",
                    0,
                    0,
                )
            )
    for _ in range(80):
        agent.update(samples)
    scores = QView(agent).score_results_structured(
        [samples[0].observation, samples[2].observation]
    )
    for cue, score in enumerate(scores):
        assert np.argmax(score.candidate_logits) == cue
        assert np.exp(score.candidate_logits[cue]) == pytest.approx(0.95)


def test_cpu_diagnostic_runs_without_optional_language_dependencies(tmp_path):
    args = parser().parse_args(
        [
            "--output",
            str(tmp_path / "learning"),
            "--skip-llm",
            "--seeds",
            "41",
            "--rounds",
            "1",
            "--train-trials",
            "1",
            "--eval-trials",
            "2",
            "--steps",
            "16",
        ]
    )
    results = run(args)
    assert {r["method"] for r in results} == {
        "independent_q",
        "dqn",
        "numeric_imitation",
    }
    for r in results:
        assert r["final_digest"] != r["initial_digest"]
        assert r["updates"] == 16
        assert len(r["evaluation"]["heldout"]) == 2
    with pytest.raises(FileExistsError):
        run(args)


def test_mirrored_training_examples_preserve_the_causal_rule():
    from examples.hybrid_adaptation import mirror_examples

    original = teacher_data([make_scenario(41, 1, 0, 16)], 41, 4, "train")
    augmented = mirror_examples(original)
    assert len(augmented) == 2 * len(original)
    for (obs, target), (mirrored, flipped) in zip(original, augmented[len(original) :]):
        np.testing.assert_array_equal(
            mirrored.entity_features, obs.entity_features[::-1]
        )
        np.testing.assert_array_equal(
            mirrored.candidate_features[:, 1:], obs.candidate_features[::-1, 1:]
        )
        np.testing.assert_array_equal(mirrored.candidate_features[:, 0], [0, 1])
        np.testing.assert_array_equal(flipped, target[::-1])
        assert mirrored.metadata["llm"]["history"] == []
        assert obs.candidate_ids == mirrored.candidate_ids


def test_ordered_features_preserve_sequence_and_empty_history_shape():
    from examples.history_benchmark import ordered_observation
    from jormungandr.llm_context import Experience

    scenario = make_scenario(41, 1, 0, 16)
    contexts = []
    for rewards in ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]):
        context = AgentContext("a", "trial", "train", max_events=4)
        for i, reward in enumerate(rewards):
            context = context.append(
                Experience(
                    str(i),
                    0,
                    "request",
                    ("route-a", "route-b")[i % 2],
                    "outcome",
                    reward,
                    0,
                )
            )
        contexts.append(context)
    first, second = [
        ordered_observation(scenario, c, representation="summary") for c in contexts
    ]
    np.testing.assert_array_equal(
        first.entity_features[:, :3], second.entity_features[:, :3]
    )
    assert not np.array_equal(
        first.entity_features[:, 3:], second.entity_features[:, 3:]
    )
    assert first.entity_features[1, 3] == 1.0  # newest outcome is route-b success
    assert first.entity_features[1, 4] == 1.0  # observed, not padding
    assert first.entity_features[0, 4] == 0.0
    empty = ordered_observation(
        scenario,
        AgentContext("a", "trial", "train", max_events=0),
        representation="summary",
    )
    assert empty.entity_features.shape == (2, 11)
    assert not np.any(empty.entity_features)


def test_history_comparison_runs_both_representations(tmp_path):
    from examples.history_benchmark import parser, run

    args = parser().parse_args(
        [
            "--output",
            str(tmp_path / "history"),
            "--seeds",
            "41",
            "--rounds",
            "1",
            "--train-trials",
            "1",
            "--eval-trials",
            "2",
            "--steps",
            "16",
        ]
    )
    records = run(args)
    assert {r["mode"] for r in records} == {"summary", "ordered"}
    assert "order_removed" in records[1]["evaluation"]
    assert len(records[0]["evaluation"]["greedy"]) == 2


def test_greedy_view_selects_maximum_and_keeps_exact_ties():
    from types import SimpleNamespace
    from examples.evaluate_hybrid_greedy import GreedyView

    agent = SimpleNamespace(
        policy=torch.nn.Identity(),
        update_steps=0,
        score_results_structured=lambda observations: [
            SimpleNamespace(candidate_logits=[1.0, 3.0, 3.0], value=0.0)
        ],
    )
    score = GreedyView(agent).score_results_structured([None])[0]
    probabilities = torch.softmax(torch.tensor(score.candidate_logits), -1).numpy()
    np.testing.assert_array_equal(probabilities, [0.0, 0.5, 0.5])


def test_coverage_comparison_pairs_initialization_sampling_and_budgets(tmp_path):
    from examples.coverage_benchmark import parser, run

    args = parser().parse_args(
        [
            "--output",
            str(tmp_path / "coverage"),
            "--seeds",
            "41",
            "--rounds",
            "2",
            "--train-trials",
            "1",
            "--eval-trials",
            "2",
            "--steps",
            "16",
        ]
    )
    repeated, fresh = run(args)
    assert repeated["initial_digest"] == fresh["initial_digest"]
    assert repeated["losses"][:16] == fresh["losses"][:16]
    assert repeated["updates"] == fresh["updates"] == 32
    assert repeated["interactions"] == fresh["interactions"] == 32
    assert len({key for group in repeated["training_world_ids"] for key in group}) == 1
    assert len({key for group in fresh["training_world_ids"] for key in group}) == 2
    assert [t["trial_id"] for t in repeated["evaluation"]["sampled"]] == [
        t["trial_id"] for t in fresh["evaluation"]["sampled"]
    ]
    assert all(
        t["trial_id"].split(":")[1] == "2" for t in fresh["evaluation"]["sampled"]
    )
    with pytest.raises(FileExistsError):
        run(args)
