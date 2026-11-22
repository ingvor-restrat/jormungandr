import numpy as np
from dataclasses import replace

from examples.multiservice_learning_control import encode, collect_arrays
from examples.multiservice_evidence import evidence_view
from examples.multiservice_adaptation import Task, world, context, append, observe


def test_features_include_other_feedback_and_relative_order():
    events = [{"action": "a", "reward": 1.0}, {"action": "b", "reward": 0.0}]
    full = encode(events, "a", 4, window=3)
    np.testing.assert_array_equal(
        full[:-1].reshape(3, 4), [[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 0]]
    )
    flipped = [{**e, "reward": 1 - e["reward"]} for e in events]
    assert not np.array_equal(full, encode(flipped, "a", 4, window=3))
    np.testing.assert_array_equal(
        full,
        encode([{**e, "action": e["action"] + "x"} for e in events], "ax", 4, window=3),
    )
    assert full[-1] == 0.25


def test_synthetic_training_is_reproducible_and_nonconstant():
    a, b = collect_arrays(12, trials=2, window=4)
    c, d = collect_arrays(12, trials=2, window=4)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert a.shape == (192, 17)
    assert set(b) == {0.0, 1.0}
    assert np.isfinite(a).all()


def test_evidence_view_preserves_audit_and_other_service_feedback():
    w = world(31, Task(4, 1.0, 0.0, 0), 0)
    c = append(append(context(w), 0, w.ids[0], 1.0), 1, w.ids[1], 0.0)
    obs = observe(w, 2, c)
    view = evidence_view(obs)
    assert view.metadata["llm"]["history"] == obs.metadata["llm"]["history"]
    desc = view.metadata["llm"]["candidate_descriptions"]
    assert "self success, other failure" in desc[w.ids[0]]
    assert "2:1" in desc[w.ids[0]]
    assert "Latest own result: untried" in desc[w.ids[2]]
    assert (
        "none"
        in evidence_view(observe(w, 2, c, "no_history")).metadata["llm"][
            "candidate_descriptions"
        ][w.ids[0]]
    )
    assert (
        desc
        == evidence_view(observe(w, 2, c, reverse=True)).metadata["llm"][
            "candidate_descriptions"
        ]
    )


def test_performance_gate_fails_regression_and_insufficient_worlds():
    from examples.multiservice_acceptance import TASKS, assess

    def row(v, n=32):
        return {"mean_reward": v, "per_trial_reward": [v] * n, "trial_count": n}

    data = {
        name: {
            k: row(0.9)
            for k in ("language", "last_outcome", "window_mean", "window_ucb")
        }
        for name in TASKS
    }
    assert assess(data)["passed"]
    data["eight_changes"]["language"] = row(0.8)
    assert not assess(data)["passed"]
    data["eight_changes"]["language"] = row(0.9)
    data["eight_stable"] = {k: row(0.9, 4) for k in data["eight_stable"]}
    assert not assess(data)["passed"]


def test_td_control_next_states_follow_observed_action_and_reward():
    x, rewards, next_x, masks, terminals = collect_arrays(
        12, trials=2, window=4, transitions=True
    )
    assert x.shape == (192, 17)
    assert next_x.shape == (192, 5, 17)
    np.testing.assert_array_equal(np.flatnonzero(terminals), [95, 191])
    assert set(masks.sum(1)) <= {3, 5}
    # Exactly one service has the just-chosen action as its newest self event.
    np.testing.assert_array_equal(next_x[:, :, 0].sum(1), np.ones(192))
    for i in range(len(x)):
        np.testing.assert_array_equal(
            next_x[i, masks[i], 2], np.full(masks[i].sum(), rewards[i])
        )
    np.testing.assert_array_equal(x[0, :-1], np.zeros(16))


def test_policy_collection_contains_only_causal_training_feedback():
    from types import SimpleNamespace
    from examples.multiservice_evidence_online import collect_policy
    from jormungandr.algorithms.llm_dqn import LLMDQNAgent

    class FixedScores:
        def action_results_structured(self, observations, deterministic=True):
            assert all(o.metadata["llm"]["split"] == "train" for o in observations)
            return [
                SimpleNamespace(candidate_values=[0.0] * len(o.candidate_ids))
                for o in observations
            ]

    rows = collect_policy(FixedScores(), 19, 2)
    assert len(rows) == 192
    for row in rows:
        LLMDQNAgent._validate_transition(row)
        assert row.split == "train"
        latest = row.next_observation.metadata["llm"]["history"][-1]
        assert latest["reward"] == row.reward
        assert latest["action"] == row.candidate_id
        assert latest["timestep"] == row.timestep
        assert all(
            e["timestep"] < row.timestep
            for e in row.observation.metadata["llm"]["history"]
        )


def test_policy_collection_covers_failure_after_long_exploitation_streak(monkeypatch):
    from types import SimpleNamespace
    from examples import multiservice_evidence_online as online

    task = Task(3, 1.0, 0.0, 0, 20)
    w = world(19, task, 0, "train")
    outcomes = np.zeros((task.steps, task.services), dtype=bool)
    outcomes[:15, 0] = True
    # Fixed actor, controlled environment: no hidden state enters the actor.
    w = replace(w, outcomes=outcomes, exploration=np.ones(task.steps))
    monkeypatch.setattr(online, "TRAIN", (task,))
    monkeypatch.setattr(online, "world", lambda *args, **kwargs: w)

    class FixedService:
        def action_results_structured(self, observations, deterministic=True):
            return [
                SimpleNamespace(
                    candidate_values=[float(k == w.ids[0]) for k in o.candidate_ids]
                )
                for o in observations
            ]

    rows = online.collect_policy(FixedService(), 19, 1)
    assert [e["reward"] for e in rows[16].observation.metadata["llm"]["history"]] == [
        1.0
    ] * 15 + [0.0]
    assert rows[16].reward == 0.0


def test_counts_evidence_uses_only_retained_feedback_and_respects_ablations():
    w = world(31, Task(4, 1.0, 0.0, 0), 0)
    c = append(
        append(append(context(w), 0, w.ids[0], 1.0), 1, w.ids[1], 0.0), 2, w.ids[0], 1.0
    )

    def description(ablation):
        return evidence_view(
            observe(w, 3, c, ablation), evidence_format="counts_v2"
        ).metadata["llm"]["candidate_descriptions"][w.ids[0]]

    assert (
        "Observed successes: 2; observed failures: 0; total observations: 2."
        in description("history")
    )
    assert (
        "Observed successes: 0; observed failures: 2; total observations: 2."
        in description("flipped_feedback")
    )
    assert (
        "Observed successes: 0; observed failures: 0; total observations: 0."
        in description("no_history")
    )
    assert "total observations: 1." in description("last_only")
