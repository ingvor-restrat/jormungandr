"""Information sufficiency, causality, and trial-boundary contracts."""

from dataclasses import replace

import numpy as np
import pytest

from examples.context_adaptation_benchmark import (
    collect,
    context_for,
    feedback,
    make_trial,
    memory_baseline,
    observation,
)
from jormungandr.algorithms.llm_dqn import LLMDQNAgent


def test_opposite_hidden_worlds_have_same_public_inputs():
    a, b = [make_trial(31, 2, i, 12) for i in (0, 1)]
    assert a.ids == b.ids and a.orders == b.orders
    assert all(x != y for x, y in zip(a.winners, b.winners))
    x = observation(a, 0, context_for(a, 1))
    y = observation(b, 0, context_for(b, 1))
    for key in ("entity_features", "candidate_features", "global_features"):
        np.testing.assert_array_equal(getattr(x, key), getattr(y, key))
        assert not np.any(getattr(x, key))
    for key in ("observation", "history", "candidate_descriptions"):
        assert x.metadata["llm"][key] == y.metadata["llm"][key]


def test_replay_feedback_is_causal_and_only_meta_session_ends_bootstrap():
    replay = collect(31, 4, 12, 1)
    assert len(replay) == 48
    assert sum(t.done for t in replay) == 4
    for t in replay:
        LLMDQNAgent._validate_transition(t)
        assert all(
            e["timestep"] < t.timestep for e in t.observation.metadata["llm"]["history"]
        )
        event = t.next_observation.metadata["llm"]["history"][-1]
        assert event["timestep"] == t.timestep and event["reward"] == t.reward
        assert event["action"] == t.candidate_id
        assert "API executed" in event["outcome"]
    assert replay[0].observation.metadata["llm"]["history"] == []
    assert replay[12].observation.metadata["llm"]["history"] == []


def test_replay_rejects_future_feedback_validation_and_cross_trial():
    t = collect(31, 4, 12, 1)[0]
    with pytest.raises(ValueError, match="future"):
        LLMDQNAgent._validate_transition(replace(t, observation=t.next_observation))
    with pytest.raises(ValueError, match="training"):
        LLMDQNAgent._validate_transition(replace(t, split="validation"))
    meta = dict(t.next_observation.metadata)
    meta["llm"] = {**meta["llm"], "trial_id": "other"}
    with pytest.raises(ValueError, match="trial boundaries"):
        LLMDQNAgent._validate_transition(
            replace(t, next_observation=replace(t.next_observation, metadata=meta))
        )


def test_feedback_ablation_does_not_mutate_stored_history():
    w = make_trial(31, 2, 0, 12)
    c = feedback(context_for(w, 1), 0, w.ids[0], 1.0)
    original = observation(w, 1, c)
    flipped = observation(w, 1, c, ablation="flipped_feedback")
    empty = observation(w, 1, c, ablation="no_history")
    assert c.events[0].reward == 1
    assert original.metadata["llm"]["history"][0]["reward"] == 1
    assert flipped.metadata["llm"]["history"][0]["reward"] == 0
    assert empty.metadata["llm"]["history"] == []


def test_memory_baseline_recovers_after_one_feedback_and_has_no_hidden_input():
    worlds = [make_trial(31, 2, i, 12) for i in range(24)]
    result = memory_baseline(worlds)
    # Balanced initial uncertainty: 12 errors; 12 changing worlds: one error each.
    assert result["mean_reward"] == 1 - 24 / (24 * 12)
    assert result["curve"][0] == 0.5


def test_grouped_history_contains_only_observed_service_results():
    w = replace(make_trial(31, 2, 0, 12), group_history=True)
    c = feedback(context_for(w, 1), 0, w.ids[0], 1.0)
    obs = observation(w, 1, c)
    descriptions = obs.metadata["llm"]["candidate_descriptions"]
    assert "Observed task reward 1.0" in descriptions[w.ids[0]]
    assert "No retained observation" in descriptions[w.ids[1]]
    altered = replace(w, winners=tuple(1 - i for i in w.winners))
    assert observation(altered, 1, c).metadata == obs.metadata
    no_history = observation(w, 1, c, ablation="no_history")
    assert all(
        "No retained observation" in d
        for d in no_history.metadata["llm"]["candidate_descriptions"].values()
    )
    flipped = observation(w, 1, c, ablation="flipped_feedback")
    assert (
        "Observed task reward 0.0"
        in flipped.metadata["llm"]["candidate_descriptions"][w.ids[0]]
    )


def test_independent_control_learns_feedback_rule(tmp_path):
    from types import SimpleNamespace
    from examples.context_numeric_control import run

    result = run(SimpleNamespace(output=str(tmp_path / "control"), seed=31))
    assert result["history"]["stable_after_first"] >= 0.95
    assert result["history"]["post_change_after_one"] >= 0.95
    assert result["no_history"]["mean_reward"] == 0.5
    assert result["flipped_feedback"]["stable_after_first"] <= 0.05


def test_confirmation_changes_wording_and_ids_without_revealing_winner():
    from examples.context_adaptation_benchmark import REQUEST, RESERVED_REQUESTS

    worlds = [make_trial(31, 3, i, 12, renamed=True) for i in range(24)]
    assert all(w.request != REQUEST for w in worlds)
    assert {w.request for w in worlds} == set(RESERVED_REQUESTS)
    for a, b in zip(worlds[::2], worlds[1::2]):
        assert a.ids == b.ids and a.orders == b.orders and a.request == b.request
        assert all(x != y for x, y in zip(a.winners, b.winners))
    assert len({w.ids for w in worlds}) == 12
    for wording in RESERVED_REQUESTS:
        assert {w.change is None for w in worlds if w.request == wording} == {
            True,
            False,
        }
