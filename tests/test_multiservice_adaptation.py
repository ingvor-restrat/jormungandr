"""Causal multi-arm task contracts and real HTTP execution, no downloads."""

from dataclasses import replace
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from examples.adaptation_tool_sandbox import SortCall, invoke, tool_server, verify
from examples.multiservice_adaptation import (
    TRAIN,
    TEST,
    Task,
    append,
    collect,
    context,
    evaluate,
    observe,
    world,
)
from jormungandr.algorithms.llm_dqn import LLMDQNAgent


def test_task_holdout_and_private_state_do_not_enter_observation():
    assert set(TRAIN).isdisjoint(TEST.values())
    assert {t.services for t in TRAIN} == {3, 5}
    assert {t.services for t in TEST.values()} == {4, 8}
    w = world(31, Task(8, 0.85, 0.15, 16), 0)
    c = append(context(w), 0, w.ids[2], 0)
    obs = observe(w, 1, c)
    hidden = replace(
        w, best=1 - w.best, outcomes=~w.outcomes, probabilities=1 - w.probabilities
    )
    assert observe(hidden, 1, c).metadata == obs.metadata
    descriptions = obs.metadata["llm"]["candidate_descriptions"]
    assert "failures: 1" in descriptions[w.ids[2]]
    assert all("failures: 0" in descriptions[key] for key in w.ids if key != w.ids[2])
    assert not np.any(obs.candidate_features)
    assert not np.any(obs.global_features)


def test_repeated_changes_and_renaming_preserve_latent_world():
    w = world(39, Task(8, 0.85, 0.15, 16), 1)
    renamed = world(39, w.task, 1, renamed=True)
    assert w.ids != renamed.ids
    np.testing.assert_array_equal(w.outcomes, renamed.outcomes)
    np.testing.assert_array_equal(w.orders, renamed.orders)
    assert np.flatnonzero(w.best[1:] != w.best[:-1]).tolist() == [15, 31, 47, 63, 79]


def test_collected_replay_is_causal_and_training_only():
    replay = collect(31, 1)
    for step in replay:
        LLMDQNAgent._validate_transition(step)
        assert all(
            e["timestep"] < step.timestep
            for e in step.observation.metadata["llm"]["history"]
        )
        event = step.next_observation.metadata["llm"]["history"][-1]
        assert event["reward"] == step.reward
        assert event["action"] == step.candidate_id
    assert sum(s.done for s in replay) == 1


def test_ablations_remove_or_flip_all_evidence_without_mutation():
    w = world(31, Task(4, 1.0, 0.0, 0), 0)
    c = append(context(w), 0, w.ids[1], 1.0)
    no = observe(w, 1, c, "no_history").metadata["llm"]
    assert no["history"] == []
    assert all(
        "successes: 0; failures: 0" in d for d in no["candidate_descriptions"].values()
    )
    flipped = observe(w, 1, c, "flipped_feedback").metadata["llm"]
    assert flipped["history"][0]["reward"] == 0
    assert "successes: 0; failures: 1" in flipped["candidate_descriptions"][w.ids[1]]
    assert c.events[0].reward == 1


def test_statistical_control_beats_random_in_stable_eight_service_task():
    worlds = [world(31, Task(8, 1.0, 0.0, 0), i) for i in range(24)]
    baseline = evaluate(None, worlds, "window_mean")
    random = evaluate(None, worlds, "random")
    assert baseline["mean_reward"] > 0.7
    assert baseline["pseudo_regret"] < random["pseudo_regret"]
    assert baseline["change_count"] == 0
    assert baseline["mean_first_best_selection_delay"] is None


def test_tool_http_contract_and_independent_verifier():
    call = SortCall("a", (3, 1, 3, -2))
    assert verify(call, {"values": [-2, 1, 3, 3]})
    assert not verify(call, {"values": [-2, 1, 3, 4]})
    assert not verify(call, {"values": [3, -2, 1, 3]})
    assert not verify(call, {"values": [True, 1, 3, 3]})
    with tool_server(["a", "b"], lambda key: key == "a") as url:
        reward, response = invoke(url, call)
        assert reward == 1 and response["values"] == [-2, 1, 3, 3]
        assert invoke(url, replace(call, service="b"))[0] == 0
        for raw in (
            {"service": "missing", "values": [1]},
            {"service": "a", "values": [True]},
            {"service": "a", "values": [], "oracle": True},
        ):
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(url, data=json.dumps(raw).encode()), timeout=5)
            assert exc.value.code == 400


def test_http_evaluation_matches_simulation_and_records_actual_calls(tmp_path):
    worlds = [world(31, Task(4, 0.85, 0.15, 8, 24), i) for i in range(4)]
    direct = evaluate(None, worlds, "window_mean")
    via = evaluate(None, worlds, "window_mean", tmp_path / "trace.jsonl", http=True)
    assert via == direct
    rows = [
        json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 96
    assert all("call" in r and "response" in r for r in rows)


def test_last_only_ablation_drops_older_evidence_from_text_and_counts():
    w = world(31, Task(4, 0.85, 0.15, 16), 0)
    c = append(context(w), 0, w.ids[0], 1.0)
    c = append(c, 1, w.ids[1], 0.0)
    obs = observe(w, 2, c, "last_only").metadata["llm"]
    assert len(obs["history"]) == 1
    assert obs["history"][0]["timestep"] == 1
    assert "successes: 0; failures: 0" in obs["candidate_descriptions"][w.ids[0]]
    assert "successes: 0; failures: 1" in obs["candidate_descriptions"][w.ids[1]]
    assert len(c.events) == 2


def test_confirmation_configs_reserved_from_training_and_development():
    from examples.multiservice_adaptation import (
        CONFIRM,
        CONFIRM_REQUEST,
        HELD_REQUEST,
        REQUEST,
    )

    assert set(CONFIRM.values()).isdisjoint(TRAIN)
    assert set(CONFIRM.values()).isdisjoint(TEST.values())
    assert CONFIRM_REQUEST not in {HELD_REQUEST, REQUEST}
    assert {t.period for t in CONFIRM.values()} == {0, 20, 28}


def test_replay_coverage_buckets_depend_on_history_not_reward():
    from examples.multiservice_adaptation import replay_buckets

    replay = collect(31, 2)
    buckets = replay_buckets(replay)
    flipped = replay_buckets([replace(t, reward=1 - t.reward) for t in replay])
    assert [len(b) for b in buckets] == [len(b) for b in flipped]
    assert sum(len(b) for b in buckets) == len(replay)
    for index, bucket in enumerate(buckets):
        for t in bucket:
            count = sum(
                e["action"] == t.candidate_id
                for e in t.observation.metadata["llm"]["history"]
            )
            assert min(count, 2) == index


def test_equal_score_ties_are_independent_of_candidate_row_order():
    from examples.multiservice_adaptation import select

    w = world(31, Task(8, 1.0, 0.0, 0), 0)
    for t in range(w.task.steps):
        assert select(w, t, [0.5] * 8, w.ids) == select(w, t, [0.5] * 8, w.ids[::-1])


def test_training_name_prefixes_do_not_encode_change_schedule():
    prefixes = {0: set(), 12: set(), 24: set()}
    for i in range(72):
        task = TRAIN[i % len(TRAIN)]
        w = world(31, task, i, "train")
        prefixes[task.period].add(w.ids[0].split("-")[0])
    assert all(values == {"svc", "worker", "node"} for values in prefixes.values())
