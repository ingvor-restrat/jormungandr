from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.delayed_credit_benchmark import Task, build, collect
from jormungandr.returns import n_step_transitions


def episode():
    return collect(17, Task(4, 1.0, 0.0), 1)[0]


def test_discounted_returns_preserve_action_identity_and_stop_at_boundary():
    raw = episode()
    rows = n_step_transitions(raw, n=13, gamma=0.5)
    assert rows[0].reward == 0.5**4 * raw[4].reward
    assert rows[0].bootstrap_steps == 13
    assert rows[-1].bootstrap_steps == 1
    assert all(r.done for r in rows)
    for original, aggregated in zip(raw, rows):
        assert original.observation is aggregated.observation
        assert original.candidate_id == aggregated.candidate_id
        assert original.timestep == aggregated.timestep
        assert aggregated.next_observation is raw[-1].next_observation
    one = n_step_transitions(raw, n=1, gamma=0.5)
    for a, b in zip(raw, one):
        assert a.reward == b.reward and a.done == b.done
        assert a.next_observation is b.next_observation


@pytest.mark.parametrize(
    "edit",
    ["actor", "episode", "split", "gap", "reorder", "boundary", "unfinished", "link"],
)
def test_rejects_cross_episode_gaps_reset_observations_and_incomplete_input(edit):
    rows = list(episode())
    if edit in ("actor", "episode", "split", "gap"):
        field, value = {
            "actor": ("actor_id", "other"),
            "episode": ("episode_id", "other"),
            "split": ("split", "validation"),
            "gap": ("timestep", 99),
        }[edit]
        rows[1] = replace(rows[1], **{field: value})
    elif edit == "reorder":
        rows[1], rows[2] = rows[2], rows[1]
    elif edit == "boundary":
        rows[1] = replace(rows[1], done=True)
    elif edit == "unfinished":
        rows[-1] = replace(rows[-1], done=False)
    else:
        rows[1] = replace(rows[1], observation=rows[3].observation)
    with pytest.raises(ValueError):
        n_step_transitions(rows, n=4, gamma=0.9)


def test_truncation_bootstraps_with_actual_horizon_and_rejects_gamma_mismatch():
    raw = episode()[:3]
    raw = (*raw[:-1], replace(raw[-1], truncated=True))
    rows = n_step_transitions(raw, n=4, gamma=0.95)
    assert all(r.truncated and not r.done for r in rows)
    agent = build(1)
    with torch.no_grad():
        for model in (agent.q, agent.q_target):
            for p in model.parameters():
                p.zero_()
            model.net[-1].bias.fill_(2.0)
    _, _, metrics, _ = agent._loss_structured(rows[:1], None)
    assert metrics["target_mean"] == pytest.approx(0.95**3 * 2)
    terminal = replace(rows[0], done=True, truncated=False)
    _, _, metrics, _ = agent._loss_structured([terminal], None)
    assert metrics["target_mean"] == 0
    with pytest.raises(ValueError, match="return_gamma"):
        agent.update_structured([replace(rows[0], return_gamma=0.5)])


def test_return_parameters_and_double_aggregation_are_rejected():
    raw = episode()
    for n in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            n_step_transitions(raw, n=n, gamma=0.9)
    for gamma in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            n_step_transitions(raw, n=2, gamma=gamma)
    with pytest.raises(ValueError):
        replace(raw[0], bootstrap_steps=3)
    with pytest.raises(ValueError):
        n_step_transitions(n_step_transitions(raw, n=1, gamma=0.9), n=2, gamma=0.9)


def test_delay_and_sparse_feedback_do_not_change_decision_inputs():
    dense = collect(13, Task(0, 1.0, 0.8), 8)
    delayed = collect(13, Task(12, 1.0, 0.8), 8)
    sparse = collect(13, Task(12, 0.25, 0.8), 8)
    for a, b, c in zip(dense, delayed, sparse):
        assert a[0].reward == b[12].reward
        assert sum(abs(t.reward) for t in b[:12]) == 0
        np.testing.assert_array_equal(
            b[0].observation.global_features, c[0].observation.global_features
        )
        assert (
            a[0].observation.global_features[2]
            == b[0].observation.global_features[2]
            == 0
        )
        assert len(a) == len(b) == len(c) == 13


def test_language_history_boundary_advances_by_return_horizon():
    from examples.multiservice_adaptation import collect as collect_language
    from jormungandr.algorithms.llm_dqn import LLMDQNAgent

    raw = collect_language(31, 1)
    row = replace(
        raw[2],
        next_observation=raw[5].next_observation,
        bootstrap_steps=4,
        return_gamma=0.95,
    )
    LLMDQNAgent._validate_transition(row)
    with pytest.raises(ValueError, match="future feedback"):
        LLMDQNAgent._validate_transition(replace(row, bootstrap_steps=3))
    with pytest.raises(ValueError, match="future feedback"):
        LLMDQNAgent._validate_transition(replace(row, observation=row.next_observation))


def test_native_dqn_learns_clean_delayed_choice_from_full_episode_returns():
    from examples.delayed_credit_benchmark import evaluation, evaluate

    torch.set_num_threads(2)
    task = Task(12, 1.0, 0.0)
    rows = [
        t
        for e in collect(10103, task, 64)
        for t in n_step_transitions(e, n=13, gamma=0.95)
    ]
    agent = build(10103)
    rng = np.random.default_rng(42)
    for _ in range(256):
        agent.update_structured([rows[i] for i in rng.integers(len(rows), size=64)])
    result = evaluate(agent, evaluation(100010103, task, 128), task)
    assert result["accuracy"] == 1.0
    assert result["expected_discounted_return"] == pytest.approx(0.95**12)
