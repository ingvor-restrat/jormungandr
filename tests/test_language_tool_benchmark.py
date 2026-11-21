"""Information boundaries and semantics for the generic language/tool example."""

from dataclasses import replace

import numpy as np

from examples.language_tool_benchmark import (
    IDS,
    TRAIN_REQUESTS,
    TEST_REQUESTS,
    TRAIN_DESCRIPTIONS,
    TEST_DESCRIPTIONS,
    append_feedback,
    epsilon_probabilities,
    language_data,
    make_observation,
    make_world,
)
from jormungandr.llm_context import AgentContext


def test_request_changes_text_but_not_numeric_inputs_or_callable_mask():
    world = make_world(41, 2, 0, 16)
    context = AgentContext("actor", world.identifier, "validation", max_events=8)
    a = make_observation(world, 0, context, kind=0)
    b = make_observation(world, 0, context, kind=1)
    for key in (
        "global_features",
        "entity_features",
        "entity_type_ids",
        "candidate_features",
        "candidate_entity_indices",
        "legal_action_mask",
    ):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    assert a.legal_action_mask.all()
    assert a.candidate_ids == b.candidate_ids
    assert a.metadata["llm"]["observation"] != b.metadata["llm"]["observation"]


def test_language_stage_never_receives_reward_history_or_hidden_quality():
    world = make_world(41, 2, 0, 16)
    context = AgentContext("actor", world.identifier, "validation", max_events=8)
    before = make_observation(world, 1, context, language=True)
    context = append_feedback(context, world, 0, 0)
    after = make_observation(world, 1, context, language=True)
    np.testing.assert_array_equal(before.entity_features, after.entity_features)
    assert not np.any(after.entity_features) and not np.any(after.candidate_features)
    assert before.metadata == after.metadata
    altered = replace(
        world, probabilities=1 - world.probabilities, successes=1 - world.successes
    )
    assert (
        make_observation(altered, 1, context, language=True).metadata == after.metadata
    )
    assert after.metadata["llm"]["history"] == []


def test_balanced_semantic_targets_and_heldout_wording():
    pairs = language_data(41, 8, 16)
    for i in range(0, len(pairs), 2):
        a, p = pairs[i]
        b, q = pairs[i + 1]
        assert a.candidate_ids == b.candidate_ids
        np.testing.assert_array_equal(p + q, np.full(4, 0.5))
        assert p.sum() == q.sum() == 1
        assert a.legal_action_mask.all() and b.legal_action_mask.all()
    assert not {s for group in TRAIN_REQUESTS for s in group} & {
        s for group in TEST_REQUESTS for s in group
    }
    assert not {s for group in TRAIN_DESCRIPTIONS for s in group} & {
        s for group in TEST_DESCRIPTIONS for s in group
    }


def test_worlds_pair_potential_outcomes_and_align_permuted_tools():
    a, b = make_world(41, 2, 1, 16), make_world(41, 2, 1, 16)
    np.testing.assert_array_equal(a.successes, b.successes)
    assert a.family == "drift"
    assert not np.array_equal(a.successes, make_world(41, 1, 1, 16).successes)
    assert not a.successes.flags.writeable
    context = AgentContext("a", a.identifier, "validation", max_events=8)
    obs = make_observation(a, 0, context, oracle_mask=True)
    for row, key in enumerate(obs.candidate_ids):
        index = IDS.index(key)
        assert obs.legal_action_mask[row] == (a.capabilities[index] == a.requests[0])
        assert (
            obs.metadata["llm"]["candidate_descriptions"][key] == a.descriptions[index]
        )
    probabilities = epsilon_probabilities(
        [0.0, 100.0, 1.0, 100.0], np.array([True, False, True, False])
    )
    np.testing.assert_allclose(probabilities, [0.05, 0.0, 0.95, 0.0])
    wrong = int(np.flatnonzero(a.capabilities != a.requests[0])[0])
    assert a.reward(0, wrong) == -a.costs[wrong] - 0.1 * a.latencies[wrong]
