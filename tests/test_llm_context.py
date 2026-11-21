from jormungandr.llm_context import AgentContext, Experience
import pytest


def test_context_persists_across_episodes_but_snapshots_and_agents_are_isolated():
    a = AgentContext("researcher", "trial-a", "train", max_events=2)
    b = AgentContext("evaluator", "trial-a", "validation")
    snapshot = a.metadata("before", {"wait": "wait"})
    for episode in range(3):
        a = a.append(Experience(str(episode), 0, "state", "wait", "result", 1.0, 0))
    assert [event.episode_id for event in a.events] == ["1", "2"]
    assert not snapshot["llm"]["history"]
    assert not b.events
    assert not AgentContext("researcher", "trial-b", "train").events
    with pytest.raises(ValueError, match="advance"):
        a.append(a.events[-1])


def test_history_zero_limit_and_invalid_feedback():
    a = AgentContext("actor", "trial", "train", max_events=0)
    assert not a.append(Experience("e", 0, "o", "a", "r", 1.0, 0)).events
    with pytest.raises(ValueError, match="finite"):
        Experience("e", 0, "o", "a", "r", float("nan"), 0)
