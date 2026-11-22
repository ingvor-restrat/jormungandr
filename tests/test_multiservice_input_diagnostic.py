from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from examples.multiservice_input_diagnostic import CanonicalServicePolicy
from examples.multiservice_adaptation import Task, append, context, observe, world


class EvidenceScorer:
    policy = SimpleNamespace(candidate_context="local")
    q = policy
    update_steps = 0

    def action_results_structured(self, observations, deterministic=True):
        return [
            SimpleNamespace(
                candidate_values=(
                    sum(e["reward"] for e in o.metadata["llm"]["history"]),
                )
            )
            for o in observations
        ]


def test_canonical_inputs_preserve_evidence_and_restore_real_service_ids():
    w = world(31, Task(4, 1.0, 0.0, 0), 0, renamed=True)
    c = append(context(w), 0, w.ids[2], 1.0)
    obs = observe(w, 1, c)
    original = deepcopy(obs.metadata)
    policy = CanonicalServicePolicy(EvidenceScorer())
    inputs = policy.inputs(obs)
    assert all(o.candidate_ids == ("svc-1000",) for o in inputs)
    assert sum(len(o.metadata["llm"]["history"]) for o in inputs) == 1
    result = policy.action_results_structured([obs])[0]
    assert result.candidate_id == w.ids[2]
    assert result.candidate_values[obs.candidate_ids.index(w.ids[2])] == 1.0
    assert obs.metadata == original
    rev = observe(w, 1, c, reverse=True)
    assert policy.action_results_structured([rev])[0].candidate_id == w.ids[2]


def test_canonicalization_removes_only_declared_nuisance_fields():
    w = world(31, Task(4, 1.0, 0.0, 0), 0, renamed=True)
    altered = replace(
        w, ids=tuple(f"other-{i}" for i in range(4)), request="Different job wording"
    )
    a = observe(w, 1, append(context(w), 0, w.ids[1], 0.0))
    b = observe(altered, 1, append(context(altered), 0, altered.ids[1], 0.0))
    policy = CanonicalServicePolicy(EvidenceScorer())
    for x, y in zip(policy.inputs(a), policy.inputs(b)):
        assert x.metadata == y.metadata
        np.testing.assert_array_equal(x.candidate_features, y.candidate_features)
        np.testing.assert_array_equal(x.entity_features, y.entity_features)
    raw = CanonicalServicePolicy(EvidenceScorer(), names=False, request=False)
    assert raw.inputs(a)[0].metadata != raw.inputs(b)[0].metadata
