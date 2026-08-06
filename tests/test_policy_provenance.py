from jormungandr import audit_python_policy_source as public_audit
from jormungandr.policy_provenance import (
    PolicyCounterfactualOutcome,
    audit_python_policy_source,
    summarize_policy_counterfactual_responses,
)


def test_source_audit_finds_direct_time_indexed_output_table() -> None:
    rows = ",\n".join("{'action': %d}" % index for index in range(64))
    source = f"""\
import copy
TRACE_ACTIONS = [{rows}]
def agent(obs):
    step = min(int(obs.get('step', 0)), len(TRACE_ACTIONS) - 1)
    return copy.deepcopy(TRACE_ACTIONS[step])
"""
    report = audit_python_policy_source(source, filename="teacher.py")

    assert public_audit is audit_python_policy_source
    assert report["classification"] == "direct_time_indexed_output_table"
    assert report["large_literal_tables"] == [
        {
            "name": "TRACE_ACTIONS",
            "line": 2,
            "entries": 64,
            "literal_type": "List",
        }
    ]
    assert report["direct_output_lookups"] == 1
    assert report["time_indexed_large_table_lookups"][0]["function"] == "agent"
    assert report["requires_human_review"] is True


def test_source_audit_does_not_flag_state_computation_or_small_lookup() -> None:
    source = """\
MOVES = ['LEFT', 'RIGHT']
def agent(obs):
    return MOVES[int(obs.get('cash', 0) > 10)]
"""
    report = audit_python_policy_source(source)

    assert report["classification"] == "no_large_literal_time_table_detected"
    assert report["time_indexed_large_table_lookups"] == []


def test_source_audit_reports_time_table_assignment_without_overclaiming() -> None:
    rows = ",".join(str(index) for index in range(64))
    source = f"""\
SCHEDULE = [{rows}]
def agent(obs):
    turn = obs['turn']
    action = {{}}
    action['quantity'] = SCHEDULE[turn]
    return action
"""
    report = audit_python_policy_source(source)

    assert report["classification"] == "time_indexed_output_lookup"
    assert report["direct_output_lookups"] == 0
    assert report["time_indexed_large_table_lookups"][0]["assignment_target"] == "action['quantity']"
    assert "not by itself proof" in report["interpretation"]


def test_source_audit_requires_review_for_opaque_embedded_payload() -> None:
    source = "PAYLOAD = " + repr("A" * 4096) + "\n"
    report = audit_python_policy_source(source)

    assert report["classification"] == "opaque_embedded_payload"
    assert report["opaque_literal_payloads"][0]["encoded_bytes"] == 4096
    assert report["requires_human_review"] is True


def test_counterfactual_summary_separates_response_from_unchanged_action() -> None:
    outcomes = (
        PolicyCounterfactualOutcome("cash", "resource", "s0", "s1", "a0", "a1"),
        PolicyCounterfactualOutcome("noise", "irrelevant", "s2", "s3", "a2", "a2"),
    )

    report = summarize_policy_counterfactual_responses(outcomes)

    assert report["responsive_probes"] == 1
    assert report["unchanged_probes"] == 1
    assert report["response_rate"] == 0.5
    assert report["classification"] == "counterfactual_state_response_observed"
    assert report["by_intervention_group"]["resource"]["responsive_probes"] == 1
