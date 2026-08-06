"""Static provenance diagnostics for Python policy sources.

The audit deliberately reports evidence rather than declaring a policy
"learned" or "intelligent".  A large table indexed by an observation-derived
turn counter can be a legitimate schedule, but it is materially different
from a policy whose output is computed from the current state.  Experiment
runners can therefore reject or explicitly approve that source before using
it as a teacher.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence


PYTHON_POLICY_SOURCE_AUDIT_SCHEMA = (
    "jormungandr.python_policy_source_audit.v1"
)
POLICY_COUNTERFACTUAL_RESPONSE_SCHEMA = (
    "jormungandr.policy_counterfactual_response.v1"
)

_TIME_KEYS = frozenset({"step", "timestep", "turn", "time_index"})
_OUTPUT_NAME_TOKENS = ("action", "decision", "command", "output")


@dataclass(frozen=True)
class PolicyCounterfactualOutcome:
    """One paired decision under a baseline and a controlled intervention."""

    probe_id: str
    intervention_group: str
    baseline_input_fingerprint: str
    counterfactual_input_fingerprint: str
    baseline_decision_fingerprint: str
    counterfactual_decision_fingerprint: str

    def __post_init__(self) -> None:
        values = (
            self.probe_id,
            self.intervention_group,
            self.baseline_input_fingerprint,
            self.counterfactual_input_fingerprint,
            self.baseline_decision_fingerprint,
            self.counterfactual_decision_fingerprint,
        )
        if any(not str(value).strip() for value in values):
            raise ValueError("counterfactual probe fields must be non-empty")
        if self.baseline_input_fingerprint == self.counterfactual_input_fingerprint:
            raise ValueError("counterfactual probe must change the policy input")

    @property
    def responded(self) -> bool:
        return self.baseline_decision_fingerprint != self.counterfactual_decision_fingerprint


def summarize_policy_counterfactual_responses(
    outcomes: Sequence[PolicyCounterfactualOutcome],
) -> Mapping[str, Any]:
    """Summarize controlled state-response evidence without owning semantics."""

    items = tuple(outcomes)
    if not items:
        raise ValueError("at least one counterfactual outcome is required")
    if len({item.probe_id for item in items}) != len(items):
        raise ValueError("counterfactual probe ids must be unique")
    groups: dict[str, Counter[str]] = {}
    for item in items:
        groups.setdefault(item.intervention_group, Counter())[
            "responded" if item.responded else "unchanged"
        ] += 1
    responded = sum(item.responded for item in items)
    return {
        "schema": POLICY_COUNTERFACTUAL_RESPONSE_SCHEMA,
        "probes": len(items),
        "responsive_probes": responded,
        "unchanged_probes": len(items) - responded,
        "response_rate": responded / len(items),
        "classification": (
            "counterfactual_state_response_observed"
            if responded
            else "no_counterfactual_state_response_observed"
        ),
        "by_intervention_group": {
            group: {
                "probes": sum(counts.values()),
                "responsive_probes": counts["responded"],
                "unchanged_probes": counts["unchanged"],
            }
            for group, counts in sorted(groups.items())
        },
    }


def _assigned_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(
            name
            for element in node.elts
            for name in _assigned_names(element)
        )
    if isinstance(node, ast.Name):
        return (node.id,)
    return ()


def _literal_entries(node: ast.AST) -> int | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts)
    if isinstance(node, ast.Dict):
        return len(node.keys)
    return None


def _root_assigned_name(node: ast.AST) -> str:
    current = node
    while isinstance(current, (ast.Subscript, ast.Attribute)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else ""


def _contains_time_observation_access(node: ast.AST) -> bool:
    """Return whether an expression reads a conventional environment clock."""

    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr == "get" and child.args:
                key = child.args[0]
                if isinstance(key, ast.Constant) and key.value in _TIME_KEYS:
                    return True
        if isinstance(child, ast.Subscript):
            key = child.slice
            if isinstance(key, ast.Constant) and key.value in _TIME_KEYS:
                return True
    return False


def _expression_names(node: ast.AST) -> set[str]:
    return {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    }


def _enclosing_function(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST]
) -> str:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parents.get(current)
    return "<module>"


def _enclosing_scope(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST], module: ast.Module
) -> ast.AST:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return current
        current = parents.get(current)
    return module


def _scope_nodes(scope: ast.AST):
    """Yield a scope without leaking same-named locals across functions."""

    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if node is not scope:
                continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _scope_time_names(scope: ast.AST) -> set[str]:
    time_names = set(_TIME_KEYS)
    assignments: list[tuple[tuple[str, ...], ast.AST]] = []
    for node in _scope_nodes(scope):
        if isinstance(node, ast.Assign):
            names = tuple(
                name for target in node.targets for name in _assigned_names(target)
            )
            assignments.append((names, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignments.append((_assigned_names(node.target), node.value))
    changed = True
    while changed:
        changed = False
        for names, value in assignments:
            if not names or set(names).issubset(time_names):
                continue
            if _contains_time_observation_access(value) or (
                _expression_names(value).intersection(time_names)
            ):
                before = len(time_names)
                time_names.update(names)
                changed = changed or len(time_names) != before
    return time_names


def _enclosing_statement(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST]
) -> ast.stmt | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.stmt):
            return current
        current = parents.get(current)
    return None


def audit_python_policy_source(
    source: str | bytes,
    *,
    filename: str = "<policy>",
    large_literal_threshold: int = 64,
    opaque_payload_threshold_bytes: int = 4096,
) -> Mapping[str, Any]:
    """Find large literal tables whose index is derived from environment time.

    This is a static, non-executing audit.  It does not decide whether a
    schedule is useful or legal; it makes the source mechanism explicit so a
    teacher-selection gate cannot silently confuse trace replay with
    state-responsive behavior.
    """

    if large_literal_threshold <= 0:
        raise ValueError("large literal threshold must be positive")
    if opaque_payload_threshold_bytes <= 0:
        raise ValueError("opaque payload threshold must be positive")
    if isinstance(source, bytes):
        payload = source
        text = source.decode("utf-8")
    else:
        text = str(source)
        payload = text.encode("utf-8")
    tree = ast.parse(text, filename=filename)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    large_tables: dict[str, dict[str, Any]] = {}
    opaque_payloads: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = tuple(
                name for target in node.targets for name in _assigned_names(target)
            )
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = _assigned_names(node.target)
            value = node.value
        else:
            continue
        if value is None:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, (str, bytes)):
            encoded = (
                value.value.encode("utf-8")
                if isinstance(value.value, str)
                else value.value
            )
            if len(encoded) >= opaque_payload_threshold_bytes:
                for name in targets:
                    opaque_payloads.append(
                        {
                            "name": name,
                            "line": int(node.lineno),
                            "encoded_bytes": len(encoded),
                            "literal_type": type(value.value).__name__,
                        }
                    )
        entries = _literal_entries(value)
        if entries is None or entries < large_literal_threshold:
            continue
        for name in targets:
            large_tables[name] = {
                "name": name,
                "line": int(node.lineno),
                "entries": int(entries),
                "literal_type": type(value).__name__.removeprefix("AST"),
            }

    scopes = [tree, *[
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
    ]]
    time_names_by_scope = {scope: _scope_time_names(scope) for scope in scopes}

    lookups: list[dict[str, Any]] = []
    all_time_lookups: list[dict[str, Any]] = []
    direct_output_lookups = 0
    output_assignment_lookups = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        scope = _enclosing_scope(node, parents, tree)
        time_names = time_names_by_scope[scope]
        index_names = _expression_names(node.slice)
        if not (
            index_names.intersection(time_names)
            or _contains_time_observation_access(node.slice)
        ):
            continue
        table_name = ast.unparse(node.value)
        table = large_tables.get(node.value.id) if isinstance(node.value, ast.Name) else None
        statement = _enclosing_statement(node, parents)
        use = "expression"
        target = ""
        output_assignment = False
        if isinstance(statement, ast.Return):
            use = "return"
            direct_output_lookups += 1
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            use = "assignment"
            assigned = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else (statement.target,)
            )
            target = ", ".join(ast.unparse(value) for value in assigned)
            assigned_roots = [_root_assigned_name(value) for value in assigned]
            output_assignment = any(
                any(token in root.lower() for token in _OUTPUT_NAME_TOKENS)
                for root in assigned_roots
            )
            output_assignment_lookups += int(output_assignment)
        record = {
            "table": table_name,
            "table_entries": table["entries"] if table is not None else None,
            "line": int(node.lineno),
            "function": _enclosing_function(node, parents),
            "index_expression": ast.unparse(node.slice),
            "use": use,
            "assignment_target": target,
            "output_assignment": output_assignment,
        }
        all_time_lookups.append(record)
        if table is not None:
            lookups.append(record)

    if direct_output_lookups:
        classification = "direct_time_indexed_output_table"
    elif output_assignment_lookups:
        classification = "time_indexed_output_lookup"
    elif lookups:
        classification = "time_indexed_large_table"
    elif opaque_payloads:
        classification = "opaque_embedded_payload"
    elif large_tables:
        classification = "large_literal_table_without_time_indexed_use"
    else:
        classification = "no_large_literal_time_table_detected"
    docstring = ast.get_docstring(tree, clean=True) or ""
    return {
        "schema": PYTHON_POLICY_SOURCE_AUDIT_SCHEMA,
        "filename": filename,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload),
        "large_literal_threshold": int(large_literal_threshold),
        "opaque_payload_threshold_bytes": int(opaque_payload_threshold_bytes),
        "module_docstring": docstring,
        "classification": classification,
        "large_literal_tables": [
            dict(value)
            for _, value in sorted(
                large_tables.items(), key=lambda item: (item[1]["line"], item[0])
            )
        ],
        "opaque_literal_payloads": sorted(
            opaque_payloads, key=lambda item: (item["line"], item["name"])
        ),
        "time_derived_names": {
            (
                scope.name
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
                else "<lambda>"
                if isinstance(scope, ast.Lambda)
                else "<module>"
            ): sorted(names)
            for scope, names in sorted(
                time_names_by_scope.items(),
                key=lambda item: getattr(item[0], "lineno", 0),
            )
        },
        "time_indexed_lookups": sorted(
            all_time_lookups, key=lambda item: (item["line"], item["table"])
        ),
        "time_indexed_large_table_lookups": sorted(
            lookups, key=lambda item: (item["line"], item["table"])
        ),
        "direct_output_lookups": int(direct_output_lookups),
        "output_assignment_lookups": int(output_assignment_lookups),
        "requires_human_review": bool(all_time_lookups or opaque_payloads),
        "interpretation": (
            "A time-indexed table is structural evidence of an open-loop "
            "schedule or trace component, not by itself proof that the entire "
            "policy ignores state."
            if all_time_lookups
            else "The source contains a large opaque literal payload. It may "
            "contain model parameters, schedules, or other data and requires "
            "provenance review before teacher use."
            if opaque_payloads
            else "No large literal output table indexed by conventional "
            "environment time was found by this static audit."
        ),
    }
