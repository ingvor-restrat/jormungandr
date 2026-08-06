"""Information-theoretic diagnostics for structured supervision corpora."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .structured_supervision import StructuredSupervisionExample


STRUCTURED_SUPERVISION_CEILING_SCHEMA = (
    "jormungandr.structured_supervision_deterministic_ceiling.v1"
)
STRUCTURED_SUPERVISION_TIME_DEPENDENCE_SCHEMA = (
    "jormungandr.structured_supervision_time_dependence.v1"
)
STRUCTURED_SUPERVISION_STRATIFIED_SUBSET_SCHEMA = (
    "jormungandr.structured_supervision_stratified_subset.v1"
)


def structured_supervision_model_input_fingerprint(
    example: StructuredSupervisionExample,
) -> str:
    """Hash exactly the numeric input and conditional choice seen by the model.

    Audit identifiers and metadata are deliberately excluded because the
    generic structured transformer does not consume them. Candidate and
    prefix identifiers are converted to their numeric row indices, matching
    the behavior-cloning loss implementation.
    """

    observation = example.observation
    digest = hashlib.sha256()

    def add_array(label: str, value: Any) -> None:
        array = np.ascontiguousarray(value)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(b"\0")
        digest.update(array.tobytes())
        digest.update(b"\0")

    add_array("global", observation.global_features)
    add_array("entities", observation.entity_features)
    add_array("entity_types", observation.entity_type_ids)
    add_array("candidates", observation.candidate_features)
    add_array("legal", observation.legal_action_mask)
    add_array("candidate_entities", observation.candidate_entity_indices)
    by_id = {
        candidate_id: index
        for index, candidate_id in enumerate(observation.candidate_ids)
    }
    add_array(
        "factor_candidates",
        np.asarray([by_id[value] for value in example.candidate_ids], dtype="<i8"),
    )
    add_array(
        "selected_prefix",
        np.asarray(
            [by_id[value] for value in example.selected_prefix_candidate_ids],
            dtype="<i8",
        ),
    )
    return digest.hexdigest()


def _structured_supervision_selection_identity(
    example: StructuredSupervisionExample,
) -> str:
    """Return a stable identity for deterministic diagnostic selection."""

    target_position = example.candidate_ids.index(example.target_candidate_id)
    fields = (
        example.actor_id,
        example.episode_id,
        str(example.timestep),
        example.factor_id,
        example.split,
        example.source_group,
        example.factor_group,
        example.target_group,
        str(target_position),
        structured_supervision_model_input_fingerprint(example),
    )
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


def structured_supervision_stratified_subset(
    examples: Sequence[StructuredSupervisionExample],
    *,
    per_group: int,
    seed: int,
    group_by: str = "target_group",
) -> tuple[tuple[StructuredSupervisionExample, ...], Mapping[str, Any]]:
    """Select a deterministic hash-ranked diagnostic subset per stratum.

    This is intended for fit and throughput probes, not for silently changing
    a training distribution.  Selection is independent of input order and the
    receipt fingerprints the exact selected records.  Supported strata are
    semantic target, factor, and source groups.
    """

    items = tuple(examples)
    if not items:
        raise ValueError("at least one supervision example is required")
    if int(per_group) <= 0:
        raise ValueError("per_group must be positive")
    if group_by not in {"target_group", "factor_group", "source_group"}:
        raise ValueError(
            "group_by must be target_group, factor_group, or source_group"
        )
    buckets: dict[str, list[tuple[str, str, StructuredSupervisionExample]]] = {}
    seed_bytes = str(int(seed)).encode("ascii")
    for example in items:
        group = str(getattr(example, group_by))
        identity = _structured_supervision_selection_identity(example)
        rank = hashlib.sha256(seed_bytes + b"\0" + identity.encode("ascii")).hexdigest()
        buckets.setdefault(group, []).append((rank, identity, example))

    selected_rows: list[tuple[str, str, str, StructuredSupervisionExample]] = []
    eligible_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for group, rows in sorted(buckets.items()):
        ordered = sorted(rows, key=lambda item: (item[0], item[1]))
        chosen = ordered[: int(per_group)]
        eligible_counts[group] = len(rows)
        selected_counts[group] = len(chosen)
        selected_rows.extend(
            (group, rank, identity, example)
            for rank, identity, example in chosen
        )
    selected_rows.sort(key=lambda item: (item[0], item[1], item[2]))
    identities = tuple(row[2] for row in selected_rows)
    receipt_digest = hashlib.sha256()
    for identity in identities:
        receipt_digest.update(identity.encode("ascii"))
        receipt_digest.update(b"\n")
    selected = tuple(row[3] for row in selected_rows)
    return selected, {
        "schema": STRUCTURED_SUPERVISION_STRATIFIED_SUBSET_SCHEMA,
        "selection": "seeded SHA-256 rank within each semantic group",
        "input_order_independent": True,
        "group_by": group_by,
        "per_group": int(per_group),
        "seed": int(seed),
        "eligible_examples": len(items),
        "selected_examples": len(selected),
        "eligible_group_counts": eligible_counts,
        "selected_group_counts": selected_counts,
        "selected_identity_sha256": receipt_digest.hexdigest(),
    }


@dataclass
class _InputCell:
    target_position_counts: Counter[int] = field(default_factory=Counter)
    target_position_weights: Counter[int] = field(default_factory=Counter)
    target_position_groups: dict[int, Counter[str]] = field(default_factory=dict)

    @property
    def examples(self) -> int:
        return int(sum(self.target_position_counts.values()))

    def add(self, position: int, target_group: str, weight: float) -> None:
        self.target_position_counts[position] += 1
        self.target_position_weights[position] += float(weight)
        self.target_position_groups.setdefault(position, Counter())[target_group] += 1


@dataclass
class _CeilingAccumulator:
    inputs: dict[str, _InputCell] = field(default_factory=dict)
    target_group_counts: Counter[str] = field(default_factory=Counter)
    examples: int = 0
    weight_sum: float = 0.0

    def add(self, example: StructuredSupervisionExample, fingerprint: str) -> None:
        target_position = example.candidate_ids.index(
            example.target_candidate_id
        )
        self.inputs.setdefault(fingerprint, _InputCell()).add(
            target_position,
            example.target_group,
            example.sample_weight,
        )
        self.target_group_counts[example.target_group] += 1
        self.examples += 1
        self.weight_sum += float(example.sample_weight)

    def summary(self) -> Mapping[str, Any]:
        raw_correct = 0
        weighted_correct = 0.0
        macro_correct_mass = 0.0
        repeated_inputs = 0
        conflicting_inputs = 0
        examples_on_conflicting_inputs = 0
        for cell in self.inputs.values():
            raw_correct += max(cell.target_position_counts.values())
            weighted_correct += max(cell.target_position_weights.values())
            if cell.examples > 1:
                repeated_inputs += 1
            if len(cell.target_position_counts) > 1:
                conflicting_inputs += 1
                examples_on_conflicting_inputs += cell.examples
            position_macro_mass = []
            for groups in cell.target_position_groups.values():
                position_macro_mass.append(
                    sum(
                        count / self.target_group_counts[group]
                        for group, count in groups.items()
                    )
                )
            macro_correct_mass += max(position_macro_mass)
        target_groups = len(self.target_group_counts)
        return {
            "examples": self.examples,
            "weight_sum": self.weight_sum,
            "model_inputs": len(self.inputs),
            "repeated_model_inputs": repeated_inputs,
            "conflicting_model_inputs": conflicting_inputs,
            "examples_on_conflicting_inputs": examples_on_conflicting_inputs,
            "raw_correct_ceiling": raw_correct,
            "raw_accuracy_ceiling": raw_correct / self.examples,
            "irreducible_raw_errors": self.examples - raw_correct,
            "weighted_correct_mass_ceiling": weighted_correct,
            "weighted_accuracy_ceiling": weighted_correct / self.weight_sum,
            "target_groups": target_groups,
            "target_macro_accuracy_ceiling": (
                macro_correct_mass / target_groups if target_groups else 0.0
            ),
        }


def structured_supervision_deterministic_ceiling(
    examples: Sequence[StructuredSupervisionExample],
) -> Mapping[str, Any]:
    """Return exact deterministic imitation ceilings for a fixed corpus.

    Examples with identical tensors, factor choice rows, and selected prefix
    are indistinguishable to the structured policy. If they request different
    target positions, no deterministic policy of any size can satisfy them
    all. The report separates this information limit from optimization or
    model-capacity failure and includes raw, sample-weighted, and target-macro
    ceilings.
    """

    items = tuple(examples)
    if not items:
        raise ValueError("at least one supervision example is required")
    overall = _CeilingAccumulator()
    factors: dict[str, _CeilingAccumulator] = {}
    sources: dict[str, _CeilingAccumulator] = {}
    for example in items:
        fingerprint = structured_supervision_model_input_fingerprint(example)
        overall.add(example, fingerprint)
        factors.setdefault(example.factor_group, _CeilingAccumulator()).add(
            example, fingerprint
        )
        sources.setdefault(example.source_group, _CeilingAccumulator()).add(
            example, fingerprint
        )
    return {
        "schema": STRUCTURED_SUPERVISION_CEILING_SCHEMA,
        "model_input": (
            "numeric global/entity/candidate tensors, masks, entity pointers, "
            "factor candidate-row indices, and selected-prefix row indices; "
            "audit identifiers and metadata excluded"
        ),
        "overall": dict(overall.summary()),
        "by_factor": {
            group: dict(accumulator.summary())
            for group, accumulator in sorted(factors.items())
        },
        "by_source": {
            group: dict(accumulator.summary())
            for group, accumulator in sorted(sources.items())
        },
    }


@dataclass
class _TimeCell:
    episodes: set[str] = field(default_factory=set)
    model_inputs: set[str] = field(default_factory=set)
    target_counts: Counter[str] = field(default_factory=Counter)
    examples: int = 0

    def add(self, example: StructuredSupervisionExample) -> None:
        self.episodes.add(example.episode_id)
        self.model_inputs.add(
            structured_supervision_model_input_fingerprint(example)
        )
        self.target_counts[example.target_group] += 1
        self.examples += 1


@dataclass
class _TimeDependenceAccumulator:
    cells: dict[tuple[int, str], _TimeCell] = field(default_factory=dict)
    examples: int = 0

    def add(self, example: StructuredSupervisionExample) -> None:
        self.cells.setdefault(
            (example.timestep, example.factor_id), _TimeCell()
        ).add(example)
        self.examples += 1

    def summary(self) -> Mapping[str, Any]:
        paired = [cell for cell in self.cells.values() if len(cell.episodes) >= 2]
        varying = [cell for cell in paired if len(cell.model_inputs) >= 2]
        responsive = [cell for cell in varying if len(cell.target_counts) >= 2]

        def accuracy(cells: Sequence[_TimeCell]) -> float | None:
            total = sum(cell.examples for cell in cells)
            if not total:
                return None
            correct = sum(max(cell.target_counts.values()) for cell in cells)
            return correct / total

        if not varying:
            classification = "insufficient_state_variation_at_matched_time"
        elif responsive:
            classification = "state_response_observed"
        else:
            classification = "open_loop_compatible_no_state_response_observed"
        return {
            "examples": self.examples,
            "time_factor_cells": len(self.cells),
            "paired_time_factor_cells": len(paired),
            "paired_examples": sum(cell.examples for cell in paired),
            "state_varying_time_factor_cells": len(varying),
            "state_varying_examples": sum(cell.examples for cell in varying),
            "responsive_time_factor_cells": len(responsive),
            "constant_target_state_varying_cells": len(varying) - len(responsive),
            "time_only_target_accuracy_on_paired_cells": accuracy(paired),
            "time_only_target_accuracy_on_state_varying_cells": accuracy(varying),
            "state_response_cell_rate": (
                len(responsive) / len(varying) if varying else None
            ),
            "classification": classification,
        }


def structured_supervision_time_dependence(
    examples: Sequence[StructuredSupervisionExample],
) -> Mapping[str, Any]:
    """Measure whether labels change with state at matched time and factor.

    The diagnostic pairs examples from different episodes by ``timestep`` and
    ``factor_id``.  It then asks whether the numeric model input changed and,
    when it did, whether the semantic target changed.  A high time-only
    accuracy with no observed response is evidence compatible with an
    open-loop teacher, not proof: many states can legitimately share an
    action.  Pair it with source inspection or deliberate counterfactual
    probes before rejecting a teacher.
    """

    items = tuple(examples)
    if not items:
        raise ValueError("at least one supervision example is required")
    overall = _TimeDependenceAccumulator()
    sources: dict[str, _TimeDependenceAccumulator] = {}
    splits: dict[str, _TimeDependenceAccumulator] = {}
    for example in items:
        overall.add(example)
        sources.setdefault(
            example.source_group, _TimeDependenceAccumulator()
        ).add(example)
        splits.setdefault(example.split, _TimeDependenceAccumulator()).add(example)
    return {
        "schema": STRUCTURED_SUPERVISION_TIME_DEPENDENCE_SCHEMA,
        "matched_key": "timestep plus factor_id across distinct episodes",
        "state_identity": (
            "the exact numeric model-input fingerprint used by structured BC"
        ),
        "decision_identity": "target_group",
        "overall": dict(overall.summary()),
        "by_source": {
            group: dict(accumulator.summary())
            for group, accumulator in sorted(sources.items())
        },
        "by_split": {
            split: dict(accumulator.summary())
            for split, accumulator in sorted(splits.items())
        },
        "caveat": (
            "No observed label change is not proof of state independence; "
            "matched counterfactual probes or source evidence are required."
        ),
    }
