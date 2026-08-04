"""Generic bounded composition of factorized actions.

The environment owns the factors, choices, resources, and any domain-specific
feasibility predicate.  Jormungandr only solves the reusable problem: choose
one option from every factor while maximizing externally supplied utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jormungandr.search import (
    SearchNode,
    UtilityFrontierPruner,
    bounded_beam_search,
)


@dataclass(frozen=True)
class JointActionChoice:
    """One option belonging to exactly one action factor."""

    key: str
    utility: float
    payload: Any = None
    resources: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.key):
            raise ValueError("joint-action choice keys cannot be empty")
        if not math.isfinite(float(self.utility)):
            raise ValueError("joint-action utilities must be finite")
        normalized: dict[str, float] = {}
        for name, amount in dict(self.resources).items():
            value = float(amount)
            if not str(name) or not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "resource names must be non-empty and usages finite/non-negative"
                )
            normalized[str(name)] = value
        object.__setattr__(self, "resources", normalized)


@dataclass(frozen=True)
class JointActionFactor:
    """A categorical factor from which exactly one choice must be selected."""

    key: str
    choices: tuple[JointActionChoice, ...]

    def __post_init__(self) -> None:
        if not str(self.key):
            raise ValueError("joint-action factor keys cannot be empty")
        choices = tuple(self.choices)
        if not choices:
            raise ValueError("every joint-action factor needs at least one choice")
        keys = [choice.key for choice in choices]
        if len(set(keys)) != len(keys):
            raise ValueError("choice keys must be unique within a factor")
        object.__setattr__(self, "choices", choices)


@dataclass(frozen=True)
class JointActionAudit:
    factor_count: int
    exhaustive_combinations: int
    beam_width: int
    generated_candidates: int
    retained_candidates: int
    pruned_candidates: int
    infeasible_candidates: int
    selected_choice_keys: tuple[str, ...]
    resource_usage: Mapping[str, float]
    utility: float


@dataclass(frozen=True)
class JointActionSolution:
    choices: tuple[JointActionChoice, ...]
    utility: float
    resource_usage: Mapping[str, float]
    audit: JointActionAudit


PartialFeasibility = Callable[[Sequence[JointActionChoice], bool], bool]


@dataclass(frozen=True)
class _PartialSelection:
    choices: tuple[JointActionChoice, ...]
    usage: Mapping[str, float]


def compose_joint_action(
    factors: Sequence[JointActionFactor],
    *,
    resource_capacities: Mapping[str, float] | None = None,
    feasible: PartialFeasibility | None = None,
    beam_width: int = 128,
    fallback_choice_keys: Mapping[str, str] | None = None,
) -> JointActionSolution:
    """Return the highest-utility feasible bounded-beam composition.

    Utilities are supplied by the caller, normally by a learned policy.
    Numeric capacities handle common additive constraints.  ``feasible`` is a
    domain-owned predicate for non-linear or coupled constraints and is called
    for every partial selection as well as terminal selections.
    """

    ordered = tuple(factors)
    if not ordered:
        raise ValueError("at least one joint-action factor is required")
    factor_keys = [factor.key for factor in ordered]
    if len(set(factor_keys)) != len(factor_keys):
        raise ValueError("joint-action factor keys must be unique")
    width = int(beam_width)
    if width < 1:
        raise ValueError("beam_width must be positive")

    capacities: dict[str, float] = {}
    for name, capacity in dict(resource_capacities or {}).items():
        value = float(capacity)
        if not str(name) or not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "resource names must be non-empty and capacities finite/non-negative"
            )
        capacities[str(name)] = value

    infeasible_candidates = 0
    root = SearchNode(
        key="root",
        utility=0.0,
        embedding=np.zeros(2, dtype=np.float32),
        payload=_PartialSelection(choices=(), usage={}),
    )

    def expand(node: SearchNode) -> tuple[SearchNode, ...]:
        nonlocal infeasible_candidates
        partial = node.payload
        if not isinstance(partial, _PartialSelection):
            raise TypeError("joint-action search payload was corrupted")
        depth = len(partial.choices)
        if depth >= len(ordered):
            return ()
        factor = ordered[depth]
        children: list[SearchNode] = []
        for choice in factor.choices:
            choices = partial.choices + (choice,)
            usage = dict(partial.usage)
            for name, amount in choice.resources.items():
                usage[name] = usage.get(name, 0.0) + float(amount)
            within_capacity = all(
                usage.get(name, 0.0) <= capacity + 1e-12
                for name, capacity in capacities.items()
            )
            complete = len(choices) == len(ordered)
            if not within_capacity or (
                feasible is not None and not feasible(choices, complete)
            ):
                infeasible_candidates += 1
                continue
            utility = float(node.utility) + float(choice.utility)
            key = f"{node.key}/{factor.key}={choice.key}"
            children.append(
                SearchNode(
                    key=key,
                    utility=utility,
                    embedding=np.asarray(
                        [len(choices) / len(ordered), utility],
                        dtype=np.float32,
                    ),
                    payload=_PartialSelection(choices=choices, usage=usage),
                )
            )
        return tuple(children)

    result = bounded_beam_search(
        root,
        expand,
        beam_width=width,
        max_depth=len(ordered),
        pruner=UtilityFrontierPruner(),
    )
    terminal = [
        node
        for node in result.frontier
        if isinstance(node.payload, _PartialSelection)
        and len(node.payload.choices) == len(ordered)
    ]
    fallback_keys = dict(fallback_choice_keys or {})
    if fallback_keys:
        fallback_choices: list[JointActionChoice] = []
        fallback_usage: dict[str, float] = {}
        for factor in ordered:
            key = fallback_keys.get(factor.key)
            choice = next(
                (candidate for candidate in factor.choices if candidate.key == key),
                None,
            )
            if choice is None:
                raise ValueError(
                    f"fallback choice for factor {factor.key!r} is absent"
                )
            fallback_choices.append(choice)
            for name, amount in choice.resources.items():
                fallback_usage[name] = fallback_usage.get(name, 0.0) + float(amount)
        fallback_valid = all(
            fallback_usage.get(name, 0.0) <= capacity + 1e-12
            for name, capacity in capacities.items()
        ) and (
            feasible is None or feasible(tuple(fallback_choices), True)
        )
        if not fallback_valid:
            raise ValueError("the supplied joint-action fallback is infeasible")
        fallback_utility = sum(choice.utility for choice in fallback_choices)
        terminal.append(
            SearchNode(
                key="fallback/" + "/".join(
                    f"{factor.key}={choice.key}"
                    for factor, choice in zip(ordered, fallback_choices)
                ),
                utility=float(fallback_utility),
                embedding=np.asarray([1.0, fallback_utility], dtype=np.float32),
                payload=_PartialSelection(
                    choices=tuple(fallback_choices), usage=fallback_usage
                ),
            )
        )
    if not terminal:
        raise ValueError("joint-action problem has no feasible complete solution")
    best = min(terminal, key=lambda node: (-float(node.utility), node.key))
    selection = best.payload
    exhaustive = math.prod(len(factor.choices) for factor in ordered)
    usage = dict(sorted(selection.usage.items()))
    audit = JointActionAudit(
        factor_count=len(ordered),
        exhaustive_combinations=int(exhaustive),
        beam_width=width,
        generated_candidates=result.generated_candidates,
        retained_candidates=result.retained_candidates,
        pruned_candidates=result.pruned_candidates,
        infeasible_candidates=infeasible_candidates,
        selected_choice_keys=tuple(choice.key for choice in selection.choices),
        resource_usage=usage,
        utility=float(best.utility),
    )
    return JointActionSolution(
        choices=selection.choices,
        utility=float(best.utility),
        resource_usage=usage,
        audit=audit,
    )
