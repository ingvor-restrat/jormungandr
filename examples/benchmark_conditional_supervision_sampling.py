"""Reference gate for weighted structured-supervision minibatches.

The corpus is a domain-neutral three-stratum classification workload.  Two
rare conditional decisions carry larger inverse-frequency weights.  Both arms
target the same exact weighted objective.  The historical arm draws records
uniformly and applies weights after the draw; the candidate arm draws in
proportion to those weights and uses unit loss weights, which is an importance-
resampled estimator of that same objective.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np

import jormungandr
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_supervision import (
    StructuredSupervisionExample,
    apply_structured_supervision_balance_weights,
    structured_supervision_balance_weights,
)
from jormungandr.structured_supervision_store import StructuredSupervisionBuffer


@dataclass(frozen=True)
class ConditionalSupervisionSamplingConfig:
    common_examples: int = 1_000
    rare_examples_per_group: int = 10
    batch_size: int = 64
    batches: int = 3_000
    balance_exponent: float = 0.5
    seed_base: int = 91_000
    maximum_importance_bias: float = 0.005
    maximum_importance_zero_rate: float = 0.01
    minimum_uniform_zero_rate: float = 0.20
    maximum_relative_mean_absolute_error: float = 0.50

    def __post_init__(self) -> None:
        if min(
            self.common_examples,
            self.rare_examples_per_group,
            self.batch_size,
            self.batches,
        ) <= 0:
            raise ValueError("sampling benchmark sizes must be positive")
        if not 0.0 <= self.balance_exponent <= 1.0:
            raise ValueError("balance exponent must be in [0, 1]")
        if min(
            self.maximum_importance_bias,
            self.maximum_importance_zero_rate,
            self.minimum_uniform_zero_rate,
            self.maximum_relative_mean_absolute_error,
        ) <= 0.0:
            raise ValueError("sampling benchmark thresholds must be positive")


def _observation() -> EntityCandidateObservation:
    return EntityCandidateObservation(
        global_features=np.asarray([0.0], dtype=np.float32),
        entity_features=np.asarray([[0.0]], dtype=np.float32),
        entity_type_ids=np.asarray([0], dtype=np.int64),
        entity_ids=("state",),
        candidate_features=np.asarray([[0.0], [1.0]], dtype=np.float32),
        candidate_ids=("choice:left", "choice:right"),
        legal_action_mask=np.ones(2, dtype=np.bool_),
        metadata={"benchmark": "ConditionalSupervisionSampling-v0"},
    )


def _example(
    index: int,
    *,
    balance_group: str,
    diagnostic_loss: float,
) -> StructuredSupervisionExample:
    observation = _observation()
    return StructuredSupervisionExample(
        actor_id="reference-expert",
        episode_id=f"example:{index}",
        timestep=0,
        observation=observation,
        factor_id="decision",
        candidate_ids=observation.candidate_ids,
        target_candidate_id="choice:left",
        target_group="reported-target",
        balance_group=balance_group,
        metadata={"diagnostic_loss": float(diagnostic_loss)},
    )


def _corpus(
    config: ConditionalSupervisionSamplingConfig,
) -> tuple[StructuredSupervisionExample, ...]:
    result: list[StructuredSupervisionExample] = []
    for _ in range(config.common_examples):
        result.append(
            _example(
                len(result),
                balance_group="common-easy",
                diagnostic_loss=0.05,
            )
        )
    for group, loss in (("rare-boundary-a", 1.0), ("rare-boundary-b", 2.0)):
        for _ in range(config.rare_examples_per_group):
            result.append(
                _example(
                    len(result),
                    balance_group=group,
                    diagnostic_loss=loss,
                )
            )
    return tuple(result)


def _arm(
    examples: Sequence[StructuredSupervisionExample],
    config: ConditionalSupervisionSamplingConfig,
    *,
    strategy: str,
    exact_objective: float,
) -> Mapping[str, float]:
    buffer = StructuredSupervisionBuffer(len(examples))
    for example in examples:
        buffer.add(example)
    estimates: list[float] = []
    rare_counts: list[int] = []
    for batch_index in range(config.batches):
        batch = buffer.sample(
            config.batch_size,
            rng=np.random.default_rng(config.seed_base + batch_index),
            strategy=strategy,
        )
        losses = np.asarray(
            [item.metadata["diagnostic_loss"] for item in batch],
            dtype=np.float64,
        )
        weights = np.asarray(
            [item.sample_weight for item in batch], dtype=np.float64
        )
        estimates.append(float(np.average(losses, weights=weights)))
        rare_counts.append(
            sum(item.balance_group != "common-easy" for item in batch)
        )
    values = np.asarray(estimates, dtype=np.float64)
    rare = np.asarray(rare_counts, dtype=np.int64)
    return {
        "objective_mean": float(values.mean()),
        "objective_absolute_bias": abs(float(values.mean()) - exact_objective),
        "objective_standard_deviation": float(values.std()),
        "objective_mean_absolute_error": float(
            np.mean(np.abs(values - exact_objective))
        ),
        "rare_examples_per_batch_mean": float(rare.mean()),
        "rare_examples_per_batch_min": int(rare.min()),
        "rare_zero_batch_rate": float(np.mean(rare == 0)),
    }


def run_benchmark(
    config: ConditionalSupervisionSamplingConfig | None = None,
) -> Mapping[str, Any]:
    resolved = config or ConditionalSupervisionSamplingConfig()
    corpus = _corpus(resolved)
    weights = structured_supervision_balance_weights(
        corpus, exponent=resolved.balance_exponent
    )
    weighted = apply_structured_supervision_balance_weights(corpus, weights)
    losses = np.asarray(
        [item.metadata["diagnostic_loss"] for item in weighted],
        dtype=np.float64,
    )
    declared_weights = np.asarray(
        [item.sample_weight for item in weighted], dtype=np.float64
    )
    exact_objective = float(np.average(losses, weights=declared_weights))
    uniform = _arm(
        weighted,
        resolved,
        strategy="uniform",
        exact_objective=exact_objective,
    )
    importance = _arm(
        weighted,
        resolved,
        strategy="sample_weight",
        exact_objective=exact_objective,
    )
    conditions = {
        "importance_matches_weighted_objective": (
            importance["objective_absolute_bias"]
            <= resolved.maximum_importance_bias
        ),
        "importance_regularly_exposes_rare_decisions": (
            importance["rare_zero_batch_rate"]
            <= resolved.maximum_importance_zero_rate
        ),
        "uniform_batches_demonstrate_rare_omission": (
            uniform["rare_zero_batch_rate"]
            >= resolved.minimum_uniform_zero_rate
        ),
        "importance_reduces_batch_objective_error": (
            importance["objective_mean_absolute_error"]
            <= resolved.maximum_relative_mean_absolute_error
            * uniform["objective_mean_absolute_error"]
        ),
    }
    return {
        "schema": "jormungandr.conditional_supervision_sampling_benchmark.v1",
        "config": asdict(resolved),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jormungandr": jormungandr.__version__,
        },
        "corpus": {
            "examples": len(weighted),
            "balance_groups": {
                group: sum(item.balance_group == group for item in weighted)
                for group in sorted(weights)
            },
            "mean_one_balance_weights": dict(weights),
            "exact_weighted_objective": exact_objective,
        },
        "arms": {
            "uniform_weighted_loss": uniform,
            "sample_weight_importance": importance,
        },
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = ConditionalSupervisionSamplingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=defaults.batches)
    parser.add_argument("--output", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(
        ConditionalSupervisionSamplingConfig(batches=args.batches)
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
