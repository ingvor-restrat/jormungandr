"""Isolated gate for prefix-conditioned structured preferences.

The two examples deliberately contain the same observation, legal mask, and
current-factor candidates. Their correct current action differs only because
an earlier factor selected a different candidate. A prefix-independent scorer
therefore cannot exceed 50 percent accuracy on the balanced corpus; the
conditioned model must fit both labels. This tests preference conditioning,
not constraint masking or environment reward.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Mapping

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.structured import EntityCandidateObservation, StructuredPolicySpec
from jormungandr.structured_supervision import StructuredSupervisionExample


SPEC = StructuredPolicySpec(
    global_dim=2,
    entity_dim=3,
    candidate_dim=4,
    entity_type_count=2,
)


@dataclass(frozen=True)
class PrefixGateConfig:
    runs: int = 5
    updates: int = 400
    learning_rate: float = 2e-2
    model_dim: int = 16
    heads: int = 4
    layers: int = 1
    feedforward_dim: int = 32
    prefix_dim: int = 4
    base_seed: int = 20260804

    def __post_init__(self) -> None:
        if self.runs <= 0 or self.updates <= 0:
            raise ValueError("runs and updates must be positive")
        if self.prefix_dim <= 0:
            raise ValueError("the conditioned arm needs a positive prefix dimension")
        if self.model_dim % self.heads:
            raise ValueError("model dimension must be divisible by attention heads")


def _observation() -> EntityCandidateObservation:
    return EntityCandidateObservation(
        global_features=np.asarray([0.25, 1.0], dtype=np.float32),
        entity_features=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        entity_type_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=("worker", "inventory"),
        candidate_features=np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        candidate_ids=(
            "intent:wait",
            "intent:produce",
            "quantity:small",
            "quantity:large",
        ),
        legal_action_mask=np.ones(4, dtype=np.bool_),
        metadata={"benchmark": "PrefixPreferenceGate-v0"},
    )


def _corpus(split: str) -> tuple[StructuredSupervisionExample, ...]:
    observation = _observation()
    common: dict[str, Any] = {
        "actor_id": "exact-teacher",
        "timestep": 0,
        "observation": observation,
        "factor_id": "quantity",
        "candidate_ids": ("quantity:small", "quantity:large"),
        "split": split,
        "source_group": "constructed_exact_case",
        "factor_group": "quantity",
        "target_group": "prefix_tradeoff",
    }
    return (
        StructuredSupervisionExample(
            **common,
            episode_id=f"wait:{split}",
            selected_prefix_candidate_ids=("intent:wait",),
            target_candidate_id="quantity:small",
        ),
        StructuredSupervisionExample(
            **common,
            episode_id=f"produce:{split}",
            selected_prefix_candidate_ids=("intent:produce",),
            target_candidate_id="quantity:large",
        ),
    )


def _agent_config(config: PrefixGateConfig, prefix_dim: int) -> Mapping[str, Any]:
    return {
        "structured_model_dim": config.model_dim,
        "structured_heads": config.heads,
        "structured_layers": config.layers,
        "structured_feedforward_dim": config.feedforward_dim,
        "structured_dropout": 0.0,
        "structured_prefix_dim": prefix_dim,
        "lr": config.learning_rate,
        "max_grad": 5.0,
    }


def _train_arm(
    config: PrefixGateConfig, *, seed: int, prefix_dim: int
) -> Mapping[str, Any]:
    torch.manual_seed(seed)
    agent = algorithm_registry.get("structured_bc").build_structured(
        SPEC, _agent_config(config, prefix_dim), "cpu"
    )
    train = _corpus("train")
    validation = _corpus("validation")
    evaluation = agent.evaluate_structured_supervision(validation)
    updates = 0
    for updates in range(1, config.updates + 1):
        agent.update_structured_supervision(train)
        if updates % 10 == 0 or updates == config.updates:
            evaluation = agent.evaluate_structured_supervision(validation)
            if prefix_dim and evaluation.accuracy == 1.0 and evaluation.nll < 0.02:
                break
    scores = agent.score_results_structured((_observation(),))[0]
    return {
        "seed": seed,
        "updates": updates,
        "accuracy": evaluation.accuracy,
        "nll": evaluation.nll,
        "entropy": evaluation.entropy,
        "parameter_count": sum(
            parameter.numel() for parameter in agent.policy.parameters()
        ),
        "prefix_output_dim": (
            len(scores.candidate_prefix_keys[0])
            if scores.candidate_prefix_keys
            else 0
        ),
    }


def run_benchmark(config: PrefixGateConfig) -> Mapping[str, Any]:
    rows = []
    for run in range(config.runs):
        seed = config.base_seed + run * 104_729
        rows.append(
            {
                "run": run,
                "prefix_independent": _train_arm(
                    config, seed=seed, prefix_dim=0
                ),
                "prefix_conditioned": _train_arm(
                    config, seed=seed, prefix_dim=config.prefix_dim
                ),
            }
        )
    flat_accuracy = [row["prefix_independent"]["accuracy"] for row in rows]
    flat_nll = [row["prefix_independent"]["nll"] for row in rows]
    conditioned_accuracy = [
        row["prefix_conditioned"]["accuracy"] for row in rows
    ]
    conditioned_nll = [row["prefix_conditioned"]["nll"] for row in rows]
    decision = {
        "flat_arm_at_information_limit": all(value == 0.5 for value in flat_accuracy)
        and statistics.median(flat_nll) >= math.log(2.0) - 0.02,
        "conditioned_arm_fits_both_labels": all(
            accuracy == 1.0 and nll < 0.02
            for accuracy, nll in zip(conditioned_accuracy, conditioned_nll)
        ),
        "conditioning_improves_nll_every_run": all(
            conditioned < flat
            for conditioned, flat in zip(conditioned_nll, flat_nll)
        ),
    }
    decision["passed"] = all(decision.values())
    return {
        "schema": "jormungandr.prefix_preference_gate.v1",
        "recorded_at_unix": time.time(),
        "config": asdict(config),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "case": {
            "observation_is_identical": True,
            "current_candidate_set_is_identical": True,
            "legal_mask_is_identical": True,
            "only_input_difference": "selected intent prefix",
            "labels": {
                "intent:wait": "quantity:small",
                "intent:produce": "quantity:large",
            },
        },
        "runs": rows,
        "aggregates": {
            "flat_accuracy_median": statistics.median(flat_accuracy),
            "flat_nll_median": statistics.median(flat_nll),
            "conditioned_accuracy_median": statistics.median(
                conditioned_accuracy
            ),
            "conditioned_nll_median": statistics.median(conditioned_nll),
        },
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--prefix-dim", type=int, default=4)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("docs/latex/figures/prefix_preference_gate.json"),
    )
    args = parser.parse_args()
    config = PrefixGateConfig(
        runs=args.runs,
        updates=args.updates,
        prefix_dim=args.prefix_dim,
    )
    torch.set_num_threads(1)
    result = run_benchmark(config)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
