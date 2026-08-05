"""Controlled gate for structured PPO critic isolation and ratio transactions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.structured import EntityCandidateObservation, StructuredPolicySpec
from jormungandr.structured_trajectory import (
    StructuredFactorChoice,
    StructuredJointTrajectoryStep,
)


SPEC = StructuredPolicySpec(2, 3, 4, 2)
RATIO_MIN = 0.5
RATIO_MAX = 2.0


def _observation(step: int, candidates: int = 2) -> EntityCandidateObservation:
    return EntityCandidateObservation(
        global_features=np.asarray([step / 10.0, 1.0], dtype=np.float32),
        entity_features=np.asarray(
            [[step, 0.0, 1.0], [0.0, 1.0, step]], dtype=np.float32
        ),
        entity_type_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=(f"entity-a:{step}", f"entity-b:{step}"),
        candidate_features=np.arange(
            candidates * 4, dtype=np.float32
        ).reshape(candidates, 4),
        candidate_ids=tuple(f"action:{index}" for index in range(candidates)),
        legal_action_mask=np.ones(candidates, dtype=np.bool_),
    )


def _agent(
    seed: int,
    *,
    learning_rate: float,
    value_coefficient: float,
    value_backbone_gradient_scale: float,
    ratio_guard: bool,
):
    torch.manual_seed(seed)
    config: dict[str, Any] = {
        "structured_model_dim": 8,
        "structured_heads": 2,
        "structured_layers": 1,
        "structured_feedforward_dim": 16,
        "structured_dropout": 0.0,
        "epochs": 2,
        "minibatch_size": 2,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "lr": learning_rate,
        "entropy_coef": 0.0,
        "value_coef": value_coefficient,
        "value_backbone_gradient_scale": value_backbone_gradient_scale,
    }
    if ratio_guard:
        config.update(
            {
                "policy_ratio_guard_min": RATIO_MIN,
                "policy_ratio_guard_max": RATIO_MAX,
                "policy_ratio_guard_backoff_factor": 0.1,
                "policy_ratio_guard_max_backtracks": 6,
            }
        )
    return algorithm_registry.get("structured_ppo").build_structured(
        SPEC, config, "cpu"
    )


def _joint_trajectory(agent, rewards: Sequence[float]):
    observation = _observation(0)
    score = agent.score_results_structured((observation,))[0]
    log_probabilities = torch.log_softmax(
        torch.as_tensor(score.candidate_logits), dim=0
    )
    return tuple(
        (
            StructuredJointTrajectoryStep(
                actor_id="safety-benchmark",
                episode_id=f"episode:{index}",
                timestep=0,
                policy_version=0,
                observation=observation,
                factors=(
                    StructuredFactorChoice(
                        factor_id="choice",
                        candidate_ids=observation.candidate_ids,
                        selected_candidate_id=(
                            observation.candidate_ids[index]
                        ),
                        behavior_log_probability=float(
                            log_probabilities[index]
                        ),
                    ),
                ),
                joint_behavior_log_probability=float(
                    log_probabilities[index]
                ),
                behavior_value=score.value,
                reward=float(reward),
                next_observation=_observation(1),
                terminated=True,
            ),
        )
        for index, reward in enumerate(rewards)
    )


def _parameter_state(agent) -> Mapping[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in agent.policy.named_parameters()
    }


def _maximum_drift(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    value_head: bool,
) -> float:
    return max(
        float((after[name] - value).abs().max())
        for name, value in before.items()
        if name.startswith("value_head.") is value_head
    )


def _critic_isolation_case(seed: int) -> Mapping[str, Any]:
    rows = {}
    for label, scale in (("shared", 1.0), ("isolated", 0.0)):
        agent = _agent(
            seed,
            learning_rate=1e-2,
            value_coefficient=1.0,
            value_backbone_gradient_scale=scale,
            ratio_guard=False,
        )
        trajectory = _joint_trajectory(agent, (1.0,))
        before = _parameter_state(agent)
        update = agent.update_joint_structured(trajectory)
        after = _parameter_state(agent)
        rows[label] = {
            "shared_backbone_max_abs_drift": _maximum_drift(
                before, after, value_head=False
            ),
            "value_head_max_abs_drift": _maximum_drift(
                before, after, value_head=True
            ),
            "update": asdict(update),
        }
    return rows


def _ratio_transaction_case(seed: int) -> Mapping[str, Any]:
    rows = {}
    for label, guarded in (("unguarded", False), ("guarded", True)):
        agent = _agent(
            seed,
            learning_rate=1.0,
            value_coefficient=0.0,
            value_backbone_gradient_scale=0.0,
            ratio_guard=guarded,
        )
        trajectories = _joint_trajectory(agent, (1.0, -1.0))
        rows[label] = asdict(agent.update_joint_structured(trajectories))
    return rows


def run_benchmark(
    seeds: Sequence[int] = (101, 103, 107, 109, 113),
) -> Mapping[str, Any]:
    declared_seeds = tuple(int(seed) for seed in seeds)
    if not declared_seeds:
        raise ValueError("at least one benchmark seed is required")
    runs = []
    for seed in declared_seeds:
        runs.append(
            {
                "seed": seed,
                "critic_isolation": _critic_isolation_case(seed),
                "ratio_transaction": _ratio_transaction_case(seed),
            }
        )
    conditions = {
        "shared_critic_moves_policy_backbone": all(
            run["critic_isolation"]["shared"][
                "shared_backbone_max_abs_drift"
            ]
            > 0.0
            for run in runs
        ),
        "isolated_critic_leaves_policy_backbone_exact": all(
            run["critic_isolation"]["isolated"][
                "shared_backbone_max_abs_drift"
            ]
            == 0.0
            for run in runs
        ),
        "isolated_value_head_still_learns": all(
            run["critic_isolation"]["isolated"][
                "value_head_max_abs_drift"
            ]
            > 0.0
            for run in runs
        ),
        "unguarded_proposal_leaves_ratio_bounds": all(
            run["ratio_transaction"]["unguarded"][
                "post_update_importance_ratio_min"
            ]
            < RATIO_MIN
            or run["ratio_transaction"]["unguarded"][
                "post_update_importance_ratio_max"
            ]
            > RATIO_MAX
            for run in runs
        ),
        "guarded_proposal_is_accepted_after_backoff": all(
            run["ratio_transaction"]["guarded"][
                "trust_region_update_accepted"
            ]
            and run["ratio_transaction"]["guarded"][
                "trust_region_backtracks"
            ]
            > 0
            for run in runs
        ),
        "guarded_full_batch_ratios_are_contained": all(
            run["ratio_transaction"]["guarded"][
                "post_update_importance_ratio_min"
            ]
            >= RATIO_MIN
            and run["ratio_transaction"]["guarded"][
                "post_update_importance_ratio_max"
            ]
            <= RATIO_MAX
            for run in runs
        ),
    }
    conditions["passed"] = all(conditions.values())
    return {
        "schema": "jormungandr.structured_ppo_safety_benchmark.v1",
        "seeds": list(declared_seeds),
        "ratio_bounds": {"min": RATIO_MIN, "max": RATIO_MAX},
        "runs": runs,
        "decision": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="101,103,107,109,113")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(
        tuple(int(value) for value in args.seeds.split(",") if value.strip())
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
