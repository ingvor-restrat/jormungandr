"""Gated oracle/BC/joint-PPO benchmark on ConstrainedWorkbench.

This is the J1 diagnostic for Jörmungandr's generic structured stack.  It is a
single predeclared setting, not a hyperparameter sweep.  All learned policies
use semantic candidates and actor-owned sequential constraint masks; every
episode carries only one nonzero terminal reward.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Mapping, Sequence

import gymnasium
import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.benchmarks.constrained_workbench import (
    CANDIDATE_DIM,
    ENTITY_DIM,
    ENTITY_TYPE_COUNT,
    GLOBAL_DIM,
    ConstrainedWorkbench,
)
from jormungandr.structured import (
    EntityCandidateObservation,
    StructuredPolicySpec,
)
from jormungandr.structured_supervision import StructuredSupervisionExample
from jormungandr.structured_trajectory import (
    StructuredJointTrajectoryStep,
    sample_structured_joint_action,
)


SPEC = StructuredPolicySpec(
    global_dim=GLOBAL_DIM,
    entity_dim=ENTITY_DIM,
    candidate_dim=CANDIDATE_DIM,
    entity_type_count=ENTITY_TYPE_COUNT,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    runs: int = 3
    horizon: int = 3
    worker_counts: tuple[int, ...] = (2, 3, 4)
    job_counts: tuple[int, ...] = (2, 3, 4)
    bc_train_episodes: int = 256
    bc_validation_episodes: int = 96
    bc_updates: int = 500
    bc_batch_size: int = 128
    ppo_total_turns: int = 12_288
    ppo_rollout_episodes: int = 32
    evaluation_every_turns: int = 1_536
    evaluation_episodes: int = 64
    model_dim: int = 32
    heads: int = 4
    layers: int = 1
    feedforward_dim: int = 64
    prefix_dim: int = 0
    bc_learning_rate: float = 1e-3
    tiny_corpus_learning_rate: float = 1e-2
    tiny_corpus_max_updates: int = 400
    ppo_learning_rate: float = 2e-4
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 96
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.005
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    base_seed: int = 20260804
    bootstrap_samples: int = 10_000

    def __post_init__(self) -> None:
        positive = (
            self.runs,
            self.horizon,
            self.bc_train_episodes,
            self.bc_validation_episodes,
            self.bc_updates,
            self.bc_batch_size,
            self.tiny_corpus_max_updates,
            self.ppo_total_turns,
            self.ppo_rollout_episodes,
            self.evaluation_every_turns,
            self.evaluation_episodes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("benchmark counts and budgets must be positive")
        if self.model_dim % self.heads:
            raise ValueError("model dimension must be divisible by attention heads")
        if self.prefix_dim < 0:
            raise ValueError("prefix dimension cannot be negative")
        if self.ppo_total_turns % self.horizon:
            raise ValueError("PPO turn budget must contain complete episodes")


def _agent_config(config: BenchmarkConfig, *, learning_rate: float) -> dict[str, Any]:
    return {
        "structured_model_dim": config.model_dim,
        "structured_heads": config.heads,
        "structured_layers": config.layers,
        "structured_feedforward_dim": config.feedforward_dim,
        "structured_prefix_dim": config.prefix_dim,
        "structured_dropout": 0.0,
        "lr": learning_rate,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_ratio": config.clip_ratio,
        "entropy_coef": config.entropy_coefficient,
        "value_coef": config.value_coefficient,
        "epochs": config.ppo_epochs,
        "minibatch_size": config.ppo_minibatch_size,
        "max_grad": config.max_grad_norm,
    }


def _environment(config: BenchmarkConfig) -> ConstrainedWorkbench:
    return ConstrainedWorkbench(
        horizon=config.horizon,
        worker_counts=config.worker_counts,
        job_counts=config.job_counts,
    )


def _masked_observation(
    observation: EntityCandidateObservation,
    legal_mask: np.ndarray,
    *,
    factor_id: str,
) -> EntityCandidateObservation:
    return EntityCandidateObservation(
        global_features=observation.global_features,
        entity_features=observation.entity_features,
        entity_type_ids=observation.entity_type_ids,
        entity_ids=observation.entity_ids,
        candidate_features=observation.candidate_features,
        candidate_ids=observation.candidate_ids,
        legal_action_mask=legal_mask,
        candidate_entity_indices=observation.candidate_entity_indices,
        metadata={**dict(observation.metadata), "supervision_factor": factor_id},
    )


def _candidate_role(candidate_id: str) -> str:
    parts = candidate_id.split(":")
    return "pass" if parts[-1] == "pass" else f"job:{parts[-1]}"


def collect_oracle_supervision(
    config: BenchmarkConfig,
    *,
    seeds: Sequence[int],
    split: str,
    actor_id: str,
) -> tuple[StructuredSupervisionExample, ...]:
    environment = _environment(config)
    examples: list[StructuredSupervisionExample] = []
    try:
        for episode_index, seed in enumerate(seeds):
            environment.reset(seed=int(seed))
            episode_id = f"{split}:{actor_id}:{episode_index}:{seed}"
            for timestep in range(config.horizon):
                observation = environment.structured_observation()
                factors = environment.action_factors()
                oracle = environment.oracle_joint_action()
                current_mask = observation.legal_action_mask.copy()
                by_id = {
                    candidate_id: index
                    for index, candidate_id in enumerate(observation.candidate_ids)
                }
                prefix: list[str] = []
                for factor_index, (factor, target) in enumerate(
                    zip(factors, oracle.selected_candidate_ids)
                ):
                    current_mask = environment.legal_mask_update(
                        factor_index, tuple(prefix), current_mask
                    )
                    legal_candidates = tuple(
                        candidate_id
                        for candidate_id in factor.candidate_ids
                        if bool(current_mask[by_id[candidate_id]])
                    )
                    worker = environment.current_turn.workers[factor_index]
                    examples.append(
                        StructuredSupervisionExample(
                            actor_id=actor_id,
                            episode_id=episode_id,
                            timestep=timestep,
                            observation=_masked_observation(
                                observation,
                                current_mask.copy(),
                                factor_id=factor.factor_id,
                            ),
                            factor_id=factor.factor_id,
                            candidate_ids=legal_candidates,
                            selected_prefix_candidate_ids=tuple(prefix),
                            target_candidate_id=target,
                            split=split,
                            source_group="exact_solver",
                            factor_group=f"worker_kind_{worker.kind}",
                            sample_weight=1.0,
                            metadata={"target_role": _candidate_role(target)},
                        )
                    )
                    prefix.append(target)
                environment.step_semantic(oracle.selected_candidate_ids)
    finally:
        environment.close()
    return tuple(examples)


def _frequency_baseline(
    train: Sequence[StructuredSupervisionExample],
    validation: Sequence[StructuredSupervisionExample],
) -> Mapping[str, float]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in train:
        counts[item.factor_group][_candidate_role(item.target_candidate_id)] += 1
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for item in validation:
        group = item.factor_group
        available = {_candidate_role(candidate_id): candidate_id for candidate_id in item.candidate_ids}
        predicted_role = max(
            available,
            key=lambda role: (counts[group][role], role == "pass", role),
        )
        correct[group] += int(available[predicted_role] == item.target_candidate_id)
        total[group] += 1
    return {
        group: correct[group] / total[group]
        for group in sorted(total)
    }


def _train_tiny_corpus(
    config: BenchmarkConfig,
    train: Sequence[StructuredSupervisionExample],
    *,
    seed: int,
) -> Mapping[str, Any]:
    tiny_train = tuple(train[: min(16, len(train))])
    tiny_validation = tuple(replace(item, split="validation") for item in tiny_train)
    torch.manual_seed(seed)
    agent = algorithm_registry.get("structured_bc").build_structured(
        SPEC,
        _agent_config(config, learning_rate=config.tiny_corpus_learning_rate),
        "cpu",
    )
    result = None
    for update in range(1, config.tiny_corpus_max_updates + 1):
        agent.update_structured_supervision(tiny_train)
        result = agent.evaluate_structured_supervision(tiny_validation)
        if result.accuracy == 1.0:
            return {
                "accuracy": result.accuracy,
                "nll": result.nll,
                "updates": update,
                "examples": len(tiny_train),
            }
    assert result is not None
    return {
        "accuracy": result.accuracy,
        "nll": result.nll,
        "updates": config.tiny_corpus_max_updates,
        "examples": len(tiny_train),
    }


def train_behavior_cloning(
    config: BenchmarkConfig,
    train: Sequence[StructuredSupervisionExample],
    validation: Sequence[StructuredSupervisionExample],
    *,
    seed: int,
) -> tuple[Any, Mapping[str, Any]]:
    torch.manual_seed(seed)
    agent = algorithm_registry.get("structured_bc").build_structured(
        SPEC,
        _agent_config(config, learning_rate=config.bc_learning_rate),
        "cpu",
    )
    rng = np.random.default_rng(seed + 97)
    curve: list[dict[str, Any]] = []
    for update in range(1, config.bc_updates + 1):
        indices = rng.choice(
            len(train), size=config.bc_batch_size, replace=len(train) < config.bc_batch_size
        )
        batch = tuple(train[int(index)] for index in np.asarray(indices).reshape(-1))
        result = agent.update_structured_supervision(batch)
        if update == 1 or update % 50 == 0 or update == config.bc_updates:
            validation_result = agent.evaluate_structured_supervision(validation)
            curve.append(
                {
                    "update": update,
                    "train_accuracy": result.accuracy,
                    "train_nll": result.nll,
                    "validation_accuracy": validation_result.accuracy,
                    "validation_nll": validation_result.nll,
                }
            )
    evaluation = agent.evaluate_structured_supervision(validation)
    return agent, {
        "updates": config.bc_updates,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "accuracy": evaluation.accuracy,
        "nll": evaluation.nll,
        "entropy": evaluation.entropy,
        "calibration_error": evaluation.calibration_error,
        "metrics": dict(evaluation.metrics),
        "curve": curve,
    }


def _evaluation_seeds(config: BenchmarkConfig) -> tuple[int, ...]:
    return tuple(
        config.base_seed + 80_000_003 + 10_007 * index
        for index in range(config.evaluation_episodes)
    )


def evaluate_policy(
    config: BenchmarkConfig,
    *,
    seeds: Sequence[int],
    agent: Any | None,
    random_seed: int = 0,
    oracle: bool = False,
) -> Mapping[str, Any]:
    environment = _environment(config)
    rng = np.random.default_rng(random_seed)
    returns: list[float] = []
    trajectory_steps = 0
    factor_choices = 0
    invalid_actions = 0
    try:
        for seed in seeds:
            environment.reset(seed=int(seed))
            episode_return = 0.0
            while True:
                if oracle:
                    selected = environment.oracle_joint_action().selected_candidate_ids
                elif agent is None:
                    selected = environment.random_legal_joint_action(rng)
                else:
                    observation = environment.structured_observation()
                    score = agent.score_results_structured((observation,))[0]
                    sampled = sample_structured_joint_action(
                        observation,
                        environment.action_factors(),
                        score.candidate_logits,
                        behavior_value=score.value,
                        candidate_prefix_keys=score.candidate_prefix_keys,
                        candidate_prefix_values=score.candidate_prefix_values,
                        deterministic=True,
                        legal_mask_update=environment.legal_mask_update,
                    )
                    selected = sampled.selected_candidate_ids
                if not environment.is_feasible(selected):
                    invalid_actions += 1
                factor_choices += len(selected)
                _, reward, terminated, truncated, _ = environment.step_semantic(selected)
                trajectory_steps += 1
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        environment.close()
    return {
        "return_mean": float(statistics.fmean(returns)),
        "return_median": float(statistics.median(returns)),
        "return_std": float(statistics.stdev(returns)) if len(returns) > 1 else 0.0,
        "returns": returns,
        "episodes": len(returns),
        "trajectory_steps": trajectory_steps,
        "factor_choices": factor_choices,
        "invalid_actions": invalid_actions,
    }


def collect_ppo_trajectories(
    config: BenchmarkConfig,
    agent: Any,
    *,
    run: int,
    first_episode: int,
    episodes: int,
) -> tuple[tuple[StructuredJointTrajectoryStep, ...], ...]:
    environment = _environment(config)
    trajectories: list[tuple[StructuredJointTrajectoryStep, ...]] = []
    try:
        for episode_offset in range(episodes):
            episode = first_episode + episode_offset
            seed = config.base_seed + run * 1_000_003 + episode * 7_919
            environment.reset(seed=seed)
            steps: list[StructuredJointTrajectoryStep] = []
            for timestep in range(config.horizon):
                observation = environment.structured_observation()
                score = agent.score_results_structured((observation,))[0]
                sampled = sample_structured_joint_action(
                    observation,
                    environment.action_factors(),
                    score.candidate_logits,
                    behavior_value=score.value,
                    candidate_prefix_keys=score.candidate_prefix_keys,
                    candidate_prefix_values=score.candidate_prefix_values,
                    deterministic=False,
                    rng=np.random.default_rng(seed + timestep * 101),
                    legal_mask_update=environment.legal_mask_update,
                )
                if not environment.is_feasible(sampled.selected_candidate_ids):
                    raise AssertionError("sequential policy produced an infeasible action")
                _, reward, terminated, truncated, _ = environment.step_semantic(
                    sampled.selected_candidate_ids
                )
                next_observation = environment.structured_observation()
                steps.append(
                    StructuredJointTrajectoryStep(
                        actor_id=f"workbench:run:{run}",
                        episode_id=f"workbench:run:{run}:episode:{episode}",
                        timestep=timestep,
                        policy_version=int(agent.update_steps),
                        observation=sampled.observation,
                        factors=sampled.factors,
                        joint_behavior_log_probability=sampled.joint_log_probability,
                        behavior_value=sampled.behavior_value,
                        reward=float(reward),
                        next_observation=next_observation,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        metadata={"benchmark": "ConstrainedWorkbench-v0"},
                    )
                )
            aligned_steps = tuple(
                replace(
                    step,
                    next_observation=(
                        steps[index + 1].observation
                        if index + 1 < len(steps)
                        else step.next_observation
                    ),
                )
                for index, step in enumerate(steps)
            )
            trajectories.append(aligned_steps)
    finally:
        environment.close()
    return tuple(trajectories)


def train_joint_ppo(
    config: BenchmarkConfig,
    *,
    run: int,
    evaluation_seeds: Sequence[int],
    initial_state: Mapping[str, Any] | None,
) -> tuple[Any, Mapping[str, Any]]:
    seed = config.base_seed + run * 1_000_003 + (41 if initial_state else 17)
    torch.manual_seed(seed)
    agent = algorithm_registry.get("structured_ppo").build_structured(
        SPEC,
        _agent_config(config, learning_rate=config.ppo_learning_rate),
        "cpu",
    )
    if initial_state is not None:
        agent.initialize_policy_from_state(initial_state)
    curve = [
        {
            "turns": 0,
            "evaluation": evaluate_policy(
                config, seeds=evaluation_seeds, agent=agent
            ),
        }
    ]
    turns = 0
    episode = 0
    next_evaluation = config.evaluation_every_turns
    last_update: Mapping[str, Any] = {}
    while turns < config.ppo_total_turns:
        remaining_episodes = (config.ppo_total_turns - turns) // config.horizon
        rollout_episodes = min(config.ppo_rollout_episodes, remaining_episodes)
        trajectories = collect_ppo_trajectories(
            config,
            agent,
            run=run,
            first_episode=episode,
            episodes=rollout_episodes,
        )
        update = agent.update_joint_structured(trajectories)
        last_update = asdict(update)
        added_turns = sum(len(trajectory) for trajectory in trajectories)
        turns += added_turns
        episode += rollout_episodes
        if turns >= next_evaluation or turns == config.ppo_total_turns:
            curve.append(
                {
                    "turns": turns,
                    "evaluation": evaluate_policy(
                        config, seeds=evaluation_seeds, agent=agent
                    ),
                    "update": last_update,
                }
            )
            while next_evaluation <= turns:
                next_evaluation += config.evaluation_every_turns
    return agent, {
        "turns": turns,
        "episodes": episode,
        "updates": int(agent.update_steps),
        "curve": curve,
        "final": curve[-1]["evaluation"],
        "last_update": last_update,
    }


def _hierarchical_paired_bootstrap_interval(
    differences_by_run: Sequence[Sequence[float]], *, samples: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(differences_by_run, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("paired differences must form a non-empty run-by-seed matrix")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        run_indices = rng.integers(0, values.shape[0], size=values.shape[0])
        seed_indices = rng.integers(0, values.shape[1], size=values.shape[1])
        draws[index] = float(values[np.ix_(run_indices, seed_indices)].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _worker_count_cardinality_check(config: BenchmarkConfig) -> Mapping[str, Any]:
    rows = []
    for worker_count in (min(config.worker_counts), max(config.worker_counts)):
        environment = ConstrainedWorkbench(
            horizon=config.horizon,
            worker_counts=(worker_count,),
            job_counts=(3,),
        )
        environment.reset(seed=config.base_seed + 123_457)
        rewards = []
        factors = []
        try:
            for timestep in range(config.horizon):
                factors.append(len(environment.action_factors()))
                selected = environment.random_legal_joint_action(
                    np.random.default_rng(config.base_seed + timestep)
                )
                _, reward, _, _, _ = environment.step_semantic(selected)
                rewards.append(float(reward))
        finally:
            environment.close()
        rows.append(
            {
                "worker_count": worker_count,
                "trajectory_steps": len(rewards),
                "nonzero_rewards": sum(value != 0.0 for value in rewards),
                "factors_per_step": factors,
            }
        )
    return {
        "rows": rows,
        "passed": all(
            row["trajectory_steps"] == config.horizon
            and row["nonzero_rewards"] == 1
            and row["factors_per_step"]
            == [row["worker_count"]] * config.horizon
            for row in rows
        ),
    }


def _aggregate_curves(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    by_turn: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        for point in run["curve"]:
            by_turn[int(point["turns"])].append(
                float(point["evaluation"]["return_mean"])
            )
    return [
        {
            "turns": turns,
            "return_mean": float(statistics.fmean(values)),
            "return_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        }
        for turns, values in sorted(by_turn.items())
    ]


def run_benchmark(config: BenchmarkConfig) -> Mapping[str, Any]:
    evaluation_seeds = _evaluation_seeds(config)
    oracle = evaluate_policy(
        config, seeds=evaluation_seeds, agent=None, oracle=True
    )
    runs: list[dict[str, Any]] = []
    for run in range(config.runs):
        train_seeds = tuple(
            config.base_seed
            + run * 1_000_003
            + 10_000_019
            + index * 3_571
            for index in range(config.bc_train_episodes)
        )
        validation_seeds = tuple(
            config.base_seed
            + run * 1_000_003
            + 30_000_023
            + index * 4_099
            for index in range(config.bc_validation_episodes)
        )
        train = collect_oracle_supervision(
            config,
            seeds=train_seeds,
            split="train",
            actor_id=f"oracle-train:{run}",
        )
        validation = collect_oracle_supervision(
            config,
            seeds=validation_seeds,
            split="validation",
            actor_id=f"oracle-validation:{run}",
        )
        frequency = _frequency_baseline(train, validation)
        tiny = _train_tiny_corpus(
            config, train, seed=config.base_seed + run * 31_337
        )
        bc_agent, bc = train_behavior_cloning(
            config,
            train,
            validation,
            seed=config.base_seed + run * 1_000_003 + 73,
        )
        bc["return"] = evaluate_policy(
            config, seeds=evaluation_seeds, agent=bc_agent
        )
        bc["frequency_baseline_by_factor"] = frequency

        random_policy = evaluate_policy(
            config,
            seeds=evaluation_seeds,
            agent=None,
            random_seed=config.base_seed + run * 1_000_003 + 89,
        )
        _, ppo = train_joint_ppo(
            config,
            run=run,
            evaluation_seeds=evaluation_seeds,
            initial_state=None,
        )
        _, bc_ppo = train_joint_ppo(
            config,
            run=run,
            evaluation_seeds=evaluation_seeds,
            initial_state=bc_agent.state_dict(),
        )
        runs.append(
            {
                "run": run,
                "tiny_corpus": tiny,
                "frequency_baseline_by_factor": frequency,
                "behavior_cloning": bc,
                "random_legal": random_policy,
                "joint_ppo": ppo,
                "bc_then_joint_ppo": bc_ppo,
            }
        )

    factor_groups = sorted(
        {
            group
            for run in runs
            for group in run["frequency_baseline_by_factor"]
        }
    )
    bc_accuracy_by_factor: dict[str, float] = {}
    frequency_by_factor: dict[str, float] = {}
    for group in factor_groups:
        key = f"group/factor/{group}/accuracy"
        bc_accuracy_by_factor[group] = float(
            statistics.median(
                run["behavior_cloning"]["metrics"][key] for run in runs
            )
        )
        frequency_by_factor[group] = float(
            statistics.median(
                run["frequency_baseline_by_factor"][group] for run in runs
            )
        )

    paired_differences_by_run = [
        [
            learned - random_return
            for learned, random_return in zip(
                run["joint_ppo"]["final"]["returns"],
                run["random_legal"]["returns"],
            )
        ]
        for run in runs
    ]
    paired_differences = [
        value for run_values in paired_differences_by_run for value in run_values
    ]
    difference_interval = _hierarchical_paired_bootstrap_interval(
        paired_differences_by_run,
        samples=config.bootstrap_samples,
        seed=config.base_seed + 91_000_003,
    )
    bc_ppo_returns = [
        value
        for run in runs
        for value in run["bc_then_joint_ppo"]["final"]["returns"]
    ]
    oracle_median = float(oracle["return_median"])
    bc_ppo_oracle_ratio = float(statistics.median(bc_ppo_returns) / oracle_median)
    invalid_actions = sum(
        int(run[arm]["final"]["invalid_actions"])
        for run in runs
        for arm in ("joint_ppo", "bc_then_joint_ppo")
    ) + sum(
        int(run[arm]["invalid_actions"])
        for run in runs
        for arm in ("random_legal",)
    ) + int(oracle["invalid_actions"])
    worker_count_check = _worker_count_cardinality_check(config)
    decision = {
        "all_actions_feasible": invalid_actions == 0,
        "tiny_corpus_accuracy_100_percent": all(
            run["tiny_corpus"]["accuracy"] == 1.0 for run in runs
        ),
        "bc_exceeds_frequency_every_factor": all(
            bc_accuracy_by_factor[group] > frequency_by_factor[group]
            for group in factor_groups
        ),
        "joint_ppo_better_than_random_95pct": difference_interval[0] > 0.0,
        "bc_then_ppo_at_least_80pct_oracle": bc_ppo_oracle_ratio >= 0.8,
        "worker_count_reward_step_invariant": bool(worker_count_check["passed"]),
    }
    decision["passed"] = all(decision.values())
    return {
        "schema": "jormungandr.constrained_workbench_benchmark.v1",
        "recorded_at_unix": time.time(),
        "config": asdict(config),
        "environment": {
            "name": "ConstrainedWorkbench-v0",
            "reward": "zero until terminal; episode utility / exact oracle utility",
            "constraints": [
                "one job or PASS per worker",
                "job selected at most once",
                "shared capacity",
                "mutually exclusive conflict groups",
            ],
            "oracle": "complete enumeration over the joint assignment",
            "policy_conditioning": (
                "low_rank_additive_v1"
                if config.prefix_dim
                else "prefix_independent"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
        },
        "parameter_count": int(
            sum(
                parameter.numel()
                for parameter in algorithm_registry.get("structured_ppo")
                .build_structured(
                    SPEC,
                    _agent_config(config, learning_rate=config.ppo_learning_rate),
                    "cpu",
                )
                .policy.parameters()
            )
        ),
        "oracle": oracle,
        "worker_count_cardinality_check": worker_count_check,
        "runs": runs,
        "aggregates": {
            "frequency_accuracy_by_factor": frequency_by_factor,
            "bc_accuracy_by_factor": bc_accuracy_by_factor,
            "joint_ppo_minus_random_mean": float(
                statistics.fmean(paired_differences)
            ),
            "joint_ppo_minus_random_hierarchical_paired_95pct": list(
                difference_interval
            ),
            "bc_then_ppo_oracle_median_ratio": bc_ppo_oracle_ratio,
            "joint_ppo_curve": _aggregate_curves(
                [run["joint_ppo"] for run in runs]
            ),
            "bc_then_joint_ppo_curve": _aggregate_curves(
                [run["bc_then_joint_ppo"] for run in runs]
            ),
            "random_legal_mean": float(
                statistics.fmean(run["random_legal"]["return_mean"] for run in runs)
            ),
            "behavior_cloning_mean": float(
                statistics.fmean(
                    run["behavior_cloning"]["return"]["return_mean"] for run in runs
                )
            ),
            "oracle_mean": float(oracle["return_mean"]),
        },
        "decision": decision,
    }


def _plot(result: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    aggregate = result["aggregates"]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for key, label, color in (
        ("joint_ppo_curve", "joint PPO, random start", "#1f77b4"),
        ("bc_then_joint_ppo_curve", "BC then joint PPO", "#d62728"),
    ):
        curve = aggregate[key]
        x = np.asarray([point["turns"] for point in curve])
        mean = np.asarray([point["return_mean"] for point in curve])
        std = np.asarray([point["return_std"] for point in curve])
        axis.plot(x, mean, marker="o", label=label, color=color)
        axis.fill_between(x, mean - std, mean + std, alpha=0.12, color=color)
    axis.axhline(aggregate["oracle_mean"], color="black", linestyle="--", label="exact oracle")
    axis.axhline(
        aggregate["random_legal_mean"], color="#7f7f7f", linestyle=":", label="random legal"
    )
    axis.axhline(
        aggregate["behavior_cloning_mean"], color="#2ca02c", linestyle="-.", label="BC before PPO"
    )
    axis.set_xlabel("environment turns")
    axis.set_ylabel("held-out terminal return / oracle")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _parse_counts(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ppo-total-turns", type=int, default=12_288)
    parser.add_argument("--bc-updates", type=int, default=500)
    parser.add_argument("--bc-train-episodes", type=int, default=256)
    parser.add_argument("--bc-validation-episodes", type=int, default=96)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--prefix-dim", type=int, default=0)
    parser.add_argument("--worker-counts", default="2,3,4")
    parser.add_argument("--job-counts", default="2,3,4")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("docs/latex/figures/constrained_workbench_j1.json"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("docs/latex/figures/constrained_workbench_j1.pdf"),
    )
    args = parser.parse_args()
    config = BenchmarkConfig(
        runs=args.runs,
        ppo_total_turns=args.ppo_total_turns,
        bc_updates=args.bc_updates,
        bc_train_episodes=args.bc_train_episodes,
        bc_validation_episodes=args.bc_validation_episodes,
        evaluation_episodes=args.evaluation_episodes,
        prefix_dim=args.prefix_dim,
        worker_counts=_parse_counts(args.worker_counts),
        job_counts=_parse_counts(args.job_counts),
    )
    torch.set_num_threads(1)
    result = run_benchmark(config)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(result, args.plot_output)
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
