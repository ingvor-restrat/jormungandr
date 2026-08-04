"""CartPole PPO parity test for Jormungandr's structured representation.

The flat reference is the controlled B1 run produced by
``benchmark_gym_ppo.py``.  This benchmark keeps its seeds, environment-turn
budget, reward, PPO controls, and deterministic evaluation panel fixed while
changing only the observation/action representation and compatible policy
encoder.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.algorithms.structured_ppo import StructuredTransition
from jormungandr.structured import (
    EntityCandidateObservation,
    StructuredPolicySpec,
    collate_entity_candidate_observations,
    entity_candidate_observation_from_payload,
    entity_candidate_observation_to_payload,
)


ENTITY_IDS = ("cart-position", "cart-velocity", "pole-angle", "pole-velocity")
CANDIDATE_IDS = ("action:left", "action:right")
SPEC = StructuredPolicySpec(
    global_dim=1,
    entity_dim=2,
    candidate_dim=2,
    entity_type_count=4,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    runs: int = 3
    total_timesteps: int = 49_152
    rollout_steps: int = 1_024
    evaluation_every_timesteps: int = 4_096
    evaluation_episodes: int = 20
    model_dim: int = 32
    heads: int = 4
    layers: int = 2
    feedforward_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.0
    value_coefficient: float = 0.5
    epochs: int = 10
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    base_seed: int = 20260804
    solved_threshold: float = 475.0

    def __post_init__(self) -> None:
        if self.runs < 1 or self.total_timesteps < 1:
            raise ValueError("runs and total timesteps must be positive")
        if self.rollout_steps < 2 or self.evaluation_episodes < 1:
            raise ValueError("rollout and evaluation sizes must be positive")
        if self.total_timesteps % self.evaluation_every_timesteps:
            raise ValueError("total timesteps must divide into evaluation intervals")
        if self.model_dim % self.heads:
            raise ValueError("model dimension must be divisible by heads")


def encode_cartpole(
    observation: Sequence[float],
    *,
    entity_order: Sequence[int] = (0, 1, 2, 3),
    candidate_order: Sequence[int] = (0, 1),
) -> EntityCandidateObservation:
    """Represent coordinates as typed entities and actions as local candidates."""

    state = np.asarray(observation, dtype=np.float32)
    if state.shape != (4,):
        raise ValueError("CartPole observations must contain four coordinates")
    entities = tuple(int(index) for index in entity_order)
    candidates = tuple(int(index) for index in candidate_order)
    if sorted(entities) != [0, 1, 2, 3] or sorted(candidates) != [0, 1]:
        raise ValueError("entity and candidate orders must be permutations")
    entity_features = np.asarray(
        [[state[index], abs(state[index])] for index in entities],
        dtype=np.float32,
    )
    action_features = np.eye(2, dtype=np.float32)[list(candidates)]
    return EntityCandidateObservation(
        global_features=np.asarray([1.0], dtype=np.float32),
        entity_features=entity_features,
        entity_type_ids=np.asarray(entities, dtype=np.int64),
        entity_ids=tuple(ENTITY_IDS[index] for index in entities),
        candidate_features=action_features,
        candidate_ids=tuple(CANDIDATE_IDS[index] for index in candidates),
        legal_action_mask=np.ones(2, dtype=np.bool_),
        candidate_entity_indices=np.full((2, 1), -1, dtype=np.int64),
        metadata={"environment": "CartPole-v1"},
    )


def _agent_config(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "structured_model_dim": config.model_dim,
        "structured_heads": config.heads,
        "structured_layers": config.layers,
        "structured_feedforward_dim": config.feedforward_dim,
        "structured_dropout": 0.0,
        "lr": config.learning_rate,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_ratio": config.clip_ratio,
        "entropy_coef": config.entropy_coefficient,
        "value_coef": config.value_coefficient,
        "epochs": config.epochs,
        "minibatch_size": config.minibatch_size,
        "max_grad": config.max_grad_norm,
    }


def _evaluation_seeds(config: BenchmarkConfig) -> tuple[int, ...]:
    return tuple(
        config.base_seed + 50_000_003 + 10_007 * index
        for index in range(config.evaluation_episodes)
    )


def _mean_return(agent: Any, seeds: Sequence[int]) -> Mapping[str, Any]:
    import gymnasium as gym

    environment = gym.make("CartPole-v1")
    returns: list[float] = []
    try:
        for seed in seeds:
            observation, _ = environment.reset(seed=int(seed))
            episode_return = 0.0
            while True:
                decision = agent.action_result_structured(
                    encode_cartpole(observation), deterministic=True
                )
                action = CANDIDATE_IDS.index(decision.candidate_id)
                observation, reward, terminated, truncated, _ = environment.step(
                    action
                )
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        environment.close()
    return {
        "return_mean": float(statistics.fmean(returns)),
        "return_std": float(statistics.stdev(returns)) if len(returns) > 1 else 0.0,
        "returns": returns,
        "solved": float(statistics.fmean(returns)) >= 475.0,
    }


def _semantic_scores(agent: Any, observation: EntityCandidateObservation):
    batch = collate_entity_candidate_observations((observation,)).to_torch("cpu")
    with torch.no_grad():
        output = agent.policy(batch)
    scores = {
        candidate_id: float(output.logits[0, index])
        for index, candidate_id in enumerate(observation.candidate_ids)
    }
    return scores, float(output.values[0])


def _identity_diagnostics(agent: Any) -> Mapping[str, Any]:
    probes = (
        np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.15, -0.7, 0.03, 0.4], dtype=np.float32),
        np.asarray([-0.4, 0.9, -0.08, -0.5], dtype=np.float32),
    )
    maximum_logit_difference = 0.0
    maximum_value_difference = 0.0
    semantic_actions_match = True
    wire_identity_match = True
    for probe in probes:
        canonical = encode_cartpole(probe)
        permuted = encode_cartpole(
            probe,
            entity_order=(2, 0, 3, 1),
            candidate_order=(1, 0),
        )
        restored = entity_candidate_observation_from_payload(
            entity_candidate_observation_to_payload(permuted), spec=SPEC
        )
        wire_identity_match &= (
            restored.entity_ids == permuted.entity_ids
            and restored.candidate_ids == permuted.candidate_ids
        )
        canonical_scores, canonical_value = _semantic_scores(agent, canonical)
        permuted_scores, permuted_value = _semantic_scores(agent, restored)
        maximum_logit_difference = max(
            maximum_logit_difference,
            *(abs(canonical_scores[key] - permuted_scores[key]) for key in CANDIDATE_IDS),
        )
        maximum_value_difference = max(
            maximum_value_difference, abs(canonical_value - permuted_value)
        )
        semantic_actions_match &= max(
            canonical_scores, key=canonical_scores.get
        ) == max(permuted_scores, key=permuted_scores.get)

    restored_agent = algorithm_registry.get("structured_ppo").build_structured(
        SPEC, _agent_config(_IDENTITY_CONFIG), "cpu"
    )
    restored_agent.load_state_dict(agent.state_dict())
    source_action = agent.action_result_structured(
        encode_cartpole(probes[1]), deterministic=True
    ).candidate_id
    restored_action = restored_agent.action_result_structured(
        encode_cartpole(probes[1]), deterministic=True
    ).candidate_id
    return {
        "wire_identity_match": bool(wire_identity_match),
        "semantic_actions_match_under_permutation": bool(semantic_actions_match),
        "max_semantic_logit_difference": maximum_logit_difference,
        "max_value_difference": maximum_value_difference,
        "checkpoint_restore_action_match": source_action == restored_action,
        "tolerance": 1e-5,
    }


# Rebound to the active configuration during each run; this keeps the restore
# construction honest without serializing benchmark-specific global state.
_IDENTITY_CONFIG = BenchmarkConfig(runs=1)


def _train(config: BenchmarkConfig, *, run: int) -> Mapping[str, Any]:
    global _IDENTITY_CONFIG
    import gymnasium as gym

    _IDENTITY_CONFIG = config
    seed = config.base_seed + run * 1_000_003
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = algorithm_registry.get("structured_ppo").build_structured(
        SPEC, _agent_config(config), "cpu"
    )
    environment = gym.make("CartPole-v1")
    observation, _ = environment.reset(seed=seed)
    evaluation_seeds = _evaluation_seeds(config)
    curve: list[dict[str, Any]] = [
        {
            "timesteps": 0,
            "updates": 0,
            "evaluation": _mean_return(agent, evaluation_seeds),
            "learner": None,
        }
    ]
    complete_trajectories: list[list[StructuredTransition]] = []
    ready_steps = 0
    current: list[StructuredTransition] = []
    episode_index = 0
    episode_timestep = 0
    updates = 0
    latest_update: Mapping[str, Any] | None = None
    started = time.perf_counter()
    try:
        for total_steps in range(1, config.total_timesteps + 1):
            encoded = encode_cartpole(observation)
            decision = agent.action_result_structured(encoded, deterministic=False)
            action = CANDIDATE_IDS.index(decision.candidate_id)
            following, reward, terminated, truncated, _ = environment.step(action)
            done = bool(terminated or truncated)
            current.append(
                StructuredTransition(
                    episode_id=f"cartpole-run-{run}:episode-{episode_index}",
                    timestep=episode_timestep,
                    observation=encoded,
                    candidate_id=decision.candidate_id,
                    candidate_index=decision.candidate_index,
                    behavior_log_probability=decision.log_probability,
                    behavior_value=decision.value,
                    reward=float(reward),
                    done=done,
                )
            )
            episode_timestep += 1
            observation = following
            if done:
                complete_trajectories.append(current)
                ready_steps += len(current)
                current = []
                episode_index += 1
                episode_timestep = 0
                observation, _ = environment.reset()
                if ready_steps >= config.rollout_steps:
                    result = agent.update_structured(complete_trajectories)
                    updates += 1
                    latest_update = asdict(result)
                    complete_trajectories = []
                    ready_steps = 0
            if total_steps % config.evaluation_every_timesteps == 0:
                curve.append(
                    {
                        "timesteps": total_steps,
                        "updates": updates,
                        "evaluation": _mean_return(agent, evaluation_seeds),
                        "learner": latest_update,
                    }
                )
    finally:
        environment.close()
    return {
        "participant": "jormungandr_structured",
        "run": run,
        "seed": seed,
        "parameter_count": int(
            sum(parameter.numel() for parameter in agent.policy.parameters())
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "unused_incomplete_steps": len(current) + ready_steps,
        "identity": _identity_diagnostics(agent),
        "curve": curve,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = []
    for index, source in enumerate(rows[0]["curve"]):
        values = [float(row["curve"][index]["evaluation"]["return_mean"]) for row in rows]
        result.append(
            {
                "timesteps": int(source["timesteps"]),
                "return_mean": float(statistics.fmean(values)),
                "return_median": float(statistics.median(values)),
                "return_std_across_runs": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                "solved_fraction": sum(
                    bool(row["curve"][index]["evaluation"]["solved"]) for row in rows
                ) / len(rows),
                "run_means": values,
            }
        )
    return result


def _first_solved(row: Mapping[str, Any]) -> int | None:
    return next(
        (
            int(point["timesteps"])
            for point in row["curve"]
            if bool(point["evaluation"]["solved"])
        ),
        None,
    )


def _flat_rows(reference: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row for row in reference.get("runs", [])
        if row.get("participant") == "jormungandr"
    ]


def _decision(
    config: BenchmarkConfig,
    rows: Sequence[Mapping[str, Any]],
    flat_reference: Mapping[str, Any],
) -> Mapping[str, Any]:
    flat = _flat_rows(flat_reference)
    if len(flat) != config.runs:
        raise ValueError("flat reference run count does not match structured runs")
    flat_final = statistics.median(
        float(row["curve"][-1]["evaluation"]["return_mean"]) for row in flat
    )
    structured_final = statistics.median(
        float(row["curve"][-1]["evaluation"]["return_mean"]) for row in rows
    )
    flat_thresholds = [_first_solved(row) for row in flat]
    structured_thresholds = [_first_solved(row) for row in rows]
    finite_flat = [value for value in flat_thresholds if value is not None]
    finite_structured = [value for value in structured_thresholds if value is not None]
    final_gate = structured_final >= 0.9 * flat_final
    threshold_gate = (
        len(finite_structured) == len(rows)
        and bool(finite_flat)
        and statistics.median(finite_structured)
        <= 2.0 * statistics.median(finite_flat)
    )
    identity_gate = all(
        row["identity"]["wire_identity_match"]
        and row["identity"]["semantic_actions_match_under_permutation"]
        and row["identity"]["checkpoint_restore_action_match"]
        and row["identity"]["max_semantic_logit_difference"] <= 1e-5
        and row["identity"]["max_value_difference"] <= 1e-5
        for row in rows
    )
    return {
        "gate": "S0",
        "passed": bool(final_gate and threshold_gate and identity_gate),
        "final_return_gate": bool(final_gate),
        "time_to_threshold_gate": bool(threshold_gate),
        "identity_and_permutation_gate": bool(identity_gate),
        "flat_final_median": float(flat_final),
        "structured_final_median": float(structured_final),
        "flat_first_solved_timesteps": flat_thresholds,
        "structured_first_solved_timesteps": structured_thresholds,
    }


def _plot(path: Path, structured, flat_reference) -> None:
    import matplotlib.pyplot as plt

    flat = flat_reference["aggregate"]["jormungandr"]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(
        [point["timesteps"] for point in flat],
        [point["return_mean"] for point in flat],
        label="Flat PPO (B1)", color="#2563EB",
    )
    axis.plot(
        [point["timesteps"] for point in structured],
        [point["return_mean"] for point in structured],
        label="Entity/candidate PPO", color="#7C3AED",
    )
    axis.axhline(475.0, color="#B45309", linestyle="--", linewidth=1.0, label="Solved threshold")
    axis.set(xlabel="Environment timesteps", ylabel="Held-out mean return", title="CartPole representation parity (S0)")
    axis.set_ylim(0, 510)
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def run(config: BenchmarkConfig, flat_reference: Mapping[str, Any]) -> Mapping[str, Any]:
    import gymnasium as gym

    torch.set_num_threads(1)
    rows = []
    for run_index in range(config.runs):
        row = _train(config, run=run_index)
        rows.append(row)
        print(json.dumps({
            "run": run_index,
            "final_return": row["curve"][-1]["evaluation"]["return_mean"],
            "first_solved": _first_solved(row),
            "elapsed_seconds": row["elapsed_seconds"],
        }, sort_keys=True), flush=True)
    aggregate = _aggregate(rows)
    return {
        "schema": "jormungandr.structured_cartpole_parity.v1",
        "benchmark": "CartPole-v1 flat versus entity/candidate PPO",
        "config": asdict(config),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gym.__version__,
        },
        "representation": {
            "spec": asdict(SPEC),
            "entity_ids": list(ENTITY_IDS),
            "candidate_ids": list(CANDIDATE_IDS),
            "policy": "EntityCandidateTransformer",
        },
        "runs": rows,
        "aggregate": {"jormungandr_structured": aggregate},
        "decision": _decision(config, rows, flat_reference),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-reference", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=49_152)
    parser.add_argument("--rollout-steps", type=int, default=1_024)
    parser.add_argument("--evaluation-every-timesteps", type=int, default=4_096)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--json-output", default="")
    parser.add_argument("--plot-output", default="")
    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        runs=args.runs,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        evaluation_every_timesteps=args.evaluation_every_timesteps,
        evaluation_episodes=args.evaluation_episodes,
        base_seed=args.seed,
    )
    reference = json.loads(Path(args.flat_reference).read_text(encoding="utf-8"))
    result = run(config, reference)
    if args.json_output:
        destination = Path(args.json_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if args.plot_output:
        _plot(Path(args.plot_output), result["aggregate"]["jormungandr_structured"], reference)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0 if result["decision"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
