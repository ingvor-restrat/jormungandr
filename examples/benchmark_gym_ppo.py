"""Controlled CartPole PPO comparison against Stable-Baselines3.

This benchmark is a formulation and implementation diagnostic, not a claim
that one library is generally better. Both participants receive the same
Gymnasium environment contract, action space, training budget, fixed held-out
evaluation seeds, and separate 64-by-64 policy/value networks. Jormungandr is
driven through its real PPO plugin; SB3 remains an independently installed
reference implementation.
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


PARTICIPANTS = ("jormungandr", "sb3")


@dataclass(frozen=True)
class BenchmarkConfig:
    participants: tuple[str, ...] = PARTICIPANTS
    runs: int = 3
    total_timesteps: int = 49_152
    rollout_steps: int = 1_024
    evaluation_every_timesteps: int = 4_096
    evaluation_episodes: int = 20
    hidden: int = 64
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

    def __post_init__(self) -> None:
        unknown = set(self.participants) - set(PARTICIPANTS)
        if not self.participants or unknown:
            raise ValueError(f"participants must come from {PARTICIPANTS}")
        if self.runs < 1 or self.total_timesteps < 1:
            raise ValueError("runs and total_timesteps must be positive")
        if self.rollout_steps < 2:
            raise ValueError("rollout_steps must be at least two")
        if self.evaluation_every_timesteps < self.rollout_steps:
            raise ValueError(
                "evaluation_every_timesteps must be at least rollout_steps"
            )
        if self.evaluation_every_timesteps % self.rollout_steps:
            raise ValueError(
                "evaluation_every_timesteps must be divisible by rollout_steps"
            )
        if self.total_timesteps % self.evaluation_every_timesteps:
            raise ValueError(
                "total_timesteps must be divisible by the evaluation interval"
            )
        if self.evaluation_episodes < 1 or self.hidden < 8:
            raise ValueError("evaluation_episodes and hidden must be positive")
        if self.minibatch_size < 2 or self.rollout_steps % self.minibatch_size:
            raise ValueError("minibatch_size must divide rollout_steps")


def _imports() -> tuple[Any, Any]:
    try:
        import gymnasium as gym
        import stable_baselines3
    except ImportError as exc:
        raise RuntimeError(
            "Run this benchmark in the pinned SB3 reference environment; "
            "see benchmarks/requirements-sb3-reference.txt"
        ) from exc
    return gym, stable_baselines3


def _evaluation_seeds(config: BenchmarkConfig) -> tuple[int, ...]:
    return tuple(
        config.base_seed + 50_000_003 + 10_007 * index
        for index in range(config.evaluation_episodes)
    )


def _mean_return(
    policy: Any,
    *,
    participant: str,
    seeds: Sequence[int],
) -> Mapping[str, Any]:
    gym, _ = _imports()
    returns: list[float] = []
    environment = gym.make("CartPole-v1")
    try:
        for seed in seeds:
            observation, _ = environment.reset(seed=int(seed))
            episode_return = 0.0
            while True:
                if participant == "jormungandr":
                    result = policy.action_result(
                        np.asarray(observation, dtype=np.float32),
                        deterministic=True,
                    )
                    action = int(result.action_idx)
                else:
                    predicted, _ = policy.predict(
                        observation, deterministic=True
                    )
                    action = int(np.asarray(predicted).item())
                observation, reward, terminated, truncated, _ = (
                    environment.step(action)
                )
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        environment.close()
    return {
        "return_mean": float(statistics.fmean(returns)),
        "return_std": (
            float(statistics.stdev(returns)) if len(returns) > 1 else 0.0
        ),
        "returns": returns,
        "solved": float(statistics.fmean(returns)) >= 475.0,
    }


def _jormungandr_config(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "action_values": [0.0, 1.0],
        "hidden": config.hidden,
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


def _train_jormungandr(
    config: BenchmarkConfig,
    *,
    run: int,
) -> Mapping[str, Any]:
    gym, _ = _imports()
    seed = config.base_seed + run * 1_000_003
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = algorithm_registry.get("ppo").build(
        4, _jormungandr_config(config), "cpu"
    )
    environment = gym.make("CartPole-v1")
    observation, _ = environment.reset(seed=seed)
    episode_index = 0
    episode_timestep = 0
    evaluation_seeds = _evaluation_seeds(config)
    curve: list[dict[str, Any]] = [
        {
            "timesteps": 0,
            "updates": 0,
            "evaluation": _mean_return(
                agent,
                participant="jormungandr",
                seeds=evaluation_seeds,
            ),
            "learner": None,
        }
    ]
    total_steps = 0
    updates = 0
    started = time.perf_counter()
    try:
        while total_steps < config.total_timesteps:
            observations: list[np.ndarray] = []
            actions: list[int] = []
            rewards: list[float] = []
            next_observations: list[np.ndarray] = []
            dones: list[float] = []
            metadata: list[dict[str, Any]] = []
            for _ in range(config.rollout_steps):
                current = np.asarray(observation, dtype=np.float32)
                decision = agent.action_result(
                    current,
                    deterministic=False,
                    epsilon=0.0,
                )
                following, reward, terminated, truncated, _ = environment.step(
                    int(decision.action_idx)
                )
                done = bool(terminated or truncated)
                observations.append(current)
                actions.append(int(decision.action_idx))
                rewards.append(float(reward))
                next_observations.append(
                    np.asarray(following, dtype=np.float32)
                )
                dones.append(float(done))
                metadata.append(
                    {
                        "actor_id": f"cartpole-run-{run}",
                        "episode_id": f"episode-{episode_index}",
                        "timestep": episode_timestep,
                        "policy_version": updates,
                        "behavior_logp": decision.log_probability,
                        "behavior_value": decision.value,
                        "action_mask": [True, True],
                        "next_action_mask": [True, True],
                    }
                )
                total_steps += 1
                episode_timestep += 1
                observation = following
                if done:
                    episode_index += 1
                    episode_timestep = 0
                    observation, _ = environment.reset()
            batch = (
                np.asarray(observations, dtype=np.float32),
                np.asarray(actions, dtype=np.float32).reshape(-1, 1),
                np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
                np.asarray(next_observations, dtype=np.float32),
                np.asarray(dones, dtype=np.float32).reshape(-1, 1),
            )
            result = agent.update(
                batch,
                np.ones(config.rollout_steps, dtype=np.float32),
                metadata=metadata,
            )
            updates += 1
            if total_steps % config.evaluation_every_timesteps == 0:
                curve.append(
                    {
                        "timesteps": total_steps,
                        "updates": updates,
                        "evaluation": _mean_return(
                            agent,
                            participant="jormungandr",
                            seeds=evaluation_seeds,
                        ),
                        "learner": dict(result.metrics),
                    }
                )
    finally:
        environment.close()
    parameter_count = sum(
        parameter.numel()
        for module in (agent.policy, agent.value)
        for parameter in module.parameters()
    )
    return {
        "participant": "jormungandr",
        "run": run,
        "seed": seed,
        "parameter_count": int(parameter_count),
        "elapsed_seconds": time.perf_counter() - started,
        "curve": curve,
    }


def _train_sb3(
    config: BenchmarkConfig,
    *,
    run: int,
) -> Mapping[str, Any]:
    gym, stable_baselines3 = _imports()
    seed = config.base_seed + run * 1_000_003
    environment = gym.make("CartPole-v1")
    model = stable_baselines3.PPO(
        "MlpPolicy",
        environment,
        learning_rate=config.learning_rate,
        n_steps=config.rollout_steps,
        batch_size=config.minibatch_size,
        n_epochs=config.epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_ratio,
        ent_coef=config.entropy_coefficient,
        vf_coef=config.value_coefficient,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={
            "net_arch": {"pi": [config.hidden, config.hidden], "vf": [config.hidden, config.hidden]}
        },
        seed=seed,
        device="cpu",
        verbose=0,
    )
    evaluation_seeds = _evaluation_seeds(config)
    curve: list[dict[str, Any]] = [
        {
            "timesteps": 0,
            "updates": 0,
            "evaluation": _mean_return(
                model, participant="sb3", seeds=evaluation_seeds
            ),
            "learner": None,
        }
    ]
    started = time.perf_counter()
    try:
        for target in range(
            config.evaluation_every_timesteps,
            config.total_timesteps + 1,
            config.evaluation_every_timesteps,
        ):
            model.learn(
                total_timesteps=config.evaluation_every_timesteps,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            curve.append(
                {
                    "timesteps": target,
                    "updates": target // config.rollout_steps,
                    "evaluation": _mean_return(
                        model,
                        participant="sb3",
                        seeds=evaluation_seeds,
                    ),
                    "learner": None,
                }
            )
    finally:
        environment.close()
    return {
        "participant": "sb3",
        "run": run,
        "seed": seed,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.policy.parameters())
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "curve": curve,
    }


def _aggregate(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    output: dict[str, list[dict[str, Any]]] = {}
    for participant in PARTICIPANTS:
        selected = [row for row in runs if row["participant"] == participant]
        if not selected:
            continue
        steps = [point["timesteps"] for point in selected[0]["curve"]]
        curve: list[dict[str, Any]] = []
        for index, timesteps in enumerate(steps):
            values = [
                float(row["curve"][index]["evaluation"]["return_mean"])
                for row in selected
            ]
            solved = [
                bool(row["curve"][index]["evaluation"]["solved"])
                for row in selected
            ]
            curve.append(
                {
                    "timesteps": int(timesteps),
                    "return_mean": float(statistics.fmean(values)),
                    "return_std_across_runs": (
                        float(statistics.stdev(values))
                        if len(values) > 1
                        else 0.0
                    ),
                    "solved_fraction": float(sum(solved) / len(solved)),
                    "run_means": values,
                }
            )
        output[participant] = curve
    return output


def _plot(path: Path, aggregate: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    colors = {"jormungandr": "#7C3AED", "sb3": "#15803D"}
    labels = {"jormungandr": "Jörmungandr PPO", "sb3": "SB3 PPO"}
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for participant, curve in aggregate.items():
        x = np.asarray([point["timesteps"] for point in curve])
        mean = np.asarray([point["return_mean"] for point in curve])
        std = np.asarray([point["return_std_across_runs"] for point in curve])
        axis.plot(x, mean, label=labels[participant], color=colors[participant])
        axis.fill_between(x, mean - std, mean + std, color=colors[participant], alpha=0.15)
    axis.axhline(475.0, color="#B45309", linestyle="--", linewidth=1.0, label="Solved threshold")
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel("Held-out mean return")
    axis.set_title("CartPole-v1 PPO implementation diagnostic")
    axis.set_ylim(bottom=0.0, top=510.0)
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def run(config: BenchmarkConfig) -> Mapping[str, Any]:
    gym, stable_baselines3 = _imports()
    torch.set_num_threads(1)
    rows: list[Mapping[str, Any]] = []
    for run_index in range(config.runs):
        for participant in config.participants:
            trainer = (
                _train_jormungandr
                if participant == "jormungandr"
                else _train_sb3
            )
            row = trainer(config, run=run_index)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "participant": participant,
                        "run": run_index,
                        "final_return": row["curve"][-1]["evaluation"][
                            "return_mean"
                        ],
                        "elapsed_seconds": row["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "schema": "jormungandr.gym_ppo_reference_benchmark.v1",
        "benchmark": "CartPole-v1 PPO implementation diagnostic",
        "config": asdict(config),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
        "contract": {
            "observation_space": "Box(4)",
            "action_space": "Discrete(2)",
            "policy_value_architecture": "separate 64x64 MLPs by default",
            "evaluation": "deterministic policy on fixed held-out seeds",
            "solved_threshold": 475.0,
            "note": (
                "Matched interface and major PPO controls; optimizer/loss "
                "implementation details remain library-specific."
            ),
        },
        "runs": rows,
        "aggregate": _aggregate(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default=",".join(PARTICIPANTS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=49_152)
    parser.add_argument("--rollout-steps", type=int, default=1_024)
    parser.add_argument("--evaluation-every-timesteps", type=int, default=4_096)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--json-output", default="")
    parser.add_argument("--plot-output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        participants=tuple(
            value.strip().lower()
            for value in args.participants.split(",")
            if value.strip()
        ),
        runs=args.runs,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        evaluation_every_timesteps=args.evaluation_every_timesteps,
        evaluation_episodes=args.evaluation_episodes,
        base_seed=args.seed,
    )
    result = run(config)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if args.plot_output:
        _plot(Path(args.plot_output), result["aggregate"])
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
