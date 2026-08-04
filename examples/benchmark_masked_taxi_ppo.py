"""Masked Taxi-v4 PPO comparison against SB3-contrib MaskablePPO.

The benchmark isolates representation and legal-action handling. Both
participants see a 500-element one-hot state, the environment-supplied six-way
action mask, matched policy/value MLPs, and the same fixed evaluation seeds.
Any masked-illegal action is a failed gate independent of episode return.
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


PARTICIPANTS = (
    "jormungandr_masked",
    "jormungandr_unmasked",
    "sb3_contrib",
    "tabular_q_masked",
)
OBSERVATION_DIM = 500
ACTION_COUNT = 6


@dataclass(frozen=True)
class BenchmarkConfig:
    participants: tuple[str, ...] = PARTICIPANTS
    runs: int = 3
    total_timesteps: int = 131_072
    rollout_steps: int = 2_048
    evaluation_every_timesteps: int = 16_384
    evaluation_episodes: int = 50
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
        if self.evaluation_every_timesteps % self.rollout_steps:
            raise ValueError("evaluation interval must divide into rollouts")
        if self.total_timesteps % self.evaluation_every_timesteps:
            raise ValueError("total_timesteps must divide into evaluations")
        if self.rollout_steps % self.minibatch_size:
            raise ValueError("minibatch_size must divide rollout_steps")
        if self.evaluation_episodes < 1 or self.hidden < 8:
            raise ValueError("evaluation_episodes and hidden must be positive")


def _imports() -> tuple[Any, Any, Any]:
    try:
        import gymnasium as gym
        import sb3_contrib
        import stable_baselines3
    except ImportError as exc:
        raise RuntimeError(
            "Run in the pinned SB3 reference environment; see "
            "benchmarks/requirements-sb3-reference.txt"
        ) from exc
    return gym, sb3_contrib, stable_baselines3


def _make_environment() -> Any:
    gym, _, _ = _imports()

    class TaxiActionMask(gym.Wrapper):
        def __init__(self, environment: Any) -> None:
            super().__init__(environment)
            self._mask = np.ones(ACTION_COUNT, dtype=np.bool_)
            self.illegal_actions = 0
            self.action_count = 0

        def reset(self, **kwargs: Any) -> tuple[Any, Mapping[str, Any]]:
            observation, info = self.env.reset(**kwargs)
            self._mask = np.asarray(
                info.get("action_mask"), dtype=np.bool_
            ).reshape(ACTION_COUNT)
            return observation, info

        def step(
            self, action: int
        ) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
            index = int(action)
            self.action_count += 1
            if not bool(self._mask[index]):
                self.illegal_actions += 1
            observation, reward, terminated, truncated, info = self.env.step(
                index
            )
            self._mask = np.asarray(
                info.get("action_mask"), dtype=np.bool_
            ).reshape(ACTION_COUNT)
            return observation, reward, terminated, truncated, info

        def action_masks(self) -> np.ndarray:
            return self._mask.copy()

    return TaxiActionMask(gym.make("Taxi-v4"))


def _one_hot(observation: Any) -> np.ndarray:
    encoded = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    encoded[int(observation)] = 1.0
    return encoded


def _evaluation_seeds(config: BenchmarkConfig) -> tuple[int, ...]:
    return tuple(
        config.base_seed + 70_000_003 + 10_007 * index
        for index in range(config.evaluation_episodes)
    )


def _evaluate(
    policy: Any,
    *,
    participant: str,
    seeds: Sequence[int],
) -> Mapping[str, Any]:
    environment = _make_environment()
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    try:
        for seed in seeds:
            observation, _ = environment.reset(seed=int(seed))
            episode_return = 0.0
            episode_length = 0
            while True:
                mask = environment.action_masks()
                if participant.startswith("jormungandr"):
                    action_mask = (
                        None
                        if participant == "jormungandr_unmasked"
                        else mask
                    )
                    decision = policy.action_result(
                        _one_hot(observation),
                        deterministic=True,
                        action_mask=action_mask,
                    )
                    action = int(decision.action_idx)
                elif participant == "tabular_q_masked":
                    action = int(
                        policy.action(
                            int(observation), mask, deterministic=True
                        )
                    )
                else:
                    predicted, _ = policy.predict(
                        observation,
                        deterministic=True,
                        action_masks=mask,
                    )
                    action = int(np.asarray(predicted).item())
                observation, reward, terminated, truncated, _ = (
                    environment.step(action)
                )
                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated:
                    successes += int(bool(terminated))
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
    finally:
        illegal = int(environment.illegal_actions)
        action_count = int(environment.action_count)
        environment.close()
    return {
        "return_mean": float(statistics.fmean(returns)),
        "return_std": (
            float(statistics.stdev(returns)) if len(returns) > 1 else 0.0
        ),
        "length_mean": float(statistics.fmean(lengths)),
        "success_rate": float(successes / len(returns)),
        "illegal_actions": illegal,
        "actions": action_count,
        "returns": returns,
        "solved": successes / len(returns) >= 0.95 and illegal == 0,
    }


def _jormungandr_config(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "action_values": list(range(ACTION_COUNT)),
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
    masked: bool = True,
) -> Mapping[str, Any]:
    participant = (
        "jormungandr_masked" if masked else "jormungandr_unmasked"
    )
    seed = config.base_seed + run * 1_000_003
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = algorithm_registry.get("ppo").build(
        OBSERVATION_DIM, _jormungandr_config(config), "cpu"
    )
    environment = _make_environment()
    observation, _ = environment.reset(seed=seed)
    episode_index = 0
    episode_timestep = 0
    evaluation_seeds = _evaluation_seeds(config)
    curve: list[dict[str, Any]] = [
        {
            "timesteps": 0,
            "updates": 0,
            "evaluation": _evaluate(
                agent,
                participant=participant,
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
            following_observations: list[np.ndarray] = []
            dones: list[float] = []
            metadata: list[dict[str, Any]] = []
            for _ in range(config.rollout_steps):
                current = _one_hot(observation)
                mask = environment.action_masks()
                decision = agent.action_result(
                    current,
                    deterministic=False,
                    epsilon=0.0,
                    action_mask=mask if masked else None,
                )
                following, reward, terminated, truncated, _ = environment.step(
                    int(decision.action_idx)
                )
                following_mask = environment.action_masks()
                done = bool(terminated or truncated)
                observations.append(current)
                actions.append(int(decision.action_idx))
                rewards.append(float(reward))
                following_observations.append(_one_hot(following))
                dones.append(float(done))
                metadata.append(
                    {
                        "actor_id": f"taxi-run-{run}",
                        "episode_id": f"episode-{episode_index}",
                        "timestep": episode_timestep,
                        "policy_version": updates,
                        "behavior_logp": decision.log_probability,
                        "behavior_value": decision.value,
                        "action_mask": (
                            mask.tolist()
                            if masked
                            else [True] * ACTION_COUNT
                        ),
                        "next_action_mask": (
                            following_mask.tolist()
                            if masked
                            else [True] * ACTION_COUNT
                        ),
                    }
                )
                total_steps += 1
                episode_timestep += 1
                observation = following
                if done:
                    episode_index += 1
                    episode_timestep = 0
                    observation, _ = environment.reset()
            result = agent.update(
                (
                    np.asarray(observations, dtype=np.float32),
                    np.asarray(actions, dtype=np.float32).reshape(-1, 1),
                    np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
                    np.asarray(following_observations, dtype=np.float32),
                    np.asarray(dones, dtype=np.float32).reshape(-1, 1),
                ),
                np.ones(config.rollout_steps, dtype=np.float32),
                metadata=metadata,
            )
            updates += 1
            if total_steps % config.evaluation_every_timesteps == 0:
                curve.append(
                    {
                        "timesteps": total_steps,
                        "updates": updates,
                        "evaluation": _evaluate(
                            agent,
                            participant=participant,
                            seeds=evaluation_seeds,
                        ),
                        "learner": dict(result.metrics),
                    }
                )
    finally:
        training_illegal = int(environment.illegal_actions)
        training_actions = int(environment.action_count)
        environment.close()
    return {
        "participant": participant,
        "run": run,
        "seed": seed,
        "parameter_count": int(
            sum(
                parameter.numel()
                for module in (agent.policy, agent.value)
                for parameter in module.parameters()
            )
        ),
        "training_illegal_actions": training_illegal,
        "training_actions": training_actions,
        "elapsed_seconds": time.perf_counter() - started,
        "curve": curve,
    }


class _MaskedTabularQPolicy:
    def __init__(self, seed: int) -> None:
        self.q = np.zeros(
            (OBSERVATION_DIM, ACTION_COUNT), dtype=np.float32
        )
        self.rng = np.random.default_rng(seed)

    def action(
        self,
        observation: int,
        mask: np.ndarray,
        *,
        deterministic: bool,
        epsilon: float = 0.0,
    ) -> int:
        legal = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
        if legal.size == 0:
            raise ValueError("Taxi mask contains no legal actions")
        if not deterministic and self.rng.random() < float(epsilon):
            return int(self.rng.choice(legal))
        values = self.q[int(observation), legal]
        return int(legal[int(np.argmax(values))])


def _train_tabular_q(
    config: BenchmarkConfig,
    *,
    run: int,
) -> Mapping[str, Any]:
    seed = config.base_seed + run * 1_000_003
    policy = _MaskedTabularQPolicy(seed)
    environment = _make_environment()
    observation, _ = environment.reset(seed=seed)
    evaluation_seeds = _evaluation_seeds(config)
    curve: list[dict[str, Any]] = [
        {
            "timesteps": 0,
            "updates": 0,
            "evaluation": _evaluate(
                policy,
                participant="tabular_q_masked",
                seeds=evaluation_seeds,
            ),
            "learner": None,
        }
    ]
    started = time.perf_counter()
    total_steps = 0
    alpha = 0.20
    try:
        while total_steps < config.total_timesteps:
            mask = environment.action_masks()
            epsilon = max(
                0.05,
                1.0 - total_steps / max(1.0, config.total_timesteps * 0.75),
            )
            action = policy.action(
                int(observation),
                mask,
                deterministic=False,
                epsilon=epsilon,
            )
            following, reward, terminated, truncated, _ = environment.step(
                action
            )
            done = bool(terminated or truncated)
            following_mask = environment.action_masks()
            target = float(reward)
            if not done:
                legal_next = np.flatnonzero(following_mask)
                target += config.gamma * float(
                    np.max(policy.q[int(following), legal_next])
                )
            current = float(policy.q[int(observation), action])
            policy.q[int(observation), action] = current + alpha * (
                target - current
            )
            total_steps += 1
            observation = following
            if done:
                observation, _ = environment.reset()
            if total_steps % config.evaluation_every_timesteps == 0:
                curve.append(
                    {
                        "timesteps": total_steps,
                        "updates": total_steps,
                        "evaluation": _evaluate(
                            policy,
                            participant="tabular_q_masked",
                            seeds=evaluation_seeds,
                        ),
                        "learner": {
                            "epsilon": float(epsilon),
                            "q_abs_mean": float(np.mean(np.abs(policy.q))),
                        },
                    }
                )
    finally:
        training_illegal = int(environment.illegal_actions)
        training_actions = int(environment.action_count)
        environment.close()
    return {
        "participant": "tabular_q_masked",
        "run": run,
        "seed": seed,
        "parameter_count": int(policy.q.size),
        "training_illegal_actions": training_illegal,
        "training_actions": training_actions,
        "elapsed_seconds": time.perf_counter() - started,
        "curve": curve,
    }


def _train_sb3_contrib(
    config: BenchmarkConfig,
    *,
    run: int,
) -> Mapping[str, Any]:
    _, sb3_contrib, _ = _imports()
    seed = config.base_seed + run * 1_000_003
    environment = _make_environment()
    model = sb3_contrib.MaskablePPO(
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
            "net_arch": {
                "pi": [config.hidden, config.hidden],
                "vf": [config.hidden, config.hidden],
            }
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
            "evaluation": _evaluate(
                model,
                participant="sb3_contrib",
                seeds=evaluation_seeds,
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
                    "evaluation": _evaluate(
                        model,
                        participant="sb3_contrib",
                        seeds=evaluation_seeds,
                    ),
                    "learner": None,
                }
            )
    finally:
        training_illegal = int(environment.illegal_actions)
        training_actions = int(environment.action_count)
        environment.close()
    return {
        "participant": "sb3_contrib",
        "run": run,
        "seed": seed,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.policy.parameters())
        ),
        "training_illegal_actions": training_illegal,
        "training_actions": training_actions,
        "elapsed_seconds": time.perf_counter() - started,
        "curve": curve,
    }


def _aggregate(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    output: dict[str, Any] = {}
    for participant in PARTICIPANTS:
        selected = [row for row in runs if row["participant"] == participant]
        if not selected:
            continue
        points = []
        for index, base in enumerate(selected[0]["curve"]):
            returns = [
                float(row["curve"][index]["evaluation"]["return_mean"])
                for row in selected
            ]
            success = [
                float(row["curve"][index]["evaluation"]["success_rate"])
                for row in selected
            ]
            points.append(
                {
                    "timesteps": int(base["timesteps"]),
                    "return_mean": float(statistics.fmean(returns)),
                    "return_std_across_runs": (
                        float(statistics.stdev(returns))
                        if len(returns) > 1
                        else 0.0
                    ),
                    "success_rate_mean": float(statistics.fmean(success)),
                    "illegal_actions": int(
                        sum(
                            int(
                                row["curve"][index]["evaluation"][
                                    "illegal_actions"
                                ]
                            )
                            for row in selected
                        )
                    ),
                }
            )
        output[participant] = points
    return output


def _plot(path: Path, aggregate: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "jormungandr_masked": "#7C3AED",
        "jormungandr_unmasked": "#DC2626",
        "sb3_contrib": "#15803D",
        "tabular_q_masked": "#B45309",
    }
    labels = {
        "jormungandr_masked": "Jörmungandr PPO (masked)",
        "jormungandr_unmasked": "Jörmungandr PPO (unmasked control)",
        "sb3_contrib": "SB3-contrib MaskablePPO",
        "tabular_q_masked": "Masked tabular Q-learning",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for participant, curve in aggregate.items():
        steps = [point["timesteps"] for point in curve]
        axes[0].plot(
            steps,
            [point["return_mean"] for point in curve],
            label=labels[participant],
            color=colors[participant],
        )
        axes[1].plot(
            steps,
            [point["success_rate_mean"] for point in curve],
            label=labels[participant],
            color=colors[participant],
        )
    axes[0].set_ylabel("Held-out mean return")
    axes[1].set_ylabel("Held-out success rate")
    for axis in axes:
        axis.set_xlabel("Environment timesteps")
        axis.grid(alpha=0.2)
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="best")
    figure.suptitle("Taxi-v4 masked-action PPO diagnostic")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def run(config: BenchmarkConfig) -> Mapping[str, Any]:
    gym, sb3_contrib, stable_baselines3 = _imports()
    torch.set_num_threads(1)
    rows: list[Mapping[str, Any]] = []
    for run_index in range(config.runs):
        for participant in config.participants:
            if participant == "jormungandr_masked":
                row = _train_jormungandr(
                    config, run=run_index, masked=True
                )
            elif participant == "jormungandr_unmasked":
                row = _train_jormungandr(
                    config, run=run_index, masked=False
                )
            elif participant == "sb3_contrib":
                row = _train_sb3_contrib(config, run=run_index)
            else:
                row = _train_tabular_q(config, run=run_index)
            rows.append(row)
            final = row["curve"][-1]["evaluation"]
            print(
                json.dumps(
                    {
                        "participant": participant,
                        "run": run_index,
                        "return_mean": final["return_mean"],
                        "success_rate": final["success_rate"],
                        "training_illegal_actions": row[
                            "training_illegal_actions"
                        ],
                        "evaluation_illegal_actions": final[
                            "illegal_actions"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    masked_illegal = sum(
        int(row["training_illegal_actions"])
        + sum(
            int(point["evaluation"]["illegal_actions"])
            for point in row["curve"]
        )
        for row in rows
        if row["participant"] != "jormungandr_unmasked"
    )
    unmasked_illegal = sum(
        int(row["training_illegal_actions"])
        + sum(
            int(point["evaluation"]["illegal_actions"])
            for point in row["curve"]
        )
        for row in rows
        if row["participant"] == "jormungandr_unmasked"
    )
    aggregate = _aggregate(rows)

    def success_auc(participant: str) -> float | None:
        curve = aggregate.get(participant)
        if not curve:
            return None
        return float(
            np.trapezoid(
                [point["success_rate_mean"] for point in curve],
                [point["timesteps"] for point in curve],
            )
            / config.total_timesteps
        )

    masked_auc = success_auc("jormungandr_masked")
    unmasked_auc = success_auc("jormungandr_unmasked")
    efficiency_gate = (
        masked_auc is None
        or unmasked_auc is None
        or masked_auc >= unmasked_auc
    )
    return {
        "schema": "jormungandr.masked_taxi_reference_benchmark.v1",
        "benchmark": "Taxi-v4 masked-action PPO diagnostic",
        "config": asdict(config),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gym.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
        },
        "contract": {
            "observation": "Discrete(500), one-hot for Jormungandr",
            "actions": "Discrete(6), environment-supplied action_mask",
            "gate": (
                "zero illegal actions for masked policies and masked PPO "
                "success AUC no worse than the unmasked control"
            ),
            "masked_illegal_actions": masked_illegal,
            "unmasked_control_illegal_actions": unmasked_illegal,
            "jormungandr_masked_success_auc": masked_auc,
            "jormungandr_unmasked_success_auc": unmasked_auc,
            "mask_gate_passed": masked_illegal == 0 and efficiency_gate,
        },
        "runs": rows,
        "aggregate": aggregate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default=",".join(PARTICIPANTS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=131_072)
    parser.add_argument("--rollout-steps", type=int, default=2_048)
    parser.add_argument("--evaluation-every-timesteps", type=int, default=16_384)
    parser.add_argument("--evaluation-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--json-output", default="")
    parser.add_argument("--plot-output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        participants=tuple(
            item.strip().lower()
            for item in args.participants.split(",")
            if item.strip()
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
