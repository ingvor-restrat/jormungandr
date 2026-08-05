"""Reference gate for terminal credit over a 719-step structured trajectory.

This benchmark contains no environment strategy and performs no learning. It
compares Jormungandr's exact structured-PPO targets with their closed form and,
when available, Stable-Baselines3's independent rollout-buffer calculation.
The two arms differ only in GAE lambda.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import jormungandr
from jormungandr.algorithms import algorithm_registry
from jormungandr.structured import EntityCandidateObservation, StructuredPolicySpec
from jormungandr.structured_trajectory import (
    StructuredFactorChoice,
    StructuredJointTrajectoryStep,
)


SPEC = StructuredPolicySpec(
    global_dim=2,
    entity_dim=2,
    candidate_dim=2,
    entity_type_count=1,
)


@dataclass(frozen=True)
class DelayedCreditConfig:
    """Frozen two-arm return-propagation comparison."""

    horizon: int = 719
    gamma: float = 1.0
    gae_lambdas: tuple[float, ...] = (0.98, 1.0)
    terminal_rewards: tuple[float, ...] = (-1.0, 1.0)
    current_lambda: float = 0.98
    candidate_lambda: float = 1.0
    tolerance: float = 2e-6
    current_opening_upper_bound: float = 1e-6
    candidate_opening_lower_bound: float = 0.999999

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("delayed-credit horizon must be at least two")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not self.gae_lambdas or any(
            not 0.0 <= value <= 1.0 for value in self.gae_lambdas
        ):
            raise ValueError("GAE lambdas must be non-empty and in [0, 1]")
        if len(set(self.gae_lambdas)) != len(self.gae_lambdas):
            raise ValueError("GAE lambda arms must be unique")
        if self.current_lambda not in self.gae_lambdas:
            raise ValueError("current lambda must be one of the declared arms")
        if self.candidate_lambda not in self.gae_lambdas:
            raise ValueError("candidate lambda must be one of the declared arms")
        if tuple(self.terminal_rewards) != (-1.0, 1.0):
            raise ValueError("the frozen diagnostic requires balanced -1/+1 returns")
        if min(
            self.tolerance,
            self.current_opening_upper_bound,
            self.candidate_opening_lower_bound,
        ) <= 0.0:
            raise ValueError("gate thresholds must be positive")


def _observation(timestep: int, horizon: int) -> EntityCandidateObservation:
    progress = float(timestep) / float(max(1, horizon - 1))
    return EntityCandidateObservation(
        global_features=np.asarray(
            [progress, float(timestep == 0)], dtype=np.float32
        ),
        entity_features=np.asarray([[progress, 1.0]], dtype=np.float32),
        entity_type_ids=np.asarray([0], dtype=np.int64),
        entity_ids=("clock",),
        candidate_features=np.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
        candidate_ids=("choice:left", "choice:right"),
        legal_action_mask=np.ones(2, dtype=np.bool_),
        candidate_entity_indices=np.asarray([0, 0], dtype=np.int64),
        metadata={"benchmark": "DelayedTerminalCredit-v0"},
    )


def _trajectory(
    config: DelayedCreditConfig,
    *,
    terminal_reward: float,
) -> tuple[StructuredJointTrajectoryStep, ...]:
    observations = tuple(
        _observation(timestep, config.horizon)
        for timestep in range(config.horizon + 1)
    )
    factor = StructuredFactorChoice(
        factor_id="decision",
        candidate_ids=("choice:left", "choice:right"),
        selected_candidate_id="choice:left",
        behavior_log_probability=-math.log(2.0),
    )
    return tuple(
        StructuredJointTrajectoryStep(
            actor_id="delayed-credit-reference",
            episode_id=f"terminal:{terminal_reward:+.0f}",
            timestep=timestep,
            policy_version=0,
            observation=observations[timestep],
            factors=(factor,),
            joint_behavior_log_probability=-math.log(2.0),
            behavior_value=0.0,
            reward=(
                float(terminal_reward)
                if timestep == config.horizon - 1
                else 0.0
            ),
            next_observation=observations[timestep + 1],
            terminated=timestep == config.horizon - 1,
            metadata={"terminal_reward": float(terminal_reward)},
        )
        for timestep in range(config.horizon)
    )


def _analytic_targets(
    config: DelayedCreditConfig,
    *,
    gae_lambda: float,
    terminal_reward: float,
) -> np.ndarray:
    distances = np.arange(config.horizon - 1, -1, -1, dtype=np.float64)
    return float(terminal_reward) * np.power(
        float(config.gamma) * float(gae_lambda), distances
    )


def _summary(
    config: DelayedCreditConfig,
    *,
    gae_lambda: float,
    advantages_by_reward: Mapping[float, np.ndarray],
    returns_by_reward: Mapping[float, np.ndarray],
) -> Mapping[str, Any]:
    advantage_errors: list[float] = []
    return_errors: list[float] = []
    raw: list[np.ndarray] = []
    terminal_rows: dict[str, Any] = {}
    for terminal_reward in config.terminal_rewards:
        expected = _analytic_targets(
            config,
            gae_lambda=gae_lambda,
            terminal_reward=terminal_reward,
        )
        advantages = np.asarray(
            advantages_by_reward[terminal_reward], dtype=np.float64
        )
        returns = np.asarray(
            returns_by_reward[terminal_reward], dtype=np.float64
        )
        if advantages.shape != (config.horizon,) or returns.shape != (
            config.horizon,
        ):
            raise ValueError("reference target vector has the wrong horizon")
        advantage_errors.append(float(np.max(np.abs(advantages - expected))))
        return_errors.append(float(np.max(np.abs(returns - expected))))
        raw.append(advantages)
        terminal_rows[f"{terminal_reward:+.0f}"] = {
            "opening_advantage": float(advantages[0]),
            "terminal_advantage": float(advantages[-1]),
            "opening_return": float(returns[0]),
            "terminal_return": float(returns[-1]),
        }
    flat = np.concatenate(raw)
    normalized = (flat - float(flat.mean())) / max(float(flat.std()), 1e-8)
    positive_offset = config.terminal_rewards.index(1.0) * config.horizon
    decay = config.gamma * gae_lambda
    half_life = (
        None
        if decay == 1.0
        else math.log(0.5) / math.log(decay)
        if decay > 0.0
        else 0.0
    )
    return {
        "gae_lambda": float(gae_lambda),
        "gae_decay": float(decay),
        "half_life_steps": half_life,
        "analytic_opening_weight": float(decay ** (config.horizon - 1)),
        "normalized_positive_opening_advantage": float(
            normalized[positive_offset]
        ),
        "maximum_advantage_absolute_error": max(advantage_errors),
        "maximum_return_absolute_error": max(return_errors),
        "terminal_rewards": terminal_rows,
    }


def _jormungandr_targets(
    config: DelayedCreditConfig,
    *,
    gae_lambda: float,
) -> Mapping[float, tuple[np.ndarray, np.ndarray]]:
    agent = algorithm_registry.get("structured_ppo").build_structured(
        SPEC,
        {
            "structured_model_dim": 8,
            "structured_heads": 2,
            "structured_layers": 1,
            "structured_feedforward_dim": 16,
            "gamma": config.gamma,
            "gae_lambda": gae_lambda,
            "epochs": 1,
            "minibatch_size": config.horizon * len(config.terminal_rewards),
        },
        "cpu",
    )
    trajectories = tuple(
        _trajectory(config, terminal_reward=reward)
        for reward in config.terminal_rewards
    )
    _, advantages, returns = agent._joint_targets(trajectories)
    return {
        reward: (
            advantages[
                index * config.horizon : (index + 1) * config.horizon
            ],
            returns[index * config.horizon : (index + 1) * config.horizon],
        )
        for index, reward in enumerate(config.terminal_rewards)
    }


def _jormungandr_arm(
    config: DelayedCreditConfig,
    *,
    gae_lambda: float,
    targets: Mapping[float, tuple[np.ndarray, np.ndarray]] | None = None,
) -> Mapping[str, Any]:
    resolved = (
        targets
        if targets is not None
        else _jormungandr_targets(config, gae_lambda=gae_lambda)
    )
    return _summary(
        config,
        gae_lambda=gae_lambda,
        advantages_by_reward={key: value[0] for key, value in resolved.items()},
        returns_by_reward={key: value[1] for key, value in resolved.items()},
    )


def _sb3_targets(
    config: DelayedCreditConfig,
    *,
    gae_lambda: float,
    terminal_reward: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import gymnasium as gym
        from stable_baselines3.common.buffers import RolloutBuffer
    except ImportError as exc:  # pragma: no cover - exercised in reference env
        raise RuntimeError(
            "Stable-Baselines3 is required for the independent arm; run this "
            "script with benchmarks/requirements-sb3-reference.txt"
        ) from exc

    buffer = RolloutBuffer(
        config.horizon,
        gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        gym.spaces.Discrete(2),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=config.gamma,
        n_envs=1,
    )
    for timestep in range(config.horizon):
        buffer.add(
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            np.asarray(
                [
                    terminal_reward
                    if timestep == config.horizon - 1
                    else 0.0
                ],
                dtype=np.float32,
            ),
            np.asarray([timestep == 0], dtype=np.bool_),
            torch.zeros(1, dtype=torch.float32),
            torch.full((1,), -math.log(2.0), dtype=torch.float32),
        )
    buffer.compute_returns_and_advantage(
        last_values=torch.zeros(1, dtype=torch.float32),
        dones=np.asarray([True], dtype=np.bool_),
    )
    return buffer.advantages[:, 0].copy(), buffer.returns[:, 0].copy()


def _sb3_arm(
    config: DelayedCreditConfig,
    *,
    gae_lambda: float,
    targets: Mapping[float, tuple[np.ndarray, np.ndarray]] | None = None,
) -> Mapping[str, Any]:
    resolved = (
        targets
        if targets is not None
        else {
            reward: _sb3_targets(
                config, gae_lambda=gae_lambda, terminal_reward=reward
            )
            for reward in config.terminal_rewards
        }
    )
    return _summary(
        config,
        gae_lambda=gae_lambda,
        advantages_by_reward={key: value[0] for key, value in resolved.items()},
        returns_by_reward={key: value[1] for key, value in resolved.items()},
    )


def run_benchmark(
    config: DelayedCreditConfig = DelayedCreditConfig(),
    *,
    include_sb3: bool = True,
) -> Mapping[str, Any]:
    """Run the frozen estimator comparison and return an auditable report."""

    jormungandr_targets = {
        str(value): _jormungandr_targets(config, gae_lambda=value)
        for value in config.gae_lambdas
    }
    jormungandr_results = {
        str(value): _jormungandr_arm(
            config,
            gae_lambda=value,
            targets=jormungandr_targets[str(value)],
        )
        for value in config.gae_lambdas
    }
    sb3_targets = (
        {
            str(value): {
                reward: _sb3_targets(
                    config,
                    gae_lambda=value,
                    terminal_reward=reward,
                )
                for reward in config.terminal_rewards
            }
            for value in config.gae_lambdas
        }
        if include_sb3
        else None
    )
    sb3_results = (
        {
            str(value): _sb3_arm(
                config,
                gae_lambda=value,
                targets=sb3_targets[str(value)],
            )
            for value in config.gae_lambdas
        }
        if sb3_targets is not None
        else None
    )
    implementation_errors = []
    if sb3_targets is not None:
        for value in config.gae_lambdas:
            key = str(value)
            for reward in config.terminal_rewards:
                for jormungandr_vector, sb3_vector in zip(
                    jormungandr_targets[key][reward],
                    sb3_targets[key][reward],
                    strict=True,
                ):
                    implementation_errors.append(
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(jormungandr_vector)
                                    - np.asarray(sb3_vector)
                                )
                            )
                        )
                    )
    reference_implementations = (jormungandr_results,) + (
        (sb3_results,) if sb3_results is not None else ()
    )
    maximum_reference_error = max(
        max(
            float(arm[metric])
            for metric in (
                "maximum_advantage_absolute_error",
                "maximum_return_absolute_error",
            )
        )
        for implementation in reference_implementations
        for arm in implementation.values()
    )
    maximum_implementation_error = (
        max(implementation_errors) if implementation_errors else None
    )
    current = jormungandr_results[str(config.current_lambda)]
    candidate = jormungandr_results[str(config.candidate_lambda)]
    conditions = {
        "jormungandr_matches_closed_form": max(
            max(
                float(arm[metric])
                for metric in (
                    "maximum_advantage_absolute_error",
                    "maximum_return_absolute_error",
                )
            )
            for arm in jormungandr_results.values()
        )
        <= config.tolerance,
        "sb3_reference_present_and_matches_closed_form": sb3_results is not None
        and max(
            max(
                float(arm[metric])
                for metric in (
                    "maximum_advantage_absolute_error",
                    "maximum_return_absolute_error",
                )
            )
            for arm in sb3_results.values()
        )
        <= config.tolerance,
        "implementations_match": maximum_implementation_error is not None
        and maximum_implementation_error <= config.tolerance,
        "current_opening_weight_below_bound": float(
            current["analytic_opening_weight"]
        )
        < config.current_opening_upper_bound,
        "episodic_opening_weight_preserved": float(
            candidate["analytic_opening_weight"]
        )
        >= config.candidate_opening_lower_bound,
    }
    packages: dict[str, str] = {
        "jormungandr": jormungandr.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if include_sb3:
        import gymnasium
        import stable_baselines3

        packages.update(
            {
                "gymnasium": gymnasium.__version__,
                "stable_baselines3": stable_baselines3.__version__,
            }
        )
    return {
        "schema": "jormungandr.delayed_terminal_credit_benchmark.v1",
        "benchmark": "DelayedTerminalCredit-v0",
        "config": asdict(config),
        "implementations": {
            "jormungandr_structured_ppo": jormungandr_results,
            "stable_baselines3_rollout_buffer": sb3_results,
        },
        "maximum_reference_error": maximum_reference_error,
        "maximum_implementation_error": maximum_implementation_error,
        "conditions": conditions,
        "passed": all(conditions.values()),
        "decision": {
            "selected_gae_lambda": (
                config.candidate_lambda if all(conditions.values()) else None
            ),
            "scope": (
                "terminal-return propagation only; no policy learning or "
                "environment-performance claim"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
    }


def _parse_lambdas(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one lambda is required")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=719)
    parser.add_argument("--gae-lambdas", type=_parse_lambdas, default=(0.98, 1.0))
    parser.add_argument("--skip-sb3", action="store_true")
    parser.add_argument(
        "--json-output",
        default="docs/latex/figures/delayed_terminal_credit.json",
    )
    args = parser.parse_args(argv)
    result = run_benchmark(
        DelayedCreditConfig(
            horizon=args.horizon,
            gae_lambdas=args.gae_lambdas,
        ),
        include_sb3=not args.skip_sb3,
    )
    output = Path(args.json_output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
