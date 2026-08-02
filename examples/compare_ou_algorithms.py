"""Compare transition-based learner plugins on fixed synthetic OU cohorts.

This benchmark deliberately compares only online algorithms that consume the
same one-step transition contract.  DQN, C51, QR-DQN, maximum-entropy Soft-Q,
and categorical SAC see the same exogenous training paths and are evaluated on
the same held-out paths.  Trajectory, asynchronous, and offline algorithms
belong in separate benchmark cohorts with data appropriate to their objective.

The synthetic mean-reverting spread is a control illustration, not a trading
strategy or a claim about market performance.  The benchmark drives the real
Jörmungandr plugin implementations directly so gradient-update counts are
deterministic and comparable; the production runtime wraps the same plugins
with distributed transport, metrics, and checkpoint envelopes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import io
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

try:  # Package import under tests and ``python -m examples...``.
    from examples.train_ou_spread import (
        ACTION_LABELS,
        ACTION_VALUES,
        OBSERVATION_KEYS,
        OrnsteinUhlenbeckSpread,
    )
except ModuleNotFoundError:  # Direct ``python examples/compare_ou_algorithms.py``.
    from train_ou_spread import (  # type: ignore[no-redef]
        ACTION_LABELS,
        ACTION_VALUES,
        OBSERVATION_KEYS,
        OrnsteinUhlenbeckSpread,
    )
from jormungandr.algorithms import algorithm_registry, normalize_update_result
from jormungandr.core import PrioritizedReplayBuffer


DEFAULT_ALGORITHMS = ("dqn", "c51", "qrdqn", "maxent", "sac")
DISPLAY_NAMES = {
    "dqn": "DQN",
    "c51": "C51",
    "qrdqn": "QR-DQN",
    "maxent": "MaxEnt Soft-Q",
    "sac": "Categorical SAC",
}
COLORS = {
    "dqn": "#4B5563",
    "c51": "#B45309",
    "qrdqn": "#7C3AED",
    "maxent": "#0891B2",
    "sac": "#15803D",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS
    runs: int = 5
    train_episodes: int = 64
    eval_interval: int = 4
    eval_episodes: int = 16
    horizon: int = 32
    batch_size: int = 32
    warmup_transitions: int = 128
    replay_capacity: int = 8192
    hidden: int = 64
    sensor_noise_std: float = 0.15
    base_seed: int = 20260802

    def __post_init__(self) -> None:
        if not self.algorithms:
            raise ValueError("at least one algorithm is required")
        if self.runs < 1 or self.train_episodes < 1:
            raise ValueError("runs and train_episodes must be positive")
        if self.eval_interval < 1 or self.eval_episodes < 1:
            raise ValueError("evaluation counts must be positive")
        if self.horizon < 2 or self.batch_size < 2:
            raise ValueError("horizon and batch_size must be at least two")
        if self.warmup_transitions < self.batch_size:
            raise ValueError("warmup_transitions cannot be smaller than batch_size")
        if self.replay_capacity < self.warmup_transitions:
            raise ValueError("replay_capacity cannot be smaller than warmup_transitions")
        if self.hidden < 8:
            raise ValueError("hidden must be at least eight")
        if self.sensor_noise_std < 0.0:
            raise ValueError("sensor_noise_std cannot be negative")


def _plugin_config(config: BenchmarkConfig) -> dict[str, Any]:
    """Common small-network profile used by every benchmark participant."""

    return {
        "action_values": list(ACTION_VALUES),
        "hidden": int(config.hidden),
        "lr": 3e-4,
        "gamma": 0.97,
        "target_update": 96,
        "max_grad": 1.0,
        "v_min": -12.0,
        "v_max": 12.0,
        "atoms": 51,
        "quantiles": 25,
        "temperature": 0.20,
        "auto_entropy": True,
        "tau": 0.02,
        "aux_enabled": False,
        "observation_noise_std": 0.0,
        "reward_clip": 0.0,
    }


def _evaluation_seeds(config: BenchmarkConfig) -> list[int]:
    return [
        int(config.base_seed + 50_000_003 + 10_007 * index)
        for index in range(config.eval_episodes)
    ]


def _training_seed(config: BenchmarkConfig, run: int, episode: int) -> int:
    return int(config.base_seed + run * 1_000_003 + episode * 10_007)


def _policy_observation(
    observation: Sequence[float],
    *,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    value = np.asarray(observation, dtype=np.float32).copy()
    if noise_std > 0.0:
        # Spread and change are sensors.  Position and remaining time are
        # controlled state and remain exact in this perturbation cohort.
        value[:2] += rng.normal(0.0, noise_std, size=2).astype(np.float32)
    return value


def evaluate_agent(
    agent: Any,
    *,
    seeds: Sequence[int],
    horizon: int,
    sensor_noise_std: float,
    capture_seed: int | None = None,
) -> dict[str, Any]:
    returns: list[float] = []
    reference_returns: list[float] = []
    captured: dict[str, Any] | None = None

    for path_seed in seeds:
        environment = OrnsteinUhlenbeckSpread(horizon)
        observation = environment.reset(int(path_seed))
        noise_rng = np.random.default_rng(int(path_seed) + 70_000_019)
        rewards: list[float] = []
        reference_rewards: list[float] = []
        spread_z: list[float] = []
        actions: list[int] = []

        for _ in range(horizon):
            policy_obs = _policy_observation(
                observation,
                noise_std=sensor_noise_std,
                rng=noise_rng,
            )
            decision = agent.action_result(
                policy_obs,
                deterministic=True,
                epsilon=0.0,
            )
            spread_z.append(float(observation[0]))
            actions.append(int(decision.action_idx))
            step = environment.step(float(decision.action))
            rewards.append(float(step.reward))
            reference_rewards.append(float(step.reference_reward))
            observation = step.observation
            if step.terminal:
                break

        returns.append(float(sum(rewards)))
        reference_returns.append(float(sum(reference_rewards)))
        if capture_seed is not None and int(path_seed) == int(capture_seed):
            captured = {
                "path_seed": int(path_seed),
                "spread_z": spread_z,
                "action_indices": actions,
                "rewards": rewards,
                "cumulative_reward": np.cumsum(rewards).astype(float).tolist(),
                "reference_rewards": reference_rewards,
                "reference_cumulative_reward": np.cumsum(reference_rewards)
                .astype(float)
                .tolist(),
            }

    result: dict[str, Any] = {
        "return_mean": float(statistics.fmean(returns)),
        "return_std": float(statistics.stdev(returns)) if len(returns) > 1 else 0.0,
        "reference_return_mean": float(statistics.fmean(reference_returns)),
        "advantage_mean": float(
            statistics.fmean(
                value - reference
                for value, reference in zip(returns, reference_returns)
            )
        ),
        "episode_returns": returns,
    }
    if captured is not None:
        result["playback"] = captured
    return result


def _checkpoint_episodes(config: BenchmarkConfig) -> set[int]:
    values = set(range(0, config.train_episodes + 1, config.eval_interval))
    values.add(config.train_episodes)
    return values


def train_run(
    algorithm: str,
    *,
    run: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    seed = int(config.base_seed + run * 1_000_003)
    np.random.seed(seed)
    torch.manual_seed(seed)
    plugin = algorithm_registry.get(algorithm)
    if plugin.replay_mode != "transition" or algorithm in {"bc", "cql"}:
        raise ValueError(
            f"{algorithm} is not an online transition-control benchmark participant"
        )
    agent = plugin.build(
        len(OBSERVATION_KEYS),
        _plugin_config(config),
        "cpu",
    )
    replay = PrioritizedReplayBuffer(
        config.replay_capacity,
        len(OBSERVATION_KEYS),
        alpha=0.6,
    )
    evaluation_seeds = _evaluation_seeds(config)
    capture_seed = evaluation_seeds[0]
    checkpoints: list[dict[str, Any]] = []
    updates = 0
    transitions = 0
    last_loss = float("nan")

    def evaluate_checkpoint(episode: int, *, capture: bool = False) -> None:
        clean = evaluate_agent(
            agent,
            seeds=evaluation_seeds,
            horizon=config.horizon,
            sensor_noise_std=0.0,
            capture_seed=capture_seed if capture else None,
        )
        perturbed = evaluate_agent(
            agent,
            seeds=evaluation_seeds,
            horizon=config.horizon,
            sensor_noise_std=config.sensor_noise_std,
        )
        row: dict[str, Any] = {
            "episode": int(episode),
            "transitions": int(transitions),
            "updates": int(updates),
            "last_loss": None if not np.isfinite(last_loss) else float(last_loss),
            "clean": clean,
            "sensor_noise": perturbed,
        }
        checkpoints.append(row)

    evaluate_checkpoint(0)
    checkpoint_episodes = _checkpoint_episodes(config)
    for episode in range(1, config.train_episodes + 1):
        environment = OrnsteinUhlenbeckSpread(config.horizon)
        observation = environment.reset(_training_seed(config, run, episode))
        fraction = (episode - 1) / max(1, config.train_episodes - 1)
        epsilon = 0.30 + fraction * (0.05 - 0.30)

        for _ in range(config.horizon):
            decision = agent.action_result(
                np.asarray(observation, dtype=np.float32),
                deterministic=False,
                epsilon=float(epsilon),
            )
            step = environment.step(float(decision.action))
            replay.add(
                observation,
                int(decision.action_idx),
                float(step.reward),
                step.observation,
                float(step.terminal),
            )
            observation = step.observation
            transitions += 1

            if len(replay) >= config.warmup_transitions:
                progress = transitions / float(config.train_episodes * config.horizon)
                beta = min(1.0, 0.4 + 0.6 * progress)
                batch, indices, weights = replay.sample(config.batch_size, beta)
                result = normalize_update_result(agent.update(batch, weights))
                replay.update_priorities(indices, result.priorities)
                last_loss = float(result.loss)
                updates += 1
            if step.terminal:
                break

        if episode in checkpoint_episodes:
            evaluate_checkpoint(
                episode,
                capture=episode == config.train_episodes,
            )

    # Exercise the stable plugin state contract without retaining large model
    # payloads in the benchmark artifact.
    state = agent.state_dict()
    restored = plugin.build(
        len(OBSERVATION_KEYS),
        _plugin_config(config),
        "cpu",
    )
    restored.load_state_dict(state)
    probe = np.asarray([0.8, 0.0, 0.0, 0.5], dtype=np.float32)
    original_action = agent.action_result(
        probe, deterministic=True, epsilon=0.0
    ).action_idx
    restored_action = restored.action_result(
        probe, deterministic=True, epsilon=0.0
    ).action_idx
    if int(original_action) != int(restored_action):
        raise RuntimeError(f"{algorithm} state round-trip changed deterministic policy")

    return {
        "algorithm": algorithm,
        "plugin_id": plugin.checkpoint_id,
        "family": plugin.family,
        "run": int(run),
        "seed": seed,
        "updates": int(updates),
        "transitions": int(transitions),
        "checkpoint_roundtrip": True,
        "checkpoints": checkpoints,
    }


def _normal_interval(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    half_width = 1.96 * float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean - half_width, mean + half_width


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for run in runs:
        for checkpoint in run["checkpoints"]:
            grouped.setdefault(
                (str(run["algorithm"]), int(checkpoint["episode"])), []
            ).append(checkpoint)

    summaries: list[dict[str, Any]] = []
    for (algorithm, episode), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "algorithm": algorithm,
            "episode": int(episode),
            "runs": len(rows),
            "transitions": int(round(statistics.fmean(row["transitions"] for row in rows))),
            "updates": int(round(statistics.fmean(row["updates"] for row in rows))),
        }
        for cohort in ("clean", "sensor_noise"):
            values = np.asarray(
                [row[cohort]["return_mean"] for row in rows], dtype=np.float64
            )
            low, high = _normal_interval(values)
            advantages = np.asarray(
                [row[cohort]["advantage_mean"] for row in rows],
                dtype=np.float64,
            )
            item[cohort] = {
                "return_mean": float(values.mean()),
                "return_95ci": [low, high],
                "advantage_mean": float(advantages.mean()),
                "reference_return_mean": float(
                    statistics.fmean(
                        row[cohort]["reference_return_mean"] for row in rows
                    )
                ),
            }
        summaries.append(item)
    return summaries


def _median_playbacks(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    algorithms = sorted({str(run["algorithm"]) for run in runs})
    for algorithm in algorithms:
        candidates = [run for run in runs if run["algorithm"] == algorithm]
        final_values = [
            float(run["checkpoints"][-1]["clean"]["return_mean"])
            for run in candidates
        ]
        median = float(statistics.median(final_values))
        chosen = min(
            candidates,
            key=lambda run: (
                abs(float(run["checkpoints"][-1]["clean"]["return_mean"]) - median),
                int(run["run"]),
            ),
        )
        final_clean = chosen["checkpoints"][-1]["clean"]
        selected[algorithm] = {
            "run": int(chosen["run"]),
            "seed": int(chosen["seed"]),
            "selection": "closest run to median final clean return",
            **dict(final_clean["playback"]),
        }
    return selected


def run_benchmark(config: BenchmarkConfig, *, quiet: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    available = set(algorithm_registry.names())
    unknown = [name for name in config.algorithms if name not in available]
    if unknown:
        raise ValueError(f"unknown algorithms: {', '.join(unknown)}")
    torch.set_num_threads(1)
    runs: list[dict[str, Any]] = []
    for algorithm in config.algorithms:
        for run in range(config.runs):
            if not quiet:
                print(
                    f"training {DISPLAY_NAMES.get(algorithm, algorithm)} "
                    f"run {run + 1}/{config.runs}",
                    flush=True,
                )
            runs.append(train_run(algorithm, run=run, config=config))
            gc.collect()
    return {
        "format": "jormungandr.ou_algorithm_benchmark.v1",
        "benchmark": "synthetic_ou_spread_online_transition_control",
        "scope": {
            "included": (
                "online plugins using the same one-step transition replay contract"
            ),
            "excluded": {
                "ppo, appo, impala": "require fresh behavior-policy trajectories",
                "bc, marwil, cql": "require an offline demonstration or behavior dataset",
                "dreamerv3": "requires a separate sequence/model-based evaluation budget",
            },
            "claim": (
                "controlled library benchmark; not an environment-general ranking "
                "or a trading-performance claim"
            ),
        },
        "config": asdict(config),
        "environment": {
            "name": "synthetic_ou_spread.v1",
            "observation_keys": list(OBSERVATION_KEYS),
            "action_values": list(ACTION_VALUES),
            "action_labels": list(ACTION_LABELS),
            "held_out_path_seeds": _evaluation_seeds(config),
            "sensor_noise": (
                "independent Gaussian noise on spread_z and spread_change_z only"
            ),
        },
        "runs": runs,
        "summaries": summarize_runs(runs),
        "playback": _median_playbacks(runs),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _plot_rows(
    report: Mapping[str, Any], algorithm: str
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            row
            for row in report["summaries"]
            if row["algorithm"] == algorithm
        ),
        key=lambda row: row["episode"],
    )


def _draw_convergence(
    report: Mapping[str, Any],
    *,
    upto_episode: int | None = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional documentation path
        raise RuntimeError("plotting requires the optional matplotlib dependency") from exc

    config = report["config"]
    figure, axes = plt.subplots(1, 2, figsize=(9.1, 3.8), sharey=True)
    cohorts = (
        ("clean", "Clean held-out paths"),
        (
            "sensor_noise",
            f"Sensor noise $\\sigma={config['sensor_noise_std']:.2f}$",
        ),
    )
    for axis, (cohort, title) in zip(axes, cohorts):
        for algorithm in config["algorithms"]:
            rows = _plot_rows(report, algorithm)
            if upto_episode is not None:
                rows = [row for row in rows if row["episode"] <= upto_episode]
            if not rows:
                continue
            x = np.asarray([row["episode"] for row in rows], dtype=np.float64)
            y = np.asarray(
                [row[cohort]["return_mean"] for row in rows], dtype=np.float64
            )
            interval = np.asarray(
                [row[cohort]["return_95ci"] for row in rows], dtype=np.float64
            )
            color = COLORS.get(algorithm, "#111827")
            axis.plot(
                x,
                y,
                label=DISPLAY_NAMES.get(algorithm, algorithm),
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3.0,
            )
            if len(rows) > 1 or int(config["runs"]) > 1:
                axis.fill_between(
                    x,
                    interval[:, 0],
                    interval[:, 1],
                    color=color,
                    alpha=0.10,
                    linewidth=0.0,
                )
        reference_rows = _plot_rows(report, config["algorithms"][0])
        reference = float(reference_rows[0][cohort]["reference_return_mean"])
        axis.axhline(
            reference,
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label="Rule reference" if cohort == "clean" else None,
        )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Training episodes")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel("Mean held-out episode return")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )
    shown = config["train_episodes"] if upto_episode is None else upto_episode
    figure.suptitle(
        f"Jörmungandr online transition agents · through episode {shown}",
        fontsize=11,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.925,
        (
            f"{config['runs']} training seeds · {config['eval_episodes']} fixed "
            "held-out paths · shaded 95% intervals across training runs"
        ),
        ha="center",
        va="center",
        fontsize=7.5,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.90), w_pad=1.4)
    return figure


def render_convergence_plot(report: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure = _draw_convergence(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Jörmungandr OU algorithm convergence",
            "Creator": "Jörmungandr reproducible benchmark",
        },
    )
    plt.close(figure)


def _figure_image(figure, *, dpi: int = 110):
    from PIL import Image

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def render_convergence_gif(report: Mapping[str, Any], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional documentation path
        raise RuntimeError("GIF rendering requires matplotlib and Pillow") from exc

    checkpoints = sorted(
        {int(row["episode"]) for row in report["summaries"]}
    )
    frames: list[Image.Image] = []
    for episode in checkpoints:
        figure = _draw_convergence(report, upto_episode=episode)
        frames.append(_figure_image(figure))
        plt.close(figure)
    frames.extend([frames[-1].copy()] * 7)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=260,
        loop=0,
        disposal=1,
        optimize=True,
    )


def _draw_playback(report: Mapping[str, Any], upto_step: int):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError as exc:  # pragma: no cover - optional documentation path
        raise RuntimeError("playback rendering requires matplotlib") from exc

    algorithms = list(report["config"]["algorithms"])
    traces = report["playback"]
    horizon = int(report["config"]["horizon"])
    steps = np.arange(horizon)
    representative = traces[algorithms[0]]
    spread = np.asarray(representative["spread_z"], dtype=np.float64)
    action_grid = np.full((len(algorithms), horizon), np.nan, dtype=np.float64)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.8, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.25, 1.75]},
    )

    visible = min(horizon, upto_step + 1)
    axes[0].plot(steps[:visible], spread[:visible], color="#2458A6", linewidth=1.8)
    axes[0].axhline(0.65, color="#9CA3AF", linestyle="--", linewidth=0.8)
    axes[0].axhline(-0.65, color="#9CA3AF", linestyle="--", linewidth=0.8)
    axes[0].axhline(0.0, color="#D1D5DB", linewidth=0.6)
    axes[0].set_ylabel("Spread z")
    axes[0].set_title(
        "Same held-out OU path: final median-run policies",
        fontsize=10.5,
        fontweight="semibold",
    )

    for row, algorithm in enumerate(algorithms):
        values = np.asarray(traces[algorithm]["action_indices"], dtype=np.float64)
        action_grid[row, :visible] = values[:visible]
    cmap = ListedColormap(["#B45309", "#E5E7EB", "#15803D"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    axes[1].imshow(
        np.ma.masked_invalid(action_grid),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(-0.5, horizon - 0.5, len(algorithms) - 0.5, -0.5),
    )
    axes[1].set_yticks(np.arange(len(algorithms)))
    axes[1].set_yticklabels(
        [DISPLAY_NAMES.get(name, name) for name in algorithms], fontsize=8
    )
    axes[1].set_ylabel("Action")
    axes[1].text(
        1.0,
        1.04,
        "SHORT     FLAT     LONG",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#4B5563",
    )

    for algorithm in algorithms:
        cumulative = np.asarray(
            traces[algorithm]["cumulative_reward"], dtype=np.float64
        )
        axes[2].plot(
            steps[:visible],
            cumulative[:visible],
            color=COLORS.get(algorithm, "#111827"),
            label=DISPLAY_NAMES.get(algorithm, algorithm),
            linewidth=1.7,
        )
    reference = np.asarray(
        representative["reference_cumulative_reward"], dtype=np.float64
    )
    axes[2].plot(
        steps[:visible],
        reference[:visible],
        color="#111827",
        linestyle="--",
        linewidth=1.1,
        label="Rule reference",
    )
    axes[2].set_ylabel("Cumulative return")
    axes[2].set_xlabel("Environment step")
    axes[2].legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=7.5,
    )

    for axis in (axes[0], axes[2]):
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.8)
    for axis in axes:
        axis.set_xlim(-0.5, horizon - 0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
    figure.text(
        0.99,
        0.01,
        f"step {visible}/{horizon} · path seed {representative['path_seed']}",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 1.0), h_pad=0.8)
    return figure


def render_policy_playback_gif(report: Mapping[str, Any], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional documentation path
        raise RuntimeError("GIF rendering requires matplotlib and Pillow") from exc

    horizon = int(report["config"]["horizon"])
    frames: list[Image.Image] = []
    for step in range(horizon):
        figure = _draw_playback(report, step)
        frames.append(_figure_image(figure, dpi=105))
        plt.close(figure)
    frames.extend([frames[-1].copy()] * 9)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
        disposal=1,
        optimize=True,
    )


def _parse_algorithms(value: str) -> tuple[str, ...]:
    algorithms = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not algorithms:
        raise argparse.ArgumentTypeError("algorithms must not be empty")
    return algorithms


def _print_summary(report: Mapping[str, Any]) -> None:
    final_episode = int(report["config"]["train_episodes"])
    print("algorithm clean_return noisy_return clean_advantage updates")
    for algorithm in report["config"]["algorithms"]:
        row = next(
            item
            for item in report["summaries"]
            if item["algorithm"] == algorithm and item["episode"] == final_episode
        )
        print(
            f"{algorithm:>9} "
            f"{row['clean']['return_mean']:>12.4f} "
            f"{row['sensor_noise']['return_mean']:>12.4f} "
            f"{row['clean']['advantage_mean']:>15.4f} "
            f"{row['updates']:>7d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms", type=_parse_algorithms, default=DEFAULT_ALGORITHMS
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-transitions", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--sensor-noise-std", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--convergence-gif-output", type=Path)
    parser.add_argument("--playback-gif-output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = BenchmarkConfig(
        algorithms=tuple(args.algorithms),
        runs=args.runs,
        train_episodes=args.train_episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        horizon=args.horizon,
        batch_size=args.batch_size,
        warmup_transitions=args.warmup_transitions,
        replay_capacity=args.replay_capacity,
        hidden=args.hidden,
        sensor_noise_std=args.sensor_noise_std,
        base_seed=args.seed,
    )
    report = run_benchmark(config, quiet=args.quiet)
    _print_summary(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.plot_output is not None:
        render_convergence_plot(report, args.plot_output)
    if args.convergence_gif_output is not None:
        render_convergence_gif(report, args.convergence_gif_output)
    if args.playback_gif_output is not None:
        render_policy_playback_gif(report, args.playback_gif_output)


if __name__ == "__main__":
    main()
