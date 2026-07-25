"""Train multiple actors on a synthetic mean-reverting spread.

This is a financial research illustration, not a market-data example or a
trading recommendation.  The actors communicate with Jormungandr exclusively
through its public HTTP API.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import statistics
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import torch

from jormungandr.service import JormungandrHttpServer, JormungandrRuntime


EXPERIENCE_SCHEMA = "jormungandr.experience.v1"
AUX_UPDATE_SCHEMA = "jormungandr.aux_update.v1"
OBSERVATION_KEYS = ["spread_z", "spread_change_z", "position", "remaining_fraction"]
ACTION_VALUES = [-1.0, 0.0, 1.0]
ACTION_LABELS = ["SHORT", "FLAT", "LONG"]
AUX_LABELS = ["DOWN", "FLAT", "UP"]
SPARKS = "▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class EpisodeJob:
    ordinal: int
    split: str
    seed: int


@dataclass(frozen=True)
class SpreadStep:
    observation: list[float]
    reward: float
    reference_reward: float
    terminal: bool
    aux_label: int
    spread_z: float
    position: float
    turnover: float


@dataclass(frozen=True)
class EpisodeResult:
    ordinal: int
    split: str
    actor_id: str
    episode_id: str
    reward_sum: float
    reference_reward_sum: float
    max_drawdown: float
    turnover: float
    mean_abs_z: float
    short_steps: int
    flat_steps: int
    long_steps: int
    steps: int
    first_policy_version: int
    last_policy_version: int

    @property
    def advantage(self) -> float:
        return self.reward_sum - self.reference_reward_sum


@dataclass(frozen=True)
class ActorSnapshot:
    actor_id: str
    ordinal: int
    split: str
    timestep: int
    horizon: int
    spread_z: float
    action_idx: int
    reward_sum: float
    policy_version: int


class SharedActorState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, ActorSnapshot] = {}

    def update(self, snapshot: ActorSnapshot) -> None:
        with self._lock:
            self._active[snapshot.actor_id] = snapshot

    def finish(self, actor_id: str) -> None:
        with self._lock:
            self._active.pop(actor_id, None)

    def snapshot(self) -> list[ActorSnapshot]:
        with self._lock:
            return sorted(self._active.values(), key=lambda item: item.actor_id)


class OrnsteinUhlenbeckSpread:
    """A discrete OU process with an explicit convergence benchmark."""

    observation_dim = len(OBSERVATION_KEYS)
    action_values = ACTION_VALUES

    def __init__(
        self,
        horizon: int,
        *,
        kappa: float = 0.18,
        sigma: float = 0.35,
        transaction_cost: float = 0.035,
        risk_penalty: float = 0.004,
        reference_threshold: float = 0.65,
    ) -> None:
        self.horizon = max(2, int(horizon))
        self.kappa = float(kappa)
        self.sigma = float(sigma)
        self.transaction_cost = float(transaction_cost)
        self.risk_penalty = float(risk_penalty)
        self.reference_threshold = float(reference_threshold)
        persistence = 1.0 - self.kappa
        self.stationary_std = self.sigma / math.sqrt(1.0 - persistence * persistence)
        self.rng = np.random.default_rng(0)
        self.timestep = 0
        self.spread = 0.0
        self.last_change = 0.0
        self.position = 0.0
        self.reference_position = 0.0

    def _observation(self) -> list[float]:
        return [
            float(self.spread / self.stationary_std),
            float(self.last_change / self.stationary_std),
            float(self.position),
            float(max(0, self.horizon - self.timestep) / self.horizon),
        ]

    def reset(self, seed: int) -> list[float]:
        self.rng = np.random.default_rng(int(seed))
        self.timestep = 0
        self.spread = float(
            np.clip(
                self.rng.normal(0.0, self.stationary_std),
                -2.8 * self.stationary_std,
                2.8 * self.stationary_std,
            )
        )
        self.last_change = 0.0
        self.position = 0.0
        self.reference_position = 0.0
        return self._observation()

    def _reference_action(self, spread_z: float) -> float:
        if spread_z > self.reference_threshold:
            return -1.0
        if spread_z < -self.reference_threshold:
            return 1.0
        return 0.0

    def step(self, action: float) -> SpreadStep:
        action = float(action)
        if action not in self.action_values:
            raise ValueError(f"unsupported action value: {action}")

        spread_z = self.spread / self.stationary_std
        reference_action = self._reference_action(spread_z)
        change = float(-self.kappa * self.spread + self.sigma * self.rng.normal())
        next_spread = self.spread + change
        pnl_units = action * change / self.stationary_std
        reference_pnl_units = reference_action * change / self.stationary_std
        turnover = abs(action - self.position)
        reference_turnover = abs(reference_action - self.reference_position)
        reward = (
            pnl_units
            - self.transaction_cost * turnover
            - self.risk_penalty * abs(action)
        )
        reference_reward = (
            reference_pnl_units
            - self.transaction_cost * reference_turnover
            - self.risk_penalty * abs(reference_action)
        )

        if spread_z > 0.35:
            aux_label = 0
        elif spread_z < -0.35:
            aux_label = 2
        else:
            aux_label = 1

        self.position = action
        self.reference_position = reference_action
        self.last_change = change
        self.spread = next_spread
        self.timestep += 1
        return SpreadStep(
            observation=self._observation(),
            reward=float(reward),
            reference_reward=float(reference_reward),
            terminal=self.timestep >= self.horizon,
            aux_label=aux_label,
            spread_z=float(self.spread / self.stationary_std),
            position=action,
            turnover=float(turnover),
        )


class JsonClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30.0) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} for {path}: {detail}") from exc
        if not result.get("ok", False):
            raise RuntimeError(str(result.get("error", "request failed")))
        return result

    def get(self, path: str) -> dict[str, Any]:
        try:
            with urlopen(f"{self.base_url}{path}", timeout=30.0) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} for {path}: {detail}") from exc
        if not result.get("ok", False):
            raise RuntimeError(str(result.get("error", "request failed")))
        return result


def interleaved_jobs(
    train_episodes: int,
    validation_episodes: int,
    seed: int,
) -> list[EpisodeJob]:
    """Place held-out episodes throughout the training schedule."""
    if train_episodes < 1 or validation_episodes < 1:
        raise ValueError("train and validation episode counts must be positive")
    order: list[str] = []
    validation_added = 0
    for train_added in range(1, train_episodes + 1):
        order.append("train")
        validation_target = math.floor(
            train_added * validation_episodes / train_episodes
        )
        while validation_added < validation_target:
            order.append("validation")
            validation_added += 1
    while validation_added < validation_episodes:
        order.append("validation")
        validation_added += 1
    return [
        EpisodeJob(
            ordinal=ordinal,
            split=split,
            # Disjoint offsets keep validation paths distinct without relying on
            # actor completion order.
            seed=int(seed + 10_007 * ordinal + (1_000_003 if split == "validation" else 0)),
        )
        for ordinal, split in enumerate(order)
    ]


def run_episode(
    *,
    client: JsonClient,
    model_id: str,
    actor_id: str,
    job: EpisodeJob,
    horizon: int,
    epsilon: float,
    shared_state: SharedActorState,
) -> EpisodeResult:
    environment = OrnsteinUhlenbeckSpread(horizon)
    observation = environment.reset(job.seed)
    episode_id = f"ou-spread-{job.ordinal:05d}"
    experiences: list[dict[str, Any]] = []
    aux_updates: list[dict[str, Any]] = []
    rewards: list[float] = []
    reference_rewards: list[float] = []
    absolute_z: list[float] = []
    policy_versions: list[int] = []
    action_counts: Counter[int] = Counter()
    turnover = 0.0
    equity = 0.0
    equity_peak = 0.0
    max_drawdown = 0.0

    for timestep in range(horizon):
        inference = client.post(
            f"/v1/models/{model_id}/policy/infer",
            {
                "obs": observation,
                "deterministic": job.split == "validation",
                "epsilon": 0.0 if job.split == "validation" else epsilon,
            },
        )
        action = float(inference["action"])
        action_idx = int(inference["action_idx"])
        policy_version = int(inference["policy_version"])
        step = environment.step(action)

        experiences.append(
            {
                "split": job.split,
                "actor_id": actor_id,
                "episode_id": episode_id,
                "timestep": timestep,
                "policy_version": policy_version,
                "obs": observation,
                "action_idx": action_idx,
                "reward": step.reward,
                "next_obs": step.observation,
                "done": step.terminal,
                "meta": {
                    "environment": "synthetic_ou_spread.v1",
                    "path_seed": job.seed,
                },
            }
        )
        aux_updates.append(
            {
                "split": job.split,
                "actor_id": actor_id,
                "episode_id": episode_id,
                "timestep": timestep,
                "aux": {"kind": "ou_drift_direction", "label": step.aux_label},
            }
        )
        rewards.append(step.reward)
        reference_rewards.append(step.reference_reward)
        absolute_z.append(abs(float(observation[0])))
        policy_versions.append(policy_version)
        action_counts[action_idx] += 1
        turnover += step.turnover
        equity += step.reward
        equity_peak = max(equity_peak, equity)
        max_drawdown = max(max_drawdown, equity_peak - equity)
        shared_state.update(
            ActorSnapshot(
                actor_id=actor_id,
                ordinal=job.ordinal,
                split=job.split,
                timestep=timestep + 1,
                horizon=horizon,
                spread_z=step.spread_z,
                action_idx=action_idx,
                reward_sum=float(sum(rewards)),
                policy_version=policy_version,
            )
        )
        observation = step.observation
        if step.terminal:
            break

    client.post(
        f"/v1/models/{model_id}/experience/add",
        {"schema": EXPERIENCE_SCHEMA, "items": experiences},
    )
    # Submit labels separately so the example exercises the delayed aux join.
    client.post(
        f"/v1/models/{model_id}/experience/aux_update",
        {"schema": AUX_UPDATE_SCHEMA, "updates": aux_updates},
    )
    client.post(
        f"/v1/models/{model_id}/metrics",
        {
            "step": job.ordinal,
            "metrics": {
                f"sampler/{job.split}/episode_reward": sum(rewards),
                f"sampler/{job.split}/reference_reward": sum(reference_rewards),
                f"sampler/{job.split}/max_drawdown": max_drawdown,
                f"sampler/{job.split}/turnover": turnover,
            },
        },
    )
    return EpisodeResult(
        ordinal=job.ordinal,
        split=job.split,
        actor_id=actor_id,
        episode_id=episode_id,
        reward_sum=float(sum(rewards)),
        reference_reward_sum=float(sum(reference_rewards)),
        max_drawdown=float(max_drawdown),
        turnover=float(turnover),
        mean_abs_z=float(statistics.fmean(absolute_z)),
        short_steps=int(action_counts[0]),
        flat_steps=int(action_counts[1]),
        long_steps=int(action_counts[2]),
        steps=len(rewards),
        first_policy_version=min(policy_versions),
        last_policy_version=max(policy_versions),
    )


def actor_loop(
    *,
    actor_number: int,
    jobs: queue.Queue[EpisodeJob | None],
    results: queue.Queue[EpisodeResult | BaseException],
    client: JsonClient,
    model_id: str,
    horizon: int,
    epsilon: float,
    pause_ms: int,
    shared_state: SharedActorState,
) -> None:
    actor_id = f"actor-{actor_number:02d}"
    while True:
        job = jobs.get()
        try:
            if job is None:
                return
            shared_state.update(
                ActorSnapshot(
                    actor_id=actor_id,
                    ordinal=job.ordinal,
                    split=job.split,
                    timestep=0,
                    horizon=horizon,
                    spread_z=0.0,
                    action_idx=1,
                    reward_sum=0.0,
                    policy_version=0,
                )
            )
            result = run_episode(
                client=client,
                model_id=model_id,
                actor_id=actor_id,
                job=job,
                horizon=horizon,
                epsilon=epsilon,
                shared_state=shared_state,
            )
            results.put(result)
            if pause_ms > 0:
                time.sleep(pause_ms / 1000.0)
        except BaseException as exc:
            results.put(exc)
        finally:
            shared_state.finish(actor_id)
            jobs.task_done()


def _mean(rows: list[EpisodeResult], name: str) -> float:
    return (
        float(statistics.fmean(float(getattr(row, name)) for row in rows))
        if rows
        else 0.0
    )


def _sparkline(values: list[float], width: int = 18) -> str:
    values = values[-width:]
    if not values:
        return "·" * width
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        chars = SPARKS[len(SPARKS) // 2] * len(values)
    else:
        chars = "".join(
            SPARKS[min(len(SPARKS) - 1, int((value - lo) / (hi - lo) * len(SPARKS)))]
            for value in values
        )
    return ("·" * (width - len(chars))) + chars


def _policy_map(client: JsonClient, model_id: str) -> dict[str, Any]:
    z_grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
    response = client.post(
        f"/v1/models/{model_id}/policy/infer",
        {
            "obs_batch": [[z, 0.0, 0.0, 0.5] for z in z_grid],
            "deterministic": True,
        },
    )
    return {
        "z_grid": z_grid,
        "items": response["items"],
        "policy_version": int(response["policy_version"]),
    }


def _color(enabled: bool, code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if enabled else text


def render_dashboard(
    *,
    args: argparse.Namespace,
    jobs: list[EpisodeJob],
    results: list[EpisodeResult],
    active: list[ActorSnapshot],
    model: dict[str, Any],
    policy_map: dict[str, Any],
    final: bool,
) -> str:
    color = not args.no_color
    bold = lambda text: _color(color, "1", text)
    dim = lambda text: _color(color, "2", text)
    cyan = lambda text: _color(color, "38;5;44", text)
    green = lambda text: _color(color, "38;5;42", text)
    yellow = lambda text: _color(color, "38;5;220", text)
    red = lambda text: _color(color, "38;5;203", text)
    blue = lambda text: _color(color, "38;5;75", text)

    by_ordinal = {row.ordinal: row for row in results}
    active_ordinals = {row.ordinal for row in active}
    schedule = "".join(
        (
            green("T")
            if job.ordinal in by_ordinal and job.split == "train"
            else blue("V")
            if job.ordinal in by_ordinal
            else yellow("t")
            if job.ordinal in active_ordinals and job.split == "train"
            else yellow("v")
            if job.ordinal in active_ordinals
            else dim("·")
        )
        for job in jobs
    )
    learner = model["learner"]
    replay = model["replay"]
    validation = model["validation"]
    train_rows = [row for row in results if row.split == "train"]
    validation_rows = [row for row in results if row.split == "validation"]

    lines = [
        bold("JORMUNGANDR  /  DISTRIBUTED OU SPREAD LEARNER"),
        dim("synthetic research path · C51 + auxiliary drift head · no market data"),
        "",
        f"progress  {len(results):>3}/{len(jobs):<3} episodes   "
        f"actors {args.actors}   horizon {args.horizon}   "
        f"status {green('COMPLETE') if final else yellow('LEARNING')}",
        f"splits    {schedule}",
        "",
        bold("ACTOR DATA PLANE"),
    ]
    if active:
        for row in active[: max(2, args.actors)]:
            split = green(f"{'TRAIN':<5}") if row.split == "train" else blue(f"{'VALID':<5}")
            action = ACTION_LABELS[row.action_idx]
            lines.append(
                f"{row.actor_id:<9} {split}  ep {row.ordinal:02d}  "
                f"step {row.timestep:02d}/{row.horizon:<2}  z {row.spread_z:+5.2f}  "
                f"{action:<5}  reward {row.reward_sum:+6.2f}  v{row.policy_version}"
            )
    else:
        lines.append(dim("actors idle; all scheduled paths have reported"))

    lines.extend(
        [
            "",
            bold("CENTRAL LEARNER"),
            f"train replay       {replay['size']:>5}/{replay['capacity']:<5}    "
            f"C51 updates       {learner['updates']:>5}    "
            f"policy version    {learner['policy_version']:>5}",
            f"validation store  {validation['size']:>5}/{validation['capacity']:<5}    "
            f"validation runs   {learner['validation_runs']:>5}    "
            f"normalizer count  {model['obs_normalizer']['count']:>5}",
            f"train loss        {learner['last_loss']:>8.4f}          "
            f"aux accuracy      {100.0 * learner['last_aux_acc']:>6.1f}%    "
            f"held-out aux      {100.0 * learner['last_validation_aux_acc']:>6.1f}%",
            "",
            bold("EPISODE STATISTICS  (reward units; transaction costs and position penalty included)"),
            "split          n    policy    reference   advantage    max DD   turnover   positive   recent",
        ]
    )

    def add_stats(label: str, rows: list[EpisodeResult], shade: str) -> None:
        positive = (
            100.0 * sum(row.reward_sum > 0.0 for row in rows) / len(rows)
            if rows
            else 0.0
        )
        raw = (
            f"{label:<12} {len(rows):>3}  {_mean(rows, 'reward_sum'):>+8.3f}  "
            f"{_mean(rows, 'reference_reward_sum'):>+10.3f}  "
            f"{_mean(rows, 'advantage'):>+10.3f}  "
            f"{_mean(rows, 'max_drawdown'):>8.3f}  "
            f"{_mean(rows, 'turnover'):>9.2f}  {positive:>7.1f}%   "
            f"{_sparkline([row.reward_sum for row in rows])}"
        )
        lines.append(_color(color, shade, raw))

    add_stats("train/explore", train_rows, "38;5;42")
    add_stats("validation", validation_rows, "38;5;75")

    z_grid = policy_map["z_grid"]
    items = policy_map["items"]
    actions = [ACTION_LABELS[int(item["action_idx"])] for item in items]
    aux = [
        AUX_LABELS[int(item.get("aux_pred", 1))]
        for item in items
    ]
    max_q = [max(float(value) for value in item["q_values"]) for item in items]
    cells = lambda values: "  ".join(f"{value:^10}" for value in values)
    lines.extend(
        [
            "",
            bold(f"POLICY PROBE  v{policy_map['policy_version']}  (flat position, deterministic inference)"),
            f"spread z     {cells([f'{z:+.1f}' for z in z_grid])}",
            f"action       {cells(actions)}",
            f"max Q        {cells([f'{value:+.3f}' for value in max_q])}",
            f"aux drift    {cells(aux)}",
            "",
            dim(
                "reference: SHORT z>+0.65, LONG z<-0.65, otherwise FLAT  |  "
                "T/V are interleaved but stored separately"
            ),
        ]
    )
    if final:
        lines.append(cyan("complete: the final frame is held for inspection"))
    else:
        lines.append(dim("lowercase t/v = active · uppercase T/V = complete"))
    return "\n".join(lines) + "\n"


def wait_for_learning(
    client: JsonClient,
    model_id: str,
    *,
    minimum_updates: int,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    model: dict[str, Any] = {}
    while time.time() < deadline:
        model = client.get(f"/v1/models/{model_id}")["model"]
        learner = model["learner"]
        if (
            int(learner.get("updates", 0)) >= minimum_updates
            and int(learner.get("validation_runs", 0)) >= 1
        ):
            return model
        time.sleep(0.05)
    raise TimeoutError(
        f"learner did not reach {minimum_updates} updates and one validation run"
    )


def build_summary(
    args: argparse.Namespace,
    jobs: list[EpisodeJob],
    results: list[EpisodeResult],
    model: dict[str, Any],
    policy_map: dict[str, Any],
) -> dict[str, Any]:
    def summarize_split(split: str) -> dict[str, Any]:
        rows = [row for row in results if row.split == split]
        return {
            "episodes": len(rows),
            "reward_mean": _mean(rows, "reward_sum"),
            "reference_reward_mean": _mean(rows, "reference_reward_sum"),
            "advantage_mean": _mean(rows, "advantage"),
            "max_drawdown_mean": _mean(rows, "max_drawdown"),
            "turnover_mean": _mean(rows, "turnover"),
            "positive_fraction": (
                sum(row.reward_sum > 0.0 for row in rows) / len(rows) if rows else 0.0
            ),
        }

    learner = model["learner"]
    return {
        "format": "jormungandr.ou_spread_result.v1",
        "config": {
            "actors": args.actors,
            "train_episodes": args.train_episodes,
            "validation_episodes": args.validation_episodes,
            "horizon": args.horizon,
            "seed": args.seed,
            "epsilon": args.epsilon,
        },
        "job_splits": [job.split for job in jobs],
        "train": summarize_split("train"),
        "validation": summarize_split("validation"),
        "learner": {
            "updates": int(learner["updates"]),
            "policy_version": int(learner["policy_version"]),
            "validation_runs": int(learner["validation_runs"]),
            "last_train_loss": float(learner["last_loss"]),
            "last_train_aux_accuracy": float(learner["last_aux_acc"]),
            "last_validation_loss": float(learner["last_validation_loss"]),
            "last_validation_aux_accuracy": float(
                learner["last_validation_aux_acc"]
            ),
        },
        "stores": {
            "training": dict(model["replay"]),
            "validation": dict(model["validation"]),
        },
        "policy_probe": [
            {
                "spread_z": float(z),
                "action": ACTION_LABELS[int(item["action_idx"])],
                "q_values": [float(value) for value in item["q_values"]],
                "aux_drift": AUX_LABELS[int(item.get("aux_pred", 1))],
            }
            for z, item in zip(policy_map["z_grid"], policy_map["items"])
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show distributed RL learning on a synthetic OU spread."
    )
    parser.add_argument("--actors", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=36)
    parser.add_argument("--validation-episodes", type=int, default=9)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--seed", type=int, default=337911)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--minimum-updates", type=int, default=60)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--actor-pause-ms", type=int, default=25)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--model-id", default="ou-spread")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--json", action="store_true", dest="print_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.actors < 1:
        raise SystemExit("--actors must be positive")
    if args.train_episodes < 1 or args.validation_episodes < 1:
        raise SystemExit("episode counts must be positive")
    if args.horizon < 2:
        raise SystemExit("--horizon must be at least 2")
    if not 0.0 <= args.epsilon <= 1.0:
        raise SystemExit("--epsilon must be in [0, 1]")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    jobs = interleaved_jobs(
        args.train_episodes,
        args.validation_episodes,
        args.seed,
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = str(output_dir / "checkpoints") if output_dir is not None else ""

    runtime = JormungandrRuntime(checkpoint_root=checkpoint_root)
    server = JormungandrHttpServer("127.0.0.1", 0, runtime=runtime)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    client = JsonClient(f"http://127.0.0.1:{server.server_address[1]}")
    workers: list[threading.Thread] = []
    shared_state = SharedActorState()
    results: list[EpisodeResult] = []
    last_model: dict[str, Any] = {}
    last_policy_map: dict[str, Any] = {}

    def emit(final: bool = False) -> None:
        nonlocal last_model, last_policy_map
        if (
            (args.print_json or (args.no_color and not args.record))
            and not final
        ):
            return
        last_model = client.get(f"/v1/models/{args.model_id}")["model"]
        last_policy_map = _policy_map(client, args.model_id)
        frame = render_dashboard(
            args=args,
            jobs=jobs,
            results=sorted(results, key=lambda item: item.ordinal),
            active=shared_state.snapshot(),
            model=last_model,
            policy_map=last_policy_map,
            final=final,
        )
        if args.record:
            sys.stdout.write(frame + "\x0c")
        elif not args.print_json:
            if not args.no_color:
                sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(frame)
        sys.stdout.flush()
        if args.interval_ms > 0 and not final:
            time.sleep(args.interval_ms / 1000.0)

    try:
        total_training_steps = args.train_episodes * args.horizon
        total_validation_steps = args.validation_episodes * args.horizon
        min_replay = min(96, total_training_steps)
        batch_size = min(64, min_replay)
        min_validation = min(32, total_validation_steps)
        client.post(
            "/v1/models",
            {
                "model_id": args.model_id,
                "obs_dim": len(OBSERVATION_KEYS),
                "replay": {
                    "capacity": max(512, 2 * total_training_steps),
                    "alpha": 0.6,
                },
                "validation": {
                    "capacity": max(128, 2 * total_validation_steps),
                },
                "tensorboard": {"enabled": False},
                "metadata": {
                    "environment": "synthetic_ou_spread.v1",
                    "feature_keys": OBSERVATION_KEYS,
                    "action_labels": ACTION_LABELS,
                    "trainer": "jormungandr.ou_spread_trainer.v1",
                    "seed": args.seed,
                    "data": "synthetic",
                },
                "learner": {
                    "enabled": True,
                    "algo": "c51",
                    "device": "cpu",
                    "hidden": 64,
                    "lr": 0.001,
                    "gamma": 0.98,
                    "v_min": -8.0,
                    "v_max": 8.0,
                    "atoms": 51,
                    "target_update": 60,
                    "batch_size": batch_size,
                    "min_replay": min_replay,
                    "updates_per_tick": 2,
                    "tick_interval_s": 0.005,
                    "checkpoint_every": 0,
                    "checkpoint_dir": "models",
                    "action_values": ACTION_VALUES,
                    "validation_every": 10,
                    "validation_batch_size": min(128, total_validation_steps),
                    "min_validation": min_validation,
                    "aux_enabled": True,
                    "aux_hidden": 32,
                    "aux_weight": 0.15,
                    "aux_classes": 3,
                    "aux_kind": "ou_drift_direction",
                    "aux_class_weighting": "balanced_batch",
                    "aux_label_smoothing": 0.02,
                },
            },
        )

        job_queue: queue.Queue[EpisodeJob | None] = queue.Queue()
        result_queue: queue.Queue[EpisodeResult | BaseException] = queue.Queue()
        for job in jobs:
            job_queue.put(job)
        for _ in range(args.actors):
            job_queue.put(None)
        for actor_number in range(args.actors):
            worker = threading.Thread(
                target=actor_loop,
                kwargs={
                    "actor_number": actor_number,
                    "jobs": job_queue,
                    "results": result_queue,
                    "client": client,
                    "model_id": args.model_id,
                    "horizon": args.horizon,
                    "epsilon": args.epsilon,
                    "pause_ms": args.actor_pause_ms,
                    "shared_state": shared_state,
                },
                daemon=True,
            )
            worker.start()
            workers.append(worker)

        emit()
        errors: list[BaseException] = []
        while len(results) + len(errors) < len(jobs):
            item = result_queue.get(timeout=args.timeout_s)
            if isinstance(item, BaseException):
                errors.append(item)
            else:
                results.append(item)
            emit()
        job_queue.join()
        if errors:
            raise RuntimeError(f"{len(errors)} actor job(s) failed; first error: {errors[0]}")

        last_model = wait_for_learning(
            client,
            args.model_id,
            minimum_updates=args.minimum_updates,
            timeout_s=args.timeout_s,
        )
        last_policy_map = _policy_map(client, args.model_id)
        emit(final=True)
        results.sort(key=lambda item: item.ordinal)
        summary = build_summary(args, jobs, results, last_model, last_policy_map)

        if output_dir is not None:
            checkpoint = client.post(
                f"/v1/models/{args.model_id}/policy/checkpoint",
                {},
            )
            summary["checkpoint"] = checkpoint["checkpoint"]
            (output_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "format": "jormungandr.ou_spread_manifest.v1",
                        "config": summary["config"],
                        "model_id": args.model_id,
                        "environment": "synthetic_ou_spread.v1",
                        "feature_keys": OBSERVATION_KEYS,
                        "action_values": ACTION_VALUES,
                        "synthetic_data_only": True,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "episode_results.jsonl").write_text(
                "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in results),
                encoding="utf-8",
            )
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.print_json:
            print(json.dumps(summary, indent=2))
    finally:
        server.shutdown()
        for worker in workers:
            worker.join(timeout=2.0)
        runtime.close_all()
        server.server_close()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
