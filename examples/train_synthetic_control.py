"""End-to-end multi-actor training against a synthetic control environment.

The example launches an in-process Jörmungandr HTTP service. Actor threads know
only its URL and the generic environment protocol below; they do not access the
learner directly.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import torch

from jormungandr.service import JormungandrHttpServer, JormungandrRuntime


EXPERIENCE_SCHEMA = "jormungandr.experience.v1"
AUX_UPDATE_SCHEMA = "jormungandr.aux_update.v1"
OBSERVATION_KEYS = ["position", "velocity", "target_error", "remaining_fraction"]
ACTION_VALUES = [-1.0, 0.0, 1.0]


@dataclass(frozen=True)
class EnvironmentStep:
    observation: list[float]
    reward: float
    terminal: bool
    aux_label: int
    absolute_error: float


class EpisodicEnvironment(Protocol):
    """The only interface an actor needs from a problem implementation."""

    observation_dim: int
    action_values: list[float]

    def reset(self, seed: int) -> list[float]:
        """Start one episode and return its first observation."""

    def step(self, action: float) -> EnvironmentStep:
        """Apply one declared action value and return the resulting transition."""


class SyntheticTrackingEnvironment:
    """Control a point so it follows a smooth, deterministic target path."""

    observation_dim = len(OBSERVATION_KEYS)
    action_values = ACTION_VALUES

    def __init__(self, horizon: int) -> None:
        self.horizon = max(2, int(horizon))
        self.timestep = 0
        self.position = 0.0
        self.velocity = 0.0
        self.phase = 0.0

    def _target(self, timestep: int) -> float:
        x = self.phase + 0.17 * float(timestep)
        return 0.75 * math.sin(x) + 0.25 * math.sin(0.37 * x + 0.4)

    def _observation(self) -> list[float]:
        target = self._target(self.timestep)
        return [
            float(self.position),
            float(self.velocity),
            float(target - self.position),
            float(max(0, self.horizon - self.timestep) / self.horizon),
        ]

    def reset(self, seed: int) -> list[float]:
        rng = np.random.default_rng(int(seed))
        self.timestep = 0
        self.phase = float(rng.uniform(-math.pi, math.pi))
        self.position = float(rng.normal(0.0, 0.2))
        self.velocity = float(rng.normal(0.0, 0.05))
        return self._observation()

    def step(self, action: float) -> EnvironmentStep:
        if float(action) not in self.action_values:
            raise ValueError(f"unsupported action value: {action}")

        acceleration = 0.18 * float(action)
        self.velocity = float(np.clip(0.82 * self.velocity + acceleration, -0.8, 0.8))
        self.position = float(np.clip(self.position + self.velocity, -2.0, 2.0))
        self.timestep += 1

        target = self._target(self.timestep)
        error = float(target - self.position)
        reward = float(np.clip(1.0 - error * error - 0.05 * self.velocity**2, -2.0, 1.0))
        terminal = self.timestep >= self.horizon

        future_target = self._target(min(self.horizon, self.timestep + 3))
        target_delta = future_target - target
        if target_delta < -0.04:
            aux_label = 0
        elif target_delta > 0.04:
            aux_label = 2
        else:
            aux_label = 1

        return EnvironmentStep(
            observation=self._observation(),
            reward=reward,
            terminal=terminal,
            aux_label=aux_label,
            absolute_error=abs(error),
        )


@dataclass(frozen=True)
class EpisodeJob:
    ordinal: int
    split: str
    seed: int


@dataclass(frozen=True)
class EpisodeResult:
    ordinal: int
    split: str
    actor_id: str
    episode_id: str
    reward_sum: float
    mean_absolute_error: float
    steps: int
    first_policy_version: int
    last_policy_version: int


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
    """Place validation episodes throughout training without randomizing splits."""
    if train_episodes < 1:
        raise ValueError("train_episodes must be positive")
    if validation_episodes < 1:
        raise ValueError("validation_episodes must be positive")

    split_order: list[str] = []
    validation_added = 0
    for train_added in range(1, train_episodes + 1):
        split_order.append("train")
        validation_target = math.floor(
            train_added * validation_episodes / train_episodes
        )
        while validation_added < validation_target:
            split_order.append("validation")
            validation_added += 1
    while validation_added < validation_episodes:
        split_order.append("validation")
        validation_added += 1

    return [
        EpisodeJob(
            ordinal=ordinal,
            split=split,
            seed=int(seed + 10_007 * ordinal),
        )
        for ordinal, split in enumerate(split_order)
    ]


def run_episode(
    *,
    client: JsonClient,
    model_id: str,
    actor_id: str,
    job: EpisodeJob,
    horizon: int,
    epsilon: float,
) -> EpisodeResult:
    environment: EpisodicEnvironment = SyntheticTrackingEnvironment(horizon)
    observation = environment.reset(job.seed)
    episode_id = f"synthetic-{job.ordinal:05d}"
    experiences: list[dict[str, Any]] = []
    aux_updates: list[dict[str, Any]] = []
    rewards: list[float] = []
    errors: list[float] = []
    policy_versions: list[int] = []

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
                "meta": {"environment": "synthetic_tracking.v1"},
            }
        )
        # These are intentionally attached after the transition batch to show
        # the delayed auxiliary-label protocol.
        aux_updates.append(
            {
                "split": job.split,
                "actor_id": actor_id,
                "episode_id": episode_id,
                "timestep": timestep,
                "aux": {
                    "kind": "target_direction",
                    "label": step.aux_label,
                },
            }
        )
        rewards.append(step.reward)
        errors.append(step.absolute_error)
        policy_versions.append(policy_version)
        observation = step.observation
        if step.terminal:
            break

    client.post(
        f"/v1/models/{model_id}/experience/add",
        {"schema": EXPERIENCE_SCHEMA, "items": experiences},
    )
    client.post(
        f"/v1/models/{model_id}/experience/aux_update",
        {"schema": AUX_UPDATE_SCHEMA, "updates": aux_updates},
    )
    client.post(
        f"/v1/models/{model_id}/metrics",
        {
            "step": job.ordinal,
            "metrics": {
                f"sampler/{job.split}/episode_reward_sum": sum(rewards),
                f"sampler/{job.split}/episode_mean_absolute_error": statistics.fmean(
                    errors
                ),
                f"sampler/{job.split}/episode_steps": len(rewards),
            },
        },
    )

    return EpisodeResult(
        ordinal=job.ordinal,
        split=job.split,
        actor_id=actor_id,
        episode_id=episode_id,
        reward_sum=float(sum(rewards)),
        mean_absolute_error=float(statistics.fmean(errors)),
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
) -> None:
    actor_id = f"actor-{actor_number:02d}"
    while True:
        job = jobs.get()
        try:
            if job is None:
                return
            result = run_episode(
                client=client,
                model_id=model_id,
                actor_id=actor_id,
                job=job,
                horizon=horizon,
                epsilon=epsilon,
            )
            results.put(result)
        except BaseException as exc:
            results.put(exc)
        finally:
            jobs.task_done()


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


def summarize(
    *,
    args: argparse.Namespace,
    jobs: list[EpisodeJob],
    results: list[EpisodeResult],
    model: dict[str, Any],
) -> dict[str, Any]:
    split_results = {
        split: [result for result in results if result.split == split]
        for split in ("train", "validation")
    }

    def split_summary(split: str) -> dict[str, Any]:
        rows = split_results[split]
        return {
            "episodes": len(rows),
            "reward_mean": float(statistics.fmean(x.reward_sum for x in rows)),
            "mean_absolute_error": float(
                statistics.fmean(x.mean_absolute_error for x in rows)
            ),
        }

    learner = model["learner"]
    return {
        "format": "jormungandr.synthetic_trainer_result.v1",
        "config": {
            "actors": args.actors,
            "train_episodes": args.train_episodes,
            "validation_episodes": args.validation_episodes,
            "horizon": args.horizon,
            "seed": args.seed,
            "epsilon": args.epsilon,
        },
        "job_splits": [job.split for job in jobs],
        "train": split_summary("train"),
        "validation": split_summary("validation"),
        "learner": {
            "updates": int(learner["updates"]),
            "policy_version": int(learner["policy_version"]),
            "validation_runs": int(learner["validation_runs"]),
            "validation_policy_version": int(
                learner["validation_policy_version"]
            ),
            "last_train_loss": float(learner["last_loss"]),
            "last_validation_loss": float(learner["last_validation_loss"]),
            "last_validation_aux_acc": float(
                learner["last_validation_aux_acc"]
            ),
        },
        "stores": {
            "training": dict(model["replay"]),
            "validation": dict(model["validation"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a generic multi-actor Jörmungandr training example."
    )
    parser.add_argument("--actors", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--minimum-updates", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--model-id", default="synthetic-control")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.actors < 1:
        raise SystemExit("--actors must be positive")
    if args.horizon < 2:
        raise SystemExit("--horizon must be at least 2")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    worker_threads: list[threading.Thread] = []
    try:
        total_training_steps = args.train_episodes * args.horizon
        total_validation_steps = args.validation_episodes * args.horizon
        min_replay = min(32, total_training_steps)
        batch_size = min(32, min_replay)
        min_validation = min(8, total_validation_steps)
        client.post(
            "/v1/models",
            {
                "model_id": args.model_id,
                "obs_dim": len(OBSERVATION_KEYS),
                "replay": {
                    "capacity": max(256, 2 * total_training_steps),
                    "alpha": 0.6,
                },
                "validation": {
                    "capacity": max(64, 2 * total_validation_steps),
                },
                "tensorboard": {"enabled": False},
                "metadata": {
                    "environment": "synthetic_tracking.v1",
                    "feature_keys": OBSERVATION_KEYS,
                    "action_labels": ["decrease", "hold", "increase"],
                    "trainer": "jormungandr.synthetic_trainer.v1",
                    "seed": args.seed,
                },
                "learner": {
                    "enabled": True,
                    "algo": "c51",
                    "device": "cpu",
                    "hidden": 64,
                    "lr": 0.0005,
                    "gamma": 0.97,
                    "v_min": -10.0,
                    "v_max": 10.0,
                    "atoms": 51,
                    "target_update": 50,
                    "batch_size": batch_size,
                    "min_replay": min_replay,
                    "tick_interval_s": 0.01,
                    "checkpoint_every": 0,
                    "checkpoint_dir": "models",
                    "action_values": ACTION_VALUES,
                    "validation_every": 5,
                    "validation_batch_size": min(64, total_validation_steps),
                    "min_validation": min_validation,
                    "aux_enabled": True,
                    "aux_hidden": 32,
                    "aux_weight": 0.1,
                    "aux_classes": 3,
                    "aux_kind": "target_direction",
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
            thread = threading.Thread(
                target=actor_loop,
                kwargs={
                    "actor_number": actor_number,
                    "jobs": job_queue,
                    "results": result_queue,
                    "client": client,
                    "model_id": args.model_id,
                    "horizon": args.horizon,
                    "epsilon": args.epsilon,
                },
                daemon=True,
            )
            thread.start()
            worker_threads.append(thread)

        job_queue.join()
        results: list[EpisodeResult] = []
        errors: list[BaseException] = []
        while not result_queue.empty():
            item = result_queue.get()
            if isinstance(item, BaseException):
                errors.append(item)
            else:
                results.append(item)
        if errors:
            raise RuntimeError(f"{len(errors)} actor job(s) failed; first error: {errors[0]}")
        if len(results) != len(jobs):
            raise RuntimeError(f"expected {len(jobs)} results, received {len(results)}")

        results.sort(key=lambda item: item.ordinal)
        model = wait_for_learning(
            client,
            args.model_id,
            minimum_updates=args.minimum_updates,
            timeout_s=args.timeout_s,
        )
        summary = summarize(args=args, jobs=jobs, results=results, model=model)

        if output_dir is not None:
            checkpoint = client.post(
                f"/v1/models/{args.model_id}/policy/checkpoint",
                {},
            )
            summary["checkpoint"] = checkpoint["checkpoint"]
            (output_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "format": "jormungandr.synthetic_trainer_manifest.v1",
                        "config": summary["config"],
                        "model_id": args.model_id,
                        "environment": "synthetic_tracking.v1",
                        "feature_keys": OBSERVATION_KEYS,
                        "action_values": ACTION_VALUES,
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

        print(json.dumps(summary, indent=2))
    finally:
        server.shutdown()
        for thread in worker_threads:
            thread.join(timeout=2.0)
        runtime.close_all()
        server.server_close()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
