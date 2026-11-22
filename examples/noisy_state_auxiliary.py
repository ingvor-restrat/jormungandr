"""Development study of predictive auxiliary objectives in a noisy contextual bandit.

Actions do not affect state or sensors. Policies receive 16 causal sensor samples,
not reward history or simulator parameters. Future sensors are training targets
only. The privileged arm alone receives latent-state training labels.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jormungandr.auxiliary import AuxiliaryLoss, compose_losses

WINDOW = 16
HORIZONS = (1, 4, 8)
METHODS = ("reward", "predictive", "change", "joint", "reward_extra", "privileged")
CHECKPOINTS = (0, 8, 16, 32, 64, 128, 512, 2048)


@dataclass(frozen=True)
class Task:
    noise: float
    hazard: float
    steps: int = 256


TASKS = {f"noise{n}_hazard{h}": Task(n, h) for n in (0.7, 1.5) for h in (0.01, 0.10)}
# Interpolation task is development-only, not a reserved confirmation set.
EVAL_TASKS = {**TASKS, "interpolation": Task(1.0, 0.05)}


def trajectory(seed, task):
    rng = np.random.default_rng(seed)
    state = np.empty(task.steps, dtype=np.int64)
    state[0] = rng.integers(2)
    for t in range(1, task.steps):
        state[t] = state[t - 1] ^ (rng.random() < task.hazard)
    sensor = 2 * state - 1 + rng.normal(0, task.noise, task.steps)
    actions = rng.integers(2, size=task.steps)
    rewards = (rng.random(task.steps) < np.where(actions == state, 0.8, 0.2)).astype(
        np.float32
    )
    return sensor.astype(np.float32), state, actions, rewards


def features(sensor):
    # Oldest to newest, with explicit presence bits; no future observations.
    x = np.zeros((len(sensor), WINDOW * 2), dtype=np.float32)
    for t in range(len(sensor)):
        n = min(t + 1, WINDOW)
        x[t, WINDOW - n : WINDOW] = sensor[t + 1 - n : t + 1]
        x[t, 2 * WINDOW - n :] = 1
    return x


def targets(sensor):
    # Forecast actual future measurements, not smoothed/full-trajectory states.
    n = len(sensor) - max(HORIZONS)
    future = np.stack([sensor[h : h + n] for h in HORIZONS], axis=1)
    # Public operational definition: next measurement differs by more than 1.5.
    # This is an observable change event, NOT a ground-truth regime switch.
    change = (np.abs(sensor[1 : n + 1] - sensor[:n]) > 1.5).astype(np.float32)
    return future, change


def collect(seed, episodes):
    arrays = [[] for _ in range(6)]
    tasks = list(TASKS.values())
    for i in range(episodes):
        sensor, state, action, reward = trajectory(seed + i, tasks[i % len(tasks)])
        future, change = targets(sensor)
        n = len(future)
        for dest, values in zip(
            arrays,
            (features(sensor)[:n], action[:n], reward[:n], future, change, state[:n]),
        ):
            dest.append(values)
    return tuple(torch.from_numpy(np.concatenate(a)) for a in arrays)


def filter_posterior(sensor, noise, hazard, window=None):
    """Exact binary HMM filter with known generator parameters, symmetric prior."""
    answer = []
    p = 0.5
    for t in range(len(sensor)):
        if window is not None:
            p = 0.5
            observations = sensor[max(0, t + 1 - window) : t + 1]
        else:
            observations = sensor[t : t + 1]
        for z in observations:
            prior = hazard + (1 - 2 * hazard) * p
            log_odds = (
                np.log(np.clip(prior, 1e-12, 1))
                - np.log(np.clip(1 - prior, 1e-12, 1))
                + 2 * float(z) / noise**2
            )
            p = 1 / (1 + np.exp(-np.clip(log_odds, -700, 700)))
        answer.append(p)
    return np.asarray(answer)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(WINDOW * 2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.reward = nn.Linear(64, 2)
        self.forecast = nn.Linear(64, len(HORIZONS) * 2)
        self.change = nn.Linear(64, 1)
        self.state = nn.Linear(64, 1)


def update(model, optimizer, batch, method, coefficient):
    x, action, reward, future, change, state = batch
    h = model.encoder(x)
    loss = F.binary_cross_entropy_with_logits(
        model.reward(h).gather(1, action[:, None]).squeeze(1), reward
    )
    terms = []
    if method in ("predictive", "joint"):
        mean, raw_var = model.forecast(h).chunk(2, dim=1)
        variance = F.softplus(raw_var) + 0.05
        # Proper Gaussian predictive scoring rule. This approximates the HMM's
        # mixture distribution by its mean/variance; variance is not epistemic.
        nll = 0.5 * (variance.log() + (future - mean).square() / variance)
        terms.append(AuxiliaryLoss("forecast", nll.mean(), coefficient))
    if method in ("change", "joint"):
        terms.append(
            AuxiliaryLoss(
                "change",
                F.binary_cross_entropy_with_logits(model.change(h).squeeze(1), change),
                coefficient,
            )
        )
    if method == "privileged":
        terms.append(
            AuxiliaryLoss(
                "state",
                F.binary_cross_entropy_with_logits(
                    model.state(h).squeeze(1), state.float()
                ),
                coefficient,
            )
        )
    total, metrics = compose_losses(loss, terms)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    return {"reward_loss": float(loss.detach()), **metrics}


def evaluation_data(seed, episodes):
    result = {}
    for j, (name, task) in enumerate(EVAL_TASKS.items()):
        rows = []
        for i in range(episodes):
            sensor, state, _, _ = trajectory(seed + j * 1000 + i, task)
            future, change = targets(sensor)
            refs = {
                "last_sensor": sensor > 0,
                "history_mean": features(sensor)[:, :WINDOW].sum(1) > 0,
                "window_filter_known_parameters": filter_posterior(
                    sensor, task.noise, task.hazard, WINDOW
                )
                > 0.5,
                "full_filter_known_parameters": filter_posterior(
                    sensor, task.noise, task.hazard
                )
                > 0.5,
                "latent_state_oracle": state,
            }
            rows.append((features(sensor), state, future, change, refs))
        result[name] = rows
    return result


@torch.no_grad()
def evaluate(model, data, method):
    result = {}
    for name, rows in data.items():
        x = torch.from_numpy(np.concatenate([r[0] for r in rows]))
        h = model.encoder(x)
        probs = model.reward(h).sigmoid().numpy().reshape(len(rows), -1, 2)
        per_episode = []
        brier = []
        for i, (_, state, _, _, _) in enumerate(rows):
            action = probs[i].argmax(1)
            per_episode.append(float(np.where(action == state, 0.8, 0.2).mean()))
            true_prob = np.where(np.arange(2)[None, :] == state[:, None], 0.8, 0.2)
            brier.append(float(((probs[i] - true_prob) ** 2).mean()))
        metrics = {
            "expected_reward": float(np.mean(per_episode)),
            "per_episode_expected_reward": per_episode,
            "reward_probability_mse": float(np.mean(brier)),
        }
        if method in ("predictive", "joint", "change"):
            means, raw = model.forecast(h).chunk(2, 1)
            variances = F.softplus(raw) + 0.05
            n = rows[0][2].shape[0]
            mean = means.reshape(len(rows), -1, len(HORIZONS))[:, :n].numpy()
            var = variances.reshape(len(rows), -1, len(HORIZONS))[:, :n].numpy()
            future = np.stack([r[2] for r in rows])
            if method in ("predictive", "joint"):
                metrics["forecast_mse"] = float(((future - mean) ** 2).mean())
                metrics["forecast_gaussian_nll_without_constant"] = float(
                    (0.5 * (np.log(var) + (future - mean) ** 2 / var)).mean()
                )
                metrics["forecast_standardized_squared_error"] = float(
                    ((future - mean) ** 2 / var).mean()
                )
            if method in ("change", "joint"):
                change_p = (
                    model.change(h).sigmoid().reshape(len(rows), -1)[:, :n].numpy()
                )
                metrics["observable_change_brier"] = float(
                    ((change_p - np.stack([r[3] for r in rows])) ** 2).mean()
                )
        result[name] = metrics
    return {
        "tasks": result,
        "mean_reward": float(np.mean([r["expected_reward"] for r in result.values()])),
    }


def run(args):
    if min(args.episodes) < 4 or min(args.seeds) < 0 or args.eval_episodes < 2:
        raise ValueError("need >=4 training episodes and >=2 evaluation episodes")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    sources = ["examples/noisy_state_auxiliary.py", "src/jormungandr/auxiliary.py"]
    protocol = {
        "stage": "development_only",
        "paper_replication": False,
        "args": vars(args),
        "tasks": {k: asdict(v) for k, v in TASKS.items()},
        "eval_tasks": {k: asdict(v) for k, v in EVAL_TASKS.items()},
        "methods": METHODS,
        "checkpoints": CHECKPOINTS,
        "coefficient_per_term": 0.1,
        "batch_size": 128,
        "learning_rate": 0.001,
        "window": WINDOW,
        "horizons": HORIZONS,
        "timing": "steady-state training only; shared warmup, collection and evaluation separate",
        "compute": "all auxiliaries share one encoder pass; reward_extra does two independent reward updates and is an upper compute control, not an exact FLOP match",
        "inputs": "causal sensor window only, excluding rewards, parameters and hidden state",
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": 2,
        },
        "sources": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources
        },
    }
    for p in sources:
        dest = out / "sources" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(p).read_bytes())
    start = time.perf_counter()
    dummy = collect(1, 4)
    for method in METHODS:
        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(3):
            update(model, optimizer, [v[:128] for v in dummy], method, 0.1)
    protocol["shared_warmup_seconds"] = time.perf_counter() - start
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    records = []
    for seed in args.seeds:
        data = evaluation_data(seed + 10000000, args.eval_episodes)
        refs = {
            name: {
                ref: float(
                    np.mean(
                        [
                            np.where(row[4][ref] == row[1], 0.8, 0.2).mean()
                            for row in rows
                        ]
                    )
                )
                for ref in rows[0][4]
            }
            for name, rows in data.items()
        }
        for episodes in args.episodes:
            start = time.perf_counter()
            arrays = collect(seed * 1000, episodes)
            collection_seconds = time.perf_counter() - start
            corpus_hash = hashlib.sha256(
                b"".join(v.numpy().tobytes() for v in arrays)
            ).hexdigest()
            tape = np.random.default_rng(seed).integers(
                len(arrays[0]), size=(max(CHECKPOINTS), 2, 128)
            )
            order = list(METHODS)
            np.random.default_rng(seed + episodes).shuffle(order)
            for method in order:
                torch.manual_seed(seed)
                model = Model()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                points, train_seconds, eval_seconds = [], 0.0, 0.0
                for step in range(max(CHECKPOINTS) + 1):
                    if step in CHECKPOINTS:
                        start = time.perf_counter()
                        point = evaluate(model, data, method)
                        eval_seconds += time.perf_counter() - start
                        point.update(
                            step=step,
                            encoder_examples=step
                            * 128
                            * (2 if method == "reward_extra" else 1),
                            training_seconds=train_seconds,
                            optimizer_updates=step
                            * (2 if method == "reward_extra" else 1),
                        )
                        points.append(point)
                        print(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "episodes": episodes,
                                    "method": method,
                                    "step": step,
                                    "reward": round(point["mean_reward"], 4),
                                }
                            ),
                            flush=True,
                        )
                    if step == max(CHECKPOINTS):
                        break
                    start = time.perf_counter()
                    for repeat in range(2 if method == "reward_extra" else 1):
                        idx = tape[step, repeat]
                        update(model, optimizer, [v[idx] for v in arrays], method, 0.1)
                    train_seconds += time.perf_counter() - start
                records.append(
                    {
                        "seed": seed,
                        "training_episodes": episodes,
                        "collected_interactions": episodes * 256,
                        "training_rows": len(arrays[0]),
                        "corpus_sha256": corpus_hash,
                        "collection_seconds": collection_seconds,
                        "evaluation_seconds": eval_seconds,
                        "method": method,
                        "method_order": order,
                        "references": refs,
                        "points": points,
                    }
                )
                torch.save(model.state_dict(), out / f"{seed}-{episodes}-{method}.pt")
                (out / "results.json").write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[9109, 9209, 9311])
    p.add_argument("--episodes", type=int, nargs="+", default=[32, 128])
    p.add_argument("--eval-episodes", type=int, default=32)
    run(p.parse_args())
