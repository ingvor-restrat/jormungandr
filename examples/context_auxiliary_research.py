"""Development-only numeric context-grounding pilot, not a ContextRL replication.

All methods share environment data and encoder capacity. Contrastive and factual
arms see the same count-preserving synthetic history pairs. Synthetic contexts
never enter reward/TD replay. The compute control spends three reward updates
per outer step, matching the contrastive arm's encoder-example count.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jormungandr.auxiliary import AuxiliaryLoss, compose_losses, context_selection_loss
from examples.multiservice_learning_control import collect_arrays, encode
from examples.multiservice_adaptation import CONFIRM, evaluate, world, WINDOW

TASKS = {
    k: CONFIRM[k]
    for k in ("eight_stable", "eight_changes", "eight_noisy", "eight_harder")
}
METHODS = ("reward", "contrastive", "factual", "reward_compute")
CHECKPOINTS = (128, 512, 2048)


def context_pairs(x, generator):
    """Swap latest own outcome with an older opposite own outcome.

    Histories retain identical action positions, counts, length and other-service
    outcomes. The question is fixed: what was this candidate's latest observed
    outcome? The anchor is that public fact, never the unobserved best action.
    """
    events = x[:, :-1].reshape(len(x), -1, 4)
    own = (events[:, :, 0] > 0.5) & (events[:, :, 3] > 0.5)
    first = own.long().argmax(1)
    rows = torch.arange(len(x), device=x.device)
    anchor = events[rows, first, 2]
    opposite = own & (events[:, :, 2] != anchor[:, None])
    valid = own.any(1) & opposite.any(1)
    alternate = opposite.long().argmax(1)
    edited = x.clone()
    edited_events = edited[:, :-1].reshape_as(events)
    active = rows[valid]
    edited_events[active, first[valid], 2] = 1 - anchor[valid]
    edited_events[active, alternate[valid], 2] = anchor[valid]
    positive = torch.randint(2, (len(x),), generator=generator, device=x.device)
    left = torch.where((positive == 0)[:, None], x, edited)
    right = torch.where((positive == 0)[:, None], edited, x)
    return left, right, positive, anchor, valid


class ContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(WINDOW * 4 + 1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU()
        )
        self.reward_head = nn.Linear(128, 1)
        self.context_head = nn.Linear(128, 1)

    def forward(self, x):
        return self.reward_head(self.encoder(x)).squeeze(-1)


class Actor:
    def __init__(self, model):
        self.policy = model
        self.policy.candidate_context = "joint"
        self.update_steps = 0

    @torch.no_grad()
    def action_results_structured(self, observations, deterministic=True):
        features = [
            encode(o.metadata["llm"]["history"], key, len(o.candidate_ids))
            for o in observations
            for key in o.candidate_ids
        ]
        values = self.policy(torch.from_numpy(np.stack(features))).sigmoid().tolist()
        offset = 0
        results = []
        for o in observations:
            local = tuple(values[offset : offset + len(o.candidate_ids)])
            results.append(
                SimpleNamespace(
                    candidate_values=local,
                    candidate_id=o.candidate_ids[int(np.argmax(local))],
                )
            )
            offset += len(local)
        return results


def probe_accuracy(model, x):
    generator = torch.Generator().manual_seed(911)
    left, right, positive, anchor, valid = context_pairs(x, generator)
    with torch.no_grad():
        a = model.context_head(model.encoder(left)).squeeze(-1)
        b = model.context_head(model.encoder(right)).squeeze(-1)
        logits = torch.stack([a, b], 1) * (2 * anchor - 1)[:, None]
        correct = logits.argmax(1) == positive
    return {
        "accuracy": float(correct[valid].float().mean()) if valid.any() else None,
        "valid_pairs": int(valid.sum()),
    }


def evaluate_model(actor, worlds, probe):
    start = time.perf_counter()
    actor.policy.eval()
    results = {
        name: evaluate(actor, ws, "language", observation_batch_size=len(ws))
        for name, ws in worlds.items()
    }
    return {
        "tasks": results,
        "mean_reward": float(np.mean([r["mean_reward"] for r in results.values()])),
        "context_probe": probe_accuracy(actor.policy, probe),
        "evaluation_seconds": time.perf_counter() - start,
    }


def train_update(model, optimizer, x, y, method, coefficient, pair_generator):
    model.train()
    loss = F.binary_cross_entropy_with_logits(model(x), y)
    terms = []
    valid_count = 0
    if method in ("contrastive", "factual"):
        left, right, positive, anchor, valid = context_pairs(x, pair_generator)
        a = model.context_head(model.encoder(left)).squeeze(-1)
        b = model.context_head(model.encoder(right)).squeeze(-1)
        if method == "contrastive":
            scores = torch.stack([a, b], 1) * (2 * anchor - 1)[:, None]
            auxiliary = context_selection_loss(
                scores, positive, valid=valid, margin_clip=5.0
            )
        else:
            target_a = torch.where(positive == 0, anchor, 1 - anchor)
            target_b = 1 - target_a
            per_item = (
                F.binary_cross_entropy_with_logits(a, target_a, reduction="none")
                + F.binary_cross_entropy_with_logits(b, target_b, reduction="none")
            ) / 2
            auxiliary = torch.where(valid, per_item, torch.zeros_like(per_item)).mean()
        terms = [AuxiliaryLoss("context", auxiliary, coefficient)]
        valid_count = int(valid.sum())
    total, metrics = compose_losses(loss, terms)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    return {"task_loss": float(loss.detach()), **metrics}, valid_count


def run(args):
    if (
        min(args.seeds) < 0
        or min(args.trials) < 1
        or args.eval_trials < 2
        or args.batch_size < 1
        or args.coefficient < 0
        or not np.isfinite(args.coefficient)
    ):
        raise ValueError("invalid experiment budget")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    sources = [
        "examples/context_auxiliary_research.py",
        "examples/multiservice_learning_control.py",
        "examples/multiservice_adaptation.py",
        "src/jormungandr/auxiliary.py",
    ]
    # Separate one-time optimizer/backend initialization from steady-state timing.
    setup_start = time.perf_counter()
    dummy = torch.zeros(args.batch_size, WINDOW * 4 + 1)
    events = dummy[:, :-1].reshape(args.batch_size, WINDOW, 4)
    events[:, :2, 0] = 1.0
    events[:, :2, 3] = 1.0
    events[:, 0, 2] = 1.0
    dummy[:, -1] = 1 / 3
    for method in METHODS:
        torch.manual_seed(0)
        warm = ContextModel()
        optimizer = torch.optim.Adam(warm.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(0)
        for _ in range(3):
            train_update(
                warm,
                optimizer,
                dummy,
                torch.zeros(args.batch_size),
                method,
                args.coefficient,
                generator,
            )
    setup_seconds = time.perf_counter() - setup_start
    protocol = {
        "stage": "development_only",
        "shared_backend_warmup_seconds": setup_seconds,
        "timing": "steady-state training wall seconds include pair generation and optimizer work; collection/evaluation/startup are recorded separately; encoder examples are a compute proxy, not FLOPs",
        "probe_interpretation": "only contrastive/factual context heads are trained; untrained-head accuracy is not a comparable representation metric",
        "paper_replication": False,
        "args": vars(args),
        "checkpoints": CHECKPOINTS,
        "methods": METHODS,
        "tasks": {k: asdict(v) for k, v in TASKS.items()},
        "target": "equal-task mean of strongest reference minus 0.05; checkpoint first-passage only",
        "auxiliary_labels": "latest observed own reward; negatives swap own outcomes, preserving counts; randomized option order",
        "compute_control": "three reward-only optimizer updates versus three encoder forwards per auxiliary update; wall time also measured",
        "sources": {
            f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources
        },
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": 2,
        },
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for f in sources:
        target = out / "sources" / f
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(f).read_bytes())
    records = []
    for seed in args.seeds:
        worlds = {
            name: [
                world(seed + 300000 + j, task, i, renamed=True)
                for i in range(args.eval_trials)
            ]
            for j, (name, task) in enumerate(TASKS.items())
        }
        references = {
            name: {
                method: evaluate(
                    None, ws, method, observation_batch_size=args.eval_trials
                )
                for method in ("last_outcome", "window_mean", "window_ucb")
            }
            for name, ws in worlds.items()
        }
        target = (
            float(
                np.mean(
                    [
                        max(r["mean_reward"] for r in methods.values())
                        for methods in references.values()
                    ]
                )
            )
            - 0.05
        )
        # A disjoint, explicitly development-only factual probe, never used for updates.
        probe = torch.from_numpy(collect_arrays(seed + 400000, 32)[0])
        torch.manual_seed(seed)
        initial = Actor(ContextModel())
        initial_eval = evaluate_model(initial, worlds, probe)
        for trials in args.trials:
            start = time.perf_counter()
            arrays = collect_arrays(seed, trials)
            collection_seconds = time.perf_counter() - start
            x, y = [torch.from_numpy(a) for a in arrays]
            rng = np.random.default_rng(seed + 500000)
            indices = rng.integers(len(x), size=(max(CHECKPOINTS), 3, args.batch_size))
            order = list(METHODS)
            np.random.default_rng(seed + trials).shuffle(order)
            for method in order:
                torch.manual_seed(seed)
                actor = Actor(ContextModel())
                optimizer = torch.optim.Adam(actor.policy.parameters(), lr=0.001)
                pair_generator = torch.Generator().manual_seed(seed + 600000)
                training_seconds = 0.0
                examples = 0
                pair_count = 0
                seen = set()
                points = [
                    {
                        "outer_step": 0,
                        "optimizer_updates": 0,
                        "encoder_examples": 0,
                        "training_seconds": 0.0,
                        "sampled_unique_transitions": 0,
                        **initial_eval,
                    }
                ]
                for step in range(max(CHECKPOINTS)):
                    start = time.perf_counter()
                    for repeat in range(3 if method == "reward_compute" else 1):
                        sample = indices[step, repeat]
                        seen.update(sample.tolist())
                        metrics, n = train_update(
                            actor.policy,
                            optimizer,
                            x[sample],
                            y[sample],
                            method,
                            args.coefficient,
                            pair_generator,
                        )
                        actor.update_steps += 1
                        pair_count += n
                        examples += len(sample) * (
                            3 if method in ("contrastive", "factual") else 1
                        )
                    training_seconds += time.perf_counter() - start
                    if step + 1 in CHECKPOINTS:
                        point = {
                            "outer_step": step + 1,
                            "optimizer_updates": actor.update_steps,
                            "encoder_examples": examples,
                            "training_seconds": training_seconds,
                            "sampled_unique_transitions": len(seen),
                            "valid_auxiliary_pairs_seen": pair_count,
                            "last_losses": metrics,
                            **evaluate_model(actor, worlds, probe),
                        }
                        points.append(point)
                        print(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "trials": trials,
                                    "method": method,
                                    "step": step + 1,
                                    "reward": round(point["mean_reward"], 4),
                                    "context_accuracy": point["context_probe"][
                                        "accuracy"
                                    ],
                                    "training_seconds": round(training_seconds, 2),
                                }
                            ),
                            flush=True,
                        )
                reached = next((p for p in points if p["mean_reward"] >= target), None)
                record = {
                    "context_head_trained": method in ("contrastive", "factual"),
                    "seed": seed,
                    "training_trials": trials,
                    "environment_interactions": len(x),
                    "collection_seconds": collection_seconds,
                    "corpus_sha256": hashlib.sha256(
                        x.numpy().tobytes() + y.numpy().tobytes()
                    ).hexdigest(),
                    "method": method,
                    "method_order": order,
                    "target": target,
                    "first_observed_target": (
                        None
                        if reached is None
                        else {
                            k: reached[k]
                            for k in (
                                "outer_step",
                                "optimizer_updates",
                                "encoder_examples",
                                "training_seconds",
                            )
                        }
                    ),
                    "references": references,
                    "points": points,
                }
                records.append(record)
                torch.save(
                    actor.policy.state_dict(), out / f"{seed}-{trials}-{method}.pt"
                )
                (out / "results.json").write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[8101, 8203, 8309])
    p.add_argument("--trials", type=int, nargs="+", default=[64, 256])
    p.add_argument("--eval-trials", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--coefficient", type=float, default=0.1)
    run(p.parse_args())
