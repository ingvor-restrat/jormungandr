"""Native structured DQN benchmark isolating reward delay, sparsity, and noise.

One consequential choice followed by forced waits avoids behavior-policy bias in
multi-step targets. This is a credit-assignment control, not general control.
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

from jormungandr.algorithms.structured_dqn import StructuredDQNAgent
from jormungandr.returns import n_step_transitions
from jormungandr.structured import (
    EntityCandidateObservation,
    EntityCandidatePolicyOutput,
    StructuredPolicySpec,
)
from jormungandr.structured_replay import StructuredReplayTransition

HORIZON = 13
GAMMA = 0.95
CHECKPOINTS = (0, 32, 128, 512)


@dataclass(frozen=True)
class Task:
    delay: int
    feedback_probability: float
    noise: float


TASKS = {
    f"delay{d}_feedback{p}_noise{s}": Task(d, p, s)
    for d in (0, 4, 12)
    for p in (1.0, 0.25)
    for s in (0.0, 0.8)
}


def observation(cue, time_index, chosen, task):
    decision = time_index == 0
    candidates = ("left", "right") if decision else ("wait",)
    return EntityCandidateObservation(
        global_features=np.array(
            [cue, time_index / HORIZON, chosen, float(decision), task.delay / HORIZON],
            dtype=np.float32,
        ),
        entity_features=np.zeros((0, 1), dtype=np.float32),
        entity_type_ids=np.zeros(0, dtype=np.int64),
        entity_ids=(),
        candidate_features=np.array(
            [[-1.0], [1.0]] if decision else [[0.0]], dtype=np.float32
        ),
        candidate_ids=candidates,
        legal_action_mask=np.ones(len(candidates), dtype=bool),
    )


def collect(seed, task, episodes):
    rng = np.random.default_rng(seed)
    result = []
    for episode in range(episodes):
        state = int(rng.choice([-1, 1]))
        cue = state + rng.normal(0, task.noise)
        chosen = int(rng.choice([-1, 1]))
        delivered = rng.random() < task.feedback_probability
        payout = float((1 if chosen == state else -1) if delivered else 0)
        observations = [
            observation(cue, t, 0 if t == 0 else chosen, task)
            for t in range(HORIZON + 1)
        ]
        result.append(
            tuple(
                StructuredReplayTransition(
                    observation=observations[t],
                    candidate_id=(
                        ("right" if chosen == 1 else "left") if t == 0 else "wait"
                    ),
                    reward=payout if t == task.delay else 0.0,
                    next_observation=observations[t + 1],
                    done=t == HORIZON - 1,
                    actor_id="collector",
                    episode_id=str(episode),
                    timestep=t,
                    policy_version=0,
                )
                for t in range(HORIZON)
            )
        )
    return result


class NumericQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1)
        )

    def forward(self, batch):
        global_features = batch.global_features[:, None, :].expand(
            -1, batch.candidate_features.shape[1], -1
        )
        logits = self.net(
            torch.cat([global_features, batch.candidate_features], dim=-1)
        ).squeeze(-1)
        logits = logits.masked_fill(
            ~(batch.candidate_mask & batch.legal_action_mask), -1e9
        )
        return EntityCandidatePolicyOutput(logits, logits.max(1).values)


class NativeAgent(StructuredDQNAgent):
    def _build_q(self, spec, config):
        return NumericQ().to(self.device)


def build(seed):
    torch.manual_seed(seed)
    return NativeAgent(
        StructuredPolicySpec(5, 1, 1, 1),
        {
            "lr": 0.003,
            "gamma": GAMMA,
            "target_update": 20,
            "max_grad": 5.0,
            "demonstration_weight": 0.0,
        },
        "cpu",
    )


def evaluation(seed, task, episodes):
    rng = np.random.default_rng(seed)
    states = rng.choice([-1, 1], size=episodes)
    cues = states + rng.normal(0, task.noise, episodes)
    return [observation(c, 0, 0, task) for c in cues], states, cues


def evaluate(agent, data, task):
    observations, states, cues = data
    results = agent.action_results_structured(observations, deterministic=True)
    actions = np.array([1 if r.candidate_id == "right" else -1 for r in results])
    correct = actions == states
    oracle = np.where(cues >= 0, 1, -1) == states
    return {
        "accuracy": float(correct.mean()),
        "per_episode_correct": correct.astype(int).tolist(),
        "cue_oracle_accuracy": float(oracle.mean()),
        "expected_undiscounted_reward": float(
            task.feedback_probability * (2 * correct.mean() - 1)
        ),
        "expected_discounted_return": float(
            GAMMA**task.delay * task.feedback_probability * (2 * correct.mean() - 1)
        ),
    }


def run(args):
    if args.episodes < 2 or args.eval_episodes < 2 or min(args.seeds) < 0:
        raise ValueError("invalid experiment budget")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    sources = [
        "examples/delayed_credit_benchmark.py",
        "src/jormungandr/returns.py",
        "src/jormungandr/structured_replay.py",
        "src/jormungandr/algorithms/structured_dqn.py",
    ]
    protocol = {
        "stage": "development_only",
        "args": vars(args),
        "tasks": {k: asdict(v) for k, v in TASKS.items()},
        "n_steps": [1, 4, 13],
        "gamma": GAMMA,
        "horizon": HORIZON,
        "checkpoints": CHECKPOINTS,
        "training": {
            "batch_size": 64,
            "lr": 0.003,
            "target_update": 20,
            "max_grad": 5.0,
            "uniform_replay": True,
        },
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": 2,
            "device": "cpu",
        },
        "sources": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources
        },
    }
    for p in sources:
        dest = out / "sources" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(p).read_bytes())
    warm = build(1)
    warm_episode = collect(1, Task(4, 1.0, 0.0), 4)
    for n in (1, 4, 13):
        warm.update_structured(
            [t for e in warm_episode for t in n_step_transitions(e, n=n, gamma=GAMMA)]
        )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    records = []
    for seed in args.seeds:
        for name, task in TASKS.items():
            start = time.perf_counter()
            episodes = collect(seed * 1000, task, args.episodes)
            collection_seconds = time.perf_counter() - start
            data = evaluation(seed + 10000000, task, args.eval_episodes)
            # Same underlying actions/outcome gates/cues across delay conditions.
            indices = np.random.default_rng(seed).integers(
                args.episodes * HORIZON, size=(max(CHECKPOINTS), 64)
            )
            order = [1, 4, 13]
            np.random.default_rng(seed + task.delay).shuffle(order)
            for n in order:
                start = time.perf_counter()
                replay = [
                    t for e in episodes for t in n_step_transitions(e, n=n, gamma=GAMMA)
                ]
                preparation_seconds = time.perf_counter() - start
                agent = build(seed)
                seconds, points, eval_seconds = 0.0, [], 0.0
                for step in range(max(CHECKPOINTS) + 1):
                    if step in CHECKPOINTS:
                        start = time.perf_counter()
                        metrics = evaluate(agent, data, task)
                        eval_seconds += time.perf_counter() - start
                        points.append(
                            {
                                "updates": step,
                                "sampled_transitions": step * 64,
                                "training_seconds": seconds,
                                **metrics,
                            }
                        )
                    if step == max(CHECKPOINTS):
                        break
                    batch = [replay[i] for i in indices[step]]
                    start = time.perf_counter()
                    agent.update_structured(batch)
                    seconds += time.perf_counter() - start
                records.append(
                    {
                        "seed": seed,
                        "task": name,
                        "n": n,
                        "method_order": order,
                        "collected_interactions": args.episodes * HORIZON,
                        "decision_opportunities": args.episodes,
                        "collection_seconds": collection_seconds,
                        "return_preparation_seconds": preparation_seconds,
                        "evaluation_seconds": eval_seconds,
                        "points": points,
                    }
                )
                torch.save(agent.state_dict(), out / f"{seed}-{name}-n{n}.pt")
                (out / "results.json").write_text(json.dumps(records))
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "task": name,
                            "n": n,
                            "accuracy": points[-1]["accuracy"],
                            "seconds": round(seconds, 2),
                        }
                    ),
                    flush=True,
                )
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[10103, 10211, 10301])
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--eval-episodes", type=int, default=1024)
    run(p.parse_args())
