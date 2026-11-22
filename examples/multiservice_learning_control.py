"""Independent learned control with causal, identity-invariant history features.

Training labels are chosen-action outcomes only. This diagnoses learnability; it
is not a replacement for the language policy or a transferred numeric actor.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from examples.multiservice_adaptation import (
    CONFIRM,
    world,
    evaluate,
    WINDOW,
)


def encode(events, key, count, *, own_only=False, window=WINDOW):
    features = np.zeros((window, 4), dtype=np.float32)
    for i, e in enumerate(reversed(events[-window:])):
        same = e["action"] == key
        features[i] = [
            same,
            not same,
            e["reward"] if (same or not own_only) else 0.0,
            1.0,
        ]
    return np.concatenate([features.reshape(-1), [1 / count]]).astype(np.float32)


def collect_arrays(
    seed,
    trials=1024,
    own_only=False,
    window=WINDOW,
    transitions=False,
    service_counts=(3, 5),
):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    next_xs, masks, terminals = [], [], []
    for trial in range(trials):
        n = int(rng.choice(service_counts))
        hi, lo = [
            (1.0, 0.0),
            (1.0, 0.0),
            (0.95, 0.05),
            (0.85, 0.15),
            (0.75, 0.25),
            (0.65, 0.35),
        ][int(rng.integers(6))]
        period = int(rng.choice([0, 8, 12, 24, 40]))
        current = int(rng.integers(n))
        events = []
        behavior = int(rng.integers(4))
        epsilon = float(rng.choice([0.1, 0.3, 0.7, 1.0]))
        for t in range(96):
            if t and period and rng.random() < 1 / period:
                current = (current + int(rng.integers(1, n))) % n
            past = [[e["reward"] for e in events if e["action"] == k] for k in range(n)]
            if behavior == 0 or rng.random() < epsilon:
                action = int(rng.integers(n))
            else:
                if behavior == 1:
                    scores = [r[-1] if r else 0.5 for r in past]
                else:
                    scores = [(sum(r) + 1) / (len(r) + 2) for r in past]
                    if behavior == 3:
                        scores = [
                            v + 0.35 * np.sqrt(np.log(17) / (len(r) + 1))
                            for v, r in zip(scores, past)
                        ]
                action = int(np.argmax(np.array(scores) + rng.uniform(0, 1e-8, n)))
            x = encode(events, action, n, own_only=own_only, window=window)
            reward = float(rng.random() < (hi if action == current else lo))
            xs.append(x)
            ys.append(reward)
            events = (events + [{"action": action, "reward": reward}])[-window:]
            if transitions:
                next_x = np.zeros(
                    (max(service_counts), window * 4 + 1), dtype=np.float32
                )
                for k in range(n):
                    next_x[k] = encode(events, k, n, own_only=own_only, window=window)
                next_xs.append(next_x)
                masks.append(np.arange(max(service_counts)) < n)
                terminals.append(t == 95)
    if transitions:
        return (
            np.stack(xs),
            np.array(ys, dtype=np.float32),
            np.stack(next_xs),
            np.stack(masks),
            np.array(terminals),
        )
    return np.stack(xs), np.array(ys, dtype=np.float32)


class LearnedControl:
    def __init__(self, device="cpu", own_only=False, window=WINDOW, probabilities=True):
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(window * 4 + 1, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 1),
        ).to(device)
        self.device = device
        self.update_steps = 0
        self.own_only = own_only
        self.window = window
        self.probabilities = probabilities
        self.policy.candidate_context = "joint"

    @torch.no_grad()
    def action_results_structured(self, observations, deterministic=True):
        result = []
        for o in observations:
            x = np.stack(
                [
                    encode(
                        o.metadata["llm"]["history"],
                        k,
                        len(o.candidate_ids),
                        own_only=self.own_only,
                        window=self.window,
                    )
                    for k in o.candidate_ids
                ]
            )
            values = self.policy(torch.tensor(x, device=self.device)).squeeze(-1)
            if self.probabilities:
                values = values.sigmoid()
            values = values.cpu().tolist()
            result.append(
                SimpleNamespace(
                    candidate_values=tuple(values),
                    candidate_id=o.candidate_ids[int(np.argmax(values))],
                )
            )
        return result


def run(args):
    if min(args.trials, args.updates, args.history) < 1 or args.eval_trials < 2:
        raise ValueError(
            "positive training budgets/window and at least two evaluation worlds required"
        )
    if min(args.services) < 2:
        raise ValueError("training requires at least two services")
    if not 0 <= args.gamma < 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    arrays = collect_arrays(
        args.seed,
        args.trials,
        args.own_only,
        args.history,
        transitions=args.objective == "td",
        service_counts=args.services,
    )
    x, y = arrays[:2]
    agent = LearnedControl(
        args.device,
        args.own_only,
        args.history,
        probabilities=args.objective == "reward",
    )
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=0.001)
    x = torch.tensor(x, device=args.device)
    y = torch.tensor(y, device=args.device)
    if args.objective == "td":
        next_x, masks, terminals = [
            torch.tensor(a, device=args.device) for a in arrays[2:]
        ]
        target = deepcopy(agent.policy).eval()
    generator = np.random.default_rng(args.seed + 1100)
    for step in range(args.updates):
        indices = torch.tensor(generator.integers(len(x), size=256), device=args.device)
        logits = agent.policy(x[indices]).squeeze(-1)
        if args.objective == "td":
            with torch.no_grad():
                next_values = (
                    agent.policy(next_x[indices])
                    .squeeze(-1)
                    .masked_fill(~masks[indices], -1e9)
                )
                actions = next_values.argmax(-1)
                target_values = (
                    target(next_x[indices])
                    .squeeze(-1)
                    .gather(1, actions[:, None])
                    .squeeze(-1)
                )
                labels = (
                    y[indices]
                    + args.gamma * (~terminals[indices]).float() * target_values
                )
            loss = torch.nn.functional.smooth_l1_loss(logits, labels)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[indices]
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.objective == "td":
            with torch.no_grad():
                for dst, src in zip(target.parameters(), agent.policy.parameters()):
                    dst.lerp_(src, 0.01)
        agent.update_steps += 1
        if (step + 1) % 500 == 0:
            print(
                json.dumps({"update": step + 1, "loss": float(loss.detach())}),
                flush=True,
            )
    torch.save(agent.policy.state_dict(), out / "policy.pt")
    results = {}
    # Previously inspected cases are now development cases, never fresh holdouts.
    for j, (name, task) in enumerate(CONFIRM.items()):
        worlds = [
            world(args.seed + 100000 + j, task, i, renamed=True)
            for i in range(args.eval_trials)
        ]
        results[name] = {}
        for method in (
            "language",
            "last_only",
            "no_history",
            "last_outcome",
            "window_mean",
            "window_ucb",
            "random",
        ):
            results[name][method] = evaluate(
                agent if method in ("language", "last_only", "no_history") else None,
                worlds,
                method,
                history_window=(
                    args.history
                    if method in ("language", "last_only", "no_history")
                    else WINDOW
                ),
            )
        print(
            json.dumps(
                {
                    "task": name,
                    "reward": {
                        k: round(v["mean_reward"], 4) for k, v in results[name].items()
                    },
                }
            ),
            flush=True,
        )
    (out / "result.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=5101)
    p.add_argument("--trials", type=int, default=1024)
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--eval-trials", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--own-only", action="store_true")
    p.add_argument("--history", type=int, default=WINDOW)
    p.add_argument("--objective", choices=["reward", "td"], default="reward")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--services", type=int, nargs="+", default=[3, 5])
    run(p.parse_args())
