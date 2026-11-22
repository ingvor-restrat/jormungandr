"""Independent tiny reward regressor with explicit action/history binding."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from examples.context_adaptation_benchmark import collect, evaluate, make_trial


def features(observation):
    history = observation.metadata["llm"]["history"]
    if not history:
        return np.zeros((len(observation.candidate_ids), 3), dtype=np.float32)
    last = history[-1]
    return np.array(
        [
            [float(key == last["action"]), last["reward"], 1.0]
            for key in observation.candidate_ids
        ],
        dtype=np.float32,
    )


class Control:
    def __init__(self):
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(3, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
        )
        self.update_steps = 0

    @torch.no_grad()
    def action_results_structured(self, observations, *, deterministic=True):
        result = []
        for obs in observations:
            values = self.policy(torch.tensor(features(obs))).squeeze(-1)
            result.append(
                SimpleNamespace(
                    candidate_id=obs.candidate_ids[int(values.argmax())],
                    candidate_values=tuple(values.tolist()),
                )
            )
        return result


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    agent = Control()
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=0.003)
    replay = collect(args.seed, 128, 12, 1)
    rng = np.random.default_rng(args.seed + 18000)
    losses = []
    for _ in range(768):
        rows = [replay[i] for i in rng.integers(len(replay), size=4)]
        x = torch.tensor(
            np.stack([features(r.observation)[r.candidate_index] for r in rows])
        )
        y = torch.tensor([r.reward for r in rows])
        loss = torch.nn.functional.mse_loss(agent.policy(x).squeeze(-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        agent.update_steps += 1
        losses.append(float(loss.detach()))
    worlds = [make_trial(args.seed + 100000, 2, i, 12) for i in range(24)]
    result = {
        method: evaluate(agent, worlds, 1, out / f"{method}.jsonl", method=method)
        for method in ("history", "no_history", "flipped_feedback")
    }
    (out / "result.json").write_text(
        json.dumps(dict(seed=args.seed, losses=losses, evaluation=result), indent=2)
    )
    print(
        json.dumps(
            {
                k: {
                    m: v[m]
                    for m in (
                        "mean_reward",
                        "stable_after_first",
                        "post_change_after_one",
                    )
                }
                for k, v in result.items()
            }
        )
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=3109)
    run(p.parse_args())
