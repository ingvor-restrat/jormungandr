"""Post-hoc greedy rollout comparison for frozen hybrid checkpoints.

No fitting or checkpoint selection: changed actions create fresh causal histories.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from examples.adaptation_benchmark import evaluate, make_scenario
from jormungandr.structured_export import load_structured_policy_bundle


class GreedyView:
    def __init__(self, agent):
        self.agent, self.policy = agent, agent.policy

    @property
    def update_steps(self):
        return self.agent.update_steps

    def score_results_structured(self, observations):
        results = []
        for score in self.agent.score_results_structured(observations):
            scores = np.asarray(score.candidate_logits)
            winners = np.flatnonzero(scores == scores.max())
            logits = np.full(len(scores), -1e9)
            logits[winners] = 0.0
            results.append(SimpleNamespace(candidate_logits=logits, value=score.value))
        return results


def run(source, output, device):
    source, output = Path(source), Path(output)
    config = json.loads((source / "configuration.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "configuration.json").open("x") as f:
        json.dump(
            dict(
                source=str(source), device=device, experiment="frozen_greedy_evaluation"
            ),
            f,
            indent=2,
        )
    torch.set_num_threads(2)
    for seed in [int(x) for x in config["seeds"].split(",")]:
        agent, _ = load_structured_policy_bundle(
            source / f"policy-{seed}", device=device
        )
        worlds = [
            make_scenario(seed, 2, i, config["steps"])
            for i in range(config["eval_trials"])
        ]
        with (output / f"trace-{seed}.jsonl").open("x") as trace:
            result = evaluate(
                GreedyView(agent),
                worlds,
                method="llm",
                representation="summary",
                history=config.get("history", 4),
                seed=seed,
                trace=trace,
                stage="greedy",
            )
        (output / f"result-{seed}.json").write_text(
            json.dumps(dict(seed=seed, greedy=result), indent=2)
        )
        print(
            json.dumps(
                dict(
                    seed=seed,
                    accuracy=float(
                        np.mean([r["optimal_action_fraction"] for r in result])
                    ),
                )
            ),
            flush=True,
        )
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    run(a.source, a.output, a.device)
