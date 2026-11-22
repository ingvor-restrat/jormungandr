"""Post-hoc information audit; does not select or update the learned model.

Rotate the hidden winning service while holding all public inputs and actor
exploration randomness fixed. History-free decisions must average exactly 1/N
in deterministic tasks, removing finite-cohort identifier luck.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from examples.multiservice_adaptation import CONFIRM_REQUEST, Task, evaluate, world
from jormungandr.structured_export import load_structured_policy_bundle


def balanced_worlds(seed, services, groups=4, steps=32):
    result = []
    for index in range(groups):
        original = world(
            seed + 900000, Task(services, 1.0, 0.0, 0, steps), index, renamed=True
        )
        for rotation in range(services):
            best = (original.best + rotation) % services
            probabilities = np.zeros_like(original.probabilities)
            probabilities[np.arange(steps), best] = 1.0
            result.append(
                replace(
                    original,
                    identifier=f"{original.identifier}:rotation:{rotation}",
                    best=best,
                    probabilities=probabilities,
                    outcomes=probabilities.astype(bool),
                    request=CONFIRM_REQUEST,
                    ids=tuple(key.replace("endpoint", "api") for key in original.ids),
                )
            )
    return result


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    agent, manifest = load_structured_policy_bundle(args.bundle, device=args.device)
    agent.q.forward_batch_size = 16
    results = {}
    for services in (4, 8):
        worlds = balanced_worlds(args.seed, services)
        result = evaluate(
            agent, worlds, "no_history", out / f"{services}-services.jsonl"
        )
        assert abs(result["mean_reward"] - 1 / services) < 1e-12
        results[str(services)] = result
    record = {
        "post_hoc_information_audit": True,
        "training_or_model_selection": False,
        "seed": args.seed,
        "bundle_sha256": manifest["artifact"]["sha256"],
        "explanation": "Hidden winners rotate; public inputs and exploration tapes remain identical within each group.",
        "results": results,
    }
    (out / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({n: r["mean_reward"] for n, r in results.items()}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
