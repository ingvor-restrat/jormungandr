"""Compare repeated and fresh training worlds with fixed DQN interaction/update budgets."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from examples.adaptation_benchmark import (
    SPEC,
    algorithm_registry,
    evaluate,
    make_scenario,
    trainable_digest,
)
from examples.learning_diagnostic import QView, train_q
from jormungandr.structured_export import export_structured_policy_bundle


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or min(seeds) < 0
        or args.rounds < 2
        or min(args.train_trials, args.eval_trials, args.history) < 1
        or args.steps < 16
    ):
        raise ValueError(
            "distinct nonnegative seeds, at least two rounds, positive budgets and steps >=16 required"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "configuration.json").open("x") as f:
        json.dump(vars(args), f, indent=2)
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for source in [
        Path(__file__),
        root / "examples/adaptation_benchmark.py",
        root / "examples/learning_diagnostic.py",
        *sorted((root / "src").rglob("*.py")),
    ]:
        rel = source.relative_to(root)
        target = output / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(rel)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "provenance.json").write_text(
        json.dumps(
            dict(
                sources=hashes,
                versions={n: metadata.version(n) for n in ("torch", "numpy")},
            ),
            indent=2,
        )
    )
    torch.set_num_threads(2)
    records = []
    for seed in seeds:
        schedule = [
            [
                make_scenario(seed, 1, r * args.train_trials + i, args.steps)
                for i in range(args.train_trials)
            ]
            for r in range(args.rounds)
        ]
        heldout = [
            make_scenario(seed, 2, i, args.steps) for i in range(args.eval_trials)
        ]
        (output / f"worlds-{seed}.json").write_text(
            json.dumps(
                [s.audit() for batch in schedule for s in batch]
                + [s.audit() for s in heldout],
                indent=2,
            )
        )
        baseline = evaluate(
            None, heldout, method="rolling", history=args.history, seed=seed
        )
        initial = None
        for mode in ("repeated", "fresh"):
            torch.manual_seed(seed)
            config = dict(
                structured_model_dim=32,
                structured_heads=4,
                structured_layers=1,
                lr=0.001,
                gamma=0.0,
            )
            agent = algorithm_registry.get("structured_dqn").build_structured(
                SPEC, config, "cpu"
            )
            view = QView(agent)
            digest = trainable_digest(view)
            if initial is not None and initial != digest:
                raise AssertionError("initial weights differ")
            initial = digest
            worlds_by_round = (
                schedule if mode == "fresh" else [schedule[0]] * args.rounds
            )
            record = dict(
                seed=seed,
                mode=mode,
                initial_digest=digest,
                training_world_ids=[
                    [s.identifier for s in batch] for batch in worlds_by_round
                ],
                interactions=args.rounds * args.train_trials * args.steps,
                evaluation={"rolling": baseline},
            )
            with (output / f"trace-{seed}-{mode}.jsonl").open("x") as trace:
                record["losses"] = train_q(
                    agent,
                    schedule[0],
                    args,
                    seed,
                    trace,
                    world_schedule=worlds_by_round,
                )
                for label, epsilon, history in [
                    ("sampled", 0.1, args.history),
                    ("greedy", 0.0, args.history),
                    ("no_history", 0.1, 0),
                ]:
                    record["evaluation"][label] = evaluate(
                        QView(agent, epsilon),
                        heldout,
                        method="numeric",
                        representation="summary",
                        history=history,
                        seed=seed,
                        trace=trace,
                        stage=label,
                    )
            record["updates"] = agent.update_steps
            record["final_digest"] = trainable_digest(view)
            assert record["updates"] == record["interactions"]
            export_structured_policy_bundle(
                output / f"policy-{seed}-{mode}",
                agent=agent,
                spec=SPEC,
                algorithm="structured_dqn",
                agent_config=config,
            )
            (output / f"result-{seed}-{mode}.json").write_text(
                json.dumps(record, indent=2)
            )
            records.append(record)
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        mode=mode,
                        accuracy={
                            k: float(np.mean([t["optimal_action_fraction"] for t in v]))
                            for k, v in record["evaluation"].items()
                        },
                    )
                ),
                flush=True,
            )
    summary = {}
    for mode in ("repeated", "fresh"):
        subset = [r for r in records if r["mode"] == mode]
        summary[mode] = {}
        for label in subset[0]["evaluation"]:
            summary[mode][label] = {}
            for metric in (
                "optimal_action_fraction",
                "mean_reward",
                "mean_expected_regret",
            ):
                values = [
                    float(np.mean([t[metric] for t in r["evaluation"][label]]))
                    for r in subset
                ]
                summary[mode][label][metric] = dict(
                    mean=float(np.mean(values)), seed_means=values
                )
    differences = {
        m: np.subtract(
            summary["fresh"]["sampled"][m]["seed_means"],
            summary["repeated"]["sampled"][m]["seed_means"],
        ).tolist()
        for m in summary["fresh"]["sampled"]
    }
    (output / "summary.json").write_text(
        json.dumps(
            dict(summary=summary, paired_fresh_minus_repeated=differences), indent=2
        )
    )
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="1709,1801,1901")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--train-trials", type=int, default=2)
    p.add_argument("--eval-trials", type=int, default=12)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--history", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
