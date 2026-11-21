"""Paired numeric DQN diagnostic: summary feedback versus ordered feedback."""

import argparse
from dataclasses import replace
from functools import partial
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
    observation,
)
from examples.learning_diagnostic import QView, train_q
from jormungandr.structured import StructuredPolicySpec
from jormungandr.structured_export import export_structured_policy_bundle


def ordered_observation(scenario, context, *, window=4, hide_order=False, **kwargs):
    """Append newest-first reward/observed pairs per route; never use latent state."""
    base = observation(scenario, context, **kwargs)
    extra = np.zeros((len(base.entity_ids), 2 * window))
    if not hide_order:
        for age, event in enumerate(reversed(context.events[-window:])):
            row = base.entity_ids.index(event.action)
            extra[row, 2 * age] = event.reward
            extra[row, 2 * age + 1] = 1.0
    return replace(
        base, entity_features=np.concatenate([base.entity_features, extra], axis=1)
    )


def run(args):
    if (
        min(args.rounds, args.train_trials, args.eval_trials, args.history) < 1
        or args.steps < 16
    ):
        raise ValueError("positive budgets and steps >=16 required")
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
    for seed in [int(s) for s in args.seeds.split(",")]:
        training = [
            make_scenario(seed, 1, i, args.steps) for i in range(args.train_trials)
        ]
        heldout = [
            make_scenario(seed, 2, i, args.steps) for i in range(args.eval_trials)
        ]
        for mode in ("summary", "ordered"):
            torch.manual_seed(seed)
            factory = (
                observation
                if mode == "summary"
                else partial(ordered_observation, window=args.history)
            )
            spec = (
                SPEC
                if mode == "summary"
                else StructuredPolicySpec(1, 3 + 2 * args.history, 3, 1)
            )
            config = dict(
                structured_model_dim=32,
                structured_heads=4,
                structured_layers=1,
                lr=0.001,
                gamma=0.0,
            )
            agent = algorithm_registry.get("structured_dqn").build_structured(
                spec, config, "cpu"
            )
            record = dict(seed=seed, mode=mode, evaluation={})
            with (output / f"trace-{seed}-{mode}.jsonl").open("x") as trace:
                record["losses"] = train_q(
                    agent, training, args, seed, trace, observation_factory=factory
                )
                for label, epsilon, history, builder in [
                    ("sampled", 0.1, args.history, factory),
                    ("greedy", 0.0, args.history, factory),
                    ("no_history", 0.1, 0, factory),
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
                        observation_factory=builder,
                    )
                if mode == "ordered":
                    record["evaluation"]["order_removed"] = evaluate(
                        QView(agent),
                        heldout,
                        method="numeric",
                        representation="summary",
                        history=args.history,
                        seed=seed,
                        trace=trace,
                        stage="order_removed",
                        observation_factory=partial(
                            ordered_observation, window=args.history, hide_order=True
                        ),
                    )
                record["evaluation"]["rolling"] = evaluate(
                    None, heldout, method="rolling", history=args.history, seed=seed
                )
            export_structured_policy_bundle(
                output / f"policy-{seed}-{mode}",
                agent=agent,
                spec=spec,
                algorithm="structured_dqn",
                agent_config=config,
            )
            (output / f"result-{seed}-{mode}.json").write_text(
                json.dumps(record, indent=2)
            )
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
            records.append(record)
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="1409,1511,1601")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--train-trials", type=int, default=2)
    p.add_argument("--eval-trials", type=int, default=12)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--history", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
