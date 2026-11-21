"""Controlled PPO input diagnostic with repeated training worlds and fresh evaluation.

Run from the repository root with python -m examples.representation_benchmark.
"""

import argparse
from dataclasses import asdict
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
    rollout,
    trainable_digest,
)
from jormungandr.structured_export import export_structured_policy_bundle


MODES = ("raw", "summary", "both")


def run(args):
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be distinct and nonnegative")
    if (
        min(args.rounds, args.train_trials, args.eval_trials, args.history) < 1
        or args.steps < 16
    ):
        raise ValueError("positive budgets and at least 16 steps required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "configuration.json").open("x") as f:
        json.dump(vars(args), f, indent=2)
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for source in [
        Path(__file__),
        root / "examples/adaptation_benchmark.py",
        *sorted((root / "src").rglob("*.py")),
    ]:
        relative = source.relative_to(root)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "sources": hashes,
                "versions": {
                    name: metadata.version(name)
                    for name in ("torch", "transformers", "peft", "numpy")
                },
                "device": (
                    torch.cuda.get_device_name()
                    if args.device.startswith("cuda")
                    else args.device
                ),
            },
            indent=2,
        )
    )
    torch.set_num_threads(2)
    config = dict(
        llm_model=args.model,
        llm_revision=args.revision,
        llm_dtype=args.dtype,
        llm_max_tokens=args.max_tokens,
        llm_forward_batch_size=3,
        llm_lora_rank=4,
        llm_lora_alpha=8,
        epochs=1,
        minibatch_size=1,
        lr=1e-4,
        gamma=0.0,
        value_backbone_gradient_scale=0.0,
    )
    results = []
    for seed in seeds:
        training = [
            make_scenario(seed, 1, i, args.steps) for i in range(args.train_trials)
        ]
        heldout = [
            make_scenario(seed, 2, i, args.steps) for i in range(args.eval_trials)
        ]
        (output / f"worlds-{seed}.json").write_text(
            json.dumps([s.audit() for s in training + heldout], indent=2)
        )
        initial = None
        for mode in MODES:
            torch.manual_seed(seed)
            agent = algorithm_registry.get("llm_ppo").build_structured(
                SPEC, config, args.device
            )
            digest = trainable_digest(agent)
            if initial is not None and digest != initial:
                raise AssertionError("input modes did not start with identical weights")
            initial = digest
            record = dict(
                seed=seed,
                representation=mode,
                initial_digest=digest,
                learning=[],
                evaluation={},
            )
            with (output / f"trace-{seed}-{mode}.jsonl").open("x") as trace:

                def assess(label, worlds, history):
                    record["evaluation"][label] = evaluate(
                        agent,
                        worlds,
                        method="llm",
                        history=history,
                        seed=seed,
                        representation=mode,
                        trace=trace,
                        stage=label,
                    )

                assess("fit_before", training, args.history)
                for update in range(args.rounds):
                    trajectories = []
                    for scenario in training:
                        episodes, _ = rollout(
                            scenario,
                            method="llm",
                            agent=agent,
                            history=args.history,
                            seed=seed + update,
                            split="train",
                            representation=mode,
                            trace=trace,
                            stage=f"fit_{update}",
                        )
                        trajectories.extend(episodes)
                    metrics = asdict(agent.update_structured(trajectories))
                    record["learning"].append(metrics)
                    print(
                        json.dumps(
                            dict(
                                seed=seed,
                                mode=mode,
                                update=update + 1,
                                loss=metrics["loss"],
                            )
                        ),
                        flush=True,
                    )
                assess("fit_after", training, args.history)
                assess("heldout", heldout, args.history)
                assess("heldout_no_history", heldout, 0)
                for label, worlds in (
                    ("fit_rolling", training),
                    ("heldout_rolling", heldout),
                ):
                    record["evaluation"][label] = evaluate(
                        None, worlds, method="rolling", history=args.history, seed=seed
                    )
            record["final_digest"] = trainable_digest(agent)
            export_structured_policy_bundle(
                output / f"policy-{seed}-{mode}",
                agent=agent,
                spec=SPEC,
                algorithm="llm_ppo",
                agent_config=config,
            )
            results.append(record)
            (output / f"result-{seed}-{mode}.json").write_text(
                json.dumps(record, indent=2)
            )
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for label in results[0]["evaluation"]:
            summary[mode][label] = {}
            for metric in (
                "mean_reward",
                "mean_expected_regret",
                "optimal_action_fraction",
            ):
                values = [
                    float(np.mean([t[metric] for t in r["evaluation"][label]]))
                    for r in results
                    if r["representation"] == mode
                ]
                summary[mode][label][metric] = dict(
                    mean=float(np.mean(values)), seed_means=values
                )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="101,211,307")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--train-trials", type=int, default=2)
    p.add_argument("--eval-trials", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--max-tokens", type=int, default=1536)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
