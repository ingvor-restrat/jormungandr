"""Frozen evaluation of compact-evidence language bundles.

Use development first. Confirmation consumes fresh cases; failed confirmation
must be recorded and cannot subsequently be represented as untouched evidence.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch

from examples.multiservice_adaptation import (
    CONFIRM,
    Task,
    world,
    evaluate,
    context,
    append,
    observe,
)
from examples.multiservice_evidence import EvidencePolicy
from examples.multiservice_balanced_control import balanced_worlds
from examples.multiservice_acceptance import TASKS, assess
from jormungandr.structured_export import load_structured_policy_bundle


def critical_history_probe(agent, *, evidence_format="v1"):
    """Public hypothetical histories; diagnostic scores, never training labels."""
    policy = EvidencePolicy(agent, evidence_format=evidence_format)
    report = {}
    for n in (4, 8):
        w = world(78, Task(n, 1.0, 0.0, 0), 0)
        for name, rewards in (
            ("empty", []),
            ("one_success", [1]),
            ("one_failure", [0]),
            ("streak", [1] * 16),
            ("failure_after_streak", [1] * 15 + [0]),
            ("two_failures_after_streak", [1] * 14 + [0, 0]),
        ):
            c = context(w)
            for t, r in enumerate(rewards):
                c = append(c, t, w.ids[0], r)
            obs = observe(w, len(rewards), c)
            result = policy.action_results_structured([obs])[0]
            scores = dict(zip(obs.candidate_ids, result.candidate_values))
            report[f"{n}_{name}"] = {
                "observed": scores[w.ids[0]],
                "untried": scores[w.ids[1]],
            }
    return report


def run(args):
    if min(args.inference_batch_size, args.observation_batch_size) < 1:
        raise ValueError("inference batch sizes must be positive")
    if args.trials < 2 or (args.stage == "confirmation" and args.trials < 32):
        raise ValueError("at least two development or 32 confirmation worlds required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source = Path(args.bundle)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["agent_config"].get("llm_candidate_context") != "evidence":
        raise ValueError("this actor requires an evidence-context bundle")
    tasks = TASKS if args.stage == "confirmation" else CONFIRM
    protocol = {
        "args": vars(args),
        "tasks": {k: asdict(v) for k, v in tasks.items()},
        "bundle_manifest": manifest,
        "sources": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "examples/multiservice_evidence_validation.py",
                "examples/multiservice_evidence.py",
                "examples/multiservice_acceptance.py",
                "examples/multiservice_adaptation.py",
                "src/jormungandr/algorithms/llm_ppo.py",
                "src/jormungandr/algorithms/llm_dqn.py",
                "src/jormungandr/algorithms/structured_dqn.py",
            )
        },
        "role": "frozen language evaluation; numeric controls are not the actor",
    }
    training_record = source.parent / "protocol.json"
    if training_record.exists():
        protocol["training_run"] = json.loads(training_record.read_text())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    agent, _ = load_structured_policy_bundle(source, device=args.device)
    agent.q.forward_batch_size = args.inference_batch_size
    evidence_format = manifest["agent_config"].get("example_evidence_format", "v1")
    wrapped = EvidencePolicy(agent, evidence_format=evidence_format)
    (out / "critical_histories.json").write_text(
        json.dumps(
            critical_history_probe(agent, evidence_format=evidence_format), indent=2
        )
    )
    results = {}
    for j, (name, task) in enumerate(tasks.items()):
        worlds = [
            world(args.seed + 900000 + j, task, i, renamed=True)
            for i in range(args.trials)
        ]
        results[name] = {
            method: evaluate(
                wrapped if method == "language" else None,
                worlds,
                method,
                observation_batch_size=args.observation_batch_size,
            )
            for method in ("language", "last_outcome", "window_mean", "window_ucb")
        }
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
    gate = assess(results)
    gate["stage"] = args.stage
    gate["confirmation_established"] = (
        False  # Requires independently trained second seed as well.
    )
    (out / "gate.json").write_text(json.dumps(gate, indent=2))
    # Matched causal ablations and actual tool execution on development cases.
    worlds = [
        world(args.seed + 950000, Task(4, 0.85, 0.15, 16, 48), i, renamed=True)
        for i in range(4)
    ]
    controls = {
        method: evaluate(wrapped, worlds, method)
        for method in ("language", "no_history", "last_only", "flipped_feedback")
    }
    controls["reordered"] = evaluate(wrapped, worlds, "language", reverse=True)
    controls["http"] = evaluate(
        wrapped, worlds, "language", out / "http.jsonl", http=True
    )
    controls["permutation_matches"] = controls["language"] == controls["reordered"]
    controls["http_matches"] = controls["language"] == controls["http"]
    controls["balanced_no_history"] = {}
    for services in (4, 8):
        result = evaluate(wrapped, balanced_worlds(args.seed, services), "no_history")
        if abs(result["mean_reward"] - 1 / services) > 1e-12:
            raise AssertionError("history-free information audit failed")
        controls["balanced_no_history"][str(services)] = result
    (out / "controls.json").write_text(json.dumps(controls, indent=2))
    if not controls["permutation_matches"] or not controls["http_matches"]:
        raise AssertionError("permutation or executable-tool equivalence failed")
    return gate


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--stage", choices=["development", "confirmation"], default="development"
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--inference-batch-size", type=int, default=32)
    p.add_argument("--observation-batch-size", type=int, default=4)
    result = run(p.parse_args())
    raise SystemExit(0 if result["passed"] else 1)
