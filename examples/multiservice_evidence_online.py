"""Native language DQN refinement using its own causal training interactions.

Only training worlds generate replay. Rewards are actual chosen-tool outcomes;
no best-action targets or evaluation outcomes enter an update.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch

from examples.multiservice_adaptation import (
    TRAIN,
    CONFIRM,
    world,
    context,
    observe,
    append,
    select,
    collect,
    evaluate,
)
from examples.multiservice_evidence import evidence_view, EvidencePolicy
from examples.adaptation_benchmark import SPEC
from jormungandr.structured_replay import StructuredReplayTransition
from jormungandr.structured_export import (
    export_structured_policy_bundle,
    load_structured_policy_bundle,
)


def collect_policy(agent, seed, trials, *, evidence_format="v1"):
    worlds = [world(seed, TRAIN[i % len(TRAIN)], i, "train") for i in range(trials)]
    contexts = [context(w) for w in worlds]
    replay = []
    wrapped = EvidencePolicy(agent, evidence_format=evidence_format)
    for t in range(TRAIN[0].steps):
        observations = [observe(w, t, c) for w, c in zip(worlds, contexts)]
        results = []
        for start in range(0, len(worlds), 4):
            results.extend(
                wrapped.action_results_structured(observations[start : start + 4])
            )
        for i, (w, c, obs, result) in enumerate(
            zip(worlds, contexts, observations, results)
        ):
            a, _ = select(w, t, result.candidate_values, obs.candidate_ids)
            reward = float(w.outcomes[t, a])
            nxt = append(c, t, w.ids[a], reward)
            replay.append(
                StructuredReplayTransition(
                    evidence_view(obs, evidence_format=evidence_format),
                    w.ids[a],
                    reward,
                    evidence_view(
                        observe(w, t + 1, nxt), evidence_format=evidence_format
                    ),
                    t == w.task.steps - 1,
                    c.actor_id,
                    "session",
                    t,
                    0,
                    split="train",
                )
            )
            contexts[i] = nxt
    return replay


def run(args):
    if (
        min(
            args.rounds,
            args.trials,
            args.updates,
            args.batch_size,
            args.microbatch_size,
        )
        < 1
        or args.eval_trials < 2
    ):
        raise ValueError("positive collection and update budgets required")
    if args.warmup_updates < 0:
        raise ValueError("warmup_updates must be nonnegative")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    agent, manifest = load_structured_policy_bundle(args.bundle, device=args.device)
    cfg = dict(manifest["agent_config"])
    cfg["example_evidence_format"] = args.evidence_format
    cfg["structured_microbatch_size"] = args.microbatch_size
    cfg["llm_forward_batch_size"] = 2 * args.microbatch_size
    agent.microbatch_size = args.microbatch_size
    if cfg.get("llm_candidate_context") != "evidence":
        raise ValueError("an evidence-context bundle is required")
    agent.q.forward_batch_size = 32
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "config": cfg,
                "source_manifest": manifest,
                "replay": "75% latest on-policy training collection, 25% initial behavior collection; actual rewards only",
                "optimizer": "fresh optimizer after loading deployment bundle",
            },
            indent=2,
        )
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    initial = [
        replace(
            r,
            observation=evidence_view(
                r.observation, evidence_format=args.evidence_format
            ),
            next_observation=evidence_view(
                r.next_observation, evidence_format=args.evidence_format
            ),
        )
        for r in collect(args.seed, 72)
    ]
    rng = np.random.default_rng(args.seed + 2000)
    losses = []
    agent.q.forward_batch_size = cfg["llm_forward_batch_size"]
    for step in range(args.warmup_updates):
        batch = [initial[i] for i in rng.integers(len(initial), size=args.batch_size)]
        result = agent.update_structured(batch)
        losses.append(float(result.loss))
        if (step + 1) % 64 == 0:
            print(
                json.dumps({"warmup_update": step + 1, "loss": float(result.loss)}),
                flush=True,
            )
    agent.q.forward_batch_size = 32
    for cycle in range(args.rounds):
        recent = collect_policy(
            agent,
            args.seed + args.collection_offset + cycle,
            args.trials,
            evidence_format=args.evidence_format,
        )
        print(
            json.dumps(
                {
                    "round": cycle + 1,
                    "collected": len(recent),
                    "collection_reward": float(np.mean([r.reward for r in recent])),
                }
            ),
            flush=True,
        )
        agent.q.forward_batch_size = cfg["llm_forward_batch_size"]
        for step in range(args.updates):
            batch = []
            for _ in range(args.batch_size):
                pool = recent if rng.random() < 0.75 else initial
                batch.append(pool[int(rng.integers(len(pool)))])
            result = agent.update_structured(batch)
            losses.append(float(result.loss))
            if (step + 1) % 64 == 0:
                print(
                    json.dumps(
                        {
                            "round": cycle + 1,
                            "update": step + 1,
                            "loss": float(result.loss),
                        }
                    ),
                    flush=True,
                )
        export_structured_policy_bundle(
            out / f"round-{cycle+1}",
            agent=agent,
            spec=SPEC,
            algorithm="llm_dqn",
            agent_config=cfg,
        )
        agent.q.forward_batch_size = 32
        (out / "losses.json").write_text(json.dumps(losses))
    wrapped = EvidencePolicy(agent, evidence_format=args.evidence_format)
    results = {}
    for j, (name, task) in enumerate(CONFIRM.items()):
        worlds = [
            world(args.seed + 100000 + j, task, i, renamed=True)
            for i in range(args.eval_trials)
        ]
        results[name] = {
            method: evaluate(wrapped if method == "language" else None, worlds, method)
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=5101)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--trials", type=int, default=36)
    p.add_argument("--updates", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--microbatch-size", type=int, default=16)
    p.add_argument("--evidence-format", choices=["v1", "counts_v2"], default="v1")
    p.add_argument("--warmup-updates", type=int, default=0)
    p.add_argument("--collection-offset", type=int, default=10000)
    p.add_argument("--eval-trials", type=int, default=12)
    p.add_argument("--device", default="cpu")
    run(p.parse_args())
