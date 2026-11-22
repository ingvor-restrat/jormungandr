"""Train language reward estimates on compact, identity-invariant public evidence.

Old confirmation cases are development data here. No oracle labels, numeric
actor, or policy recommendation is supplied to the language model.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch

from examples.multiservice_adaptation import collect, CONFIRM, evaluate, world
from examples.adaptation_benchmark import SPEC, algorithm_registry
from jormungandr.structured_export import export_structured_policy_bundle


def evidence_view(obs, *, evidence_format="v1"):
    if evidence_format not in {"v1", "counts_v2"}:
        raise ValueError("unknown example evidence format")
    raw = obs.metadata["llm"]
    descriptions = {}
    for key in obs.candidate_ids:
        events = raw["history"]
        own = [
            (len(events) - i, e["reward"])
            for i, e in enumerate(events)
            if e["action"] == key
        ]
        latest = "untried" if not own else ("success" if own[-1][1] else "failure")
        descriptions[key] = (
            f"Latest own result: {latest}. Own observations (age:reward): "
            + (", ".join(f"{age}:{int(r)}" for age, r in own) or "none")
            + ". All calls oldest to newest (self or other, success or failure): "
            + (
                ", ".join(
                    ("self" if e["action"] == key else "other")
                    + (" success" if e["reward"] else " failure")
                    for e in events
                )
                or "none"
            )
        )
        if evidence_format == "counts_v2":
            successes = int(sum(r for _, r in own))
            descriptions[key] = (
                f"Observed successes: {successes}; observed failures: {len(own)-successes}; "
                f"total observations: {len(own)}. " + descriptions[key]
            )
    return replace(
        obs,
        metadata={
            **obs.metadata,
            "llm": {
                **raw,
                "observation": f"Estimate this service's probability of completing the job. {len(obs.candidate_ids)} services available.",
                "candidate_descriptions": descriptions,
            },
        },
    )


class EvidencePolicy:
    def __init__(self, agent, *, evidence_format="v1"):
        self.agent = agent
        self.evidence_format = evidence_format

    @property
    def policy(self):
        return self.agent.policy

    @property
    def update_steps(self):
        return self.agent.update_steps

    def action_results_structured(self, observations, deterministic=True):
        return self.agent.action_results_structured(
            [
                evidence_view(o, evidence_format=self.evidence_format)
                for o in observations
            ],
            deterministic=deterministic,
        )


def run(args):
    if min(args.trials, args.updates, args.batch_size) < 1 or args.eval_trials < 2:
        raise ValueError(
            "positive training budgets and at least two evaluation worlds required"
        )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    cfg = dict(
        llm_model="HuggingFaceTB/SmolLM2-135M",
        llm_revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        llm_dtype="bfloat16",
        llm_max_tokens=512,
        llm_forward_batch_size=8,
        llm_attention_implementation="sdpa",
        llm_lora_rank=16,
        llm_lora_alpha=32,
        llm_candidate_context="evidence",
        lr=1e-4,
        structured_microbatch_size=8,
        gamma=0.0,
        demonstration_weight=0.0,
    )
    (out / "config.json").write_text(
        json.dumps({"args": vars(args), "config": cfg}, indent=2)
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    rows = [
        replace(
            r,
            observation=evidence_view(r.observation),
            next_observation=evidence_view(r.next_observation),
        )
        for r in collect(args.seed, args.trials)
    ]
    agent = algorithm_registry.get("llm_dqn").build_structured(SPEC, cfg, args.device)
    rng = np.random.default_rng(args.seed + 2200)
    losses = []
    for step in range(args.updates):
        batch = [rows[i] for i in rng.integers(len(rows), size=args.batch_size)]
        result = agent.update_structured(batch)
        losses.append(float(result.loss))
        if (step + 1) % 64 == 0:
            print(json.dumps({"update": step + 1, "loss": losses[-1]}), flush=True)
    (out / "losses.json").write_text(json.dumps(losses))
    export_structured_policy_bundle(
        out / "language", agent=agent, spec=SPEC, algorithm="llm_dqn", agent_config=cfg
    )
    agent.q.forward_batch_size = 32
    wrapped = EvidencePolicy(agent)
    results = {}
    for j, (name, task) in enumerate(CONFIRM.items()):
        worlds = [
            world(args.seed + 100000 + j, task, i, renamed=True)
            for i in range(args.eval_trials)
        ]
        results[name] = {
            method: evaluate(
                wrapped if method in ("language", "no_history", "last_only") else None,
                worlds,
                method,
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=5101)
    p.add_argument("--trials", type=int, default=216)
    p.add_argument("--updates", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-trials", type=int, default=12)
    p.add_argument("--device", default="cpu")
    run(p.parse_args())
