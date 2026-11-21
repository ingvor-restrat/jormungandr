"""Train and ablate a language/numeric hybrid on causal route feedback."""

import argparse
from dataclasses import replace
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from examples.adaptation_benchmark import (
    SPEC,
    algorithm_registry,
    evaluate,
    make_scenario,
    trainable_digest,
)
from examples.learning_diagnostic import (
    teacher_data,
    teach,
    imitation_metrics,
    train_q,
    QView,
)
from jormungandr.structured_export import export_structured_policy_bundle


class ComponentView:
    def __init__(self, agent, language=1.0, numeric=1.0):
        self.agent, self.policy = agent, agent.policy
        self.language, self.numeric = language, numeric

    @property
    def update_steps(self):
        return self.agent.update_steps

    def score_results_structured(self, observations):
        was_training = self.policy.training
        self.policy.eval()
        with torch.no_grad():
            output = self.policy(
                self.agent._collate_observations(observations),
                language_scale=self.language,
                numeric_scale=self.numeric,
            )
        self.policy.train(was_training)
        return [
            SimpleNamespace(
                candidate_logits=logits[: len(obs.candidate_ids)].cpu().tolist(),
                value=float(value),
            )
            for logits, value, obs in zip(output.logits, output.values, observations)
        ]


def mirror_examples(data):
    """Balance route identity without inventing rewards or revealing outcomes."""
    augmented = list(data)
    for obs, target in data:
        candidates = obs.candidate_features[::-1].copy()
        candidates[:, 0] = np.arange(2)
        meta = dict(obs.metadata)
        meta["llm"] = dict(
            meta["llm"],
            candidate_descriptions={
                key: f"Call tool {i}. Cost {candidates[i,1]:.4f}; latency {candidates[i,2]:.4f}."
                for i, key in enumerate(obs.candidate_ids)
            },
        )
        augmented.append(
            (
                replace(
                    obs,
                    entity_features=obs.entity_features[::-1].copy(),
                    candidate_features=candidates,
                    metadata=meta,
                ),
                target[::-1].copy(),
            )
        )
    return augmented


def warmup_numeric(agent, data, epochs, seed):
    """Teach structured features first, without language forward/backward passes."""
    rng = np.random.default_rng(seed)
    branch = agent.policy.numeric_branch
    losses = []
    for _ in range(epochs):
        values = []
        branch.train()
        order = rng.permutation(len(data))
        for start in range(0, len(data), 16):
            examples = [data[i] for i in order[start : start + 16]]
            if any(o.metadata["llm"]["split"] != "train" for o, _ in examples):
                raise ValueError("warmup requires training context")
            batch = agent._collate_observations([o for o, _ in examples]).numeric
            targets = torch.tensor(
                np.stack([p for _, p in examples]),
                dtype=torch.float32,
                device=agent.device,
            )
            loss = (
                -(
                    targets
                    * torch.log_softmax(
                        agent.policy.numeric_logit_scale * branch(batch).logits, -1
                    )
                )
                .sum(-1)
                .mean()
            )
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
            agent.optimizer.step()
            agent.update_steps += 1
            values.append(float(loss.detach()))
        losses.append(float(np.mean(values)))
    return losses


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    if (
        len(set(seeds)) != len(seeds)
        or min(seeds) < 0
        or min(args.rounds, args.train_trials, args.eval_trials) < 1
        or args.steps < 16
    ):
        raise ValueError("invalid seeds or budgets")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "configuration.json").open("x") as f:
        json.dump(vars(args), f, indent=2)
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for source in [
        Path(__file__),
        root / "examples/learning_diagnostic.py",
        root / "examples/adaptation_benchmark.py",
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
                versions={
                    n: metadata.version(n)
                    for n in ("torch", "numpy", "transformers", "peft")
                },
            ),
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
        llm_numeric_branch=True,
        llm_numeric_lr=0.0001 if args.teacher == "dqn" else 0.001,
        llm_numeric_logit_scale=10.0 if args.teacher == "dqn" else 1.0,
        lr=0.0001,
        gamma=0.0,
        value_backbone_gradient_scale=0.0,
    )
    records = []
    for seed in seeds:
        training = [
            make_scenario(seed, 1, i, args.steps) for i in range(args.train_trials)
        ]
        heldout = [
            make_scenario(seed, 2, i, args.steps) for i in range(args.eval_trials)
        ]
        train_data = teacher_data(training, seed, args.history, "train")
        test_data = teacher_data(heldout, seed, args.history, "validation")
        (output / f"worlds-{seed}.json").write_text(
            json.dumps([s.audit() for s in training + heldout], indent=2)
        )
        teacher = None
        teacher_losses = None
        if args.teacher == "dqn":
            torch.manual_seed(seed)
            teacher = algorithm_registry.get("structured_dqn").build_structured(
                SPEC,
                dict(
                    structured_model_dim=32,
                    structured_heads=4,
                    structured_layers=1,
                    lr=0.001,
                    gamma=0.0,
                ),
                "cpu",
            )
            with (output / f"teacher-trace-{seed}.jsonl").open("x") as trace:
                teacher_losses = train_q(teacher, training, args, seed, trace)

            def relabel(data):
                scores = teacher.action_results_structured(
                    [o for o, _ in data], deterministic=True
                )
                targets = torch.softmax(
                    10.0 * torch.tensor([r.candidate_values for r in scores]), -1
                ).numpy()
                return [(o, p) for (o, _), p in zip(data, targets)]

            train_data, test_data = relabel(train_data), relabel(test_data)
        torch.manual_seed(seed)
        agent = algorithm_registry.get("llm_ppo").build_structured(
            SPEC, config, args.device
        )
        if teacher is not None:
            agent.initialize_numeric_actor_from_state(teacher.state_dict())
        record = dict(
            seed=seed,
            initial_digest=trainable_digest(agent),
            imitation_before=imitation_metrics(agent, train_data),
        )
        record["teacher_losses"] = teacher_losses
        if args.mirror:
            train_data = mirror_examples(train_data)
        record["numeric_warmup_losses"] = warmup_numeric(
            agent, train_data, args.warmup_epochs, seed
        )
        record["after_warmup"] = imitation_metrics(agent, train_data)
        record["losses"] = teach(agent, train_data, args.rounds, seed)
        record["imitation_after"] = imitation_metrics(agent, train_data)
        record["imitation_heldout"] = imitation_metrics(agent, test_data)
        record["evaluation"] = {}
        with (output / f"trace-{seed}.jsonl").open("x") as trace:
            for label, view, history in [
                ("hybrid", agent, args.history),
                ("no_history", agent, 0),
                ("numeric_only", ComponentView(agent, language=0), args.history),
                ("language_only", ComponentView(agent, numeric=0), args.history),
            ]:
                record["evaluation"][label] = evaluate(
                    view,
                    heldout,
                    method="llm",
                    representation="summary",
                    history=history,
                    seed=seed,
                    trace=trace,
                    stage=label,
                )
            if teacher is not None:
                record["evaluation"]["dqn_teacher"] = evaluate(
                    QView(teacher),
                    heldout,
                    method="numeric",
                    representation="summary",
                    history=args.history,
                    seed=seed,
                    trace=trace,
                    stage="dqn_teacher",
                )
            for baseline in ("random", "rolling"):
                record["evaluation"][baseline] = evaluate(
                    None, heldout, method=baseline, history=args.history, seed=seed
                )
        record["final_digest"] = trainable_digest(agent)
        export_structured_policy_bundle(
            output / f"policy-{seed}",
            agent=agent,
            spec=SPEC,
            algorithm="llm_ppo",
            agent_config=config,
        )
        (output / f"result-{seed}.json").write_text(json.dumps(record, indent=2))
        records.append(record)
        print(
            json.dumps(
                dict(
                    seed=seed,
                    imitation=record["imitation_heldout"],
                    accuracy={
                        label: float(
                            np.mean([t["optimal_action_fraction"] for t in trials])
                        )
                        for label, trials in record["evaluation"].items()
                    },
                )
            ),
            flush=True,
        )
        del agent, view
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {}
    for label in records[0]["evaluation"]:
        summary[label] = {}
        for metric in (
            "mean_reward",
            "mean_expected_regret",
            "optimal_action_fraction",
        ):
            values = [
                float(np.mean([t[metric] for t in record["evaluation"][label]]))
                for record in records
            ]
            summary[label][metric] = {
                "mean": float(np.mean(values)),
                "seed_means": values,
            }
    differences = {
        metric: summary["hybrid"][metric]["mean"] - summary["rolling"][metric]["mean"]
        for metric in summary["hybrid"]
    }
    result = {
        "summary": summary,
        "hybrid_minus_rolling": differences,
        "aggregate_target_met": differences["mean_reward"] >= 0
        and differences["mean_expected_regret"] <= 0
        and differences["optimal_action_fraction"] >= 0,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="701,809,907")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--teacher", choices=("rolling", "dqn"), default="rolling")
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--train-trials", type=int, default=2)
    p.add_argument("--eval-trials", type=int, default=12)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--max-tokens", type=int, default=1536)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
