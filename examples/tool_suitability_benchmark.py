"""Independent tool suitability, abstention, and frozen numeric composition.

A generic simulated example, not an API executor or a calibrated safety gate.
"""

import argparse
from dataclasses import replace
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path
import re

import numpy as np
import torch

from examples.adaptation_benchmark import (
    SPEC,
    algorithm_registry,
    seeded,
    trainable_digest,
)
from examples.language_tool_benchmark import (
    IDS,
    TRAIN_REQUESTS,
    TEST_REQUESTS,
    TRAIN_DESCRIPTIONS,
    TEST_DESCRIPTIONS,
    ToolWorld,
    make_world,
    make_observation,
    append_feedback,
    epsilon_probabilities,
    evaluate_policy,
)
from examples.learning_diagnostic import QView
from jormungandr.llm_context import AgentContext
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_export import (
    export_structured_policy_bundle,
    load_structured_policy_bundle,
)

# Previous benchmark test wording is now DEVELOPMENT data, never a new holdout.
TRAIN_TEXT = tuple(
    TRAIN_REQUESTS[i]
    + TEST_REQUESTS[i]
    + (
        ("Do not summarize; translate this text.",)
        if i == 0
        else ("Do not translate; summarize this text.",)
    )
    for i in range(2)
) + (
    (
        "Check this text for spelling errors.",
        "Proofread this document.",
        "Correct grammatical mistakes in this passage.",
    ),
)
HOLDOUT_TEXT = (
    (
        "Rewrite the passage in Spanish.",
        "Express this document in German.",
        "I want another language, not a shorter version.",
    ),
    (
        "Shorten the passage while preserving its key ideas.",
        "Distill the document into a few main takeaways.",
        "I want fewer words, not another language.",
    ),
    (
        "Find typographical errors in this document.",
        "Repair the punctuation in this passage.",
        "Check whether the grammar is correct.",
    ),
)
TRAIN_DESC = tuple(TRAIN_DESCRIPTIONS[i] + TEST_DESCRIPTIONS[i] for i in range(2))
HOLDOUT_DESC = (
    (
        "Rewrites documents in a target language while preserving meaning.",
        "Renders prose in the language specified by the caller.",
    ),
    (
        "Creates a brief digest of a longer document.",
        "Extracts the key ideas into a shortened version of the input.",
    ),
)


def pair_observation(request, description, split="validation"):
    """Opaque service IDs, catalog order, and outcomes never enter the model."""
    context = AgentContext("suitability", "pair", split, max_events=0)
    return EntityCandidateObservation(
        global_features=np.zeros(1),
        entity_features=np.zeros((2, 3)),
        entity_type_ids=np.zeros(2, dtype=np.int64),
        entity_ids=("no", "yes"),
        candidate_features=np.zeros((2, 3)),
        candidate_ids=("no", "yes"),
        candidate_entity_indices=np.arange(2),
        legal_action_mask=np.ones(2, dtype=bool),
        metadata=context.metadata(
            f"Request: {request}\nTool description: {description}\n"
            "Does this tool perform the requested task? Respect exclusions in the request.",
            {
                "no": "This tool does not satisfy the request.",
                "yes": "This tool satisfies the request.",
            },
        ),
    )


def training_data():
    return [
        (
            pair_observation(request, description, "train"),
            np.array([float(kind != capability), float(kind == capability)]),
        )
        for kind, requests in enumerate(TRAIN_TEXT)
        for request in requests
        for capability, descriptions in enumerate(TRAIN_DESC)
        for description in descriptions
    ]


def score_pairs(agent, pairs):
    # Deduplicate only identical public inputs, never labels or hidden world state.
    unique = list(dict.fromkeys(pairs))
    scores = agent.score_results_structured([pair_observation(*p) for p in unique])
    values = {
        p: np.asarray(s.candidate_logits, dtype=float) for p, s in zip(unique, scores)
    }
    logits = np.array([values[p] for p in pairs])
    margins = logits[:, 1] - logits[:, 0]
    probabilities = 1 / (1 + np.exp(-np.clip(margins, -80, 80)))
    return logits, probabilities


def train_suitability(agent, data, epochs, seed, batch_size=8):
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(data))
        total = 0.0
        for start in range(0, len(data), batch_size):
            examples = [data[i] for i in order[start : start + batch_size]]
            if any(o.metadata["llm"]["split"] != "train" for o, _ in examples):
                raise ValueError("training observations required")
            agent.policy.train()
            output = agent.policy(agent._collate_observations([o for o, _ in examples]))
            target = torch.tensor(
                np.stack([y for _, y in examples]),
                dtype=torch.float32,
                device=agent.device,
            )
            loss = -(target * torch.log_softmax(output.logits, -1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite suitability loss")
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in agent.policy.parameters() if p.requires_grad], 1.0
            )
            agent.optimizer.step()
            agent.update_steps += 1
            total += float(loss.detach()) * len(examples)
        losses.append(total / len(data))
        print(
            json.dumps(dict(seed=seed, epoch=epoch + 1, training_loss=losses[-1])),
            flush=True,
        )
    return losses


def parser_accepts(request, description):
    """Declared lexical baseline; exclusions outside this grammar remain a limitation."""
    text = request.lower()
    text = re.sub(r"do not [^;]+;", "", text)
    text = text.split(", not ")[0]
    patterns = (
        r"translat|language|french|spanish|german",
        r"summar|short|condens|overview|main|key|digest|distill|fewer",
    )
    kinds = [i for i, p in enumerate(patterns) if re.search(p, text)]
    return len(kinds) == 1 and bool(re.search(patterns[kinds[0]], description.lower()))


def catalog_decision(ids, probabilities, threshold=0.5):
    if len(ids) != len(probabilities) or len(set(ids)) != len(ids):
        raise ValueError("aligned unique tool IDs required")
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not 0 < threshold < 1:
        raise ValueError("finite probabilities and threshold in (0,1) required")
    allowed = tuple(key for key, value in zip(ids, p) if value >= threshold)
    return {"action": "select" if allowed else "abstain", "eligible_ids": allowed}


def semantic_audit(agent, output):
    pairs, labels = [], []
    for kind, requests in enumerate(HOLDOUT_TEXT):
        for request in requests:
            for capability, descriptions in enumerate(HOLDOUT_DESC):
                for description in descriptions:
                    pairs.append((request, description))
                    labels.append(kind == capability)
    logits, p = score_pairs(agent, pairs)
    labels = np.asarray(labels)
    predicted = p >= 0.5
    parser = np.array([parser_accepts(*pair) for pair in pairs])
    records = [
        dict(
            request=r,
            description=d,
            expected=bool(y),
            logits=s.tolist(),
            suitability_score=float(v),
            accepted=bool(v >= 0.5),
            parser=bool(b),
        )
        for (r, d), y, s, v, b in zip(pairs, labels, logits, p, parser)
    ]
    (output / "semantic-traces.json").write_text(json.dumps(records, indent=2))
    # Controlled catalogs have 0..4 valid tools, including repeated capabilities.
    catalog_results = []
    for count in range(5):
        correct = []
        for kind in range(2):
            for request in HOLDOUT_TEXT[kind]:
                descriptions = [
                    HOLDOUT_DESC[kind if i < count else 1 - kind][i % 2]
                    for i in range(4)
                ]
                _, scores = score_pairs(agent, [(request, d) for d in descriptions])
                ids = tuple(f"service-{i}" for i in range(4))
                decision = catalog_decision(ids, scores)
                expected = set(ids[:count])
                correct.append(set(decision["eligible_ids"]) == expected)
                # Reordering/renaming is an architectural contract of the adapter.
                reversed_decision = catalog_decision(ids[::-1], scores[::-1])
                assert set(reversed_decision["eligible_ids"]) == set(
                    decision["eligible_ids"]
                )
                renamed = catalog_decision(
                    tuple(f"renamed-{i}" for i in range(4)), scores
                )
                assert len(renamed["eligible_ids"]) == len(decision["eligible_ids"])
        catalog_results.append(
            dict(
                suitable_count=count,
                exact_set_accuracy=float(np.mean(correct)),
                cases=len(correct),
            )
        )
    return dict(
        pair_accuracy=float(np.mean(predicted == labels)),
        positive_recall=float(np.mean(predicted[labels])),
        negative_recall=float(np.mean(~predicted[~labels])),
        brier_score=float(np.mean((p - labels) ** 2)),
        parser_pair_accuracy=float(np.mean(parser == labels)),
        catalogs=catalog_results,
        cases=len(labels),
    )


class HeldoutWorld(ToolWorld):
    def request(self, turn, kind=None):
        kind = int(self.requests[turn]) if kind is None else int(kind)
        return HOLDOUT_TEXT[kind][int(self.wording[turn])]


def compose(numeric, world, probabilities, seed, trace):
    context = AgentContext("numeric", world.identifier, "validation", max_events=8)
    rng = seeded(seed, 33, int(world.identifier.split(":")[-1]))
    rewards, optimal, compatible, regrets = [], [], [], []
    for turn, scores in enumerate(probabilities):
        decision = catalog_decision(IDS, scores)
        chosen = None
        if decision["eligible_ids"]:
            obs = make_observation(world, turn, context)
            obs = replace(
                obs,
                legal_action_mask=np.array(
                    [i in decision["eligible_ids"] for i in obs.candidate_ids]
                ),
            )
            result = numeric.action_result_structured(obs, deterministic=True)
            distribution = epsilon_probabilities(
                result.candidate_values, obs.legal_action_mask
            )
            canonical = np.zeros(4)
            for key, value in zip(obs.candidate_ids, distribution):
                canonical[IDS.index(key)] = value
            chosen = int(rng.choice(4, p=canonical))
        utility = world.utility(turn)
        rewards.append(0.0 if chosen is None else world.reward(turn, chosen))
        optimal.append(chosen == int(np.argmax(utility)))
        compatible.append(
            chosen is not None and world.capabilities[chosen] == world.requests[turn]
        )
        regrets.append(
            float(max(utility) - (0.0 if chosen is None else utility[chosen]))
        )
        trace.write(
            json.dumps(
                dict(
                    world=world.identifier,
                    turn=turn,
                    suitability_scores=scores.tolist(),
                    eligible_ids=decision["eligible_ids"],
                    action=None if chosen is None else IDS[chosen],
                    reward=rewards[-1],
                )
            )
            + "\n"
        )
        if chosen is not None:
            context = append_feedback(context, world, turn, chosen)
    return dict(
        mean_reward=float(np.mean(rewards)),
        optimal_action_fraction=float(np.mean(optimal)),
        compatibility=float(np.mean(compatible)),
        mean_expected_regret=float(np.mean(regrets)),
        abstention_rate=float(np.mean(np.max(probabilities, axis=1) < 0.5)),
    )


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("distinct nonnegative seeds required")
    if min(args.epochs, args.eval_trials) < 1:
        raise ValueError("positive training and evaluation budgets required")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("positive finite learning rate required")
    for seed in seeds:
        if not (
            Path(args.numeric_root) / f"numeric-{seed}" / "manifest.json"
        ).is_file():
            raise ValueError(f"missing frozen numeric bundle for seed {seed}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "configuration.json").write_text(json.dumps(vars(args), indent=2))
    source = Path(__file__).read_bytes()
    (out / "source.py").write_bytes(source)
    (out / "source.sha256").write_text(hashlib.sha256(source).hexdigest())
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "examples" / name
        for name in (
            "adaptation_benchmark.py",
            "learning_diagnostic.py",
            "language_tool_benchmark.py",
        )
    ] + sorted((root / "src").rglob("*.py"))
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                sources={
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
                versions={
                    name: metadata.version(name)
                    for name in ("torch", "numpy", "transformers", "peft")
                },
            ),
            indent=2,
        )
    )
    torch.set_num_threads(2)
    records = []
    for seed in seeds:
        dest = out / str(seed)
        dest.mkdir()
        torch.manual_seed(seed)
        config = dict(
            llm_model=args.model,
            llm_revision=args.revision,
            llm_dtype="float32",
            llm_max_tokens=768,
            llm_forward_batch_size=24,
            llm_lora_rank=4,
            llm_lora_alpha=8,
            lr=args.learning_rate,
            value_backbone_gradient_scale=0.0,
        )
        agent = algorithm_registry.get("llm_ppo").build_structured(
            SPEC, config, args.device
        )
        data = training_data()
        losses = train_suitability(agent, data, args.epochs, seed)
        training_scores = agent.score_results_structured([o for o, _ in data])
        training_accuracy = float(
            np.mean(
                [
                    np.argmax(s.candidate_logits) == np.argmax(y)
                    for s, (_, y) in zip(training_scores, data)
                ]
            )
        )
        before, updates = trainable_digest(agent), agent.update_steps
        audit = semantic_audit(agent, dest)
        numeric, manifest = load_structured_policy_bundle(
            Path(args.numeric_root) / f"numeric-{seed}"
        )
        digest, numeric_updates = trainable_digest(QView(numeric)), numeric.update_steps
        results = {k: [] for k in ("learned", "parser", "oracle_numeric")}
        worlds = []
        with (dest / "composition.jsonl").open("w") as trace:
            for trial in range(args.eval_trials):
                base = make_world(seed + 100000, 2, trial, 32)
                world = HeldoutWorld(
                    **{
                        **vars(base),
                        "descriptions": tuple(
                            HOLDOUT_DESC[int(c)][i % 2]
                            for i, c in enumerate(base.capabilities)
                        ),
                    }
                )
                worlds.append(world.audit())
                pairs = [
                    (world.request(t), d) for t in range(32) for d in world.descriptions
                ]
                _, probabilities = score_pairs(agent, pairs)
                results["learned"].append(
                    compose(numeric, world, probabilities.reshape(32, 4), seed, trace)
                )
                parser_scores = np.array(
                    [parser_accepts(*p) for p in pairs], dtype=float
                ).reshape(32, 4)
                results["parser"].append(
                    compose(numeric, world, parser_scores, seed, io.StringIO())
                )
                results["oracle_numeric"].append(
                    evaluate_policy(
                        numeric,
                        world,
                        np.zeros((32, 4)),
                        "oracle_numeric",
                        seed,
                        8,
                        io.StringIO(),
                    )
                )
        assert trainable_digest(agent) == before and agent.update_steps == updates
        assert (
            trainable_digest(QView(numeric)) == digest
            and numeric.update_steps == numeric_updates
        )
        export_structured_policy_bundle(
            dest / "language",
            agent=agent,
            spec=SPEC,
            algorithm="llm_ppo",
            agent_config=config,
        )
        (dest / "worlds.json").write_text(json.dumps(worlds, indent=2))
        record = dict(
            seed=seed,
            training_accuracy=training_accuracy,
            training_losses=losses,
            semantics=audit,
            numeric_artifact=manifest["artifact"],
            evaluation=results,
        )
        (dest / "result.json").write_text(json.dumps(record, indent=2))
        records.append(record)
        print(
            json.dumps(
                dict(
                    seed=seed,
                    semantics=audit,
                    rewards={
                        k: float(np.mean([r["mean_reward"] for r in rows]))
                        for k, rows in results.items()
                    },
                )
            ),
            flush=True,
        )
        del agent, numeric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (out / "summary.json").write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--numeric-root",
        required=True,
        help="Prior language_tool_benchmark output with frozen numeric-SEED bundles",
    )
    p.add_argument("--seeds", default="2111,2129,2131")
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--eval-trials", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    run(p.parse_args())
