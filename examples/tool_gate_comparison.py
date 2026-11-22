"""Matched ranking/binary and narrow/broad language coverage experiment.

Development selects a recipe; confirmation uses new wording and training seeds.
Neither stage trains the numeric actor. No real APIs are executed.
"""

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from examples.adaptation_benchmark import SPEC, algorithm_registry, trainable_digest
from examples.language_tool_benchmark import IDS, ToolWorld, make_world
from examples.learning_diagnostic import QView
from examples.tool_suitability_benchmark import (
    TRAIN_TEXT,
    TRAIN_DESC,
    HOLDOUT_TEXT,
    HOLDOUT_DESC,
    pair_observation,
    compose,
    parser_accepts,
)
from jormungandr.llm_context import AgentContext
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_export import (
    export_structured_policy_bundle,
    load_structured_policy_bundle,
)

BROAD_TEXT = tuple(
    TRAIN_TEXT[i] + extra
    for i, extra in enumerate(
        (
            (
                "Translate, rather than summarize, this passage.",
                "I need a translation, not a summary.",
                "Keep the full content but change its language.",
                "Write this material in Japanese.",
                "Make an English version of this text.",
                "Change the language without shortening the document.",
            ),
            (
                "Summarize, rather than translate, this passage.",
                "I need a summary, not a translation.",
                "Keep the language but reduce the amount of text.",
                "Extract the important ideas from this material.",
                "Make a shorter version of this text.",
                "Reduce the length without changing the language.",
            ),
        )
    )
)
BROAD_DESC = tuple(
    TRAIN_DESC[i] + extra
    for i, extra in enumerate(
        (
            (
                "Changes the language of a document without omitting its contents.",
                "Writes an equivalent version of the text in a different language.",
                "Supports translation of full passages into a chosen language.",
            ),
            (
                "Reduces document length while retaining important information.",
                "Writes an abbreviated version containing the main ideas.",
                "Supports summarization of full passages into short notes.",
            ),
        )
    )
)
# Frozen before any development runs. These strings are never used for training
# or development selection. This is a small authored suite, not a population sample.
FINAL_TEXT = (
    (
        "Produce an Italian rendition of the entire passage.",
        "Preserve every detail, but put the document into Portuguese.",
        "The output should be in Korean; do not condense the original.",
        "Instead of a digest, give me the text in Dutch.",
    ),
    (
        "Boil this document down to its central message.",
        "Keep the original language and give me the gist in a few lines.",
        "The output should be brief; do not convert it to another language.",
        "Instead of a translation, give me a compact account of the key points.",
    ),
    (
        "Identify misspelled words without rewriting this passage.",
        "Check the commas and apostrophes in this document.",
        "Tell me which sentences contain grammatical errors.",
        "Correct the typos while keeping the document otherwise intact.",
    ),
)
FINAL_DESC = (
    (
        "Provides a complete rendition of a document in a specified foreign tongue.",
        "Takes prose in one language and outputs equivalent prose in another.",
    ),
    (
        "Produces an abridged account of the central ideas in a document.",
        "Turns lengthy prose into a brief digest retaining the salient points.",
    ),
)
FORMATS = ("ranking", "binary")
COVERAGES = ("narrow", "broad")


@dataclass(frozen=True)
class Case:
    request: str
    descriptions: tuple
    labels: tuple
    ids: tuple = IDS


def observation(case, split="validation"):
    return EntityCandidateObservation(
        global_features=np.zeros(1),
        entity_features=np.zeros((4, 3)),
        entity_type_ids=np.zeros(4, dtype=np.int64),
        entity_ids=case.ids,
        candidate_features=np.zeros((4, 3)),
        candidate_ids=case.ids,
        candidate_entity_indices=np.arange(4),
        legal_action_mask=np.ones(4, dtype=bool),
        metadata=AgentContext("comparison", "catalog", split, max_events=0).metadata(
            "Request: "
            + case.request
            + " Choose a tool whose described capability fulfills the request.",
            dict(zip(case.ids, case.descriptions)),
        ),
    )


def training_cases(seed, coverage, count):
    requests = TRAIN_TEXT[:2] if coverage == "narrow" else BROAD_TEXT
    descriptions = TRAIN_DESC if coverage == "narrow" else BROAD_DESC
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        kind = index % 2
        capabilities = rng.permutation([0, 0, 1, 1])
        cases.append(
            Case(
                requests[kind][int(rng.integers(len(requests[kind])))],
                tuple(
                    descriptions[c][int(rng.integers(len(descriptions[c])))]
                    for c in capabilities
                ),
                tuple(bool(c == kind) for c in capabilities),
            )
        )
    return cases


def evaluation_cases(stage):
    requests, descriptions = (
        (HOLDOUT_TEXT, HOLDOUT_DESC)
        if stage == "development"
        else (FINAL_TEXT, FINAL_DESC)
    )
    rng = np.random.default_rng(4903)
    cases = []
    for kind, texts in enumerate(requests):
        for request in texts:
            for count in range(5) if kind < 2 else (0,):
                for _ in range(2):
                    if kind < 2:
                        capabilities = rng.permutation(
                            [kind] * count + [1 - kind] * (4 - count)
                        )
                    else:
                        capabilities = rng.permutation([0, 0, 1, 1])
                    cases.append(
                        Case(
                            request,
                            tuple(
                                descriptions[c][int(rng.integers(2))]
                                for c in capabilities
                            ),
                            tuple(bool(c == kind) for c in capabilities),
                        )
                    )
    return cases


def train(agent, cases, representation):
    losses = []
    # Every optimizer update consumes exactly the same two semantic catalogs.
    for start in range(0, len(cases), 2):
        observations, targets = [], []
        for case in cases[start : start + 2]:
            if representation == "ranking":
                observations.append(observation(case, "train"))
                targets.append(np.asarray(case.labels, dtype=float) / 2)
            else:
                for description, label in zip(case.descriptions, case.labels):
                    observations.append(
                        pair_observation(case.request, description, "train")
                    )
                    targets.append(np.array([float(not label), float(label)]))
        agent.policy.train()
        output = agent.policy(agent._collate_observations(observations))
        target = torch.tensor(
            np.stack(targets), dtype=torch.float32, device=agent.device
        )
        loss = -(target * torch.log_softmax(output.logits, -1)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.policy.parameters() if p.requires_grad], 1.0
        )
        agent.optimizer.step()
        agent.update_steps += 1
        losses.append(float(loss.detach()))
        if len(losses) % 64 == 0:
            print(
                json.dumps(
                    dict(updates=len(losses), recent_loss=float(np.mean(losses[-64:])))
                ),
                flush=True,
            )
    return losses


def score(agent, cases, representation):
    if representation == "ranking":
        return np.array(
            [
                s.candidate_logits
                for s in agent.score_results_structured([observation(c) for c in cases])
            ]
        )
    pairs = list(dict.fromkeys((c.request, d) for c in cases for d in c.descriptions))
    results = agent.score_results_structured([pair_observation(*p) for p in pairs])
    margins = {
        p: float(s.candidate_logits[1] - s.candidate_logits[0])
        for p, s in zip(pairs, results)
    }
    return np.array([[margins[(c.request, d)] for d in c.descriptions] for c in cases])


def top_two(scores):
    selected = np.zeros_like(scores, dtype=bool)
    for row, values in zip(selected, scores):
        row[np.argsort(values, kind="stable")[-2:]] = True
    return selected


def metrics(cases, scores, representation):
    labels = np.asarray([c.labels for c in cases])
    ranked = top_two(scores)
    primary = labels.sum(-1) == 2
    result = dict(
        primary_cases=int(primary.sum()),
        primary_exact_set=float(
            np.mean(np.all(ranked[primary] == labels[primary], axis=1))
        ),
        primary_selected_precision=float(np.mean(labels[primary][ranked[primary]])),
    )
    # Fixed-two ranking has no native abstention/variable-cardinality decision.
    if representation == "binary":
        native = scores >= 0
        result["native"] = dict(
            exact_set=float(np.mean(np.all(native == labels, axis=1))),
            positive_recall=float(np.mean(native[labels])),
            negative_recall=float(np.mean(~native[~labels])),
            none_fit_accuracy=(
                float(np.mean(~native[labels.sum(-1) == 0].any(-1)))
                if np.any(labels.sum(-1) == 0)
                else None
            ),
            by_suitable_count={
                str(n): (
                    float(
                        np.mean(
                            np.all(
                                native[labels.sum(-1) == n]
                                == labels[labels.sum(-1) == n],
                                axis=1,
                            )
                        )
                    )
                    if np.any(labels.sum(-1) == n)
                    else None
                )
                for n in range(5)
            },
        )
    return result


def audit(agent, cases, representation, output):
    before, updates = trainable_digest(agent), agent.update_steps
    scores = score(agent, cases, representation)
    primary = [c for c in cases if sum(c.labels) == 2]
    reversed_cases = [
        Case(c.request, c.descriptions[::-1], c.labels[::-1], c.ids[::-1])
        for c in primary
    ]
    renamed_cases = [
        Case(
            c.request,
            c.descriptions,
            c.labels,
            tuple(f"service-{i+100}" for i in range(4)),
        )
        for c in primary
    ]
    original = score(agent, primary, representation)
    reversed_scores = score(agent, reversed_cases, representation)[:, ::-1]
    renamed_scores = score(agent, renamed_cases, representation)
    result = metrics(cases, scores, representation)
    result["invariance"] = dict(
        cases=len(primary),
        reordered_changed_sets=int(
            np.any(top_two(original) != top_two(reversed_scores), axis=1).sum()
        ),
        renamed_changed_sets=int(
            np.any(top_two(original) != top_two(renamed_scores), axis=1).sum()
        ),
    )
    parser = np.array(
        [[parser_accepts(c.request, d) for d in c.descriptions] for c in cases]
    )
    result["parser_exact_set"] = float(
        np.mean(np.all(parser == np.asarray([c.labels for c in cases]), axis=1))
    )
    primary_mask = np.array([sum(c.labels) == 2 for c in cases])
    result["parser_primary_exact_set"] = float(
        np.mean(
            np.all(
                parser[primary_mask]
                == np.asarray([c.labels for c in cases])[primary_mask],
                axis=1,
            )
        )
    )
    (output / "semantic-traces.json").write_text(
        json.dumps(
            [
                dict(
                    request=c.request,
                    descriptions=c.descriptions,
                    labels=c.labels,
                    scores=s.tolist(),
                    top_two=t.tolist(),
                    parser=p.tolist(),
                )
                for c, s, t, p in zip(cases, scores, top_two(scores), parser)
            ],
            indent=2,
        )
    )
    assert before == trainable_digest(agent) and updates == agent.update_steps
    return result


def composition(agent, numeric, representation, stage, seed, output):
    texts, descriptions = (
        (HOLDOUT_TEXT, HOLDOUT_DESC)
        if stage == "development"
        else (FINAL_TEXT, FINAL_DESC)
    )

    class World(ToolWorld):
        def request(self, turn, kind=None):
            kind = int(self.requests[turn]) if kind is None else int(kind)
            return texts[kind][int(self.wording[turn])]

    worlds = []
    for trial in range(12):
        base = make_world(
            seed + (200000 if stage == "development" else 300000), 2, trial, 32
        )
        rng = np.random.default_rng(seed + trial)
        worlds.append(
            World(
                **{
                    **vars(base),
                    "wording": rng.integers(len(texts[0]), size=32),
                    "descriptions": tuple(
                        descriptions[int(c)][i % 2]
                        for i, c in enumerate(base.capabilities)
                    ),
                }
            )
        )
    (output / "worlds.json").write_text(
        json.dumps([w.audit() for w in worlds], indent=2)
    )
    before, updates = trainable_digest(QView(numeric)), numeric.update_steps
    language_before, language_updates = trainable_digest(agent), agent.update_steps
    result = {k: [] for k in ("top_two", "parser", "oracle")}
    if representation == "binary":
        result["native"] = []
    with (output / "composition.jsonl").open("w") as trace:
        for world in worlds:
            cases = [
                Case(
                    world.request(t),
                    world.descriptions,
                    tuple(world.capabilities == world.requests[t]),
                )
                for t in range(32)
            ]
            values = score(agent, cases, representation)
            masks = dict(
                top_two=top_two(values),
                parser=np.array(
                    [
                        [parser_accepts(c.request, d) for d in c.descriptions]
                        for c in cases
                    ]
                ),
                oracle=np.asarray([c.labels for c in cases]),
            )
            if representation == "binary":
                masks["native"] = values >= 0
            for method, mask in masks.items():
                buffer = io.StringIO()
                result[method].append(
                    compose(numeric, world, mask.astype(float), seed, buffer)
                )
                for line in buffer.getvalue().splitlines():
                    trace.write(
                        json.dumps(dict(method=method, **json.loads(line))) + "\n"
                    )
    assert (
        before == trainable_digest(QView(numeric)) and updates == numeric.update_steps
    )
    assert (
        language_before == trainable_digest(agent)
        and language_updates == agent.update_steps
    )
    return {
        k: {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
        for k, rows in result.items()
    }


def digest_cases(cases):
    return hashlib.sha256(
        json.dumps([vars(c) for c in cases], sort_keys=True).encode()
    ).hexdigest()


def choose_recipe(records):
    # Primary shared fixed-two metric only. Ties prefer narrow coverage, then ranking.
    best = max(
        records,
        key=lambda r: (
            r["evaluation"]["primary_exact_set"],
            r["coverage"] == "narrow",
            r["representation"] == "ranking",
        ),
    )
    return dict(representation=best["representation"], coverage=best["coverage"])


def validate_controls(records):
    for seed in {r["seed"] for r in records}:
        rows = [r for r in records if r["seed"] == seed]
        if len({r["initial_digest"] for r in rows}) != 1:
            raise ValueError("initial weights differ across comparison cells")
        for coverage in {r["coverage"] for r in rows}:
            hashes = {
                r["training_catalog_digest"] for r in rows if r["coverage"] == coverage
            }
            if len(hashes) != 1:
                raise ValueError("training catalogs differ across representations")
    if len({r["numeric_artifact"]["sha256"] for r in records}) != 1:
        raise ValueError("numeric actor changed during comparison")


def run(args):
    if args.updates < 1:
        raise ValueError("positive update budget required")
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("distinct nonnegative seeds required")
    if args.stage == "development" and len(seeds) != 1:
        raise ValueError("development selection requires one seed")
    variants = [(r, c) for r in FORMATS for c in COVERAGES]
    if args.stage == "confirmation":
        if not args.selection:
            raise ValueError("confirmation requires a development selection")
        selection = json.loads(Path(args.selection).read_text())
        if args.updates != selection["updates"]:
            raise ValueError("confirmation must preserve selected training budget")
        variants = [(selection["representation"], selection["coverage"])]
        if variants[0] not in [(r, c) for r in FORMATS for c in COVERAGES]:
            raise ValueError("invalid selected recipe")
        if selection["reserved_confirmation_digest"] != digest_cases(
            evaluation_cases("confirmation")
        ):
            raise ValueError("reserved confirmation suite changed")
        development = json.loads(
            (Path(args.selection).parent / "protocol.json").read_text()
        )["arguments"]
        if any(getattr(args, key) != development[key] for key in ("model", "revision")):
            raise ValueError("confirmation must preserve the model and revision")
        if set(seeds) & set(map(int, development["seeds"].split(","))):
            raise ValueError("confirmation requires new training seeds")
    if not (
        Path(args.numeric_root) / f"numeric-{args.numeric_seed}" / "manifest.json"
    ).is_file():
        raise ValueError("missing frozen numeric bundle")
    if args.stage == "confirmation":
        development_records = json.loads(
            (Path(args.selection).parent / "results.json").read_text()
        )
        selected = next(
            r
            for r in development_records
            if (r["representation"], r["coverage"]) == variants[0]
        )
        numeric_manifest = json.loads(
            (
                Path(args.numeric_root)
                / f"numeric-{args.numeric_seed}"
                / "manifest.json"
            ).read_text()
        )
        if (
            numeric_manifest["artifact"]["sha256"]
            != selected["numeric_artifact"]["sha256"]
        ):
            raise ValueError("confirmation must preserve the frozen numeric actor")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cases = evaluation_cases(args.stage)
    protocol = dict(
        arguments=vars(args),
        variants=variants,
        evaluation_digest=digest_cases(cases),
        reserved_confirmation_digest=digest_cases(evaluation_cases("confirmation")),
        selection_rule="development primary exact-set accuracy; ties narrow then ranking",
        batch_catalogs=2,
        learning_rate=0.0001,
        threshold=0.0,
        note="primary uses top two for both formats; native binary threshold evaluated separately",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    records = []
    for seed in seeds:
        numeric, manifest = load_structured_policy_bundle(
            Path(args.numeric_root) / f"numeric-{args.numeric_seed}"
        )
        for representation, coverage in variants:
            dest = out / f"{seed}-{representation}-{coverage}"
            dest.mkdir()
            data = training_cases(seed, coverage, args.updates * 2)
            torch.manual_seed(seed)
            config = dict(
                llm_model=args.model,
                llm_revision=args.revision,
                llm_dtype="float32",
                llm_max_tokens=1536,
                llm_forward_batch_size=24,
                llm_lora_rank=4,
                llm_lora_alpha=8,
                lr=0.0001,
                value_backbone_gradient_scale=0.0,
            )
            agent = algorithm_registry.get("llm_ppo").build_structured(
                SPEC, config, args.device
            )
            initial_digest = trainable_digest(agent)
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        representation=representation,
                        coverage=coverage,
                        phase="train",
                    )
                ),
                flush=True,
            )
            losses = train(agent, data, representation)
            training = metrics(
                data[:64], score(agent, data[:64], representation), representation
            )
            evaluation = audit(agent, cases, representation, dest)
            composed = composition(
                agent, numeric, representation, args.stage, seed, dest
            )
            export_structured_policy_bundle(
                dest / "language",
                agent=agent,
                spec=SPEC,
                algorithm="llm_ppo",
                agent_config=config,
            )
            record = dict(
                seed=seed,
                representation=representation,
                coverage=coverage,
                initial_digest=initial_digest,
                training_catalog_digest=digest_cases(data),
                losses=losses,
                training_sample=training,
                evaluation=evaluation,
                composition=composed,
                numeric_artifact=manifest["artifact"],
            )
            (dest / "result.json").write_text(json.dumps(record, indent=2))
            records.append(record)
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in (
                            "seed",
                            "representation",
                            "coverage",
                            "evaluation",
                            "composition",
                        )
                    }
                ),
                flush=True,
            )
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del numeric
    validate_controls(records)
    (out / "results.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    if args.stage == "development":
        if len(set(r["seed"] for r in records)) != 1:
            raise ValueError("development selection currently requires one seed")
        (out / "selection.json").write_text(
            json.dumps(
                dict(
                    **choose_recipe(records),
                    updates=args.updates,
                    reserved_confirmation_digest=protocol[
                        "reserved_confirmation_digest"
                    ],
                ),
                indent=2,
            )
        )
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--numeric-root", required=True)
    p.add_argument("--numeric-seed", type=int, default=2111)
    p.add_argument(
        "--stage", choices=("development", "confirmation"), default="development"
    )
    p.add_argument("--selection")
    p.add_argument("--seeds", default="2111")
    p.add_argument("--updates", type=int, default=544)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
