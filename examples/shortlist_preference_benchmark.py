"""Synthetic preference study: explicit comparisons versus position-biased choices.

The simulated user, acceptability rule, features, and scorer are example-owned.
Core records/losses are exercised without an HTTP service or external users.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from jormungandr.preferences import (
    RankedExposure,
    PreferenceFeedback,
    PairwisePreference,
    PreferenceLedger,
    exposure_to_payload,
    feedback_to_payload,
    preference_examples,
    preference_batch_loss,
)
from jormungandr.structured import EntityCandidateObservation

METHODS = ("explicit_pairs", "biased_choices", "randomized_choices")
CHECKPOINTS = (0, 32, 128, 512)
CANDIDATES = 6
DIM = 4
UTILITY_WEIGHTS = np.array([1.0, -0.7, 0.5, -1.2], dtype=np.float32)


def generate(seed, count, split):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        context = rng.normal(size=DIM).astype(np.float32)
        candidates = rng.normal(size=(CANDIDATES, DIM)).astype(np.float32)
        utility = (candidates * context) @ UTILITY_WEIGHTS
        ids = tuple(f"candidate-{j}" for j in range(CANDIDATES))
        obs = EntityCandidateObservation(
            global_features=context,
            entity_features=np.zeros((0, 1), dtype=np.float32),
            entity_type_ids=np.zeros(0, dtype=np.int64),
            entity_ids=(),
            candidate_features=candidates,
            candidate_ids=ids,
            legal_action_mask=np.ones(CANDIDATES, dtype=bool),
        )
        biased = np.argsort(-candidates[:, 0], kind="stable")
        random = rng.permutation(CANDIDATES)
        # A strong position effect; the remaining choices favor true utility.
        choose_first = rng.random() < 0.8
        probabilities = np.exp((utility - utility.max()) / 0.5)
        true_choice = int(rng.choice(CANDIDATES, p=probabilities / probabilities.sum()))
        first, second = map(int, rng.choice(CANDIDATES, 2, replace=False))
        difference = float(utility[first] - utility[second])
        target = 0.5 if abs(difference) < 0.15 else float(difference > 0)
        # Annotation errors, distinct from the much stronger click-position bias.
        if target != 0.5 and rng.random() < 0.05:
            target = 1 - target
        exposures, feedback = {}, {}
        for kind, order in [("biased", biased), ("randomized", random)]:
            exposure_id = f"{seed}:{i}:{kind}"
            exposures[kind] = RankedExposure(
                exposure_id=exposure_id,
                actor_id="simulated-presenter",
                episode_id=str(i),
                timestep=0,
                policy_version=0,
                shown_at_ns=i * 10,
                observation=obs,
                displayed_ids=tuple(ids[j] for j in order),
                displayed_scores=tuple(float(candidates[j, 0]) for j in order),
                split=split,
                display_probability=(
                    1.0 if kind == "biased" else 1 / math.factorial(CANDIDATES)
                ),
            )
            selected = int(order[0]) if choose_first else true_choice
            feedback[kind] = PreferenceFeedback(
                feedback_id=f"{exposure_id}:choice",
                exposure_id=exposure_id,
                received_at_ns=i * 10 + 1,
                selected_id=ids[selected],
            )
        feedback["explicit"] = PreferenceFeedback(
            feedback_id=f"{seed}:{i}:pair",
            exposure_id=exposures["biased"].exposure_id,
            received_at_ns=i * 10 + 2,
            comparisons=(PairwisePreference(ids[first], ids[second], target),),
        )
        result.append(
            {"exposures": exposures, "feedback": feedback, "utility": utility}
        )
    return result


def features(rows):
    return torch.from_numpy(
        np.stack(
            [
                np.concatenate(
                    [
                        o.candidate_features,
                        np.broadcast_to(o.global_features, (CANDIDATES, DIM)),
                        o.candidate_features * o.global_features,
                    ],
                    axis=1,
                )
                for o in [r["exposures"]["biased"].observation for r in rows]
            ]
        )
    )


def examples_for(rows, method):
    kind = "randomized" if method == "randomized_choices" else "biased"
    ledger = PreferenceLedger(capacity=len(rows))
    for i, row in enumerate(rows):
        exposure = row["exposures"][kind]
        feedback = row["feedback"]["explicit" if method == "explicit_pairs" else kind]
        # Exercise both arrival orders. This changes delivery order, not event time.
        if i % 2:
            ledger.add_feedback(feedback)
            ledger.add_exposure(exposure)
        else:
            ledger.add_exposure(exposure)
            ledger.add_feedback(feedback)
        assert not ledger.add_feedback(feedback)  # exact retry is idempotent
    assert ledger.pending_count == 0
    resolved = {e.exposure_id: (e, f) for e, f in ledger.resolved(split="train")}
    return [
        preference_examples(
            *resolved[r["exposures"][kind].exposure_id],
            infer_selection=method != "explicit_pairs",
        )
        for r in rows
    ]


@torch.no_grad()
def evaluate(model, x, utility):
    scores = model(x).squeeze(-1).numpy()
    order = np.argsort(-scores, axis=1, kind="stable")
    ideal = np.argsort(-utility, axis=1, kind="stable")
    acceptable = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(acceptable, ideal[:, :3], True, axis=1)
    index = np.arange(len(scores))
    pairs = np.triu_indices(CANDIDATES, k=1)
    difference = utility[:, pairs[0]] - utility[:, pairs[1]]
    prediction = scores[:, pairs[0]] - scores[:, pairs[1]]
    informative = np.abs(difference) >= 0.15
    pair_correct = (np.sign(prediction) == np.sign(difference))[informative]
    return {
        "top1_acceptable_rate": float(acceptable[index, order[:, 0]].mean()),
        "top3_acceptable_recall": float(
            np.take_along_axis(acceptable, order[:, :3], axis=1).mean()
        ),
        "best_option_accuracy": float((order[:, 0] == ideal[:, 0]).mean()),
        "pair_order_accuracy": float(pair_correct.mean()),
        "utility_regret": float((utility.max(1) - utility[index, order[:, 0]]).mean()),
    }


def run(args):
    if min(args.exposures) < 2 or args.eval_exposures < 2 or min(args.seeds) < 0:
        raise ValueError("invalid study size")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    sources = [
        "examples/shortlist_preference_benchmark.py",
        "src/jormungandr/preferences.py",
        "src/jormungandr/structured.py",
    ]
    protocol = {
        "stage": "development_only",
        "args": vars(args),
        "methods": METHODS,
        "checkpoints": CHECKPOINTS,
        "candidates": CANDIDATES,
        "acceptable_per_context": 3,
        "position_choice_probability": 0.8,
        "explicit_comparison_error_probability": 0.05,
        "tie_threshold": 0.15,
        "optimizer": "Adam",
        "learning_rate": 0.03,
        "batch_size": 64,
        "model": "linear scorer on candidate, context and their elementwise product (12 features)",
        "label_budget": "one user event per exposure: one explicit comparison or one choice; choice yields 5 assumed pairwise preferences. Annotation effort/quality are not equalized.",
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": 2,
        },
        "sources": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources
        },
    }
    for p in sources:
        dest = out / "sources" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(p).read_bytes())
    warm = nn.Linear(12, 1, bias=False)
    optimizer = torch.optim.Adam(warm.parameters(), lr=0.03)
    # Remove one-time optimizer startup from method timing.
    for _ in range(3):
        optimizer.zero_grad()
        warm(torch.ones(64, 12)).square().mean().backward()
        optimizer.step()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    records = []
    for seed in args.seeds:
        rows = generate(seed, max(args.exposures), "train")
        validation = generate(seed + 10000000, args.eval_exposures, "validation")
        x_eval = features(validation)
        utility_eval = np.stack([r["utility"] for r in validation])
        # Archive actual exposures and signals, excluding evaluator-only utilities.
        corpus = "\n".join(
            json.dumps(
                {
                    "exposures": {
                        k: exposure_to_payload(v) for k, v in r["exposures"].items()
                    },
                    "feedback": {
                        k: feedback_to_payload(v) for k, v in r["feedback"].items()
                    },
                },
                sort_keys=True,
            )
            for r in rows
        )
        (out / f"{seed}-events.jsonl").write_text(corpus + "\n")
        for budget in args.exposures:
            train = rows[:budget]
            x = features(train)
            tape = np.random.default_rng(seed + budget).integers(
                budget, size=(max(CHECKPOINTS), 64)
            )
            order = list(METHODS)
            np.random.default_rng(seed + budget + 1).shuffle(order)
            for method in order:
                start = time.perf_counter()
                examples = examples_for(train, method)
                preparation = time.perf_counter() - start
                torch.manual_seed(seed)
                model = nn.Linear(12, 1, bias=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
                points, seconds, eval_seconds = [], 0.0, 0.0
                for step in range(max(CHECKPOINTS) + 1):
                    if step in CHECKPOINTS:
                        start = time.perf_counter()
                        metrics = evaluate(model, x_eval, utility_eval)
                        eval_seconds += time.perf_counter() - start
                        points.append(
                            {"updates": step, "training_seconds": seconds, **metrics}
                        )
                    if step == max(CHECKPOINTS):
                        break
                    start = time.perf_counter()
                    selected = tape[step]
                    batch = [e for i in selected for e in examples[i]]
                    repeats = torch.tensor([len(examples[i]) for i in selected])
                    scores = (
                        model(x[selected]).squeeze(-1).repeat_interleave(repeats, dim=0)
                    )
                    loss = preference_batch_loss(scores, batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    seconds += time.perf_counter() - start
                records.append(
                    {
                        "seed": seed,
                        "exposures": budget,
                        "method": method,
                        "method_order": order,
                        "user_events": budget,
                        "derived_pairs": sum(map(len, examples)),
                        "event_corpus_sha256": hashlib.sha256(
                            corpus.encode()
                        ).hexdigest(),
                        "pair_preparation_seconds": preparation,
                        "evaluation_seconds": eval_seconds,
                        "points": points,
                    }
                )
                torch.save(model.state_dict(), out / f"{seed}-{budget}-{method}.pt")
                (out / "results.json").write_text(json.dumps(records, indent=2))
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "exposures": budget,
                            "method": method,
                            **points[-1],
                        }
                    ),
                    flush=True,
                )
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11113, 11213, 11311])
    p.add_argument("--exposures", type=int, nargs="+", default=[128, 1024])
    p.add_argument("--eval-exposures", type=int, default=2048)
    run(p.parse_args())
