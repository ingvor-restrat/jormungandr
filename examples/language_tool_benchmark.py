"""Complementary language interpretation and numeric adaptation in a simulated tool catalog.

Requests and catalog descriptions are public text. Reliability is hidden. All
four tools are callable; suitability must be inferred at evaluation time.
"""

import argparse
from dataclasses import dataclass, replace
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
    make_scenario,
    seeded,
    trainable_digest,
)
from examples.learning_diagnostic import QView, teach
from jormungandr.llm_context import AgentContext, Experience
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_export import export_structured_policy_bundle
from jormungandr.structured_replay import StructuredReplayTransition

IDS = ("tool-a", "tool-b", "tool-c", "tool-d")
TRAIN_REQUESTS = (
    (
        "Translate this passage into another language.",
        "Please translate the supplied text.",
        "I need a translation of this document.",
    ),
    (
        "Summarize this passage.",
        "Please write a summary of the supplied text.",
        "I need the main points of this document.",
    ),
)
TEST_REQUESTS = (
    (
        "Give me a translation of the following passage.",
        "Render this text in a different language.",
        "Convert the document into French.",
    ),
    (
        "Give me a concise summary of the following passage.",
        "Condense this text to its essential points.",
        "Provide a brief overview of the document.",
    ),
)
TRAIN_DESCRIPTIONS = (
    (
        "Translates text from one language to another.",
        "Produces translations of supplied documents.",
    ),
    (
        "Summarizes text by extracting its main points.",
        "Produces short summaries of supplied documents.",
    ),
)
TEST_DESCRIPTIONS = (
    (
        "Converts written passages between languages.",
        "Returns the supplied text translated into the requested language.",
    ),
    (
        "Condenses written passages into concise overviews.",
        "Returns a short summary highlighting essential information.",
    ),
)


@dataclass(frozen=True)
class ToolWorld:
    identifier: str
    family: str
    probabilities: np.ndarray
    successes: np.ndarray
    costs: np.ndarray
    latencies: np.ndarray
    capabilities: np.ndarray
    requests: np.ndarray
    orders: np.ndarray
    wording: np.ndarray
    descriptions: tuple[str, ...]
    split: int

    def request(self, turn, kind=None):
        kind = int(self.requests[turn]) if kind is None else int(kind)
        templates = TRAIN_REQUESTS if self.split == 1 else TEST_REQUESTS
        return templates[kind][int(self.wording[turn])]

    def utility(self, turn):
        compatible = self.capabilities == self.requests[turn]
        return self.probabilities[turn] * compatible - self.costs - 0.1 * self.latencies

    def reward(self, turn, action, *, operational=False):
        compatible = operational or self.capabilities[action] == self.requests[turn]
        return float(
            self.successes[turn, action] * compatible
            - self.costs[action]
            - 0.1 * self.latencies[action]
        )

    def audit(self):
        return {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in vars(self).items()
        }


def make_world(seed, split, trial, steps):
    groups = [make_scenario(seed + 100 * g, split, trial, steps) for g in range(2)]
    rng = seeded(seed, split + 20, trial)
    permutation = rng.permutation(4)
    capabilities = np.repeat(np.arange(2), 2)[permutation]
    descriptions = TRAIN_DESCRIPTIONS if split == 1 else TEST_DESCRIPTIONS
    values = dict(
        probabilities=np.concatenate([s.probabilities for s in groups], axis=1)[
            :, permutation
        ],
        successes=np.concatenate([s.successes for s in groups], axis=1)[:, permutation],
        costs=np.concatenate([s.costs for s in groups])[permutation],
        latencies=np.concatenate([s.latencies for s in groups])[permutation],
        capabilities=capabilities,
        requests=rng.integers(2, size=steps),
        orders=np.stack([rng.permutation(4) for _ in range(steps)]),
        wording=rng.integers(3, size=steps),
    )
    for array in values.values():
        array.setflags(write=False)
    return ToolWorld(
        f"{seed}:{split}:{trial}",
        groups[0].family,
        **values,
        descriptions=tuple(descriptions[g][int(rng.integers(2))] for g in capabilities),
        split=split,
    )


def make_observation(
    world, turn, context, *, language=False, kind=None, oracle_mask=False
):
    order = world.orders[turn]
    kind = int(world.requests[turn]) if kind is None else kind
    features = []
    for index in order:
        recent = [e for e in context.events if e.action == IDS[index]]
        age = next(
            (
                i
                for i, e in enumerate(reversed(context.events))
                if e.action == IDS[index]
            ),
            len(context.events),
        )
        features.append(
            [
                float(np.mean([e.reward for e in recent])) if recent else 0.0,
                len(recent) / max(1, context.max_events),
                age / max(1, context.max_events),
            ]
        )
    ids = tuple(IDS[i] for i in order)
    # The language stage never sees numeric histories, costs or reward labels.
    numeric = (
        np.zeros((4, 3))
        if language
        else np.array([[i, world.costs[i], world.latencies[i]] for i in order])
    )
    empty = replace(context, events=())
    return EntityCandidateObservation(
        global_features=np.zeros(1),
        entity_features=np.zeros((4, 3)) if language else np.asarray(features),
        entity_type_ids=np.zeros(4, dtype=np.int64),
        entity_ids=ids,
        candidate_features=numeric,
        candidate_ids=ids,
        candidate_entity_indices=np.arange(4),
        legal_action_mask=(
            (world.capabilities[order] == kind)
            if oracle_mask
            else np.ones(4, dtype=bool)
        ),
        metadata=empty.metadata(
            "Request: "
            + world.request(turn, kind)
            + " Choose a tool whose described capability fulfills the request.",
            {IDS[i]: world.descriptions[i] for i in order},
        ),
    )


def append_feedback(context, world, turn, action, version=0):
    return context.append(
        Experience(
            f"request-{turn}",
            0,
            "tool call",
            IDS[action],
            "success" if world.successes[turn, action] else "failure",
            world.reward(turn, action, operational=True),
            version,
        )
    )


def epsilon_probabilities(values, allowed, epsilon=0.1):
    indices = np.flatnonzero(allowed)
    if not len(indices):
        raise ValueError("at least one eligible tool required")
    winners = indices[
        np.isclose(
            np.asarray(values)[indices],
            max(np.asarray(values)[indices]),
            rtol=0,
            atol=1e-8,
        )
    ]
    probabilities = np.zeros(len(values))
    probabilities[indices] = epsilon / len(indices)
    probabilities[winners] += (1 - epsilon) / len(winners)
    return probabilities


def rolling_probabilities(context, allowed):
    rewards = [[e.reward for e in context.events if e.action == key] for key in IDS]
    unseen = [i for i in np.flatnonzero(allowed) if not rewards[i]]
    if unseen:
        probabilities = np.zeros(4)
        probabilities[unseen] = 1 / len(unseen)
        return probabilities
    return epsilon_probabilities([np.mean(r) if r else 0 for r in rewards], allowed)


def language_data(seed, count, steps):
    if count % 2:
        raise ValueError("language examples must be even for paired requests")
    data = []
    for i in range(count):
        world = make_world(seed, 1, 10000 + i // 2, steps)
        context = AgentContext("language", world.identifier, "train", max_events=0)
        obs = make_observation(world, 0, context, language=True, kind=i % 2)
        target = (world.capabilities[world.orders[0]] == i % 2).astype(float) / 2
        data.append((obs, target))
    return data


def train_numeric(seed, args, trace):
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
    rng = seeded(seed, 31, 0)
    replay = []
    losses = []
    for round_index in range(args.rounds):
        for trial in range(args.train_trials):
            world = make_world(
                seed, 1, round_index * args.train_trials + trial, args.steps
            )
            context = AgentContext(
                "numeric", world.identifier, "train", max_events=args.history
            )
            for turn in range(args.steps):
                # Training supplies suitability labels, never hidden reliability.
                obs = make_observation(world, turn, context, oracle_mask=True)
                score = agent.action_result_structured(obs, deterministic=True)
                probabilities = epsilon_probabilities(
                    score.candidate_values, obs.legal_action_mask, 0.2
                )
                row = int(rng.choice(4, p=probabilities))
                action = IDS.index(obs.candidate_ids[row])
                reward = world.reward(turn, action, operational=True)
                replay.append(
                    StructuredReplayTransition(
                        obs,
                        IDS[action],
                        reward,
                        obs,
                        True,
                        "numeric",
                        f"{world.identifier}:{turn}",
                        0,
                        agent.update_steps,
                    )
                )
                trace.write(
                    json.dumps(
                        dict(
                            trial=world.identifier,
                            turn=turn,
                            action=IDS[action],
                            reward=reward,
                            semantic_mask=obs.legal_action_mask.tolist(),
                        )
                    )
                    + "\n"
                )
                context = append_feedback(
                    context, world, turn, action, agent.update_steps
                )
        for _ in range(args.train_trials * args.steps):
            loss = agent.update_structured(
                [replay[i] for i in rng.integers(len(replay), size=32)]
            ).loss
            losses.append(loss)
    return agent, config, losses


def training_semantics(agent, data):
    before = trainable_digest(agent)
    scores = agent.score_results_structured([obs for obs, _ in data])
    exact, mass = [], []
    for score, (_, target) in zip(scores, data):
        allowed = target > 0
        top = np.argsort(score.candidate_logits, kind="stable")[-2:]
        exact.append(bool(np.all(allowed[top])))
        probabilities = torch.softmax(torch.tensor(score.candidate_logits), -1).numpy()
        mass.append(float(probabilities[allowed].sum()))
    assert before == trainable_digest(agent)
    return dict(
        exact_suitable_pair=float(np.mean(exact)),
        compatible_probability_mass=float(np.mean(mass)),
    )


def semantic_scores(agent, worlds):
    before = trainable_digest(agent)
    updates = agent.update_steps
    result = {}
    exact = []
    mass = []
    for world in worlds:
        context = AgentContext("language", world.identifier, "validation", max_events=0)
        observations = [
            make_observation(world, t, context, language=True)
            for t in range(len(world.requests))
        ]
        for obs in observations:
            diagnostics = agent.policy.context_diagnostics(obs)
            if diagnostics["retained_events"] != 0:
                raise AssertionError("language stage should not use history")
        scores = agent.score_results_structured(observations)
        rows = []
        for turn, (obs, score) in enumerate(zip(observations, scores)):
            canonical = np.zeros(4)
            for key, value in zip(obs.candidate_ids, score.candidate_logits):
                canonical[IDS.index(key)] = value
            rows.append(canonical)
            top = world.orders[turn][
                np.argsort(score.candidate_logits, kind="stable")[-2:]
            ]
            exact.append(bool(np.all(world.capabilities[top] == world.requests[turn])))
            probabilities = torch.softmax(torch.tensor(canonical), -1).numpy()
            mass.append(
                float(probabilities[world.capabilities == world.requests[turn]].sum())
            )
        result[world.identifier] = np.asarray(rows)
    assert before == trainable_digest(agent) and updates == agent.update_steps
    return result, dict(
        exact_suitable_pair=float(np.mean(exact)),
        compatible_probability_mass=float(np.mean(mass)),
    )


def evaluate_policy(numeric, world, language_logits, method, seed, history, trace):
    context = AgentContext(
        "evaluation", world.identifier, "validation", max_events=history
    )
    rng = seeded(seed, 33, int(world.identifier.split(":")[-1]))
    rewards = []
    regret = []
    suitable = []
    optimal = []
    for turn in range(len(world.requests)):
        oracle = world.capabilities == world.requests[turn]
        language_allowed = np.zeros(4, dtype=bool)
        order = world.orders[turn]
        top = order[np.argsort(language_logits[turn, order], kind="stable")[-2:]]
        language_allowed[top] = True
        allowed = oracle if method.startswith("oracle_") else language_allowed
        if method in ("numeric_only", "random"):
            allowed = np.ones(4, dtype=bool)
        if method in ("random", "language_only"):
            probabilities = allowed.astype(float) / allowed.sum()
        elif method.endswith("rolling"):
            probabilities = rolling_probabilities(context, allowed)
        else:
            obs = make_observation(world, turn, context)
            obs = replace(
                obs,
                legal_action_mask=np.array(
                    [allowed[IDS.index(key)] for key in obs.candidate_ids]
                ),
            )
            scores = numeric.action_result_structured(obs, deterministic=True)
            values = np.zeros(4)
            for key, value in zip(obs.candidate_ids, scores.candidate_values):
                values[IDS.index(key)] = value
            probabilities = epsilon_probabilities(values, allowed)
        action = int(rng.choice(4, p=probabilities))
        reward = world.reward(turn, action)
        utility = world.utility(turn)
        best = int(np.argmax(utility))
        rewards.append(reward)
        regret.append(float(utility[best] - utility[action]))
        suitable.append(bool(oracle[action]))
        optimal.append(action == best)
        trace.write(
            json.dumps(
                dict(
                    trial=world.identifier,
                    turn=turn,
                    method=method,
                    request=world.request(turn),
                    action=IDS[action],
                    probabilities=probabilities.tolist(),
                    compatible=bool(oracle[action]),
                    reward=reward,
                    history_limit=history,
                    history_before_action=[
                        dict(action=e.action, reward=e.reward) for e in context.events
                    ],
                )
            )
            + "\n"
        )
        context = append_feedback(context, world, turn, action)
    return dict(
        trial_id=world.identifier,
        family=world.family,
        compatibility=float(np.mean(suitable)),
        optimal_action_fraction=float(np.mean(optimal)),
        mean_reward=float(np.mean(rewards)),
        mean_expected_regret=float(np.mean(regret)),
    )


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    if (
        len(set(seeds)) != len(seeds)
        or min(seeds) < 0
        or min(
            args.rounds,
            args.train_trials,
            args.eval_trials,
            args.history,
            args.language_epochs,
            args.language_examples,
        )
        < 1
        or args.language_examples % 2
        or args.steps < 16
    ):
        raise ValueError(
            "positive budgets, steps >=16, even language examples and distinct nonnegative seeds required"
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
                versions={
                    n: metadata.version(n)
                    for n in ("torch", "numpy", "transformers", "peft")
                },
            ),
            indent=2,
        )
    )
    torch.set_num_threads(2)
    records = []
    for seed in seeds:
        with (output / f"numeric-training-{seed}.jsonl").open("x") as trace:
            numeric, numeric_config, numeric_losses = train_numeric(seed, args, trace)
        torch.manual_seed(seed)
        config = dict(
            llm_model=args.model,
            llm_revision=args.revision,
            llm_dtype=args.dtype,
            llm_max_tokens=args.max_tokens,
            llm_forward_batch_size=5,
            llm_lora_rank=4,
            llm_lora_alpha=8,
            lr=0.0001,
            value_backbone_gradient_scale=0.0,
        )
        language = algorithm_registry.get("llm_ppo").build_structured(
            SPEC, config, args.device
        )
        data = language_data(seed, args.language_examples, args.steps)
        training_before = training_semantics(language, data)
        losses = teach(language, data, args.language_epochs, seed)
        training_after = training_semantics(language, data)
        worlds = [make_world(seed, 2, i, args.steps) for i in range(args.eval_trials)]
        (output / f"worlds-{seed}.json").write_text(
            json.dumps([w.audit() for w in worlds], indent=2)
        )
        logits, semantics = semantic_scores(language, worlds)
        record = dict(
            seed=seed,
            numeric_losses=numeric_losses,
            language_losses=losses,
            training_semantics_before=training_before,
            training_semantics_after=training_after,
            semantics=semantics,
            evaluation={},
        )
        before = trainable_digest(QView(numeric))
        updates = numeric.update_steps
        with (output / f"evaluation-{seed}.jsonl").open("x") as trace:
            for method in (
                "random",
                "numeric_only",
                "language_only",
                "combined",
                "language_rolling",
                "oracle_numeric",
                "oracle_rolling",
            ):
                record["evaluation"][method] = [
                    evaluate_policy(
                        numeric,
                        w,
                        logits[w.identifier],
                        method,
                        seed,
                        args.history,
                        trace,
                    )
                    for w in worlds
                ]
            record["evaluation"]["combined_no_history"] = [
                evaluate_policy(
                    numeric, w, logits[w.identifier], "combined", seed, 0, trace
                )
                for w in worlds
            ]
        assert (
            before == trainable_digest(QView(numeric))
            and updates == numeric.update_steps
        )
        export_structured_policy_bundle(
            output / f"numeric-{seed}",
            agent=numeric,
            spec=SPEC,
            algorithm="structured_dqn",
            agent_config=numeric_config,
        )
        export_structured_policy_bundle(
            output / f"language-{seed}",
            agent=language,
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
                    semantics=semantics,
                    reward={
                        k: float(np.mean([t["mean_reward"] for t in v]))
                        for k, v in record["evaluation"].items()
                    },
                )
            ),
            flush=True,
        )
        del language, numeric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {}
    for label in records[0]["evaluation"]:
        summary[label] = {}
        for metric in (
            "compatibility",
            "optimal_action_fraction",
            "mean_reward",
            "mean_expected_regret",
        ):
            values = [
                float(np.mean([t[metric] for t in r["evaluation"][label]]))
                for r in records
            ]
            summary[label][metric] = dict(
                mean=float(np.mean(values)), seed_means=values
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="2003,2011,2027")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--train-trials", type=int, default=2)
    p.add_argument("--eval-trials", type=int, default=12)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--language-examples", type=int, default=128)
    p.add_argument("--language-epochs", type=int, default=4)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--max-tokens", type=int, default=1536)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
