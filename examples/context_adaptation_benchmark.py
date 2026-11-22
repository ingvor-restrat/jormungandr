"""Causal feedback adaptation with a frozen language Double-DQN at evaluation.

Two callable services, one fulfills the job. Hidden availability can swap.
This is an information-sufficient sanity task, not a full paper replication.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from examples.adaptation_benchmark import SPEC, algorithm_registry, trainable_digest
from jormungandr.llm_context import AgentContext, Experience
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_replay import StructuredReplayTransition
from jormungandr.structured_export import export_structured_policy_bundle

REQUEST = "Complete the job. Choose the service most likely to fulfill it using observed task outcomes."
RESERVED_REQUESTS = (
    "Please get this job done successfully. Select a service using the feedback from earlier attempts.",
    "Which service should handle this request? Use the recorded results to guide your choice.",
)


@dataclass(frozen=True)
class Trial:
    identifier: str
    ids: tuple
    winners: tuple
    orders: tuple
    change: int | None
    split: str
    group_history: bool = False
    request: str = REQUEST


def make_trial(seed, split, index, steps=12, *, renamed=False):
    # Opposite hidden winners share identical public inputs within each pair.
    rng = np.random.default_rng(np.random.SeedSequence([seed, split, index // 2]))
    change = int(rng.integers(3, steps - 2)) if (index // 2) % 2 else None
    winners = tuple(
        (index % 2) ^ int(change is not None and t >= change) for t in range(steps)
    )
    ids = ("node-x", "node-y") if renamed else ("service-a", "service-b")
    if renamed and split == 3:
        names = np.random.default_rng(
            np.random.SeedSequence([seed, split, index // 2, 817])
        )
        ids = tuple(
            f"endpoint-{i}" for i in names.choice(9000, size=2, replace=False) + 1000
        )
    return Trial(
        f"{seed}:{split}:{index}",
        ids,
        winners,
        tuple(tuple(int(i) for i in rng.permutation(2)) for _ in range(steps + 1)),
        change,
        "train" if split == 1 else "validation",
        request=(
            RESERVED_REQUESTS[(index // 4) % len(RESERVED_REQUESTS)]
            if split == 3
            else REQUEST
        ),
    )


def observation(world, turn, context, *, ablation="history"):
    events = context.events
    if ablation == "no_history":
        events = ()
    elif ablation == "flipped_feedback":
        events = tuple(
            replace(
                e,
                reward=1 - e.reward,
                outcome=(
                    "API executed; task succeeded"
                    if e.reward == 0
                    else "API executed; task failed"
                ),
            )
            for e in events
        )
    context = replace(context, events=events)
    ids = tuple(world.ids[i] for i in world.orders[turn])
    descriptions = {
        key: "Execute the job through this callable service." for key in ids
    }
    if world.group_history:
        for key in ids:
            matching = [e for e in events if e.action == key]
            evidence = "; ".join(
                f"Observed task reward {e.reward}: {e.outcome}" for e in matching
            )
            descriptions[key] += " Observed results for this service: " + (
                evidence or "No retained observation."
            )
    request = world.request
    return EntityCandidateObservation(
        global_features=np.zeros(1),
        entity_features=np.zeros((2, 3)),
        entity_type_ids=np.zeros(2, dtype=np.int64),
        entity_ids=ids,
        candidate_features=np.zeros((2, 3)),
        candidate_ids=ids,
        candidate_entity_indices=np.arange(2),
        legal_action_mask=np.ones(2, dtype=bool),
        metadata=context.metadata(
            request,
            descriptions,
        ),
    )


def context_for(world, history):
    return AgentContext(
        "actor-" + world.identifier, world.identifier, world.split, max_events=history
    )


def feedback(context, turn, key, reward, version=0):
    # No hidden trial/label index is serialized into the language history.
    return context.append(
        Experience(
            "session",
            turn,
            "Complete the job.",
            key,
            "API executed; task succeeded" if reward else "API executed; task failed",
            float(reward),
            version,
        )
    )


def collect(seed, trials, steps, history, *, group_history=False):
    rng = np.random.default_rng(seed + 9000)
    transitions = []
    for index in range(trials):
        world = replace(make_trial(seed, 1, index, steps), group_history=group_history)
        context = context_for(world, history)
        for t in range(steps):
            obs = observation(world, t, context)
            action = int(rng.integers(2))
            reward = float(action == world.winners[t])
            following = feedback(context, t, world.ids[action], reward)
            transitions.append(
                StructuredReplayTransition(
                    obs,
                    world.ids[action],
                    reward,
                    observation(world, t + 1, following),
                    t == steps - 1,
                    context.actor_id,
                    "session",
                    t,
                    0,
                    split="train",
                )
            )
            context = following
    return transitions


def evaluate(agent, worlds, history, output, *, method="history"):
    contexts = [context_for(w, history) for w in worlds]
    rewards = np.zeros((len(worlds), len(worlds[0].winners)))
    before, updates = trainable_digest(agent), agent.update_steps
    with output.open("w") as trace:
        for t in range(rewards.shape[1]):
            observations = [
                observation(w, t, c, ablation=method) for w, c in zip(worlds, contexts)
            ]
            results = agent.action_results_structured(observations, deterministic=True)
            for i, (world, obs, result) in enumerate(
                zip(worlds, observations, results)
            ):
                # Common forced initial probe; all later decisions come from the model.
                key = world.ids[0] if t == 0 else result.candidate_id
                reward = float(world.ids.index(key) == world.winners[t])
                rewards[i, t] = reward
                trace.write(
                    json.dumps(
                        dict(
                            trial=world.identifier,
                            turn=t,
                            action=key,
                            reward=reward,
                            q_values=result.candidate_values,
                            candidate_ids=obs.candidate_ids,
                            history=obs.metadata["llm"]["history"],
                            change=world.change,
                        )
                    )
                    + "\n"
                )
                contexts[i] = feedback(contexts[i], t, key, reward, agent.update_steps)
    assert before == trainable_digest(agent) and updates == agent.update_steps
    stable = [i for i, w in enumerate(worlds) if w.change is None]
    changed = [(i, w.change) for i, w in enumerate(worlds) if w.change is not None]
    return dict(
        first=float(rewards[:, 0].mean()),
        mean_reward=float(rewards.mean()),
        stable_after_first=float(rewards[stable, 1:].mean()),
        post_change_after_one=float(
            np.mean([rewards[i, c + 1 :].mean() for i, c in changed])
        ),
        curve=rewards.mean(0).tolist(),
        trials=rewards.tolist(),
    )


def memory_baseline(worlds):
    rewards = []
    for world in worlds:
        action = 0
        row = []
        for winner in world.winners:
            success = action == winner
            row.append(float(success))
            if not success:
                action = 1 - action
        rewards.append(row)
    return dict(
        mean_reward=float(np.mean(rewards)), curve=np.mean(rewards, axis=0).tolist()
    )


def run(args):
    if (
        args.steps < 8
        or args.trials < 4
        or args.trials % 4
        or args.eval_trials < 4
        or args.eval_trials % 4
    ):
        raise ValueError("steps >=8 and trial counts divisible by four required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "configuration.json").write_text(json.dumps(vars(args), indent=2))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config = dict(
        llm_model=args.model,
        llm_revision=args.revision,
        llm_dtype="float32",
        llm_max_tokens=1536,
        llm_forward_batch_size=12,
        llm_lora_rank=4,
        llm_lora_alpha=8,
        lr=args.lr,
        gamma=args.gamma,
        llm_target_tau=0.1,
        demonstration_weight=0.0,
    )
    agent = algorithm_registry.get("llm_dqn").build_structured(
        SPEC, config, args.device
    )
    split = 2 if args.stage == "development" else 3
    worlds = [
        replace(
            make_trial(args.seed + 100000, split, i, args.steps),
            group_history=args.group_history,
        )
        for i in range(args.eval_trials)
    ]
    before = evaluate(agent, worlds, args.history, out / "initial.jsonl")
    replay = collect(
        args.seed,
        args.trials,
        args.steps,
        args.history,
        group_history=args.group_history,
    )
    rng = np.random.default_rng(args.seed + 18000)
    metrics = []
    for step in range(args.updates):
        batch = [replay[i] for i in rng.integers(len(replay), size=args.batch_size)]
        result = agent.update_structured(batch)
        metrics.append(dict(result.metrics))
        if (step + 1) % 64 == 0:
            print(
                json.dumps(
                    dict(
                        update=step + 1,
                        loss=result.loss,
                        q=result.metrics["q_mean"],
                        target=result.metrics["target_mean"],
                    )
                ),
                flush=True,
            )
    evaluations = {}
    for method in ("history", "no_history", "flipped_feedback"):
        evaluations[method] = evaluate(
            agent, worlds, args.history, out / f"{method}.jsonl", method=method
        )
    renamed = [
        replace(
            make_trial(args.seed + 100000, split, i, args.steps, renamed=True),
            group_history=args.group_history,
        )
        for i in range(args.eval_trials)
    ]
    evaluations["renamed"] = evaluate(
        agent, renamed, args.history, out / "renamed.jsonl"
    )
    reordered = [
        replace(w, orders=tuple(order[::-1] for order in w.orders)) for w in worlds
    ]
    evaluations["reordered"] = evaluate(
        agent, reordered, args.history, out / "reordered.jsonl"
    )
    export_structured_policy_bundle(
        out / "language",
        agent=agent,
        spec=SPEC,
        algorithm="llm_dqn",
        agent_config=config,
    )
    result = dict(
        initial=before,
        evaluation=evaluations,
        memory_baseline=memory_baseline(worlds),
        training=metrics,
        targets=dict(stable_after_first=0.9, post_change_after_one=0.85),
        target_met=bool(
            all(
                evaluations[k]["stable_after_first"] >= 0.9
                and evaluations[k]["post_change_after_one"] >= 0.85
                for k in ("history", "renamed", "reordered")
            )
            and abs(evaluations["no_history"]["mean_reward"] - 0.5) < 1e-8
        ),
        weights_frozen_during_evaluation=True,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            dict(
                target_met=result["target_met"],
                evaluation={
                    k: {
                        m: v[m]
                        for m in (
                            "mean_reward",
                            "stable_after_first",
                            "post_change_after_one",
                        )
                    }
                    for k, v in evaluations.items()
                },
            )
        ),
        flush=True,
    )
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=3109)
    p.add_argument("--trials", type=int, default=128)
    p.add_argument("--eval-trials", type=int, default=24)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--history", type=int, default=1)
    p.add_argument(
        "--group-history",
        action="store_true",
        help="Attach only observed outcomes to the matching service description",
    )
    p.add_argument("--updates", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument(
        "--stage", choices=("development", "confirmation"), default="development"
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
