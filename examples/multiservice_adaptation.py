"""Frozen language scores for multi-service, noisy, nonstationary tool routing.

Exploration is an explicit shared actor rule; it is not claimed to be learned.
Training and validation task tuples are disjoint. Confirmation uses new seeds.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from examples.adaptation_benchmark import SPEC, algorithm_registry, trainable_digest
from examples.adaptation_tool_sandbox import SortCall, invoke, tool_server
from jormungandr.llm_context import AgentContext, Experience
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_replay import StructuredReplayTransition
from jormungandr.structured_export import (
    export_structured_policy_bundle,
    load_structured_policy_bundle,
)


@dataclass(frozen=True)
class Task:
    services: int
    high: float
    low: float
    period: int  # zero means stable
    steps: int = 96


TRAIN = tuple(
    Task(n, hi, lo, period)
    for n in (3, 5)
    for hi, lo in ((1.0, 0.0), (0.9, 0.1), (0.75, 0.25))
    for period in (0, 12, 24)
)
TEST = {
    "four_stable": Task(4, 1.0, 0.0, 0),
    "eight_stable": Task(8, 1.0, 0.0, 0),
    "four_changes": Task(4, 1.0, 0.0, 16),
    "eight_changes": Task(8, 1.0, 0.0, 32),
    "four_noisy": Task(4, 0.85, 0.15, 16),
    "eight_noisy": Task(8, 0.85, 0.15, 32),
    "eight_harder": Task(8, 0.7, 0.3, 32),
}
CONFIRM = {
    "four_stable": Task(4, 1.0, 0.0, 0, 112),
    "eight_stable": Task(8, 1.0, 0.0, 0, 112),
    "four_changes": Task(4, 1.0, 0.0, 20, 112),
    "eight_changes": Task(8, 1.0, 0.0, 28, 112),
    "four_noisy": Task(4, 0.8, 0.2, 20, 112),
    "eight_noisy": Task(8, 0.8, 0.2, 28, 112),
    "eight_harder": Task(8, 0.65, 0.35, 28, 112),
}
CONFIRM_REQUEST = "Route this job to a service, taking account of its recorded results."
REQUEST = "Choose a service to complete the job using its observed outcomes."
HELD_REQUEST = (
    "Select an endpoint for this job. Consider the evidence from previous calls."
)
WINDOW = 16
EPSILON = 0.1


@dataclass
class World:
    identifier: str
    task: Task
    ids: tuple
    best: np.ndarray
    probabilities: np.ndarray
    outcomes: np.ndarray
    orders: np.ndarray
    exploration: np.ndarray
    probes: np.ndarray
    split: str
    request: str


def world(seed, task, index, split="validation", renamed=False):
    namespace = 1 if split == "train" else 2
    rng = np.random.default_rng(np.random.SeedSequence([seed, namespace, index]))
    n, steps = task.services, task.steps
    best = np.empty(steps, dtype=int)
    current = int(rng.integers(n))
    for t in range(steps):
        if t and task.period and t % task.period == 0:
            current = (current + int(rng.integers(1, n))) % n
        best[t] = current
    probabilities = np.full((steps, n), task.low)
    probabilities[np.arange(steps), best] = task.high
    outcomes = rng.random((steps, n)) < probabilities
    # Naming/order/exploration RNGs are independent of latent task generation.
    public = np.random.default_rng(
        np.random.SeedSequence([seed, namespace, index, 811])
    )
    ids = tuple(f"svc-{i}" for i in range(n))
    if split == "train":
        prefix = ("svc", "worker", "node")[int(public.integers(3))]
        ids = tuple(
            f"{prefix}-{x}" for x in public.choice(9000, n, replace=False) + 1000
        )
    if renamed:
        ids = tuple(
            f"endpoint-{x}" for x in public.choice(9000, n, replace=False) + 1000
        )
    order_rng = np.random.default_rng(
        np.random.SeedSequence([seed, namespace, index, 823])
    )
    return World(
        f"{seed}:{namespace}:{index}",
        task,
        ids,
        best,
        probabilities,
        outcomes,
        np.stack([order_rng.permutation(n) for _ in range(steps + 1)]),
        order_rng.random(steps),
        order_rng.integers(n, size=steps),
        split,
        REQUEST if split == "train" else HELD_REQUEST,
    )


def context(w):
    return AgentContext(
        "actor-" + w.identifier, w.identifier, w.split, max_events=WINDOW
    )


def evidence(c, key, turn, ablation="history"):
    events = () if ablation == "no_history" else c.events
    matching = [e for e in events if e.action == key]
    rewards = [
        1 - e.reward if ablation == "flipped_feedback" else e.reward for e in matching
    ]
    return rewards, (turn - matching[-1].timestep if matching else WINDOW + 1)


def observe(w, turn, c, ablation="history", reverse=False):
    if ablation == "last_only":
        c = replace(c, events=c.events[-1:])
    ids = tuple(w.ids[i] for i in w.orders[turn])
    if reverse:
        ids = ids[::-1]
    descriptions = {}
    for key in ids:
        rewards, age = evidence(c, key, turn, ablation)
        descriptions[key] = (
            f"Execute job. Recent observed successes: {int(sum(rewards))}; "
            f"failures: {len(rewards)-int(sum(rewards))}; "
            f"calls since last observation: {age}. "
            "Results oldest to newest: "
            + (",".join(str(int(r)) for r in rewards) or "none")
        )
    # Counts are public, deterministic summaries. Full causal source events are
    # retained for auditing; the optional local serializer binds each to its ID.
    events = c.events
    if ablation == "no_history":
        events = ()
    elif ablation == "flipped_feedback":
        events = tuple(
            replace(e, reward=1 - e.reward, outcome="pass" if e.reward == 0 else "fail")
            for e in events
        )
    meta = replace(c, events=events).metadata(w.request, descriptions)
    n = len(ids)
    return EntityCandidateObservation(
        global_features=np.zeros(1),
        entity_features=np.zeros((n, 3)),
        entity_type_ids=np.zeros(n, dtype=np.int64),
        entity_ids=ids,
        candidate_features=np.zeros((n, 3)),
        candidate_ids=ids,
        candidate_entity_indices=np.arange(n),
        legal_action_mask=np.ones(n, dtype=bool),
        metadata=meta,
    )


def append(c, t, key, reward):
    return c.append(
        Experience(
            "session", t, "job", key, "pass" if reward else "fail", float(reward), 0
        )
    )


def collect(seed, trials):
    rng = np.random.default_rng(seed + 1900)
    replay = []
    for i in range(trials):
        task = TRAIN[i % len(TRAIN)]
        w = world(seed, task, i, "train")
        c = context(w)
        for t in range(task.steps):
            obs = observe(w, t, c)
            # Mixture of random and evidence-seeking behavior covers both sparse
            # exploration and repeated sampling; labels are actual outcomes only.
            values = [score(c, key, t, "window_mean") for key in w.ids]
            a = (
                int(rng.integers(task.services))
                if rng.random() < 0.5
                else int(np.argmax(values))
            )
            reward = float(w.outcomes[t, a])
            nxt = append(c, t, w.ids[a], reward)
            replay.append(
                StructuredReplayTransition(
                    obs,
                    w.ids[a],
                    reward,
                    observe(w, t + 1, nxt),
                    t == task.steps - 1,
                    c.actor_id,
                    "session",
                    t,
                    0,
                    split="train",
                )
            )
            c = nxt
    return replay


def score(c, key, t, method):
    values, age = evidence(c, key, t)
    if method == "last_outcome":
        return values[-1] if values else 0.5
    if method == "window_mean":
        return (sum(values) + 1) / (len(values) + 2)
    if method == "window_ucb":
        return (sum(values) + 1) / (len(values) + 2) + 0.35 * np.sqrt(
            np.log(WINDOW + 1) / (len(values) + 1)
        )
    raise ValueError(method)


def replay_buckets(replay):
    """Stratify by evidence count, never by reward or the hidden best service."""
    buckets = [[], [], []]
    for row in replay:
        count = sum(
            e["action"] == row.candidate_id
            for e in row.observation.metadata["llm"]["history"]
        )
        buckets[min(count, 2)].append(row)
    if not all(buckets):
        raise ValueError(
            "training collection must cover zero, one and multiple observations"
        )
    return buckets


def select(w, t, scores, ids):
    # Same epsilon exploration for language and statistical controls. No hidden
    # state, forced oracle probe, or reset on change is used by the actor.
    if w.exploration[t] < EPSILON:
        return int(w.probes[t]), True
    # Exact score ties must not turn catalog order into an exploration policy.
    chosen = max(range(len(ids)), key=lambda i: (float(scores[i]), ids[i]))
    return w.ids.index(ids[chosen]), False


def evaluate(agent, worlds, method, output=None, *, reverse=False, http=False):
    contexts = [context(w) for w in worlds]
    before = None if agent is None else (trainable_digest(agent), agent.update_steps)
    rows, rewards, regrets, best_choices, exploration_count = [], [], [], [], 0
    slot = {}
    no_history_scores = None
    server = (
        tool_server(
            {key for w in worlds for key in w.ids},
            lambda key: slot["world"].outcomes[
                slot["turn"], slot["world"].ids.index(key)
            ],
        )
        if http
        else None
    )
    from contextlib import nullcontext

    with server if server is not None else nullcontext(None) as url:
        for t in range(worlds[0].task.steps):
            obs = [
                observe(
                    w,
                    t,
                    c,
                    (
                        method
                        if method in ("no_history", "flipped_feedback", "last_only")
                        else "history"
                    ),
                    reverse,
                )
                for w, c in zip(worlds, contexts)
            ]
            if agent is not None:
                # Bounded outer batches avoid retaining an entire test suite on GPU.
                if no_history_scores is None:
                    results = []
                    for start in range(0, len(obs), 4):
                        results.extend(
                            agent.action_results_structured(
                                obs[start : start + 4], deterministic=True
                            )
                        )
                    if (
                        method == "no_history"
                        and agent.policy.candidate_context == "local"
                    ):
                        # Public inputs are constant without history. Local scores
                        # are independent of row order, covered by offline tests.
                        no_history_scores = [
                            dict(zip(o.candidate_ids, r.candidate_values))
                            for o, r in zip(obs, results)
                        ]
                else:
                    results = [
                        SimpleNamespace(
                            candidate_values=tuple(
                                scores[key] for key in o.candidate_ids
                            )
                        )
                        for o, scores in zip(obs, no_history_scores)
                    ]
            rr, gg, bb = [], [], []
            for i, (w, c, o) in enumerate(zip(worlds, contexts, obs)):
                values = (
                    results[i].candidate_values
                    if agent is not None
                    else (
                        [0.0] * len(o.candidate_ids)
                        if method == "random"
                        else [score(c, key, t, method) for key in o.candidate_ids]
                    )
                )
                a, explored = select(w, t, values, o.candidate_ids)
                if method == "random":
                    a, explored = int(w.probes[t]), True
                reward = float(w.outcomes[t, a])
                if http:
                    slot.update(world=w, turn=t)
                    call = SortCall(w.ids[a], (3, -t, 2, 3, 0))
                    reward, response = invoke(url, call)
                    assert reward == float(w.outcomes[t, a])
                regret = float(w.task.high - w.probabilities[t, a])
                contexts[i] = append(c, t, w.ids[a], reward)
                rr.append(reward)
                gg.append(regret)
                bb.append(float(a == w.best[t]))
                exploration_count += explored
                if output is not None:
                    rows.append(
                        dict(
                            trial=w.identifier,
                            turn=t,
                            action=w.ids[a],
                            reward=reward,
                            regret=regret,
                            best_selected=a == int(w.best[t]),
                            explored=explored,
                            history=o.metadata["llm"]["history"],
                            candidate_ids=o.candidate_ids,
                            scores=tuple(float(v) for v in values),
                            descriptions=o.metadata["llm"]["candidate_descriptions"],
                            **(
                                {"call": asdict(call), "response": response}
                                if http
                                else {}
                            ),
                        )
                    )
            rewards.append(rr)
            regrets.append(gg)
            best_choices.append(bb)
    if agent is not None:
        assert before == (trainable_digest(agent), agent.update_steps)
    reward_array = np.array(rewards).T
    best_array = np.array(best_choices).T
    delays = []
    censored = 0
    for i, w in enumerate(worlds):
        for change in (
            range(w.task.period, w.task.steps, w.task.period) if w.task.period else []
        ):
            end = min(change + w.task.period, w.task.steps)
            hit = np.flatnonzero(best_array[i, change:end])
            delays.append(int(hit[0]) if len(hit) else end - change)
            censored += not len(hit)
    per_trial = reward_array.mean(1)
    result = dict(
        trial_count=len(worlds),
        mean_reward=float(per_trial.mean()),
        reward_standard_error=(
            float(per_trial.std(ddof=1) / np.sqrt(len(worlds)))
            if len(worlds) > 1
            else 0.0
        ),
        cumulative_reward=float(reward_array.sum(1).mean()),
        pseudo_regret=float(np.array(regrets).sum(0).mean()),
        best_service_fraction=float(best_array.mean()),
        exploration_fraction=exploration_count / reward_array.size,
        mean_first_best_selection_delay=float(np.mean(delays)) if delays else None,
        unrecovered_change_count=int(censored),
        change_count=len(delays),
        curve=reward_array.mean(0).tolist(),
        per_trial_reward=per_trial.tolist(),
        weights_frozen=True,
    )
    if output is not None:
        Path(output).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return result


def suite(
    agent, seed, trials, out, *, controls=True, sandbox=True, stage="development"
):
    if agent is not None:
        # Inference retains no backward graph, so it can use larger microbatches.
        agent.q.forward_batch_size = 16
    result = {}
    tasks = TEST if stage == "development" else CONFIRM
    for j, (name, task) in enumerate(tasks.items()):
        worlds = [
            world(seed + 100000 + j, task, i, renamed=True) for i in range(trials)
        ]
        if stage == "confirmation":
            worlds = [
                replace(
                    w,
                    request=CONFIRM_REQUEST,
                    ids=tuple(key.replace("endpoint", "api") for key in w.ids),
                )
                for w in worlds
            ]
        methods = (
            ["language", "no_history", "flipped_feedback", "last_only"]
            if agent is not None
            else []
        )
        if controls:
            methods += ["random", "last_outcome", "window_mean", "window_ucb"]
        result[name] = {}
        for method in methods:
            model = (
                agent
                if method in ("language", "no_history", "flipped_feedback", "last_only")
                else None
            )
            cohort = (
                worlds[:4] if method in ("flipped_feedback", "last_only") else worlds
            )
            result[name][method] = evaluate(
                model, cohort, method, out / f"{name}-{method}.jsonl"
            )
        if agent is not None:
            result[name]["reordered"] = evaluate(
                agent, worlds[:4], "language", reverse=True
            )
            result[name]["control_reference"] = {
                "trial_count": min(4, len(worlds)),
                "mean_reward": float(
                    np.mean(result[name]["language"]["per_trial_reward"][:4])
                ),
            }
        print(
            json.dumps(
                {
                    "task": name,
                    "reward": {
                        k: round(v["mean_reward"], 4) for k, v in result[name].items()
                    },
                }
            ),
            flush=True,
        )
        (out / "partial-results.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
    if sandbox and agent is not None:
        task = Task(4, 0.85, 0.15, 16, 48)
        worlds = [world(seed + 800000, task, i, renamed=True) for i in range(4)]
        direct = evaluate(agent, worlds, "language")
        via_http = evaluate(
            agent, worlds, "language", out / "http-tools.jsonl", http=True
        )
        assert direct == via_http
        result["http_tools"] = {
            "evaluation": via_http,
            "matches_simulator": True,
            "calls": task.steps * len(worlds),
        }
    return result


def run(args):
    if (
        args.trials < 1
        or args.eval_trials < 2
        or args.updates < 0
        or args.batch_size < 1
    ):
        raise ValueError("positive budgets and at least two evaluation trials required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config = dict(
        llm_model="HuggingFaceTB/SmolLM2-135M",
        llm_revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        llm_dtype="bfloat16",
        llm_max_tokens=320,
        llm_forward_batch_size=2,
        llm_attention_implementation="sdpa",
        llm_lora_rank=4,
        llm_lora_alpha=8,
        llm_candidate_context="local",
        lr=3e-5,
        structured_microbatch_size=2,
        gamma=0.0,
        llm_target_tau=0.1,
        demonstration_weight=0.0,
    )
    bundle_manifest = None
    if args.bundle:
        bundle_manifest = json.loads((Path(args.bundle) / "manifest.json").read_text())
        config = dict(bundle_manifest["agent_config"])
    protocol = dict(
        arguments=vars(args),
        config=config,
        training_tasks=[asdict(t) for t in TRAIN],
        test_tasks={
            k: asdict(v)
            for k, v in (TEST if args.stage == "development" else CONFIRM).items()
        },
        window=WINDOW,
        epsilon=EPSILON,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        supporting_sources={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "src/jormungandr/algorithms/llm_ppo.py",
                "src/jormungandr/algorithms/llm_dqn.py",
                "src/jormungandr/algorithms/structured_dqn.py",
                "examples/adaptation_tool_sandbox.py",
            )
        },
        note="Reward prediction (gamma=0); exploration actor rule is shared and not learned.",
        source_bundle_manifest=bundle_manifest,
        replay_sampling="25% zero, 25% one, 50% multiple chosen-service observations; not reward-stratified",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    for filename in protocol["supporting_sources"]:
        snapshot = out / "sources" / filename
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(Path(filename).read_bytes())
    if args.baselines_only:
        result = suite(
            None, args.seed, args.eval_trials, out, sandbox=False, stage=args.stage
        )
    else:
        agent = (
            load_structured_policy_bundle(args.bundle, device=args.device)[0]
            if args.bundle
            else algorithm_registry.get("llm_dqn").build_structured(
                SPEC, config, args.device
            )
        )
        losses = []
        if not args.bundle:
            replay = collect(args.seed, args.trials)
            buckets = replay_buckets(replay)
            rng = np.random.default_rng(args.seed + 2200)
            for step in range(args.updates):
                batch = []
                for _ in range(args.batch_size):
                    bucket = buckets[int(rng.choice(3, p=(0.25, 0.25, 0.5)))]
                    batch.append(bucket[int(rng.integers(len(bucket)))])
                update = agent.update_structured(batch)
                losses.append(float(update.loss))
                if (step + 1) % 64 == 0:
                    print(
                        json.dumps({"update": step + 1, "loss": float(update.loss)}),
                        flush=True,
                    )
            (out / "losses.json").write_text(json.dumps(losses))
            export_structured_policy_bundle(
                out / "language",
                agent=agent,
                spec=SPEC,
                algorithm="llm_dqn",
                agent_config=config,
            )
        result = suite(agent, args.seed, args.eval_trials, out, stage=args.stage)
        (out / "losses.json").write_text(json.dumps(losses))
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=4109)
    p.add_argument("--trials", type=int, default=72)
    p.add_argument("--eval-trials", type=int, default=12)
    p.add_argument("--updates", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--bundle")
    p.add_argument(
        "--stage", choices=("development", "confirmation"), default="development"
    )
    return p


if __name__ == "__main__":
    run(parser().parse_args())
