"""Generic switching-tool benchmark; domain logic stays in this example.

Frozen-policy evaluation uses paired potential outcomes, fresh trial memory,
held-out seeds and both abrupt and gradual changes. No financial semantics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from importlib import metadata
import shutil
import sys
from pathlib import Path
import time

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.algorithms.structured_ppo import StructuredTransition
from jormungandr.llm_context import AgentContext, Experience
from jormungandr.structured import EntityCandidateObservation, StructuredPolicySpec
from jormungandr.structured_export import export_structured_policy_bundle


SCHEMA = "jormungandr.example.adaptation.v1"
SPEC = StructuredPolicySpec(1, 3, 3, 1)
ACTION_IDS = ("route-a", "route-b")


def seeded(seed, split, trial):
    """Independent RNG streams with disjoint train/evaluation namespaces."""
    return np.random.default_rng(np.random.SeedSequence([seed, split, trial]))


@dataclass(frozen=True)
class Scenario:
    identifier: str
    family: str
    probabilities: np.ndarray
    costs: np.ndarray
    latencies: np.ndarray
    successes: np.ndarray

    @property
    def utilities(self):
        return self.probabilities - self.costs - 0.1 * self.latencies

    @property
    def changes(self):
        best = self.utilities.argmax(axis=1)
        return np.flatnonzero(best[1:] != best[:-1]) + 1

    def reward(self, turn, action):
        return float(
            self.successes[turn, action]
            - self.costs[action]
            - 0.1 * self.latencies[action]
        )

    def public_request(self):
        return {"costs": self.costs.tolist(), "latencies": self.latencies.tolist()}

    def audit(self):
        return {
            "id": self.identifier,
            "family": self.family,
            "probabilities": self.probabilities.tolist(),
            "costs": self.costs.tolist(),
            "latencies": self.latencies.tolist(),
            "potential_successes": self.successes.tolist(),
            "best_action_changes": self.changes.tolist(),
        }


def make_scenario(seed, split, trial, steps):
    if steps < 16:
        raise ValueError("at least 16 steps are needed for changing environments")
    rng = seeded(seed, split, trial)
    family = "drift" if split == 2 and trial % 2 else "abrupt"
    high, low = rng.uniform(0.8, 0.95), rng.uniform(0.05, 0.2)
    first = rng.integers(2)
    if family == "abrupt":
        if split == 1:
            switches = [int(rng.integers(steps // 3, 2 * steps // 3))]
        else:
            switches = [
                int(rng.integers(steps // 4, steps // 2)),
                int(rng.integers(2 * steps // 3, 5 * steps // 6)),
            ]
        phase = (np.searchsorted(switches, np.arange(steps), side="right") + first) % 2
        mixture = phase.astype(float)
    else:
        # Unseen change family: a gradual inversion with randomly placed center.
        center = rng.uniform(0.4, 0.6) * steps
        width = rng.uniform(0.35, 0.6) * steps
        mixture = np.clip((np.arange(steps) - center) / width + 0.5, 0, 1)
        if first:
            mixture = 1 - mixture
    p0 = high * (1 - mixture) + low * mixture
    probabilities = np.column_stack([p0, high + low - p0])
    costs, latencies = rng.uniform(0.02, 0.12, 2), rng.uniform(0.05, 0.3, 2)
    # Draw all potential outcomes before any action. Different policies cannot
    # change the future random stream by selecting different actions.
    successes = (rng.random((steps, 2)) < probabilities).astype(float)
    for array in (probabilities, costs, latencies, successes):
        array.setflags(write=False)
    return Scenario(
        f"{seed}:{split}:{trial}", family, probabilities, costs, latencies, successes
    )


def observation(scenario, context, *, numeric_history=False, representation=None):
    if representation not in (None, "raw", "summary", "both"):
        raise ValueError("unknown history representation")
    if representation is not None:
        numeric_history = representation in ("summary", "both")
    entities = []
    for action in ACTION_IDS:
        recent = [e for e in context.events if e.action == action]
        mean = np.mean([e.reward for e in recent]) if recent else 0.0
        age = next(
            (i for i, e in enumerate(reversed(context.events)) if e.action == action),
            len(context.events),
        )
        entities.append(
            [
                float(mean),
                len(recent) / max(1, context.max_events),
                age / max(1, context.max_events),
            ]
        )
    # Numeric policy gets a causal compressed history. LLM gets raw history;
    # disabling LLM history also removes every history-derived numeric feature.
    if not numeric_history:
        entities = np.zeros((2, 3)).tolist()
    visible_context = context
    if representation == "summary":
        from dataclasses import replace

        visible_context = replace(context, events=())
    instruction = "Select a route. Reward = success - cost - 0.1 * latency. Reliability can change without notice."
    if representation is not None:
        instruction += (
            " Entity features are recent mean reward, observed count divided by history capacity,"
            " and requests since last observation divided by history capacity."
            " Zero counts indicate no observations. Features are zero when summaries are disabled."
        )
    return EntityCandidateObservation(
        global_features=np.zeros(1),
        entity_features=np.asarray(entities),
        entity_type_ids=np.zeros(2, dtype=np.int64),
        entity_ids=ACTION_IDS,
        candidate_features=np.array(
            [[i, scenario.costs[i], scenario.latencies[i]] for i in range(2)]
        ),
        candidate_ids=ACTION_IDS,
        candidate_entity_indices=np.arange(2),
        legal_action_mask=np.ones(2, dtype=bool),
        metadata=visible_context.metadata(
            instruction,
            {
                action: f"Call tool {i}. Cost {scenario.costs[i]:.4f}; latency {scenario.latencies[i]:.4f}."
                for i, action in enumerate(ACTION_IDS)
            },
        ),
    )


def rolling_probabilities(context, *, epsilon=0.1):
    estimates = []
    counts = []
    for action in ACTION_IDS:
        rewards = [e.reward for e in context.events if e.action == action]
        counts.append(len(rewards))
        estimates.append(float(np.mean(rewards)) if rewards else 0.0)
    untried = np.flatnonzero(np.asarray(counts) == 0)
    probabilities = np.zeros(2)
    if len(untried):
        probabilities[untried] = 1 / len(untried)
    else:
        winners = np.flatnonzero(np.isclose(estimates, max(estimates)))
        probabilities[:] = epsilon / 2
        probabilities[winners] += (1 - epsilon) / len(winners)
    return probabilities


def trainable_digest(agent):
    digest = hashlib.sha256()
    for name, parameter in agent.policy.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(
                parameter.detach()
                .cpu()
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
    return digest.hexdigest()


def trial_metrics(scenario, actions, rewards, probabilities, latencies):
    best = scenario.utilities.argmax(axis=1)
    actions = np.asarray(actions)
    regret = (
        scenario.utilities.max(axis=1)
        - scenario.utilities[np.arange(len(actions)), actions]
    )
    recovery = []
    postchange_regret = []
    ends = list(scenario.changes[1:]) + [len(actions)]
    for change, end in zip(scenario.changes, ends):
        delay = None
        # Descriptive recovery: first completed four-turn window with >=75%
        # optimal choices. Unrecovered changes remain explicitly censored.
        for start in range(change, end - 3):
            if np.mean(actions[start : start + 4] == best[start : start + 4]) >= 0.75:
                delay = int(start + 4 - change)
                break
        recovery.append(
            {"change": int(change), "delay": delay, "followup": int(end - change)}
        )
        postchange_regret.append(float(regret[change : min(end, change + 8)].sum()))
    return {
        "trial_id": scenario.identifier,
        "family": scenario.family,
        "mean_reward": float(np.mean(rewards)),
        "mean_expected_regret": float(regret.mean()),
        "optimal_action_fraction": float(np.mean(actions == best)),
        "nongreedy_action_fraction": float(
            np.mean([p[a] < max(p) - 1e-8 for p, a in zip(probabilities, actions)])
        ),
        "postchange_regret_8": float(np.mean(postchange_regret)),
        "recovery": recovery,
        "recovery_censored_fraction": float(
            np.mean([r["delay"] is None for r in recovery])
        ),
        "mean_decision_ms": float(np.mean(latencies)),
    }


def rollout(
    scenario,
    *,
    method,
    agent=None,
    history=4,
    seed=0,
    split="validation",
    trace=None,
    stage="",
    representation=None,
    observation_factory=observation,
):
    context = AgentContext("actor", scenario.identifier, split, max_events=history)
    rng = seeded(seed, 3, int(scenario.identifier.split(":")[-1]))
    trajectories, actions, rewards, distributions, timings = [], [], [], [], []
    # Explicit warm-up is not scored; it only initializes kernels/caches and
    # observes the same public initial state, with no environment feedback.
    if agent is not None:
        agent.score_results_structured(
            [
                observation_factory(
                    scenario,
                    context,
                    numeric_history=method == "numeric",
                    representation=representation,
                )
            ]
        )
    for turn in range(len(scenario.probabilities)):
        obs = observation_factory(
            scenario,
            context,
            numeric_history=method == "numeric",
            representation=representation,
        )
        if method == "llm":
            diagnostic = agent.policy.context_diagnostics(obs)
            if diagnostic["retained_events"] != len(obs.metadata["llm"]["history"]):
                raise ValueError(
                    "benchmark history would be truncated; increase --max-tokens"
                )
        start = time.perf_counter()
        if agent is not None:
            score = agent.score_results_structured([obs])[0]
            logits = np.asarray(score.candidate_logits, dtype=np.float64)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            value = score.value
        elif method == "rolling":
            probabilities, value = rolling_probabilities(context), 0.0
        elif method == "random":
            probabilities, value = np.full(2, 0.5), 0.0
        else:
            raise ValueError(f"unknown method: {method}")
        action = int(rng.choice(2, p=probabilities))
        timings.append((time.perf_counter() - start) * 1000)
        reward = scenario.reward(turn, action)
        episode_id = f"request-{turn}"
        if agent is not None and split == "train":
            trajectories.append(
                [
                    StructuredTransition(
                        episode_id,
                        0,
                        obs,
                        ACTION_IDS[action],
                        action,
                        float(np.log(probabilities[action])),
                        value,
                        reward,
                        True,
                    )
                ]
            )
        if trace is not None:
            trace.write(
                json.dumps(
                    {
                        "trial": scenario.identifier,
                        "split": split,
                        "turn": turn,
                        "method": method,
                        "stage": stage,
                        "policy_updates": (
                            agent.update_steps if agent is not None else None
                        ),
                        "history_limit": history,
                        "representation": representation,
                        "history_before_action": [asdict(e) for e in context.events],
                        "action": action,
                        "probabilities": probabilities.tolist(),
                        "reward": reward,
                        "success": float(scenario.successes[turn, action]),
                    }
                )
                + "\n"
            )
        context = context.append(
            Experience(
                episode_id,
                0,
                "route request",
                ACTION_IDS[action],
                "success" if scenario.successes[turn, action] else "failure",
                reward,
                agent.update_steps if agent is not None else 0,
            )
        )
        actions.append(action)
        rewards.append(reward)
        distributions.append(probabilities)
    return trajectories, trial_metrics(
        scenario, actions, rewards, distributions, timings
    )


def evaluate(
    agent,
    scenarios,
    *,
    method,
    history,
    seed,
    trace=None,
    stage="evaluation",
    representation=None,
    observation_factory=observation,
):
    before = trainable_digest(agent) if agent is not None else None
    updates = agent.update_steps if agent is not None else None
    results = [
        rollout(
            s,
            method=method,
            agent=agent,
            history=history,
            seed=seed,
            trace=trace,
            stage=stage,
            representation=representation,
            observation_factory=observation_factory,
        )[1]
        for s in scenarios
    ]
    if agent is not None and (
        trainable_digest(agent) != before or agent.update_steps != updates
    ):
        raise AssertionError(
            "evaluation modified learned parameters or learner counters"
        )
    return results


def train(agent, *, seed, rounds, trials, steps, history, method, trace=None):
    metrics = []
    for update in range(rounds):
        trajectories = []
        for trial in range(trials):
            scenario = make_scenario(seed, 1, update * trials + trial, steps)
            episodes, _ = rollout(
                scenario,
                method=method,
                agent=agent,
                history=history,
                seed=seed,
                split="train",
                trace=trace,
                stage="training",
            )
            trajectories.extend(episodes)
        result = agent.update_structured(trajectories)
        metrics.append(asdict(result))
        print(
            json.dumps(
                {
                    "method": method,
                    "seed": seed,
                    "update": update + 1,
                    "loss": result.loss,
                }
            ),
            flush=True,
        )
    return metrics


METRICS = (
    "mean_reward",
    "mean_expected_regret",
    "optimal_action_fraction",
    "nongreedy_action_fraction",
    "postchange_regret_8",
    "recovery_censored_fraction",
    "mean_decision_ms",
)


def summarize(seed_results):
    labels = sorted(seed_results[0]["evaluation"])
    output = {}
    for family in ("all", "abrupt", "drift"):
        output[family] = {}
        for label in labels:
            summary = {}
            for metric in METRICS:
                means = [
                    float(
                        np.mean(
                            [
                                t[metric]
                                for t in item["evaluation"][label]
                                if family == "all" or t["family"] == family
                            ]
                        )
                    )
                    for item in seed_results
                ]
                summary[metric] = {
                    "mean": float(np.mean(means)),
                    "seed_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
                    "seed_means": means,
                }
            output[family][label] = summary
    paired = {}
    if "llm_trained_history" in labels:
        for reference in (
            "rolling",
            "numeric_trained",
            "llm_untrained_history",
            "llm_trained_no_history",
        ):
            differences = []
            for item in seed_results:
                a = {
                    t["trial_id"]: t for t in item["evaluation"]["llm_trained_history"]
                }
                b = {t["trial_id"]: t for t in item["evaluation"][reference]}
                if a.keys() != b.keys():
                    raise ValueError(
                        "paired comparisons require identical scenario IDs"
                    )
                differences.append(
                    float(
                        np.mean(
                            [a[key]["mean_reward"] - b[key]["mean_reward"] for key in a]
                        )
                    )
                )
            paired[reference] = {
                "reward_difference_mean": float(np.mean(differences)),
                "seed_differences": differences,
            }
    return {
        "by_family": output,
        "paired_llm_reward_differences": paired,
        "uncertainty": "Seed SD is descriptive. No significance or confidence claim from this pilot.",
    }


def run(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be unique nonnegative integers")
    if min(args.rounds, args.train_trials, args.history) < 1 or args.eval_trials < 2:
        raise ValueError(
            "positive training budgets/history and at least two evaluation trials required"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "configuration.json").open("x") as file:
        json.dump({"schema": SCHEMA, **vars(args)}, file, indent=2)
    # Archive code and versions without embedding machine-specific paths.
    import jormungandr

    package_root = Path(jormungandr.__file__).parent
    sources = [(Path(__file__), Path("examples") / Path(__file__).name)]
    sources += [
        (path, Path("src/jormungandr") / path.relative_to(package_root))
        for path in sorted(package_root.rglob("*.py"))
    ]
    hashes = {}
    for source, relative in sources:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "numpy", "transformers", "peft"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    (output / "provenance.json").write_text(
        json.dumps({"versions": versions, "source_sha256": hashes}, indent=2)
    )
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    seed_results = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for seed in seeds:
        scenarios = [
            make_scenario(seed, 2, trial, args.steps)
            for trial in range(args.eval_trials)
        ]
        (output / f"scenarios-{seed}.json").write_text(
            json.dumps([s.audit() for s in scenarios], indent=2)
        )
        evaluation = {}
        learning = {}
        with (output / f"traces-{seed}.jsonl").open("w") as trace:
            for method in ("random", "rolling"):
                evaluation[method] = evaluate(
                    None,
                    scenarios,
                    method=method,
                    history=args.history,
                    seed=seed,
                    trace=trace,
                )
            torch.manual_seed(seed)
            numeric_config = {
                "structured_model_dim": 32,
                "structured_heads": 4,
                "structured_layers": 1,
                "epochs": 2,
                "minibatch_size": 16,
                "lr": 0.001,
                "gamma": 0.0,
            }
            numeric = algorithm_registry.get("structured_ppo").build_structured(
                SPEC, numeric_config, args.device
            )
            evaluation["numeric_untrained"] = evaluate(
                numeric,
                scenarios,
                method="numeric",
                history=args.history,
                seed=seed,
                trace=trace,
                stage="numeric_untrained",
            )
            learning["numeric"] = train(
                numeric,
                seed=seed,
                rounds=args.rounds,
                trials=args.train_trials,
                steps=args.steps,
                history=args.history,
                method="numeric",
                trace=trace,
            )
            evaluation["numeric_trained"] = evaluate(
                numeric,
                scenarios,
                method="numeric",
                history=args.history,
                seed=seed,
                trace=trace,
                stage="numeric_trained",
            )
            export_structured_policy_bundle(
                output / f"numeric-{seed}",
                agent=numeric,
                spec=SPEC,
                algorithm="structured_ppo",
                agent_config=numeric_config,
            )
            del numeric
            if not args.no_llm:
                torch.manual_seed(seed)
                llm_config = {
                    "llm_model": args.model,
                    "llm_revision": args.revision,
                    "llm_dtype": args.dtype,
                    "llm_max_tokens": args.max_tokens,
                    "llm_forward_batch_size": 3,
                    "llm_lora_rank": 4,
                    "llm_lora_alpha": 8,
                    "epochs": 1,
                    "minibatch_size": 1,
                    "lr": args.learning_rate,
                    "gamma": 0.0,
                    "value_backbone_gradient_scale": 0.0,
                }
                llm = algorithm_registry.get("llm_ppo").build_structured(
                    SPEC, llm_config, args.device
                )
                evaluation["llm_untrained_history"] = evaluate(
                    llm,
                    scenarios,
                    method="llm",
                    history=args.history,
                    seed=seed,
                    trace=trace,
                    stage="llm_untrained_history",
                )
                learning["llm"] = train(
                    llm,
                    seed=seed,
                    rounds=args.rounds,
                    trials=args.train_trials,
                    steps=args.steps,
                    history=args.history,
                    method="llm",
                    trace=trace,
                )
                evaluation["llm_trained_history"] = evaluate(
                    llm,
                    scenarios,
                    method="llm",
                    history=args.history,
                    seed=seed,
                    trace=trace,
                    stage="llm_trained_history",
                )
                evaluation["llm_trained_no_history"] = evaluate(
                    llm,
                    scenarios,
                    method="llm",
                    history=0,
                    seed=seed,
                    trace=trace,
                    stage="llm_trained_no_history",
                )
                export_structured_policy_bundle(
                    output / f"llm-{seed}",
                    agent=llm,
                    spec=SPEC,
                    algorithm="llm_ppo",
                    agent_config=llm_config,
                )
                del llm
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        seed_results.append(
            {"seed": seed, "evaluation": evaluation, "learning": learning}
        )
        (output / f"result-{seed}.json").write_text(
            json.dumps(seed_results[-1], indent=2, allow_nan=False)
        )
        print(json.dumps({"completed_seed": seed}), flush=True)
    result = {
        "schema": SCHEMA,
        "configuration": vars(args),
        "results": summarize(seed_results),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else None
        ),
        "training_steps_per_method_per_seed": args.rounds
        * args.train_trials
        * args.steps,
        "evaluation_trials_per_seed": args.eval_trials,
        "financial_result": False,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"summary": str(output / "summary.json")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/adaptation-benchmark")
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--train-trials", type=int, default=2)
    parser.add_argument("--eval-trials", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument(
        "--revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
