"""Learning sanity ladder: independent reward regression, DQN and rule imitation.

All task logic stays in examples. Run with python -m examples.learning_diagnostic.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from examples.adaptation_benchmark import (
    ACTION_IDS,
    SPEC,
    algorithm_registry,
    evaluate,
    make_scenario,
    observation,
    rolling_probabilities,
    rollout,
    seeded,
    trainable_digest,
)
from jormungandr.llm_context import AgentContext, Experience
from jormungandr.structured_export import export_structured_policy_bundle
from jormungandr.structured_replay import StructuredReplayTransition


def numeric_inputs(observations):
    return torch.tensor(
        np.stack(
            [
                np.concatenate([o.entity_features, o.candidate_features[:, 1:]], axis=1)
                for o in observations
            ]
        ),
        dtype=torch.float32,
    )


class IndependentQ:
    """Plain PyTorch reward regressor: no Jormungandr learner or model code.

    A shared per-route MLP learns chosen-action reward with uniform replay.
    All transitions terminate, so Q targets are simply observed rewards.
    """

    def __init__(self):
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(5, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.003)
        self.update_steps = 0

    def values(self, observations):
        with torch.no_grad():
            return self.policy(numeric_inputs(observations)).squeeze(-1).numpy()

    def update(self, transitions):
        values = self.policy(
            numeric_inputs([t.observation for t in transitions])
        ).squeeze(-1)
        actions = torch.tensor([t.candidate_index for t in transitions])
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        loss = torch.nn.functional.smooth_l1_loss(
            values[torch.arange(len(actions)), actions], rewards
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1
        return float(loss.detach())


class QView:
    """Expose explicit epsilon-greedy distributions to the paired evaluator."""

    def __init__(self, agent, epsilon=0.1):
        self.agent, self.epsilon = agent, epsilon
        self.policy = agent.policy if isinstance(agent, IndependentQ) else agent.q

    @property
    def update_steps(self):
        return self.agent.update_steps

    def score_results_structured(self, observations):
        if isinstance(self.agent, IndependentQ):
            values = self.agent.values(observations)
        else:
            values = [
                r.candidate_values
                for r in self.agent.action_results_structured(
                    observations, deterministic=True
                )
            ]
        results = []
        for row in values:
            probabilities = np.full(len(row), self.epsilon / len(row))
            winners = np.flatnonzero(np.isclose(row, max(row), rtol=0, atol=1e-8))
            probabilities[winners] += (1 - self.epsilon) / len(winners)
            results.append(
                SimpleNamespace(
                    candidate_logits=np.log(np.maximum(probabilities, 1e-30)),
                    value=float(max(row)),
                )
            )
        return results


def teacher_data(worlds, seed, history, split):
    """Teacher labels depend only on past chosen-action feedback."""
    data = []
    for scenario in worlds:
        context = AgentContext(
            "teacher", scenario.identifier, split, max_events=history
        )
        rng = seeded(seed, 5, int(scenario.identifier.split(":")[-1]))
        for turn in range(len(scenario.probabilities)):
            target = rolling_probabilities(context)
            data.append(
                (observation(scenario, context, representation="summary"), target)
            )
            action = int(rng.choice(2, p=target))
            context = context.append(
                Experience(
                    f"request-{turn}",
                    0,
                    "route request",
                    ACTION_IDS[action],
                    "success" if scenario.successes[turn, action] else "failure",
                    scenario.reward(turn, action),
                    0,
                )
            )
    return data


def cue_data(seed):
    """An explicit cue removes uncertainty about whether the input is useful."""
    scenario = make_scenario(seed, 1, 0, 16)
    data = []
    for action in (0, 1):
        context = AgentContext("cue", f"cue-{action}", "train", max_events=0)
        obs = observation(scenario, context, representation="summary")
        meta = context.metadata(
            f"The correct action is {ACTION_IDS[action]}. Select that action.",
            {key: key for key in ACTION_IDS},
        )
        data.append(
            (
                replace(
                    obs,
                    entity_features=np.zeros((2, 3)),
                    candidate_features=np.zeros((2, 3)),
                    metadata=meta,
                ),
                np.eye(2)[action],
            )
        )
    return data


def imitation_metrics(agent, data):
    before, updates = trainable_digest(agent), agent.update_steps
    scores = agent.score_results_structured([o for o, _ in data])
    logits = torch.tensor([s.candidate_logits for s in scores])
    targets = torch.tensor(np.stack([p for _, p in data]), dtype=torch.float32)
    logp = torch.log_softmax(logits, -1)
    decisive = targets.max(-1).values > 0.5
    agreement = (logp.argmax(-1) == targets.argmax(-1))[decisive]
    assert trainable_digest(agent) == before and agent.update_steps == updates
    return dict(
        cross_entropy=float(-(targets * logp).sum(-1).mean()),
        teacher_agreement=float(agreement.float().mean()) if len(agreement) else None,
        decisive_examples=int(decisive.sum()),
        examples=len(data),
    )


def teach(agent, data, epochs, seed):
    rng = seeded(seed, 6, 0)
    losses = []
    for _ in range(epochs):
        total = 0.0
        for index in rng.permutation(len(data)):
            obs, target = data[index]
            if obs.metadata["llm"]["split"] != "train":
                raise ValueError("supervision requires training context")
            agent.policy.train()
            output = agent.policy(agent._collate_observations([obs]))
            target = torch.tensor(target, dtype=torch.float32, device=agent.device)
            loss = -(target * torch.log_softmax(output.logits[0], -1)).sum()
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in agent.policy.parameters() if p.requires_grad], 1.0
            )
            agent.optimizer.step()
            agent.update_steps += 1
            total += float(loss.detach())
        losses.append(total / len(data))
    return losses


def train_q(agent, worlds, args, seed, trace, observation_factory=observation):
    view = QView(agent, epsilon=0.2)
    replay, losses = [], []
    rng = seeded(seed, 7, 0)
    for round_index in range(args.rounds):
        for scenario in worlds:
            episodes, _ = rollout(
                scenario,
                method="numeric",
                agent=view,
                history=args.history,
                representation="summary",
                split="train",
                seed=seed + round_index,
                trace=trace,
                stage="training",
                observation_factory=observation_factory,
            )
            for episode in episodes:
                t = episode[0]
                replay.append(
                    StructuredReplayTransition(
                        t.observation,
                        t.candidate_id,
                        t.reward,
                        t.observation,
                        True,
                        "actor",
                        t.episode_id,
                        0,
                        view.update_steps,
                    )
                )
        for _ in range(args.train_trials * args.steps):
            batch = [replay[i] for i in rng.integers(len(replay), size=32)]
            losses.append(
                agent.update(batch)
                if isinstance(agent, IndependentQ)
                else agent.update_structured(batch).loss
            )
    return losses


def run(args):
    seeds = [int(x) for x in args.seeds.split(",")]
    if (
        len(set(seeds)) != len(seeds)
        or min(seeds) < 0
        or min(args.rounds, args.train_trials, args.eval_trials, args.history) < 1
        or args.steps < 16
    ):
        raise ValueError(
            "distinct nonnegative seeds, positive budgets and steps >=16 required"
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
                    name: metadata.version(name)
                    for name in (
                        ["torch", "numpy"]
                        if args.skip_llm
                        else ["torch", "numpy", "transformers", "peft"]
                    )
                },
            ),
            indent=2,
        )
    )
    torch.set_num_threads(2)
    numeric_config = dict(
        structured_model_dim=32,
        structured_heads=4,
        structured_layers=1,
        lr=0.001,
        gamma=0.0,
    )
    llm_config = dict(
        llm_model=args.model,
        llm_revision=args.revision,
        llm_dtype=args.dtype,
        llm_max_tokens=1536,
        llm_forward_batch_size=3,
        llm_lora_rank=4,
        llm_lora_alpha=8,
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
        baselines = {
            name: evaluate(None, heldout, method=name, history=args.history, seed=seed)
            for name in ("random", "rolling")
        }
        (output / f"baselines-{seed}.json").write_text(json.dumps(baselines, indent=2))
        for method in ["independent_q", "dqn", "numeric_imitation"] + (
            [] if args.skip_llm else ["llm_imitation", "llm_cue"]
        ):
            torch.manual_seed(seed)
            algorithm = (
                "llm_ppo"
                if method.startswith("llm")
                else "structured_dqn" if method == "dqn" else "structured_ppo"
            )
            config = llm_config if method.startswith("llm") else numeric_config
            agent = (
                IndependentQ()
                if method == "independent_q"
                else algorithm_registry.get(algorithm).build_structured(
                    SPEC, config, args.device if method.startswith("llm") else "cpu"
                )
            )
            q_method = method in ("independent_q", "dqn")
            view = QView(agent) if q_method else agent
            record = dict(
                seed=seed,
                method=method,
                initial_digest=trainable_digest(view),
                evaluation={},
            )
            data = cue_data(seed) if method == "llm_cue" else train_data
            if not q_method:
                record["imitation_before"] = imitation_metrics(agent, data)
            with (output / f"trace-{seed}-{method}.jsonl").open("x") as trace:
                if q_method:
                    record["losses"] = train_q(agent, training, args, seed, trace)
                else:
                    record["losses"] = teach(
                        agent, data, 32 if method == "llm_cue" else args.rounds, seed
                    )
                    record["imitation_after"] = imitation_metrics(agent, data)
                    if method != "llm_cue":
                        record["imitation_heldout"] = imitation_metrics(
                            agent, test_data
                        )
                if method != "llm_cue":
                    for label, worlds, history in [
                        ("fit", training, args.history),
                        ("heldout", heldout, args.history),
                        ("heldout_no_history", heldout, 0),
                    ]:
                        record["evaluation"][label] = evaluate(
                            view,
                            worlds,
                            method="llm" if method.startswith("llm") else "numeric",
                            representation="summary",
                            history=history,
                            seed=seed,
                            trace=trace,
                            stage=label,
                        )
            record["final_digest"] = trainable_digest(view)
            record["updates"] = view.update_steps
            if method == "independent_q":
                torch.save(
                    agent.policy.state_dict(), output / f"independent-q-{seed}.pt"
                )
            else:
                export_structured_policy_bundle(
                    output / f"policy-{seed}-{method}",
                    agent=agent,
                    spec=SPEC,
                    algorithm=algorithm,
                    agent_config=config,
                )
            (output / f"result-{seed}-{method}.json").write_text(
                json.dumps(record, indent=2)
            )
            records.append(record)
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        method=method,
                        updates=view.update_steps,
                        imitation=record.get("imitation_after"),
                    )
                ),
                flush=True,
            )
            del view, agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="401,503,607")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--train-trials", type=int, default=2)
    p.add_argument("--eval-trials", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
