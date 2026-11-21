"""Portable LLM-agent smoke: shared learner, isolated actors, fixed tool/scorer.

Synthetic routing rewards test plumbing, not financial performance or learned
adaptation. Run with --help; supply a local model or a pinned Hub revision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from jormungandr.llm_context import AgentContext, Experience
from jormungandr.service import JormungandrRuntime
from jormungandr.structured import (
    EntityCandidateObservation,
    entity_candidate_observation_to_payload,
)
from jormungandr.structured_trajectory import (
    StructuredActionFactor,
    StructuredJointTrajectoryStep,
    sample_structured_joint_action,
    structured_joint_step_to_payload,
)


def observe(context):
    return EntityCandidateObservation(
        global_features=np.zeros(1),
        entity_features=np.zeros((1, 1)),
        entity_type_ids=np.zeros(1, dtype=np.int64),
        entity_ids=("request",),
        candidate_features=np.array([[0.0], [1.0]]),
        candidate_ids=("route-a", "route-b"),
        legal_action_mask=np.ones(2, dtype=bool),
        metadata=context.metadata(
            "Choose a tool route. Recent outcomes may indicate which route currently succeeds.",
            {
                "route-a": "Execute the request through tool A.",
                "route-b": "Execute the request through tool B.",
            },
        ),
    )


def collect(runtime, model_id, contexts, *, episodes, episode_offset, rng):
    trajectories, records = [], []
    for episode_index in range(episodes):
        # Fixed executor/scorer changes an unobserved condition midway through
        # each collection block. Actor observations do not reveal the answer.
        for actor_index, context in enumerate(contexts):
            episode_id = f"{context.trial_id}:{context.actor_id}:{episode_offset + episode_index}"
            observation = observe(context)
            response = runtime.structured_policy_score(
                model_id,
                observations=[entity_candidate_observation_to_payload(observation)],
            )
            score = response["items"][0]
            selected = sample_structured_joint_action(
                observation,
                [StructuredActionFactor("tool-route", observation.candidate_ids)],
                score["candidate_logits"],
                behavior_value=score["behavior_value"],
                rng=rng,
            )
            action = selected.selected_candidate_ids[0]
            preferred = (actor_index + int(episode_index >= episodes // 2)) % 2
            reward = 1.0 if action == observation.candidate_ids[preferred] else -1.0
            outcome = "request succeeded" if reward > 0 else "request failed"
            next_context = context.append(
                Experience(
                    episode_id,
                    0,
                    "tool request",
                    action,
                    outcome,
                    reward,
                    response["policy_version"],
                )
            )
            step = StructuredJointTrajectoryStep(
                actor_id=context.actor_id,
                episode_id=episode_id,
                timestep=0,
                policy_version=response["policy_version"],
                observation=selected.observation,
                factors=selected.factors,
                joint_behavior_log_probability=selected.joint_log_probability,
                behavior_value=selected.behavior_value,
                reward=reward,
                next_observation=observe(next_context),
                terminated=True,
                split=context.split,
            )
            payload = structured_joint_step_to_payload(step)
            trajectories.append([payload])
            records.append(payload)
            contexts[actor_index] = next_context
    return trajectories, records


def run(args):
    if min(args.actors, args.rounds, args.episodes) < 1:
        raise ValueError("actors, rounds and episodes must be positive")
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError(
            "output already contains a completed run; choose another directory"
        )
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    representation = dict(
        global_dim=1, entity_dim=1, candidate_dim=1, entity_type_count=1
    )
    learner = {
        "algo": "llm_ppo",
        "enabled": True,
        "device": args.device,
        "min_trajectory_steps": args.actors * args.episodes,
        "max_trajectory_batch_steps": args.actors * args.episodes,
        "max_policy_lag": 0,
        "max_updates": args.rounds,
        "checkpoint_every": 0,
        "seed": args.seed,
        "plugin_config": {
            "llm_model": args.model,
            "llm_revision": args.revision,
            "llm_dtype": args.dtype,
            "llm_max_tokens": args.max_tokens,
            "llm_forward_batch_size": 2,
            "llm_lora_rank": 4,
            "epochs": 1,
            "minibatch_size": 1,
            "lr": args.learning_rate,
            "value_backbone_gradient_scale": 0.0,
        },
    }
    runtime = JormungandrRuntime(checkpoint_root=str(output / "checkpoints"))
    started = time.monotonic()
    all_records = []
    try:
        created = runtime.create_structured_model(
            model_id="language-agent",
            representation=representation,
            learner=learner,
            tensorboard_enabled=False,
        )
        print(
            json.dumps({"trainable_parameters": created["trainable_parameters"]}),
            flush=True,
        )
        contexts = [
            AgentContext(f"researcher-{i}", "training", "train", max_events=4)
            for i in range(args.actors)
        ]
        for round_index in range(args.rounds):
            episodes, records = collect(
                runtime,
                "language-agent",
                contexts,
                episodes=args.episodes,
                episode_offset=round_index * args.episodes,
                rng=np.random.default_rng(args.seed + round_index),
            )
            all_records.extend(records)
            runtime.structured_trajectory_add("language-agent", episodes)
            deadline = time.monotonic() + args.update_timeout
            while True:
                status = runtime.get_structured_model("language-agent")
                if status["last_error"]:
                    raise RuntimeError(status["last_error"])
                if status["updates"] >= round_index + 1:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("learner did not finish before update timeout")
                time.sleep(0.05)
            print(
                json.dumps({"round": round_index + 1, "updates": status["updates"]}),
                flush=True,
            )
        # Validation ingress exercises the runtime's separate held-out store.
        validation = [
            AgentContext(f"validation-{i}", "held-out", "validation", max_events=4)
            for i in range(args.actors)
        ]
        episodes, records = collect(
            runtime,
            "language-agent",
            validation,
            episodes=args.episodes,
            episode_offset=0,
            rng=np.random.default_rng(args.seed + 1000),
        )
        runtime.structured_trajectory_add("language-agent", episodes)
        all_records.extend(records)
        metrics = runtime.get_structured_metrics("language-agent")
        status = runtime.get_structured_model("language-agent")
        assert status["updates"] == args.rounds
        checkpoint = runtime.force_structured_checkpoint("language-agent")["checkpoint"]
        reference = runtime.structured_policy_score(
            "language-agent",
            observations=[
                entity_candidate_observation_to_payload(observe(validation[0]))
            ],
        )
    finally:
        runtime.close_all()
    # Reload through the same public runtime in frozen mode. No optimizer step
    # or training queue exists for this deployment.
    frozen = JormungandrRuntime(checkpoint_root=str(output / "checkpoints"))
    try:
        frozen.create_structured_model(
            model_id="frozen-language-agent",
            representation=representation,
            learner={**learner, "enabled": False},
            checkpoint_path=checkpoint,
            tensorboard_enabled=False,
        )
        frozen_updates = frozen.get_structured_model("frozen-language-agent")["updates"]
        restored = frozen.structured_policy_score(
            "frozen-language-agent",
            observations=[
                entity_candidate_observation_to_payload(observe(validation[0]))
            ],
        )
        if not np.allclose(
            reference["items"][0]["candidate_logits"],
            restored["items"][0]["candidate_logits"],
            atol=1e-5,
        ):
            raise AssertionError("checkpoint reload changed scores")
        evaluation = {}
        for history_limit in (0, 4):
            contexts = [
                AgentContext(
                    f"eval-{i}",
                    f"frozen-{history_limit}",
                    "validation",
                    max_events=history_limit,
                )
                for i in range(args.actors)
            ]
            _, records = collect(
                frozen,
                "frozen-language-agent",
                contexts,
                episodes=args.episodes,
                episode_offset=0,
                rng=np.random.default_rng(args.seed + 2000),
            )
            evaluation[f"history_{history_limit}_mean_reward"] = float(
                np.mean([r["reward"] for r in records])
            )
            all_records.extend(records)
        assert (
            frozen.get_structured_model("frozen-language-agent")["updates"]
            == frozen_updates
        )
    finally:
        frozen.close_all()
    summary = {
        "algorithm": "llm_ppo",
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "updates": args.rounds,
        "actors": args.actors,
        "checkpoint": checkpoint,
        "checkpoint_reload_verified": True,
        "elapsed_seconds": time.monotonic() - started,
        "evaluation": evaluation,
        "learning_metrics": metrics,
        "runtime_status": status,
        "financial_result": False,
        "adaptation_claim": False,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else None
        ),
    }
    (output / "transitions.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in all_records)
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({"summary": str(output / "summary.json"), "evaluation": evaluation}),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Local Hugging Face model directory or Hub model ID",
    )
    parser.add_argument(
        "--revision",
        help="Immutable 40-character Hub commit; required for remote models",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--output", default="runs/llm-agent-prototype")
    parser.add_argument("--actors", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--update-timeout", type=float, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
