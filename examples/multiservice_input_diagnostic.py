"""Actor-side canonical inputs for the single-job service-reliability example.

Names remain in the actor registry. The learned scorer receives a canonical ID
and task description, plus only that service's actual observed evidence. This
is not a semantic parser for arbitrary jobs and does not compute a policy score.
"""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from examples.multiservice_adaptation import REQUEST


class CanonicalServicePolicy:
    def __init__(self, agent, *, names=True, request=True):
        if agent.policy.candidate_context != "local":
            raise ValueError(
                "canonical service scoring requires local candidate context"
            )
        self.agent = agent
        self.names = names
        self.request = request

    @property
    def policy(self):
        return self.agent.policy

    @property
    def q(self):
        return self.agent.q

    @property
    def update_steps(self):
        return self.agent.update_steps

    def inputs(self, observation):
        raw = observation.metadata["llm"]
        result = []
        for i, key in enumerate(observation.candidate_ids):
            alias = "svc-1000" if self.names else key
            entity = int(
                np.asarray(observation.candidate_entity_indices[i]).reshape(-1)[0]
            )
            if entity < 0:
                raise ValueError("this example requires one associated service entity")
            context = {
                **raw,
                "observation": REQUEST if self.request else raw["observation"],
                "candidate_descriptions": {alias: raw["candidate_descriptions"][key]},
                "history": [
                    {**e, "action": alias} for e in raw["history"] if e["action"] == key
                ],
            }
            result.append(
                replace(
                    observation,
                    entity_features=observation.entity_features[[entity]],
                    entity_type_ids=observation.entity_type_ids[[entity]],
                    entity_ids=(
                        alias if self.names else observation.entity_ids[entity],
                    ),
                    candidate_features=observation.candidate_features[[i]],
                    candidate_ids=(alias,),
                    candidate_entity_indices=np.array([0]),
                    legal_action_mask=observation.legal_action_mask[[i]],
                    metadata={**observation.metadata, "llm": context},
                )
            )
        return result

    def action_results_structured(self, observations, *, deterministic=True):
        grouped = [self.inputs(o) for o in observations]
        inputs = [o for group in grouped for o in group]
        scores = []
        for start in range(0, len(inputs), 8):
            scores.extend(
                r.candidate_values[0]
                for r in self.agent.action_results_structured(
                    inputs[start : start + 8], deterministic=True
                )
            )
        results, offset = [], 0
        for obs in observations:
            values = tuple(scores[offset : offset + len(obs.candidate_ids)])
            offset += len(values)
            results.append(
                SimpleNamespace(
                    candidate_ids=obs.candidate_ids,
                    candidate_values=values,
                    candidate_id=obs.candidate_ids[
                        max(
                            range(len(values)),
                            key=lambda i: (values[i], obs.candidate_ids[i]),
                        )
                    ],
                )
            )
        return results


def run(args):
    """Matched post-hoc diagnostic, never used to update or select weights."""
    import json
    from pathlib import Path
    import torch
    from examples.multiservice_adaptation import CONFIRM_REQUEST, Task, evaluate, world
    from jormungandr.structured_export import load_structured_policy_bundle

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    agent, manifest = load_structured_policy_bundle(args.bundle, device=args.device)
    agent.q.forward_batch_size = 16
    worlds = [
        world(args.seed + 100002, Task(4, 1.0, 0.0, 20, 64), i, renamed=True)
        for i in range(4)
    ]
    worlds = [
        replace(
            w,
            request=CONFIRM_REQUEST,
            ids=tuple(key.replace("endpoint", "api") for key in w.ids),
        )
        for w in worlds
    ]
    evaluations = {}
    for names, request in ((False, False), (False, True), (True, False), (True, True)):
        key = f"names={names},request={request}"
        policy = CanonicalServicePolicy(agent, names=names, request=request)
        evaluations[key] = evaluate(policy, worlds, "language")
    result = {
        "post_hoc_diagnostic": True,
        "seed": args.seed,
        "bundle_sha256": manifest["artifact"]["sha256"],
        "evaluation": evaluations,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["mean_reward"] for k, v in evaluations.items()}), flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
