"""Send interleaved training and validation experience to Jormungandr."""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8811")
    parser.add_argument("--model-id", default="distributed-demo")
    parser.add_argument("--actor-id", default="actor-0")
    args = parser.parse_args()

    post_json(
        f"{args.url}/v1/models",
        {
            "model_id": args.model_id,
            "obs_dim": 4,
            "replay": {"capacity": 10_000},
            "validation": {"capacity": 1_000},
            "tensorboard": {"enabled": False},
            "learner": {
                "enabled": True,
                "min_replay": 8,
                "batch_size": 8,
                "validation_every": 4,
                "min_validation": 2,
                "checkpoint_every": 0,
            },
        },
    )

    items = []
    for timestep in range(12):
        split = "validation" if timestep % 5 == 4 else "train"
        items.append(
            {
                "split": split,
                "actor_id": args.actor_id,
                "episode_id": "demo-episode",
                "timestep": timestep,
                "policy_version": 0,
                "obs": [timestep / 12.0, 0.0, 1.0, -1.0],
                "action_idx": timestep % 3,
                "reward": float(timestep % 3 - 1),
                "next_obs": [(timestep + 1) / 12.0, 0.0, 1.0, -1.0],
                "done": timestep == 11,
                "aux": {"kind": "direction", "label": timestep % 3},
            }
        )

    result = post_json(
        f"{args.url}/v1/models/{args.model_id}/experience/add",
        {"schema": "jormungandr.experience.v1", "items": items},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
