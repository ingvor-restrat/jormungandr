"""Archive paired delayed-credit results and plot accuracy by reward delay."""

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.run / "protocol.json").read_text())
    records = json.loads((args.run / "results.json").read_text())
    assert len(records) == len(protocol["args"]["seeds"]) * len(
        protocol["tasks"]
    ) * len(protocol["n_steps"]), "Run incomplete"
    summaries = []
    for name in protocol["tasks"]:
        for n in protocol["n_steps"]:
            rows = [r for r in records if r["task"] == name and r["n"] == n]
            baseline = {
                r["seed"]: r for r in records if r["task"] == name and r["n"] == 1
            }
            points = []
            for i, step in enumerate(protocol["checkpoints"]):
                points.append(
                    {
                        "updates": step,
                        "mean_accuracy": float(
                            np.mean([r["points"][i]["accuracy"] for r in rows])
                        ),
                        "mean_training_seconds": float(
                            np.mean([r["points"][i]["training_seconds"] for r in rows])
                        ),
                        "paired_accuracy_deltas": [
                            r["points"][i]["accuracy"]
                            - baseline[r["seed"]]["points"][i]["accuracy"]
                            for r in rows
                        ],
                    }
                )
            summaries.append({"task": name, "n": n, "points": points})
    archive = copy.deepcopy(records)
    for row in archive:
        for point in row["points"]:
            bits = point.pop("per_episode_correct")
            point["evaluation_episode_count"] = len(bits)
            point["per_episode_correct_bits_hex"] = (
                np.packbits(bits, bitorder="big").tobytes().hex()
            )
    artifact = {
        "protocol": protocol,
        "correctness_encoding": 'np.unpackbits(np.frombuffer(bytes.fromhex(hex_string), dtype=np.uint8), bitorder="big")[:evaluation_episode_count]',
        "summaries": summaries,
        "records": archive,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(artifact, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, delay in zip(axes, (0, 4, 12)):
        for n in protocol["n_steps"]:
            rows = [
                r
                for r in records
                if protocol["tasks"][r["task"]]["delay"] == delay and r["n"] == n
            ]
            # Average conditions within each seed before computing variability.
            ys = np.array(
                [
                    [
                        np.mean(
                            [
                                r["points"][i]["accuracy"]
                                for r in rows
                                if r["seed"] == seed
                            ]
                        )
                        for i in range(len(protocol["checkpoints"]))
                    ]
                    for seed in protocol["args"]["seeds"]
                ]
            )
            (line,) = ax.plot(
                protocol["checkpoints"], ys.mean(0), marker="o", label=f"n={n}"
            )
            ax.fill_between(
                protocol["checkpoints"],
                ys.min(0),
                ys.max(0),
                alpha=0.12,
                color=line.get_color(),
            )
        oracle = np.mean([r["points"][-1]["cue_oracle_accuracy"] for r in rows])
        ax.axhline(oracle, color="black", linestyle=":", label="Cue-sign oracle")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"Reward delay: {delay} steps")
        ax.set_xlabel("Optimizer updates (64 sampled transitions/update)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Initial-choice accuracy")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Native DQN: equal-weight conditions, 3-seed means and min–max bands")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
