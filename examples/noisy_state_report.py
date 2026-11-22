"""Archive a noisy-state pilot and plot early learning against measured work."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--previous-run", type=Path)
    args = p.parse_args()
    records = json.loads((args.run / "results.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    budgets = sorted({r["training_episodes"] for r in records})
    methods = protocol["methods"]
    expected = len(protocol["args"]["seeds"]) * len(budgets) * len(methods)
    assert len(records) == expected, "Run incomplete"
    summaries = []
    for budget in budgets:
        for method in methods:
            rows = [
                r
                for r in records
                if r["training_episodes"] == budget and r["method"] == method
            ]
            baseline = {
                r["seed"]: r
                for r in records
                if r["training_episodes"] == budget and r["method"] == "reward"
            }
            points = []
            for i in range(len(rows[0]["points"])):
                rewards = [r["points"][i]["mean_reward"] for r in rows]
                delta = [
                    r["points"][i]["mean_reward"]
                    - baseline[r["seed"]]["points"][i]["mean_reward"]
                    for r in rows
                ]
                points.append(
                    {
                        "step": rows[0]["points"][i]["step"],
                        "mean_reward": float(np.mean(rewards)),
                        "seed_min": min(rewards),
                        "seed_max": max(rewards),
                        "paired_reward_deltas": delta,
                        "mean_training_seconds": float(
                            np.mean([r["points"][i]["training_seconds"] for r in rows])
                        ),
                    }
                )
            summaries.append(
                {"training_episodes": budget, "method": method, "points": points}
            )
    artifact = {"protocol": protocol, "summaries": summaries, "records": records}
    if args.previous_run:
        previous = json.loads((args.previous_run / "results.json").read_text())
        key = lambda r: (r["seed"], r["training_episodes"], r["method"])
        lookup = {
            key(r): {p["step"]: p["mean_reward"] for p in r["points"]} for r in records
        }
        assert all(
            lookup[key(r)][p["step"]] == p["mean_reward"]
            for r in previous
            for p in r["points"]
        )
        artifact["measurement_refinement"] = {
            "reason": "Initial checkpoints missed early convergence; repeat adds 8/16/32/64 without changing training",
            "shared_checkpoint_rewards_identical": True,
            "previous_protocol": json.loads(
                (args.previous_run / "protocol.json").read_text()
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(artifact, indent=2) + "\n")
    fig, axes = plt.subplots(len(budgets), 2, figsize=(11, 7), squeeze=False)
    for row, budget in enumerate(budgets):
        for method in methods:
            rows = [
                r
                for r in records
                if r["training_episodes"] == budget and r["method"] == method
            ]
            ys = np.array(
                [
                    [p["mean_reward"] for p in r["points"] if p["step"] <= 128]
                    for r in rows
                ]
            )
            for col, field in enumerate(("encoder_examples", "training_seconds")):
                xs = np.array(
                    [[p[field] for p in r["points"] if p["step"] <= 128] for r in rows]
                ).mean(0)
                ax = axes[row, col]
                (line,) = ax.plot(
                    xs, ys.mean(0), marker=".", label=method.replace("_", " ")
                )
                ax.fill_between(
                    xs, ys.min(0), ys.max(0), color=line.get_color(), alpha=0.10
                )
                ax.set_xlabel("Encoder examples" if col == 0 else "Training seconds")
                ax.set_ylabel("Expected reward")
                ax.set_title(
                    f"{budget * 256:,} collected interactions; first 128 outer updates"
                )
                ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Noisy state: early learning (3-seed means and min–max bands)")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
