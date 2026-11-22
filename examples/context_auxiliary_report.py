"""Package a completed development pilot and draw scientific learning curves."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k != "curve"}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path)
    args = parser.parse_args()
    records = json.loads((args.run / "results.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    artifact = {"protocol": protocol, "records": compact(records)}
    if args.previous_run:
        previous = json.loads((args.previous_run / "results.json").read_text())
        key = lambda r: (r["seed"], r["training_trials"], r["method"])
        curves = lambda rs: {
            key(r): [p["mean_reward"] for p in r["points"]] for r in rs
        }
        assert curves(previous) == curves(records), "Repeat changed reward curves"
        artifact["timing_correction"] = {
            "reason": "Shared backend warmup removes first-method startup skew",
            "all_mean_reward_curves_identical": True,
            "previous_protocol": json.loads(
                (args.previous_run / "protocol.json").read_text()
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(artifact, indent=2) + "\n")
    budgets = sorted({r["training_trials"] for r in records})
    fig, axes = plt.subplots(
        len(budgets), 2, figsize=(11, 4 * len(budgets)), squeeze=False
    )
    labels = {
        "reward": "Reward only",
        "contrastive": "Contrastive",
        "factual": "Factual",
        "reward_compute": "Extra reward updates",
    }
    for row, budget in enumerate(budgets):
        for method, label in labels.items():
            selected = [
                r
                for r in records
                if r["training_trials"] == budget and r["method"] == method
            ]
            ys = np.array([[p["mean_reward"] for p in r["points"]] for r in selected])
            for col, field in enumerate(("encoder_examples", "training_seconds")):
                xs = np.array([[p[field] for p in r["points"]] for r in selected]).mean(
                    axis=0
                )
                ax = axes[row, col]
                (line,) = ax.plot(xs, ys.mean(axis=0), marker="o", label=label)
                ax.fill_between(
                    xs,
                    ys.min(axis=0),
                    ys.max(axis=0),
                    alpha=0.12,
                    color=line.get_color(),
                )
                ax.set_xlabel(
                    "Encoder examples" if col == 0 else "Steady-state training seconds"
                )
                if col == 0:
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                ax.set_ylabel("Mean evaluation reward")
                ax.set_title(
                    f"{selected[0]['environment_interactions']:,} collected interactions"
                )
                ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Development pilot: means and seed min–max bands (3 seeds)")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
