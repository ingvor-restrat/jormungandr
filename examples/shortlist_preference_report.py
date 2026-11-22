"""Archive a completed preference study and plot shortlist quality."""

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
    protocol = json.loads((args.run / "protocol.json").read_text())
    records = json.loads((args.run / "results.json").read_text())
    budgets = sorted(protocol["args"]["exposures"])
    methods = protocol["methods"]
    assert len(records) == len(budgets) * len(methods) * len(
        protocol["args"]["seeds"]
    ), "incomplete study"
    if args.previous_run:
        previous = json.loads((args.previous_run / "results.json").read_text())

        def quality_curves(rows):
            return {
                (r["seed"], r["exposures"], r["method"]): [
                    {k: v for k, v in point.items() if k != "training_seconds"}
                    for point in r["points"]
                ]
                for r in rows
            }

        assert quality_curves(previous) == quality_curves(records)
        protocol = {
            **protocol,
            "repeat_verification": "All recorded quality metrics identical after shortlist illegal-mask compatibility fix; training recipe unchanged.",
        }
    metrics = (
        "top1_acceptable_rate",
        "top3_acceptable_recall",
        "best_option_accuracy",
        "pair_order_accuracy",
        "utility_regret",
    )
    summaries = []
    for budget in budgets:
        for method in methods:
            rows = [
                r for r in records if r["exposures"] == budget and r["method"] == method
            ]
            points = []
            for i, step in enumerate(protocol["checkpoints"]):
                point = {"updates": step}
                for metric in metrics:
                    values = [r["points"][i][metric] for r in rows]
                    point[metric] = {
                        "mean": float(np.mean(values)),
                        "seed_min": min(values),
                        "seed_max": max(values),
                    }
                points.append(point)
            summaries.append({"exposures": budget, "method": method, "points": points})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "timing_note": "CPU wall times are diagnostic only. No speed comparison is claimed.",
                "summaries": summaries,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, budget in zip(axes, budgets):
        for method in methods:
            rows = [
                r for r in records if r["exposures"] == budget and r["method"] == method
            ]
            ys = np.array(
                [[p["top3_acceptable_recall"] for p in r["points"]] for r in rows]
            )
            (line,) = ax.plot(
                protocol["checkpoints"],
                ys.mean(0),
                marker="o",
                label=method.replace("_", " "),
            )
            ax.fill_between(
                protocol["checkpoints"],
                ys.min(0),
                ys.max(0),
                color=line.get_color(),
                alpha=0.12,
            )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{budget:,} user events")
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Recall of 3 acceptable options in top 3")
    axes[1].legend(fontsize=8)
    fig.suptitle("Synthetic preference study: 3-seed means and min–max bands")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
