"""Build vector figures for the paper from committed results; no retraining."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "latex" / "figures"


def main():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8})
    artifact = json.loads((ROOT / "benchmarks/delayed_credit.json").read_text())
    records, protocol = artifact["records"], artifact["protocol"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    for ax, delay in zip(axes, (0, 4, 12)):
        for n in protocol["n_steps"]:
            rows = [
                r
                for r in records
                if protocol["tasks"][r["task"]]["delay"] == delay and r["n"] == n
            ]
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
                protocol["checkpoints"],
                ys.mean(0),
                marker="o",
                markersize=3,
                label=f"{n}-step return",
            )
            ax.fill_between(
                protocol["checkpoints"],
                ys.min(0),
                ys.max(0),
                color=line.get_color(),
                alpha=0.12,
            )
        ax.axhline(
            np.mean([r["points"][-1]["cue_oracle_accuracy"] for r in rows]),
            color="black",
            linestyle=":",
            label="Observation rule",
        )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        ax.set_title(f"Reward delay: {delay} steps")
        ax.set_xlabel("Training updates")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Initial-choice accuracy")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "delayed_credit_learning.pdf")
    plt.close(fig)
    artifact = json.loads((ROOT / "benchmarks/shortlist_preference.json").read_text())
    records, protocol = artifact["records"], artifact["protocol"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.1), sharey=True)
    for ax, budget in zip(axes, sorted(protocol["args"]["exposures"])):
        for method in protocol["methods"]:
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
                markersize=3,
                label={"explicit_pairs": "Explicit comparisons", "biased_choices": "Biased clicks", "randomized_choices": "Clicks, randomized order"}[method],
            )
            ax.fill_between(
                protocol["checkpoints"],
                ys.min(0),
                ys.max(0),
                color=line.get_color(),
                alpha=0.12,
            )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        ax.set_title(f"{budget:,} feedback events")
        ax.set_xlabel("Training updates")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction of acceptable options recovered")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "shortlist_preference_learning.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
