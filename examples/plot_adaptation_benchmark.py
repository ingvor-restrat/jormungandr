"""Plot recorded benchmark results; requires the optional [plot] dependencies."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "random": "Random",
    "rolling": "Rolling estimator",
    "numeric_untrained": "Numeric PPO, untrained",
    "numeric_trained": "Numeric PPO, trained",
    "llm_untrained_history": "LLM, untrained + history",
    "llm_trained_history": "LLM, trained + history",
    "llm_trained_no_history": "LLM, trained, no history",
}


def plot(summary_path, destination):
    summary = json.loads(Path(summary_path).read_text())
    data = summary["results"]["by_family"]["all"]
    names = [name for name in LABELS if name in data]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("mean_reward", "mean_expected_regret"),
        ("Reward per request — higher is better", "Expected regret — lower is better"),
    ):
        means = [data[name][metric]["mean"] for name in names]
        colors = ["#20756d" if name == "rolling" else "#2966a3" for name in names]
        ax.barh(y, means, color=colors, alpha=0.65, height=0.65)
        for row, name in enumerate(names):
            values = data[name][metric]["seed_means"]
            ax.scatter(
                values,
                row + np.linspace(-0.12, 0.12, len(values)),
                s=24,
                color="#17202a",
                zorder=3,
            )
        ax.set_title(title, fontsize=11)
        ax.axvline(0, color="#555", linewidth=0.7)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [LABELS[name] for name in names])
    axes[0].invert_yaxis()
    fig.suptitle("Frozen-policy adaptation pilot", fontsize=15)
    fig.text(
        0.5,
        0.02,
        "Bars: mean across training seeds. Dots: individual seed means.\n"
        "Paired held-out scenarios; this small pilot does not establish statistical significance.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--output", default="runs/adaptation-comparison.png")
    args = parser.parse_args()
    print(plot(args.summary, args.output))
