"""Measure QUBO frontier pruning on a seeded noisy branching search.

This is an algorithmic benchmark, not a learning-quality claim.  Every trial
constructs a complete tree so the best terminal branch is known.  Search sees
only a noisy, correlated proxy for each prefix and must decide which fixed-size
frontier to expand.  The comparison asks whether utility/diversity QUBO needs a
smaller beam than ordinary utility-only search for comparable oracle recovery.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence

import numpy as np

from jormungandr.search import (
    SearchNode,
    bounded_beam_search,
    build_frontier_pruner,
)


@dataclass(frozen=True)
class TreeConfig:
    branching_factor: int = 4
    depth: int = 6
    noise_scale: float = 0.9

    def __post_init__(self) -> None:
        if self.branching_factor < 2:
            raise ValueError("branching_factor must be at least two")
        if self.depth < 2:
            raise ValueError("depth must be at least two")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale cannot be negative")


@dataclass(frozen=True)
class TrialResult:
    seed: int
    strategy: str
    beam_width: int
    oracle_value: float
    selected_value: float
    regret: float
    exact_recovery: bool
    generated_candidates: int
    expanded_nodes: int
    terminal_evaluations: int
    selector_time_ms: float
    search_time_ms: float


class CorrelatedNoisyTree:
    """A finite tree with an exact oracle and a fallible search heuristic.

    Leaf value is accumulated from progressively smaller random branch
    increments.  A node's cheap utility is its subtree optimum plus noise.  A
    shared first-branch error makes sibling estimates correlated, modelling a
    value model that is systematically optimistic about one region.  QUBO can
    hedge that error by retaining dissimilar prefixes.
    """

    def __init__(self, seed: int, config: TreeConfig) -> None:
        self.seed = int(seed)
        self.config = config
        rng = np.random.default_rng(self.seed)
        self.paths_by_depth: dict[int, list[tuple[int, ...]]] = {0: [()]}
        path_value: dict[tuple[int, ...], float] = {(): 0.0}

        for depth in range(1, config.depth + 1):
            paths: list[tuple[int, ...]] = []
            increment_scale = 1.0 / math.sqrt(float(depth))
            for parent in self.paths_by_depth[depth - 1]:
                for branch in range(config.branching_factor):
                    path = parent + (branch,)
                    paths.append(path)
                    path_value[path] = path_value[parent] + float(
                        rng.normal(0.0, increment_scale)
                    )
            self.paths_by_depth[depth] = paths

        leaves = self.paths_by_depth[config.depth]
        self.leaf_values = {path: path_value[path] for path in leaves}
        self.subtree_best: dict[tuple[int, ...], float] = dict(self.leaf_values)
        for depth in range(config.depth - 1, -1, -1):
            for path in self.paths_by_depth[depth]:
                self.subtree_best[path] = max(
                    self.subtree_best[path + (branch,)]
                    for branch in range(config.branching_factor)
                )

        shared_error = rng.normal(
            0.0, config.noise_scale, size=config.branching_factor
        )
        self.utility: dict[tuple[int, ...], float] = {(): self.subtree_best[()]}
        for depth in range(1, config.depth + 1):
            for path in self.paths_by_depth[depth]:
                independent_error = float(
                    rng.normal(0.0, config.noise_scale * 0.35)
                )
                self.utility[path] = (
                    self.subtree_best[path]
                    + float(shared_error[path[0]])
                    + independent_error
                )

    @property
    def oracle_value(self) -> float:
        return float(self.subtree_best[()])

    @property
    def terminal_path_count(self) -> int:
        return len(self.paths_by_depth[self.config.depth])

    @property
    def exhaustive_candidate_count(self) -> int:
        return sum(
            len(self.paths_by_depth[depth])
            for depth in range(1, self.config.depth + 1)
        )

    def _embedding(self, path: tuple[int, ...]) -> np.ndarray:
        embedding = np.zeros(
            self.config.depth * self.config.branching_factor,
            dtype=np.float32,
        )
        for depth, branch in enumerate(path):
            embedding[depth * self.config.branching_factor + branch] = 1.0
        return embedding

    def node(self, path: tuple[int, ...]) -> SearchNode:
        return SearchNode(
            key="root" if not path else ".".join(str(item) for item in path),
            utility=float(self.utility[path]),
            embedding=self._embedding(path),
            payload=path,
        )

    def expand(self, node: SearchNode) -> tuple[SearchNode, ...]:
        path = tuple(node.payload)
        if len(path) >= self.config.depth:
            return ()
        return tuple(
            self.node(path + (branch,))
            for branch in range(self.config.branching_factor)
        )

    def terminal_value(self, node: SearchNode) -> float:
        path = tuple(node.payload)
        if len(path) != self.config.depth:
            raise ValueError("only terminal nodes have an expensive value")
        return float(self.leaf_values[path])


def run_trial(
    *,
    seed: int,
    strategy: str,
    beam_width: int,
    tree_config: TreeConfig,
    qubo_config: dict[str, Any],
) -> TrialResult:
    tree = CorrelatedNoisyTree(seed, tree_config)
    pruner = build_frontier_pruner(
        strategy, qubo_config if strategy == "qubo" else {}
    )
    result = bounded_beam_search(
        tree.node(()),
        tree.expand,
        beam_width=beam_width,
        max_depth=tree_config.depth,
        pruner=pruner,
    )
    selected_value = max(tree.terminal_value(node) for node in result.frontier)
    regret = max(0.0, tree.oracle_value - selected_value)
    return TrialResult(
        seed=int(seed),
        strategy=strategy,
        beam_width=int(beam_width),
        oracle_value=tree.oracle_value,
        selected_value=float(selected_value),
        regret=float(regret),
        exact_recovery=bool(regret <= 1e-12),
        generated_candidates=result.generated_candidates,
        expanded_nodes=result.expanded_nodes,
        terminal_evaluations=len(result.frontier),
        selector_time_ms=result.selector_time_ms,
        search_time_ms=result.search_time_ms,
    )


def _mean(rows: Sequence[TrialResult], attribute: str) -> float:
    return float(statistics.fmean(float(getattr(row, attribute)) for row in rows))


def _normal_mean_interval(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean
    half_width = 1.96 * float(values.std(ddof=1) / math.sqrt(values.size))
    return mean - half_width, mean + half_width


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1:
        raise ValueError("trials must be positive")
    z = 1.96
    probability = float(successes) / float(trials)
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator
    return center - half_width, center + half_width


def summarize(rows: Iterable[TrialResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[TrialResult]] = {}
    for row in rows:
        grouped.setdefault((row.beam_width, row.strategy), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (width, strategy), group in sorted(grouped.items()):
        regrets = np.asarray([row.regret for row in group], dtype=np.float64)
        regret_interval = _normal_mean_interval(regrets)
        successes = sum(int(row.exact_recovery) for row in group)
        recovery_interval = _wilson_interval(successes, len(group))
        summaries.append(
            {
                "strategy": strategy,
                "beam_width": width,
                "trials": len(group),
                "exact_recovery_rate": float(
                    statistics.fmean(float(row.exact_recovery) for row in group)
                ),
                "mean_regret": float(regrets.mean()),
                "mean_regret_95ci": list(regret_interval),
                "p95_regret": float(np.quantile(regrets, 0.95)),
                "exact_recovery_95ci": list(recovery_interval),
                "mean_generated_candidates": _mean(
                    group, "generated_candidates"
                ),
                "mean_expanded_nodes": _mean(group, "expanded_nodes"),
                "mean_terminal_evaluations": _mean(
                    group, "terminal_evaluations"
                ),
                "mean_selector_time_ms": _mean(group, "selector_time_ms"),
                "mean_search_time_ms": _mean(group, "search_time_ms"),
            }
        )
    return summaries


def matched_efficiency(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match each QUBO row to the smallest utility beam with no more regret."""

    utility_rows = [row for row in summaries if row["strategy"] == "utility"]
    matches: list[dict[str, Any]] = []
    for qubo in (row for row in summaries if row["strategy"] == "qubo"):
        eligible = [
            row
            for row in utility_rows
            if row["mean_regret"] <= qubo["mean_regret"] + 1e-12
        ]
        if not eligible:
            continue
        baseline = min(eligible, key=lambda row: row["beam_width"])
        saved_terminal = (
            baseline["mean_terminal_evaluations"]
            - qubo["mean_terminal_evaluations"]
        )
        additional_selector_ms = max(
            0.0,
            qubo["mean_selector_time_ms"]
            - baseline["mean_selector_time_ms"],
        )
        break_even = (
            additional_selector_ms / saved_terminal
            if saved_terminal > 0.0
            else None
        )
        matches.append(
            {
                "qubo_beam_width": qubo["beam_width"],
                "utility_beam_width": baseline["beam_width"],
                "qubo_mean_regret": qubo["mean_regret"],
                "utility_mean_regret": baseline["mean_regret"],
                "qubo_exact_recovery_rate": qubo["exact_recovery_rate"],
                "utility_exact_recovery_rate": baseline["exact_recovery_rate"],
                "generated_candidate_delta": (
                    baseline["mean_generated_candidates"]
                    - qubo["mean_generated_candidates"]
                ),
                "terminal_evaluation_delta": saved_terminal,
                "qubo_additional_selector_ms": additional_selector_ms,
                "break_even_terminal_evaluation_ms": break_even,
            }
        )
    return matches


def paired_comparisons(rows: Sequence[TrialResult]) -> list[dict[str, Any]]:
    """Compare selectors on identical trees and report a paired confidence interval."""

    indexed: dict[tuple[int, int], dict[str, TrialResult]] = {}
    for row in rows:
        indexed.setdefault((row.seed, row.beam_width), {})[row.strategy] = row

    comparisons: list[dict[str, Any]] = []
    widths = sorted({row.beam_width for row in rows})
    for width in widths:
        pairs = [
            strategies
            for (seed, candidate_width), strategies in sorted(indexed.items())
            if candidate_width == width
            and "utility" in strategies
            and "qubo" in strategies
        ]
        if not pairs:
            continue
        # Positive means QUBO returned a better terminal value (lower regret).
        regret_improvement = np.asarray(
            [
                pair["utility"].regret - pair["qubo"].regret
                for pair in pairs
            ],
            dtype=np.float64,
        )
        mean_improvement = float(regret_improvement.mean())
        standard_error = (
            float(regret_improvement.std(ddof=1) / math.sqrt(len(pairs)))
            if len(pairs) > 1
            else 0.0
        )
        comparisons.append(
            {
                "beam_width": width,
                "paired_trials": len(pairs),
                "utility_minus_qubo_mean_regret": mean_improvement,
                "mean_regret_improvement_95ci": [
                    mean_improvement - 1.96 * standard_error,
                    mean_improvement + 1.96 * standard_error,
                ],
                "qubo_better_rate": float(
                    np.mean(regret_improvement > 1e-12)
                ),
                "equal_rate": float(np.mean(np.abs(regret_improvement) <= 1e-12)),
                "qubo_exact_recovery_rate_delta": float(
                    statistics.fmean(
                        float(pair["qubo"].exact_recovery)
                        - float(pair["utility"].exact_recovery)
                        for pair in pairs
                    )
                ),
                "qubo_additional_selector_ms": float(
                    statistics.fmean(
                        pair["qubo"].selector_time_ms
                        - pair["utility"].selector_time_ms
                        for pair in pairs
                    )
                ),
            }
        )
    return comparisons


def run_benchmark(
    *,
    trials: int,
    widths: Sequence[int],
    base_seed: int,
    tree_config: TreeConfig,
    qubo_config: dict[str, Any],
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be positive")
    normalized_widths = sorted({int(width) for width in widths})
    if not normalized_widths or normalized_widths[0] < 1:
        raise ValueError("beam widths must be positive")

    rows = [
        run_trial(
            seed=base_seed + trial,
            strategy=strategy,
            beam_width=width,
            tree_config=tree_config,
            qubo_config=qubo_config,
        )
        for trial in range(trials)
        for width in normalized_widths
        for strategy in ("utility", "qubo")
    ]
    summaries = summarize(rows)
    reference_tree = CorrelatedNoisyTree(base_seed, tree_config)
    return {
        "benchmark": "correlated_noisy_branch_search",
        "tree": asdict(tree_config),
        "trials": int(trials),
        "base_seed": int(base_seed),
        "widths": normalized_widths,
        "qubo": dict(qubo_config),
        "oracle": {
            "terminal_paths": reference_tree.terminal_path_count,
            "generated_candidates": reference_tree.exhaustive_candidate_count,
        },
        "summaries": summaries,
        "paired_comparisons": paired_comparisons(rows),
        "matched_efficiency": matched_efficiency(summaries),
    }


def render_plot(report: dict[str, Any], output: Path) -> None:
    """Render the paired quality/efficiency frontier as a vector plot."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional documentation path
        raise RuntimeError(
            "plotting requires the optional matplotlib dependency"
        ) from exc

    styles = {
        "utility": {
            "label": "Utility-only",
            "color": "#4B5563",
            "marker": "o",
        },
        "qubo": {
            "label": "QUBO utility + diversity",
            "color": "#2458A6",
            "marker": "s",
        },
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for strategy in ("utility", "qubo"):
        rows = sorted(
            (
                row
                for row in report["summaries"]
                if row["strategy"] == strategy
            ),
            key=lambda row: row["beam_width"],
        )
        x = np.asarray([row["beam_width"] for row in rows], dtype=np.float64)
        regret = np.asarray([row["mean_regret"] for row in rows])
        regret_ci = np.asarray([row["mean_regret_95ci"] for row in rows])
        recovery = 100.0 * np.asarray(
            [row["exact_recovery_rate"] for row in rows]
        )
        recovery_ci = 100.0 * np.asarray(
            [row["exact_recovery_95ci"] for row in rows]
        )
        style = styles[strategy]
        axes[0].errorbar(
            x,
            regret,
            yerr=np.vstack((regret - regret_ci[:, 0], regret_ci[:, 1] - regret)),
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            linewidth=1.35,
            capsize=2.5,
        )
        axes[1].errorbar(
            x,
            recovery,
            yerr=np.vstack(
                (recovery - recovery_ci[:, 0], recovery_ci[:, 1] - recovery)
            ),
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            linewidth=1.35,
            capsize=2.5,
        )

    axes[0].set_title("Terminal regret (lower is better)", fontsize=9.5)
    axes[0].set_ylabel("Mean oracle regret")
    axes[1].set_title("Exact optimum recovery (higher is better)", fontsize=9.5)
    axes[1].set_ylabel("Recovery rate (%)")
    for axis in axes:
        axis.set_xlabel("Terminal evaluations (beam width)")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    figure.tight_layout(pad=0.8, w_pad=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "QUBO frontier-search efficiency",
            "Creator": "Jörmungandr reproducible benchmark",
        },
    )
    plt.close(figure)


def _parse_widths(value: str) -> list[int]:
    try:
        widths = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("widths must be comma-separated integers") from exc
    if not widths or min(widths) < 1:
        raise argparse.ArgumentTypeError("widths must contain positive integers")
    return widths


def _print_table(report: dict[str, Any]) -> None:
    print(
        "strategy width recovery mean_regret p95_regret generated leaves selector_ms"
    )
    for row in report["summaries"]:
        print(
            f"{row['strategy']:>8} "
            f"{row['beam_width']:>5d} "
            f"{row['exact_recovery_rate']:>8.1%} "
            f"{row['mean_regret']:>11.4f} "
            f"{row['p95_regret']:>10.4f} "
            f"{row['mean_generated_candidates']:>9.0f} "
            f"{row['mean_terminal_evaluations']:>6.0f} "
            f"{row['mean_selector_time_ms']:>11.3f}"
        )
    if report["matched_efficiency"]:
        print("\nmean-regret-matched comparisons")
        for match in report["matched_efficiency"]:
            break_even = match["break_even_terminal_evaluation_ms"]
            break_even_text = "n/a" if break_even is None else f"{break_even:.3f} ms"
            print(
                f"QUBO width {match['qubo_beam_width']} vs utility width "
                f"{match['utility_beam_width']}: "
                f"utility-minus-QUBO delta "
                f"{match['generated_candidate_delta']:+.0f} candidates and "
                f"{match['terminal_evaluation_delta']:+.0f} terminal evaluations; "
                f"break-even terminal cost {break_even_text}"
            )
    if report["paired_comparisons"]:
        print("\npaired same-width comparisons")
        for comparison in report["paired_comparisons"]:
            low, high = comparison["mean_regret_improvement_95ci"]
            print(
                f"width {comparison['beam_width']}: utility-minus-QUBO mean regret "
                f"{comparison['utility_minus_qubo_mean_regret']:+.4f} "
                f"(95% CI {low:+.4f} to {high:+.4f}); exact-recovery delta "
                f"{comparison['qubo_exact_recovery_rate_delta']:+.1%}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--widths", type=_parse_widths, default=[4, 6, 8, 12, 16])
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--noise-scale", type=float, default=0.9)
    parser.add_argument("--qubo-diversity-weight", type=float, default=0.12)
    parser.add_argument("--qubo-local-search-passes", type=int, default=4)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    args = parser.parse_args()

    report = run_benchmark(
        trials=args.trials,
        widths=args.widths,
        base_seed=args.seed,
        tree_config=TreeConfig(
            branching_factor=args.branching_factor,
            depth=args.depth,
            noise_scale=args.noise_scale,
        ),
        qubo_config={
            "qubo_utility_weight": 1.0,
            "qubo_diversity_weight": args.qubo_diversity_weight,
            "qubo_cardinality_penalty": 4.0,
            "qubo_local_search_passes": args.qubo_local_search_passes,
        },
    )
    _print_table(report)
    if args.plot_output is not None:
        render_plot(report, args.plot_output)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
