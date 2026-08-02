"""Render an auditable QUBO branch-frontier selection trace.

The visualization uses the real correlated-noise tree, bounded beam search,
and QUBO frontier pruner from the empirical benchmark.  It emphasizes the
important unit of optimization: after every retained beam is expanded, one
binary problem is constructed over the complete candidate frontier.  QUBO is
not solved independently inside each branch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Package import under tests and ``python -m examples...``.
    from examples.benchmark_qubo_branch_search import (
        CorrelatedNoisyTree,
        TreeConfig,
    )
except ModuleNotFoundError:  # Direct ``python examples/visualize...``.
    from benchmark_qubo_branch_search import (  # type: ignore[no-redef]
        CorrelatedNoisyTree,
        TreeConfig,
    )

from jormungandr.search import (
    BeamSearchResult,
    QUBOFrontierPruner,
    SearchNode,
    bounded_beam_search,
)
from jormungandr.selectors import RolloutCandidate


SELECTED = "#2458A6"
PRUNED = "#D1D5DB"
PARENT = "#334155"
ORACLE = "#D97706"
TEXT = "#172033"


@dataclass(frozen=True)
class FrontierTraceLevel:
    depth: int
    parents: tuple[SearchNode, ...]
    candidates: tuple[SearchNode, ...]
    decisions: np.ndarray
    normalized_utility: np.ndarray
    similarity: np.ndarray
    qubo: np.ndarray
    metrics: Mapping[str, float]

    @property
    def selected(self) -> tuple[SearchNode, ...]:
        return tuple(
            candidate
            for candidate, decision in zip(self.candidates, self.decisions)
            if int(decision) == 1
        )


@dataclass(frozen=True)
class FrontierTrace:
    seed: int
    beam_width: int
    tree: CorrelatedNoisyTree
    search: BeamSearchResult
    levels: tuple[FrontierTraceLevel, ...]
    oracle_path: tuple[int, ...]


def build_trace(
    *,
    seed: int,
    beam_width: int,
    tree_config: TreeConfig,
    qubo_config: Mapping[str, Any],
) -> FrontierTrace:
    tree = CorrelatedNoisyTree(seed, tree_config)
    pruner = QUBOFrontierPruner(qubo_config)
    root = tree.node(())
    search = bounded_beam_search(
        root,
        tree.expand,
        beam_width=beam_width,
        max_depth=tree_config.depth,
        pruner=pruner,
    )
    oracle_path = max(
        tree.leaf_values,
        key=lambda path: (tree.leaf_values[path], tuple(-item for item in path)),
    )

    parents: tuple[SearchNode, ...] = (root,)
    levels: list[FrontierTraceLevel] = []
    for recorded in search.levels:
        candidates = tuple(
            child for parent in parents for child in tree.expand(parent)
        )
        if tuple(candidate.key for candidate in candidates) != recorded.candidate_keys:
            raise RuntimeError("reconstructed candidate frontier does not match audit")
        rollout_candidates = [
            RolloutCandidate(
                key=candidate.key,
                utility=float(candidate.utility),
                embedding=np.asarray(candidate.embedding, dtype=np.float64),
                payload=candidate,
            )
            for candidate in candidates
        ]
        qubo, utility, similarity = pruner.solver.build_qubo(
            rollout_candidates,
            recorded.selected_count,
        )
        decisions = np.asarray(recorded.decisions, dtype=np.int8)
        level = FrontierTraceLevel(
            depth=recorded.depth,
            parents=parents,
            candidates=candidates,
            decisions=decisions,
            normalized_utility=utility,
            similarity=similarity,
            qubo=qubo,
            metrics=recorded.selector_metrics,
        )
        levels.append(level)
        parents = level.selected

    return FrontierTrace(
        seed=int(seed),
        beam_width=int(beam_width),
        tree=tree,
        search=search,
        levels=tuple(levels),
        oracle_path=tuple(oracle_path),
    )


def trace_report(trace: FrontierTrace) -> dict[str, Any]:
    return {
        "format": "jormungandr.qubo_frontier_trace.v1",
        "seed": trace.seed,
        "beam_width": trace.beam_width,
        "branching_factor": trace.tree.config.branching_factor,
        "depth": trace.tree.config.depth,
        "oracle_path": list(trace.oracle_path),
        "oracle_value": trace.tree.oracle_value,
        "selection_unit": (
            "one QUBO over the complete candidate frontier after each beam expansion"
        ),
        "levels": [
            {
                "depth": level.depth,
                "parent_keys": [node.key for node in level.parents],
                "candidate_keys": [node.key for node in level.candidates],
                "candidate_utility": [
                    float(node.utility) for node in level.candidates
                ],
                "normalized_utility": level.normalized_utility.astype(float).tolist(),
                "decisions": level.decisions.astype(int).tolist(),
                "selected_keys": [node.key for node in level.selected],
                "similarity_mean": float(
                    level.similarity[np.triu_indices(len(level.candidates), 1)].mean()
                )
                if len(level.candidates) > 1
                else 0.0,
                "qubo_energy": float(
                    level.decisions.astype(float)
                    @ level.qubo
                    @ level.decisions.astype(float)
                ),
                "selector_metrics": {
                    key: float(value) for key, value in level.metrics.items()
                },
            }
            for level in trace.levels
        ],
        "search": {
            "expanded_nodes": trace.search.expanded_nodes,
            "generated_candidates": trace.search.generated_candidates,
            "retained_candidates": trace.search.retained_candidates,
            "pruned_candidates": trace.search.pruned_candidates,
            "selector_time_ms": trace.search.selector_time_ms,
        },
    }


def _short_key(node: SearchNode, *, width: int = 9) -> str:
    if node.key == "root":
        return "root"
    return node.key if len(node.key) <= width else "…" + node.key[-(width - 1) :]


def _is_oracle_prefix(node: SearchNode, oracle_path: Sequence[int]) -> bool:
    path = tuple(node.payload)
    return tuple(oracle_path[: len(path)]) == path


def _draw_branch_level(
    axis,
    level: FrontierTraceLevel,
    *,
    oracle_path: Sequence[int],
    detailed: bool,
) -> None:
    candidate_count = len(level.candidates)
    child_x = np.arange(candidate_count, dtype=np.float64)
    child_by_parent = max(1, candidate_count // len(level.parents))
    parent_x = np.asarray(
        [
            float(child_x[index * child_by_parent : (index + 1) * child_by_parent].mean())
            for index in range(len(level.parents))
        ]
    )

    for parent_index, parent in enumerate(level.parents):
        start = parent_index * child_by_parent
        stop = min(candidate_count, start + child_by_parent)
        for index in range(start, stop):
            axis.plot(
                [parent_x[parent_index], child_x[index]],
                [1.0, 0.0],
                color="#CBD5E1",
                linewidth=0.55,
                zorder=1,
            )
        parent_oracle = _is_oracle_prefix(parent, oracle_path)
        axis.scatter(
            [parent_x[parent_index]],
            [1.0],
            s=52 if detailed else 38,
            color=PARENT,
            edgecolor=ORACLE if parent_oracle else "white",
            linewidth=2.0 if parent_oracle else 0.8,
            zorder=3,
        )
        if detailed:
            axis.text(
                parent_x[parent_index],
                1.18,
                _short_key(parent),
                ha="center",
                va="bottom",
                fontsize=6.6,
                color=TEXT,
            )

    for index, candidate in enumerate(level.candidates):
        chosen = bool(level.decisions[index])
        oracle = _is_oracle_prefix(candidate, oracle_path)
        axis.scatter(
            [child_x[index]],
            [0.0],
            s=62 if detailed else 42,
            color=SELECTED if chosen else PRUNED,
            edgecolor=ORACLE if oracle else "white",
            linewidth=2.2 if oracle else 0.75,
            zorder=4,
        )
        axis.text(
            child_x[index],
            0.0,
            "1" if chosen else "0",
            ha="center",
            va="center",
            fontsize=6.2 if detailed else 5.5,
            color="white" if chosen else "#64748B",
            fontweight="bold",
            zorder=5,
        )
        if detailed and chosen:
            axis.text(
                child_x[index],
                -0.18,
                f"u={level.normalized_utility[index]:.2f}",
                ha="center",
                va="top",
                rotation=60,
                fontsize=5.7,
                color=TEXT,
            )

    energy = float(
        level.decisions.astype(float) @ level.qubo @ level.decisions.astype(float)
    )
    axis.set_title(
        (
            f"depth {level.depth}: {len(level.parents)} retained parents "
            f"→ {candidate_count} candidates → {int(level.decisions.sum())} retained"
        ),
        fontsize=8.7 if detailed else 7.8,
        loc="left",
        pad=7,
    )
    axis.text(
        0.995,
        0.97,
        f"Σx={int(level.decisions.sum())}  ·  E(x)={energy:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#475569",
    )
    axis.set_xlim(-0.8, max(1.0, candidate_count - 0.2))
    axis.set_ylim(-0.58 if detailed else -0.32, 1.42)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def render_static(trace: FrontierTrace, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional documentation path
        raise RuntimeError("plotting requires the optional matplotlib dependency") from exc

    chosen_depths = sorted(
        {min(2, len(trace.levels)), max(1, len(trace.levels) // 2), len(trace.levels)}
    )
    figure, axes = plt.subplots(len(chosen_depths), 1, figsize=(7.2, 5.15))
    axes_array = np.atleast_1d(axes)
    for axis, depth in zip(axes_array, chosen_depths):
        _draw_branch_level(
            axis,
            trace.levels[depth - 1],
            oracle_path=trace.oracle_path,
            detailed=False,
        )
    figure.suptitle(
        "QUBO is applied once to each expanded candidate frontier",
        fontsize=11,
        fontweight="semibold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.945,
        (
            r"$\mathcal{B}_{t-1}\;\longrightarrow\;\mathcal{C}_{t}"
            r"\;\longrightarrow\;x^*=\arg\min E(x)"
            r"\;\longrightarrow\;\mathcal{B}_{t}$  "
            r"with $x\in\{0,1\}^{|\mathcal{C}_t|}$"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT,
    )
    figure.text(
        0.5,
        0.015,
        (
            f"Actual seeded trace {trace.seed}; blue xᵢ=1, grey xᵢ=0, "
            "gold outline = oracle-path prefix. The trace is illustrative; "
            "aggregate evidence is reported separately."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#475569",
    )
    figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.91), h_pad=1.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "QUBO frontier branch expansion trace",
            "Creator": "Jörmungandr reproducible benchmark",
        },
    )
    plt.close(figure)


def _draw_animation_frame(trace: FrontierTrace, level: FrontierTraceLevel):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover - optional documentation path
        raise RuntimeError("animation rendering requires matplotlib") from exc

    figure = plt.figure(figsize=(9.2, 5.4))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.28)
    branch_axis = figure.add_subplot(grid[0, :])
    similarity_axis = figure.add_subplot(grid[1, 0])
    score_axis = figure.add_subplot(grid[1, 1])
    _draw_branch_level(
        branch_axis,
        level,
        oracle_path=trace.oracle_path,
        detailed=True,
    )

    image = similarity_axis.imshow(
        level.similarity,
        vmin=0.0,
        vmax=1.0,
        cmap="Greys",
        interpolation="nearest",
        aspect="auto",
    )
    chosen = np.flatnonzero(level.decisions)
    for index in chosen:
        similarity_axis.add_patch(
            Rectangle(
                (index - 0.5, index - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=SELECTED,
                linewidth=1.4,
            )
        )
    similarity_axis.set_title(
        "Pairwise path similarity $s_{ij}$ (selected diagonal outlined)",
        fontsize=8.4,
    )
    similarity_axis.set_xlabel("candidate j", fontsize=7.5)
    similarity_axis.set_ylabel("candidate i", fontsize=7.5)
    similarity_axis.tick_params(labelsize=6.5)
    colorbar = figure.colorbar(image, ax=similarity_axis, fraction=0.046, pad=0.03)
    colorbar.ax.tick_params(labelsize=6.5)

    candidate_indices = np.arange(len(level.candidates))
    colors = [SELECTED if decision else PRUNED for decision in level.decisions]
    score_axis.bar(
        candidate_indices,
        level.normalized_utility,
        color=colors,
        width=0.82,
        edgecolor="white",
        linewidth=0.3,
    )
    score_axis.set_ylim(0.0, 1.08)
    score_axis.set_xlim(-0.8, len(level.candidates) - 0.2)
    score_axis.set_title(
        "Robustly scaled utility $u_i$ and binary decision $x_i$",
        fontsize=8.4,
    )
    score_axis.set_xlabel("candidate i", fontsize=7.5)
    score_axis.set_ylabel("$u_i$", fontsize=7.5)
    score_axis.tick_params(labelsize=6.5)
    score_axis.grid(axis="y", color="#E2E8F0", linewidth=0.5)
    score_axis.spines["top"].set_visible(False)
    score_axis.spines["right"].set_visible(False)

    figure.suptitle(
        (
            "Expand retained beam → build one frontier QUBO → keep xᵢ=1  "
            f"({level.depth}/{len(trace.levels)})"
        ),
        y=0.995,
        fontsize=11,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.012,
        (
            r"$E(x)=-\lambda_u\sum_i u_ix_i+\lambda_s\sum_{i<j}s_{ij}x_ix_j"
            r"+\lambda_k(\sum_i x_i-k)^2$"
        ),
        ha="center",
        va="bottom",
        fontsize=9.1,
        color=TEXT,
    )
    figure.tight_layout(rect=(0.01, 0.055, 0.99, 0.965))
    return figure


def _figure_image(figure, *, dpi: int = 105):
    from PIL import Image

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def render_gif(trace: FrontierTrace, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional documentation path
        raise RuntimeError("GIF rendering requires matplotlib and Pillow") from exc

    frames: list[Image.Image] = []
    for level in trace.levels:
        figure = _draw_animation_frame(trace, level)
        frames.append(_figure_image(figure))
        plt.close(figure)
    frames.extend([frames[-1].copy()] * 7)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=650,
        loop=0,
        disposal=1,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--noise-scale", type=float, default=0.9)
    parser.add_argument("--qubo-diversity-weight", type=float, default=0.12)
    parser.add_argument("--qubo-local-search-passes", type=int, default=4)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--gif-output", type=Path)
    args = parser.parse_args()

    trace = build_trace(
        seed=args.seed,
        beam_width=args.beam_width,
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
    print(
        f"seed {trace.seed}: {trace.search.generated_candidates} candidates, "
        f"{trace.search.pruned_candidates} pruned, "
        f"oracle path {'.'.join(str(value) for value in trace.oracle_path)}"
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(trace_report(trace), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.plot_output is not None:
        render_static(trace, args.plot_output)
    if args.gif_output is not None:
        render_gif(trace, args.gif_output)


if __name__ == "__main__":
    main()
