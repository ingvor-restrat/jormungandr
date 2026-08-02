import numpy as np

from examples.benchmark_qubo_branch_search import (
    CorrelatedNoisyTree,
    TreeConfig,
    run_benchmark,
    run_trial,
)
from examples.visualize_qubo_frontier import build_trace, trace_report
from jormungandr.search import (
    QUBOFrontierPruner,
    SearchNode,
    UtilityFrontierPruner,
    bounded_beam_search,
    build_frontier_pruner,
)
from jormungandr.selectors import QUBORolloutSelector, RolloutCandidate


def _binary_node(path: tuple[int, ...], depth: int = 3) -> SearchNode:
    embedding = np.zeros(depth * 2, dtype=np.float32)
    for level, branch in enumerate(path):
        embedding[level * 2 + branch] = 1.0
    return SearchNode(
        key="root" if not path else ".".join(str(item) for item in path),
        utility=float(sum((level + 1) * branch for level, branch in enumerate(path))),
        embedding=embedding,
        payload=path,
    )


def test_bounded_beam_search_prunes_and_accounts_for_frontier_work() -> None:
    def expand(node: SearchNode) -> tuple[SearchNode, ...]:
        path = tuple(node.payload)
        if len(path) == 3:
            return ()
        return (_binary_node(path + (0,)), _binary_node(path + (1,)))

    result = bounded_beam_search(
        _binary_node(()),
        expand,
        beam_width=2,
        max_depth=3,
        pruner=UtilityFrontierPruner(),
    )

    assert result.generated_candidates == 10
    assert result.expanded_nodes == 5
    assert result.retained_candidates == 6
    assert result.pruned_candidates == 4
    assert len(result.frontier) == 2
    assert [level.candidate_count for level in result.levels] == [2, 4, 4]
    assert result.pruner_name == "utility"
    assert all(sum(level.decisions) == level.selected_count for level in result.levels)
    assert all(
        len(level.candidate_keys) == level.candidate_count for level in result.levels
    )


def test_qubo_frontier_pruner_is_fixed_width_and_avoids_one_cluster() -> None:
    candidates = [
        SearchNode(str(index), utility, np.asarray(embedding, dtype=np.float32))
        for index, (utility, embedding) in enumerate(
            [
                (10.0, [0.0, 0.0]),
                (9.0, [0.0, 0.0]),
                (8.0, [0.0, 0.0]),
                (7.0, [5.0, 0.0]),
                (6.0, [0.0, 5.0]),
                (5.0, [5.0, 5.0]),
            ]
        )
    ]
    utility = UtilityFrontierPruner().select(candidates, 3)
    qubo = QUBOFrontierPruner(
        {
            "qubo_diversity_weight": 5.0,
            "qubo_local_search_passes": 16,
        }
    ).select(candidates, 3)

    assert [node.key for node in utility.selected] == ["0", "1", "2"]
    assert len({tuple(node.embedding) for node in qubo.selected}) == 3
    assert int(qubo.decisions.sum()) == 3
    assert qubo.metrics["selector_qubo_matrix_bytes"] == 6 * 6 * 8
    assert sum(qubo.audit["decisions"]) == 3


def test_vectorized_qubo_swaps_finish_at_a_one_swap_local_optimum() -> None:
    rng = np.random.default_rng(73)
    candidates = [
        RolloutCandidate(
            key=str(index),
            utility=float(rng.normal()),
            embedding=rng.normal(size=4),
        )
        for index in range(10)
    ]
    solver = QUBORolloutSelector(
        utility_weight=1.0,
        diversity_weight=0.4,
        local_search_passes=512,
    )
    result = solver.select(candidates, 4)
    _qubo, utility, similarity = solver.build_qubo(candidates, 4)

    def score(indices: list[int]) -> float:
        chosen = np.asarray(indices, dtype=np.int64)
        return float(utility[chosen].sum()) - 0.4 * float(
            np.triu(similarity[np.ix_(chosen, chosen)], 1).sum()
        )

    selected = result.selected_indices.tolist()
    current = score(selected)
    for position in range(len(selected)):
        for replacement in sorted(set(range(len(candidates))) - set(selected)):
            proposal = list(selected)
            proposal[position] = replacement
            assert score(proposal) <= current + 1e-12


def test_noisy_tree_has_exact_oracle_and_seeded_search_results() -> None:
    config = TreeConfig(branching_factor=3, depth=4, noise_scale=0.7)
    tree = CorrelatedNoisyTree(17, config)
    assert tree.terminal_path_count == 3**4
    assert tree.exhaustive_candidate_count == sum(3**depth for depth in range(1, 5))
    assert tree.oracle_value == max(tree.leaf_values.values())

    settings = {
        "qubo_diversity_weight": 0.12,
        "qubo_local_search_passes": 4,
    }
    first = run_trial(
        seed=17,
        strategy="qubo",
        beam_width=5,
        tree_config=config,
        qubo_config=settings,
    )
    second = run_trial(
        seed=17,
        strategy="qubo",
        beam_width=5,
        tree_config=config,
        qubo_config=settings,
    )
    assert first.selected_value == second.selected_value
    assert first.regret == second.regret
    assert first.exact_recovery == second.exact_recovery
    assert first.generated_candidates == second.generated_candidates
    assert first.terminal_evaluations == 5


def test_branch_benchmark_reports_both_controls_without_claiming_speedup() -> None:
    report = run_benchmark(
        trials=3,
        widths=[3, 5],
        base_seed=91,
        tree_config=TreeConfig(branching_factor=3, depth=4, noise_scale=0.7),
        qubo_config={
            "qubo_diversity_weight": 0.12,
            "qubo_local_search_passes": 2,
        },
    )

    assert report["oracle"]["terminal_paths"] == 3**4
    assert len(report["summaries"]) == 4
    assert {row["strategy"] for row in report["summaries"]} == {
        "utility",
        "qubo",
    }
    for row in report["summaries"]:
        assert row["mean_selector_time_ms"] >= 0.0
        assert row["mean_regret_95ci"][0] <= row["mean_regret"]
        assert row["mean_regret"] <= row["mean_regret_95ci"][1]
        assert row["exact_recovery_95ci"][0] <= row["exact_recovery_rate"]
        assert row["exact_recovery_rate"] <= row["exact_recovery_95ci"][1]
    assert isinstance(build_frontier_pruner("top-k", {}), UtilityFrontierPruner)


def test_qubo_frontier_visual_trace_replays_audited_binary_decisions() -> None:
    trace = build_trace(
        seed=23,
        beam_width=4,
        tree_config=TreeConfig(branching_factor=3, depth=4, noise_scale=0.7),
        qubo_config={
            "qubo_diversity_weight": 0.12,
            "qubo_local_search_passes": 2,
        },
    )
    report = trace_report(trace)

    assert report["selection_unit"].startswith("one QUBO")
    assert len(report["levels"]) == 4
    assert report["search"]["generated_candidates"] == trace.search.generated_candidates
    for recorded, rendered in zip(trace.search.levels, report["levels"]):
        assert rendered["candidate_keys"] == list(recorded.candidate_keys)
        assert rendered["decisions"] == list(recorded.decisions)
        assert sum(rendered["decisions"]) == recorded.selected_count
        assert rendered["selected_keys"] == list(recorded.selected_keys)
