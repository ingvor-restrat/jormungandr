import pytest

from jormungandr.path_planning import (
    DirectedRouteEdge,
    solve_shortest_path,
)


def test_shortest_path_minimizes_declared_cost() -> None:
    result = solve_shortest_path(
        ("start", "a", "b", "goal"),
        (
            DirectedRouteEdge("start", "a", "to-a", 1.0),
            DirectedRouteEdge("a", "goal", "a-goal", 4.0),
            DirectedRouteEdge("start", "b", "to-b", 2.0),
            DirectedRouteEdge("b", "goal", "b-goal", 1.0),
        ),
        source="start",
        targets=("goal",),
    )

    assert result.reachable
    assert result.node_path == ("start", "b", "goal")
    assert result.action_path == ("to-b", "b-goal")
    assert result.total_cost == pytest.approx(3.0)
    assert result.to_payload()["method"] == "exact_dijkstra"


def test_shortest_path_uses_caller_edge_order_for_equal_cost_paths() -> None:
    first = DirectedRouteEdge("start", "north", "NORTH")
    second = DirectedRouteEdge("start", "west", "WEST")
    tails = (
        DirectedRouteEdge("north", "goal", "WEST"),
        DirectedRouteEdge("west", "goal", "NORTH"),
    )
    north_first = solve_shortest_path(
        ("start", "north", "west", "goal"),
        (first, second, *tails),
        source="start",
        targets=("goal",),
    )
    west_first = solve_shortest_path(
        ("start", "north", "west", "goal"),
        (second, first, *tails),
        source="start",
        targets=("goal",),
    )

    assert north_first.action_path[0] == "NORTH"
    assert west_first.action_path[0] == "WEST"
    assert north_first.total_cost == west_first.total_cost == 2.0


def test_shortest_path_selects_the_nearest_of_multiple_targets() -> None:
    result = solve_shortest_path(
        ("start", "near", "middle", "far"),
        (
            DirectedRouteEdge("start", "near", "near"),
            DirectedRouteEdge("start", "middle", "middle"),
            DirectedRouteEdge("middle", "far", "far"),
        ),
        source="start",
        targets=("far", "near"),
    )

    assert result.reached_target == "near"
    assert result.action_path == ("near",)


def test_shortest_path_reports_an_unreachable_certificate() -> None:
    result = solve_shortest_path(
        ("start", "island"),
        (),
        source="start",
        targets=("island",),
    )

    assert not result.reachable
    assert result.node_path == ()
    assert result.action_path == ()
    assert result.total_cost is None
    assert result.to_payload()["reachable"] is False


def test_shortest_path_accepts_source_as_target() -> None:
    result = solve_shortest_path(
        ("start",), (), source="start", targets=("start",)
    )

    assert result.reachable
    assert result.node_path == ("start",)
    assert result.action_path == ()
    assert result.total_cost == 0.0


@pytest.mark.parametrize(
    "edge, message",
    (
        (DirectedRouteEdge("start", "goal", "move"), "undeclared"),
        (None, "target"),
    ),
)
def test_shortest_path_rejects_invalid_problem(edge, message) -> None:
    with pytest.raises(ValueError, match=message):
        solve_shortest_path(
            ("start",),
            () if edge is None else (edge,),
            source="start",
            targets=() if edge is None else ("start",),
        )


def test_route_edge_requires_positive_finite_cost() -> None:
    with pytest.raises(ValueError, match="positive"):
        DirectedRouteEdge("a", "b", "move", 0.0)
