"""Exact shortest-path planning over caller-declared directed graphs.

Jormungandr owns only the generic calculation. Applications define nodes,
edges, action meanings, traversability, costs, and edge order. The latter is
an explicit deterministic tie-break, so a recorded problem can be replayed
without importing application policy or environment rules here.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Mapping, Sequence


SHORTEST_PATH_SCHEMA = "jormungandr.shortest_path.v1"


@dataclass(frozen=True)
class DirectedRouteEdge:
    """One feasible directed transition supplied by an application."""

    source: str
    target: str
    action_id: str
    cost: float = 1.0
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not str(self.source).strip() or not str(self.target).strip():
            raise ValueError("route edge endpoints cannot be empty")
        if not str(self.action_id).strip():
            raise ValueError("route edge action_id cannot be empty")
        if not math.isfinite(float(self.cost)) or float(self.cost) <= 0.0:
            raise ValueError("route edge cost must be finite and positive")


@dataclass(frozen=True)
class ShortestPathResult:
    """An auditable optimum or an explicit unreachable certificate."""

    source: str
    requested_targets: tuple[str, ...]
    reached_target: str | None
    node_path: tuple[str, ...]
    action_path: tuple[str, ...]
    total_cost: float | None
    explored_nodes: int
    candidate_edges: int
    tie_break: str = "caller_edge_order_lexicographic"

    @property
    def reachable(self) -> bool:
        return self.reached_target is not None

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "schema": SHORTEST_PATH_SCHEMA,
            "method": "exact_dijkstra",
            "objective": "minimize_total_declared_edge_cost",
            "tie_break": self.tie_break,
            "source": self.source,
            "requested_targets": list(self.requested_targets),
            "reachable": self.reachable,
            "reached_target": self.reached_target,
            "node_path": list(self.node_path),
            "action_path": list(self.action_path),
            "total_cost": self.total_cost,
            "explored_nodes": self.explored_nodes,
            "candidate_edges": self.candidate_edges,
        }


def solve_shortest_path(
    nodes: Sequence[str],
    edges: Sequence[DirectedRouteEdge],
    *,
    source: str,
    targets: Sequence[str],
) -> ShortestPathResult:
    """Return the least-cost route using caller edge order for exact ties.

    All costs are required to be positive. Among equal-cost paths, the tuple
    of input edge positions is minimized lexicographically. This makes the
    result deterministic while leaving tie-breaking policy with the caller.
    """

    node_ids = tuple(str(value) for value in nodes)
    if not node_ids or any(not value.strip() for value in node_ids):
        raise ValueError("route nodes must be nonempty strings")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("route nodes must be unique")
    source_id = str(source)
    if source_id not in set(node_ids):
        raise ValueError("route source must be a declared node")
    requested_targets = tuple(str(value) for value in targets)
    if not requested_targets:
        raise ValueError("at least one route target is required")
    if len(set(requested_targets)) != len(requested_targets):
        raise ValueError("route targets must be unique")
    unknown_targets = set(requested_targets) - set(node_ids)
    if unknown_targets:
        raise ValueError("route target must be a declared node")

    records = tuple(edges)
    known = set(node_ids)
    identities: set[tuple[str, str, str]] = set()
    adjacency: dict[str, list[tuple[int, DirectedRouteEdge]]] = {
        node: [] for node in node_ids
    }
    for index, edge in enumerate(records):
        if edge.source not in known or edge.target not in known:
            raise ValueError("route edge refers to an undeclared node")
        identity = (edge.source, edge.target, edge.action_id)
        if identity in identities:
            raise ValueError("duplicate route edge identity")
        identities.add(identity)
        adjacency[edge.source].append((index, edge))

    if source_id in set(requested_targets):
        return ShortestPathResult(
            source=source_id,
            requested_targets=requested_targets,
            reached_target=source_id,
            node_path=(source_id,),
            action_path=(),
            total_cost=0.0,
            explored_nodes=1,
            candidate_edges=len(records),
        )

    # Heap order is total cost followed by the exact input-edge signature.
    frontier: list[tuple[float, tuple[int, ...], str]] = [
        (0.0, (), source_id)
    ]
    best: dict[str, tuple[float, tuple[int, ...]]] = {source_id: (0.0, ())}
    parent: dict[str, tuple[str, DirectedRouteEdge]] = {}
    explored = 0
    target_set = set(requested_targets)
    reached: str | None = None
    while frontier:
        cost, signature, node = heapq.heappop(frontier)
        if best.get(node) != (cost, signature):
            continue
        explored += 1
        if node in target_set:
            reached = node
            break
        for edge_index, edge in adjacency[node]:
            proposed = (cost + float(edge.cost), signature + (edge_index,))
            current = best.get(edge.target)
            if current is not None and not proposed < current:
                continue
            best[edge.target] = proposed
            parent[edge.target] = (node, edge)
            heapq.heappush(
                frontier, (proposed[0], proposed[1], edge.target)
            )

    if reached is None:
        return ShortestPathResult(
            source=source_id,
            requested_targets=requested_targets,
            reached_target=None,
            node_path=(),
            action_path=(),
            total_cost=None,
            explored_nodes=explored,
            candidate_edges=len(records),
        )

    reversed_nodes = [reached]
    reversed_actions: list[str] = []
    cursor = reached
    while cursor != source_id:
        previous, edge = parent[cursor]
        reversed_actions.append(edge.action_id)
        reversed_nodes.append(previous)
        cursor = previous
    return ShortestPathResult(
        source=source_id,
        requested_targets=requested_targets,
        reached_target=reached,
        node_path=tuple(reversed(reversed_nodes)),
        action_path=tuple(reversed(reversed_actions)),
        total_cost=float(best[reached][0]),
        explored_nodes=explored,
        candidate_edges=len(records),
    )


__all__ = [
    "SHORTEST_PATH_SCHEMA",
    "DirectedRouteEdge",
    "ShortestPathResult",
    "solve_shortest_path",
]
