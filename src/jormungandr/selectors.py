"""Replay/rollout selectors, including an auditable classical QUBO solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from jormungandr.core import PrioritizedReplayBuffer


SELECTOR_ENTRY_POINT_GROUP = "jormungandr.rollout_selectors"


@dataclass(frozen=True)
class RolloutCandidate:
    """A rollout or transition represented for binary subset selection."""

    key: str
    utility: float
    embedding: np.ndarray
    payload: Any = None


@dataclass(frozen=True)
class QUBOSelection:
    selected_indices: np.ndarray
    decisions: np.ndarray
    energy: float
    utility_sum: float
    redundancy: float
    qubo: np.ndarray
    solve_time_ms: float


@dataclass(frozen=True)
class ReplaySelection:
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    indices: np.ndarray
    weights: np.ndarray
    metrics: Mapping[str, float] = field(default_factory=dict)
    audit: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _TrajectoryFragment:
    key: str
    indices: np.ndarray
    utility: float
    embedding: np.ndarray


class ReplaySelector(Protocol):
    name: str

    def select(
        self,
        replay: PrioritizedReplayBuffer,
        batch_size: int,
        beta: float,
    ) -> ReplaySelection: ...


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    finite = np.where(np.isfinite(values), values, np.nan)
    if np.isnan(finite).all():
        return np.zeros_like(values)
    fill = float(np.nanmedian(finite))
    finite = np.nan_to_num(finite, nan=fill, posinf=fill, neginf=fill)
    low, high = np.quantile(finite, [0.05, 0.95])
    if high - low < 1e-12:
        return np.full_like(finite, 0.5)
    return np.clip((finite - low) / (high - low), 0.0, 1.0)


def _similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 2:
        raise ValueError("candidate embeddings must form a 2D matrix")
    median = np.median(values, axis=0, keepdims=True)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0, keepdims=True)
    scale = np.where((q75 - q25) > 1e-8, q75 - q25, 1.0)
    normalized = (values - median) / scale
    squared = np.sum((normalized[:, None, :] - normalized[None, :, :]) ** 2, axis=-1)
    positive = squared[squared > 1e-12]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    similarity = np.exp(-squared / max(2.0 * bandwidth, 1e-8))
    np.fill_diagonal(similarity, 0.0)
    return similarity


class QUBORolloutSelector:
    """Select exactly ``k`` candidates using a QUBO utility/diversity objective.

    The built-in solver uses greedy construction followed by deterministic
    one-for-one swaps.  The generated Q matrix and binary decisions are exposed
    so a future simulated/quantum annealer or Rust solver can replace only the
    solve step.
    """

    name = "qubo"

    def __init__(
        self,
        *,
        utility_weight: float = 1.0,
        diversity_weight: float = 0.35,
        cardinality_penalty: float = 4.0,
        local_search_passes: int = 8,
    ) -> None:
        self.utility_weight = max(0.0, float(utility_weight))
        self.diversity_weight = max(0.0, float(diversity_weight))
        self.cardinality_penalty = max(0.0, float(cardinality_penalty))
        self.local_search_passes = max(0, int(local_search_passes))

    def build_qubo(
        self, candidates: Sequence[RolloutCandidate], select_count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(candidates)
        k = int(select_count)
        if not 0 < k <= n:
            raise ValueError("select_count must be in [1, number of candidates]")
        utility = _robust_unit_scale(
            np.asarray([candidate.utility for candidate in candidates], dtype=np.float64)
        )
        embeddings = np.stack(
            [np.asarray(candidate.embedding, dtype=np.float64).reshape(-1) for candidate in candidates]
        )
        similarity = _similarity_matrix(embeddings)
        # Energy is x'Qx.  Off-diagonal coefficients are halved because the
        # symmetric matrix contributes Q_ij and Q_ji.
        qubo = np.zeros((n, n), dtype=np.float64)
        diagonal = (
            -self.utility_weight * utility
            + self.cardinality_penalty * (1.0 - 2.0 * k)
        )
        np.fill_diagonal(qubo, diagonal)
        pair_coefficients = (
            self.diversity_weight * similarity + 2.0 * self.cardinality_penalty
        )
        upper = np.triu_indices(n, k=1)
        qubo[upper] = pair_coefficients[upper] * 0.5
        qubo[(upper[1], upper[0])] = qubo[upper]
        return qubo, utility, similarity

    def select(
        self, candidates: Sequence[RolloutCandidate], select_count: int
    ) -> QUBOSelection:
        started = time.perf_counter()
        qubo, utility, similarity = self.build_qubo(candidates, select_count)
        n = len(candidates)
        k = int(select_count)
        selected: list[int] = []
        remaining = set(range(n))
        # Exact-cardinality search can omit the cardinality term while ranking
        # feasible choices; it remains in Q for external unconstrained solvers.
        while len(selected) < k:
            best = max(
                remaining,
                key=lambda index: (
                    self.utility_weight * utility[index]
                    - self.diversity_weight
                    * sum(similarity[index, chosen] for chosen in selected),
                    -index,
                ),
            )
            selected.append(int(best))
            remaining.remove(best)

        for _ in range(self.local_search_passes):
            selected_pass = np.asarray(selected, dtype=np.int64)
            unselected_pass = np.asarray(
                sorted(set(range(n)) - set(selected)), dtype=np.int64
            )
            if unselected_pass.size == 0:
                break

            # Evaluate every feasible one-for-one exchange together.  For a
            # selected item ``old`` and candidate ``new``, only their utility
            # and similarity edges touching the retained set can change.  The
            # vectorized delta avoids rebuilding and rescoring a k-by-k matrix
            # for each of the k(n-k) proposals, which is material when this
            # selector is used to prune a live search frontier.
            selected_similarity = similarity[
                np.ix_(selected_pass, selected_pass)
            ]
            unselected_to_selected = similarity[
                np.ix_(unselected_pass, selected_pass)
            ]
            old_edge_sum = selected_similarity.sum(axis=1)
            new_edge_sum = unselected_to_selected.sum(axis=1)
            redundancy_delta = (
                new_edge_sum[None, :]
                - unselected_to_selected.T
                - old_edge_sum[:, None]
            )
            score_delta = self.utility_weight * (
                utility[unselected_pass][None, :]
                - utility[selected_pass][:, None]
            ) - self.diversity_weight * redundancy_delta
            best_flat = int(np.argmax(score_delta))
            best_delta = float(score_delta.reshape(-1)[best_flat])
            if best_delta <= 1e-12:
                break
            position, candidate_position = np.unravel_index(
                best_flat, score_delta.shape
            )
            selected[int(position)] = int(unselected_pass[int(candidate_position)])

        selected_array = np.asarray(sorted(selected), dtype=np.int64)
        decisions = np.zeros(n, dtype=np.int8)
        decisions[selected_array] = 1
        energy = float(decisions.astype(np.float64) @ qubo @ decisions.astype(np.float64))
        selected_similarity = similarity[np.ix_(selected_array, selected_array)]
        pairs = k * (k - 1) / 2
        redundancy = (
            float(np.triu(selected_similarity, 1).sum() / pairs) if pairs > 0 else 0.0
        )
        return QUBOSelection(
            selected_indices=selected_array,
            decisions=decisions,
            energy=energy,
            utility_sum=float(utility[selected_array].sum()),
            redundancy=redundancy,
            qubo=qubo,
            solve_time_ms=(time.perf_counter() - started) * 1000.0,
        )


class PrioritizedReplaySelector:
    name = "prioritized"

    def select(self, replay: PrioritizedReplayBuffer, batch_size: int, beta: float) -> ReplaySelection:
        batch, indices, weights = replay.sample(batch_size, beta)
        return ReplaySelection(
            batch=batch,
            indices=np.asarray(indices, dtype=np.int64),
            weights=np.asarray(weights, dtype=np.float32),
            metrics={"selector_candidate_count": float(batch_size)},
        )


class QUBOReplaySelector:
    """Apply rollout QUBO selection to a prioritized replay candidate pool."""

    name = "qubo"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.pool_factor = max(1.0, float(config.get("qubo_pool_factor", 4.0)))
        self.reward_utility_weight = max(
            0.0, float(config.get("qubo_reward_utility_weight", 0.0))
        )
        self.solver = QUBORolloutSelector(
            utility_weight=float(config.get("qubo_utility_weight", 1.0)),
            diversity_weight=float(config.get("qubo_diversity_weight", 0.35)),
            cardinality_penalty=float(config.get("qubo_cardinality_penalty", 4.0)),
            local_search_passes=int(config.get("qubo_local_search_passes", 8)),
        )

    @staticmethod
    def _candidate_batch(
        replay: PrioritizedReplayBuffer, count: int, beta: float
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        size = len(replay)
        count = min(int(count), size)
        raw = np.asarray(replay.priorities[:size], dtype=np.float64)
        probabilities = np.power(np.maximum(raw, 1e-6), replay.alpha)
        probabilities /= probabilities.sum()
        indices = np.random.choice(size, size=count, replace=False, p=probabilities)
        weights = np.power(size * probabilities[indices], -float(beta))
        weights /= max(float(weights.max()), 1e-12)
        batch = (
            replay.obs[indices],
            replay.action[indices],
            replay.reward[indices],
            replay.next_obs[indices],
            replay.done[indices],
        )
        return batch, indices.astype(np.int64), weights.astype(np.float32)

    def select(self, replay: PrioritizedReplayBuffer, batch_size: int, beta: float) -> ReplaySelection:
        started = time.perf_counter()
        candidate_count = min(
            len(replay), max(int(batch_size), int(np.ceil(batch_size * self.pool_factor)))
        )
        candidate_batch, candidate_indices, _candidate_weights = self._candidate_batch(
            replay, candidate_count, beta
        )
        obs, actions, rewards, next_obs, dones = candidate_batch
        priorities = np.asarray(replay.priorities[candidate_indices], dtype=np.float64)
        utilities = np.log1p(np.maximum(priorities, 0.0))
        if self.reward_utility_weight > 0.0:
            utilities += self.reward_utility_weight * np.log1p(np.abs(rewards.reshape(-1)))
        delta = next_obs - obs
        embeddings = np.concatenate(
            (obs, delta, actions.astype(np.float32), rewards, dones), axis=1
        )
        candidates = [
            RolloutCandidate(
                key=str(int(candidate_indices[index])),
                utility=float(utilities[index]),
                embedding=embeddings[index],
                payload=int(candidate_indices[index]),
            )
            for index in range(candidate_count)
        ]
        selection = self.solver.select(candidates, int(batch_size))
        chosen = selection.selected_indices
        batch = tuple(np.asarray(component)[chosen] for component in candidate_batch)
        indices = candidate_indices[chosen]
        # The deterministic utility/diversity solve changes inclusion
        # probabilities in a way that ordinary PER weights do not correct.
        # Use neutral weights rather than presenting a partial correction as
        # unbiased importance sampling.
        weights = np.ones(len(chosen), dtype=np.float32)
        return ReplaySelection(
            batch=batch,  # type: ignore[arg-type]
            indices=indices,
            weights=weights,
            metrics={
                "selector_candidate_count": float(candidate_count),
                "selector_selected_count": float(len(chosen)),
                "selector_qubo_energy": float(selection.energy),
                "selector_utility_sum": float(selection.utility_sum),
                "selector_redundancy": float(selection.redundancy),
                "selector_importance_correction": 0.0,
                "selector_solve_time_ms": float(selection.solve_time_ms),
                "selector_time_ms": (time.perf_counter() - started) * 1000.0,
            },
            audit={
                "candidate_indices": [int(x) for x in candidate_indices.tolist()],
                "decisions": [int(x) for x in selection.decisions.tolist()],
            },
        )


def build_replay_selector(name: str, config: Mapping[str, Any]) -> ReplaySelector:
    normalized = str(name or "prioritized").strip().lower().replace("-", "_")
    if normalized in {"", "prioritized", "per"}:
        return PrioritizedReplaySelector()
    if normalized == "qubo":
        return QUBOReplaySelector(config)
    try:
        discovered = importlib_metadata.entry_points()
        entries = (
            discovered.select(group=SELECTOR_ENTRY_POINT_GROUP, name=normalized)
            if hasattr(discovered, "select")
            else [
                item
                for item in discovered.get(SELECTOR_ENTRY_POINT_GROUP, ())
                if item.name == normalized
            ]
        )
        for entry in entries:
            factory = entry.load()
            selector = factory(config)
            if hasattr(selector, "select"):
                return selector
    except Exception as exc:
        raise ValueError(f"could not load replay selector {name}: {exc}") from exc
    raise ValueError(f"unsupported replay selector: {name}")


def _trajectory_fragments(
    replay: PrioritizedReplayBuffer,
    metadata_by_idx: Mapping[int, Mapping[str, Any]],
    *,
    rollout_length: int,
    current_policy_version: int,
    max_policy_lag: int,
    reward_utility_weight: float,
) -> tuple[list[_TrajectoryFragment], Mapping[str, float]]:
    """Build contiguous, policy-fresh episode fragments from circular replay."""

    size = len(replay)
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    missing_metadata = 0
    stale = 0
    for index in range(size):
        item = metadata_by_idx.get(index)
        if not isinstance(item, Mapping):
            missing_metadata += 1
            continue
        actor_id = str(item.get("actor_id", "")).strip()
        episode_id = str(item.get("episode_id", "")).strip()
        try:
            timestep = int(item.get("timestep"))
        except Exception:
            missing_metadata += 1
            continue
        if not actor_id or not episode_id or timestep < 0:
            missing_metadata += 1
            continue
        raw_version = item.get("policy_version")
        if raw_version is not None:
            try:
                policy_version = int(raw_version)
            except Exception:
                policy_version = current_policy_version
            if (
                max_policy_lag >= 0
                and current_policy_version - policy_version > max_policy_lag
            ):
                stale += 1
                continue
        grouped.setdefault((actor_id, episode_id), []).append((timestep, index))

    length = max(1, int(rollout_length))
    raw_fragments: list[tuple[str, np.ndarray]] = []
    for (actor_id, episode_id), entries in sorted(grouped.items()):
        entries.sort(key=lambda pair: (pair[0], pair[1]))
        run: list[tuple[int, int]] = []

        def flush() -> None:
            if not run:
                return
            for start in range(0, len(run), length):
                block = run[start : start + length]
                indices = np.asarray([pair[1] for pair in block], dtype=np.int64)
                first_step = block[0][0]
                last_step = block[-1][0]
                raw_fragments.append(
                    (
                        f"{actor_id}/{episode_id}/{first_step}-{last_step}",
                        indices,
                    )
                )

        for timestep, replay_index in entries:
            if run and (
                timestep != run[-1][0] + 1
                or bool(replay.done[run[-1][1], 0])
            ):
                flush()
                run = []
            run.append((timestep, replay_index))
            if bool(replay.done[replay_index, 0]):
                flush()
                run = []
        flush()

    fragments: list[_TrajectoryFragment] = []
    for key, indices in raw_fragments:
        obs = replay.obs[indices]
        next_obs = replay.next_obs[indices]
        actions = replay.action[indices].reshape(-1)
        rewards = replay.reward[indices].reshape(-1)
        dones = replay.done[indices].reshape(-1)
        priorities = np.maximum(replay.priorities[indices], 1e-6)
        utility = float(np.log1p(priorities).mean())
        if reward_utility_weight > 0.0:
            utility += float(reward_utility_weight) * float(
                np.log1p(np.abs(rewards).sum())
            )
        embedding = np.concatenate(
            (
                obs.mean(axis=0),
                (next_obs - obs).mean(axis=0),
                np.asarray(
                    [
                        actions.mean(),
                        actions.std(),
                        rewards.mean(),
                        rewards.std(),
                        rewards.sum(),
                        dones.mean(),
                        float(len(indices)) / float(length),
                    ],
                    dtype=np.float32,
                ),
            )
        )
        fragments.append(
            _TrajectoryFragment(
                key=key,
                indices=indices,
                utility=utility,
                embedding=embedding,
            )
        )
    return fragments, {
        "selector_missing_trajectory_metadata": float(missing_metadata),
        "selector_stale_transition_count": float(stale),
    }


def select_trajectory_replay(
    replay: PrioritizedReplayBuffer,
    metadata_by_idx: Mapping[int, Mapping[str, Any]],
    *,
    batch_size: int,
    beta: float,
    rollout_length: int,
    current_policy_version: int,
    max_policy_lag: int,
    selector_name: str,
    config: Mapping[str, Any],
) -> ReplaySelection | None:
    """Select complete contiguous fragments for trajectory-mode plugins.

    A QUBO configuration operates on fragment utility and similarity, making
    each binary variable an auditable rollout yes/no decision.  The returned
    transition count can differ from ``batch_size`` because a selected
    fragment is never cut merely to meet a row count; the returned metrics
    expose the actual count.
    """

    started = time.perf_counter()
    fragments, base_metrics = _trajectory_fragments(
        replay,
        metadata_by_idx,
        rollout_length=rollout_length,
        current_policy_version=current_policy_version,
        max_policy_lag=max_policy_lag,
        reward_utility_weight=max(
            0.0, float(config.get("qubo_reward_utility_weight", 0.0))
        ),
    )
    if not fragments:
        return None
    mean_fragment_length = max(
        1.0, float(np.mean([len(fragment.indices) for fragment in fragments]))
    )
    desired = max(
        1,
        min(
            len(fragments),
            int(np.ceil(float(batch_size) / mean_fragment_length)),
        ),
    )
    normalized_selector = str(selector_name or "prioritized").strip().lower().replace("-", "_")
    selector_metrics: dict[str, float] = dict(base_metrics)
    audit: dict[str, Any] = {}

    if normalized_selector == "qubo":
        pool_factor = max(1.0, float(config.get("qubo_pool_factor", 4.0)))
        pool_count = min(
            len(fragments), max(desired, int(np.ceil(desired * pool_factor)))
        )
        fragment_priority = np.asarray(
            [
                float(np.maximum(replay.priorities[item.indices], 1e-6).mean())
                for item in fragments
            ],
            dtype=np.float64,
        )
        pool_probability = np.power(fragment_priority, replay.alpha)
        pool_probability /= pool_probability.sum()
        pool_indices = np.random.choice(
            len(fragments), size=pool_count, replace=False, p=pool_probability
        )
        pool = [fragments[int(index)] for index in pool_indices.tolist()]
        solver = QUBORolloutSelector(
            utility_weight=float(config.get("qubo_utility_weight", 1.0)),
            diversity_weight=float(config.get("qubo_diversity_weight", 0.35)),
            cardinality_penalty=float(config.get("qubo_cardinality_penalty", 4.0)),
            local_search_passes=int(config.get("qubo_local_search_passes", 8)),
        )
        qubo_candidates = [
            RolloutCandidate(
                key=item.key,
                utility=item.utility,
                embedding=item.embedding,
                payload=item,
            )
            for item in pool
        ]
        result = solver.select(qubo_candidates, min(desired, len(pool)))
        chosen = [pool[int(index)] for index in result.selected_indices.tolist()]
        selector_metrics.update(
            {
                "selector_candidate_rollouts": float(len(pool)),
                "selector_selected_rollouts": float(len(chosen)),
                "selector_qubo_energy": float(result.energy),
                "selector_utility_sum": float(result.utility_sum),
                "selector_redundancy": float(result.redundancy),
                "selector_solve_time_ms": float(result.solve_time_ms),
            }
        )
        audit = {
            "unit": "rollout_fragment",
            "candidate_keys": [item.key for item in pool],
            "decisions": [int(value) for value in result.decisions.tolist()],
        }
    else:
        fragment_priority = np.asarray(
            [
                float(np.maximum(replay.priorities[item.indices], 1e-6).mean())
                for item in fragments
            ],
            dtype=np.float64,
        )
        probabilities = np.power(fragment_priority, replay.alpha)
        probabilities /= probabilities.sum()
        selected_indices = np.random.choice(
            len(fragments), size=desired, replace=False, p=probabilities
        )
        chosen = [fragments[int(index)] for index in selected_indices.tolist()]
        selector_metrics.update(
            {
                "selector_candidate_rollouts": float(len(fragments)),
                "selector_selected_rollouts": float(len(chosen)),
            }
        )

    flat_indices = np.concatenate([item.indices for item in chosen]).astype(
        np.int64, copy=False
    )
    if normalized_selector == "qubo":
        weights = np.ones(len(flat_indices), dtype=np.float32)
        selector_metrics["selector_importance_correction"] = 0.0
    else:
        raw = np.maximum(replay.priorities[: len(replay)], 1e-6).astype(np.float64)
        transition_probability = np.power(raw, replay.alpha)
        transition_probability /= transition_probability.sum()
        weights = np.power(
            len(replay) * transition_probability[flat_indices], -float(beta)
        )
        weights /= max(float(weights.max()), 1e-12)
    batch = (
        replay.obs[flat_indices],
        replay.action[flat_indices],
        replay.reward[flat_indices],
        replay.next_obs[flat_indices],
        replay.done[flat_indices],
    )
    selector_metrics["selector_selected_count"] = float(len(flat_indices))
    selector_metrics["selector_time_ms"] = (
        time.perf_counter() - started
    ) * 1000.0
    return ReplaySelection(
        batch=batch,
        indices=flat_indices,
        weights=np.asarray(weights, dtype=np.float32),
        metrics=selector_metrics,
        audit=audit,
    )
