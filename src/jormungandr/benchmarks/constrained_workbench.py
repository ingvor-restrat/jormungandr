"""A small exact-oracle benchmark for constrained joint actions.

The environment deliberately contains no application-specific concepts.  A
variable set of workers chooses among a variable set of jobs or PASS.  Jobs
consume a shared capacity, can be selected at most once, and may conflict with
other jobs.  Step utilities accumulate internally; Gymnasium reward is zero
until the final turn, when it is normalized by the exact episode oracle.

The ordinary Gymnasium interface uses a padded ``MultiDiscrete`` action.  The
structured helpers expose typed entities, state-local semantic candidates,
ordered factors, and a sequential legal-mask callback for Jörmungandr.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_trajectory import StructuredActionFactor


GLOBAL_DIM = 6
ENTITY_DIM = 9
CANDIDATE_DIM = 11
ENTITY_TYPE_COUNT = 3


@dataclass(frozen=True)
class WorkbenchWorker:
    worker_id: int
    kind: int
    efficiency: float

    def __post_init__(self) -> None:
        if int(self.worker_id) < 0 or int(self.kind) not in {0, 1}:
            raise ValueError("worker ID and binary kind must be valid")
        if not math.isfinite(float(self.efficiency)) or not (
            0.5 <= float(self.efficiency) <= 1.5
        ):
            raise ValueError("worker efficiency must be finite and in [0.5, 1.5]")


@dataclass(frozen=True)
class WorkbenchJob:
    job_id: int
    kind: int
    value: float
    capacity_cost: int
    conflict_group: int

    def __post_init__(self) -> None:
        if int(self.job_id) < 0 or int(self.kind) not in {0, 1}:
            raise ValueError("job ID and binary kind must be valid")
        if not math.isfinite(float(self.value)) or float(self.value) <= 0.0:
            raise ValueError("job value must be finite and positive")
        if int(self.capacity_cost) <= 0 or int(self.conflict_group) < -1:
            raise ValueError("job cost must be positive and conflict group >= -1")


@dataclass(frozen=True)
class WorkbenchTurn:
    workers: tuple[WorkbenchWorker, ...]
    jobs: tuple[WorkbenchJob, ...]
    capacity: int

    def __post_init__(self) -> None:
        if not self.workers or not self.jobs or int(self.capacity) <= 0:
            raise ValueError("a turn requires workers, jobs, and positive capacity")


@dataclass(frozen=True)
class WorkbenchOracleDecision:
    selected_candidate_ids: tuple[str, ...]
    utility: float


def _worker_id(worker: WorkbenchWorker) -> str:
    return f"worker:{worker.worker_id}"


def _job_id(job: WorkbenchJob) -> str:
    return f"job:{job.job_id}"


def _pass_candidate(worker: WorkbenchWorker) -> str:
    return f"worker:{worker.worker_id}:pass"


def _job_candidate(worker: WorkbenchWorker, job: WorkbenchJob) -> str:
    return f"worker:{worker.worker_id}:job:{job.job_id}"


class ConstrainedWorkbench(gym.Env[Mapping[str, np.ndarray], np.ndarray]):
    """Finite-horizon constrained assignment task with an exact oracle."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        horizon: int = 3,
        worker_counts: Sequence[int] = (2, 3, 4),
        job_counts: Sequence[int] = (2, 3, 4),
        max_workers: int = 4,
        max_jobs: int = 4,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.worker_counts = tuple(sorted({int(value) for value in worker_counts}))
        self.job_counts = tuple(sorted({int(value) for value in job_counts}))
        self.max_workers = int(max_workers)
        self.max_jobs = int(max_jobs)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not self.worker_counts or not self.job_counts:
            raise ValueError("worker and job count sets cannot be empty")
        if min(self.worker_counts) <= 0 or max(self.worker_counts) > self.max_workers:
            raise ValueError("worker counts must fit max_workers")
        if min(self.job_counts) <= 0 or max(self.job_counts) > self.max_jobs:
            raise ValueError("job counts must fit max_jobs")

        self.action_space = spaces.MultiDiscrete(
            np.full(self.max_workers, self.max_jobs + 1, dtype=np.int64)
        )
        self.observation_space = spaces.Dict(
            {
                "turn": spaces.Box(0, self.horizon, shape=(1,), dtype=np.int32),
                "worker_count": spaces.Box(
                    0, self.max_workers, shape=(1,), dtype=np.int32
                ),
                "job_count": spaces.Box(
                    0, self.max_jobs, shape=(1,), dtype=np.int32
                ),
                "capacity": spaces.Box(0, 16, shape=(1,), dtype=np.int32),
                "workers": spaces.Box(
                    low=-1.0,
                    high=2.0,
                    shape=(self.max_workers, 3),
                    dtype=np.float32,
                ),
                "worker_mask": spaces.MultiBinary(self.max_workers),
                "jobs": spaces.Box(
                    low=-1.0,
                    high=16.0,
                    shape=(self.max_jobs, 5),
                    dtype=np.float32,
                ),
                "job_mask": spaces.MultiBinary(self.max_jobs),
                "initial_action_mask": spaces.MultiBinary(
                    (self.max_workers, self.max_jobs + 1)
                ),
            }
        )
        self._turns: tuple[WorkbenchTurn, ...] = ()
        self._turn_index = 0
        self._episode_utility = 0.0
        self._oracle_terminal_utility = 0.0
        self._terminated = False
        self._invalid_action_attempts = 0

    @property
    def current_turn(self) -> WorkbenchTurn:
        if not self._turns:
            raise RuntimeError("reset must be called before accessing a turn")
        return self._turns[min(self._turn_index, len(self._turns) - 1)]

    @property
    def invalid_action_attempts(self) -> int:
        return self._invalid_action_attempts

    @property
    def oracle_terminal_utility(self) -> float:
        return self._oracle_terminal_utility

    @staticmethod
    def assignment_utility(worker: WorkbenchWorker, job: WorkbenchJob) -> float:
        match_multiplier = 1.0 if worker.kind == job.kind else 0.25
        return float(job.value * worker.efficiency * match_multiplier)

    def _generate_turns(self) -> tuple[WorkbenchTurn, ...]:
        worker_count = int(self.np_random.choice(self.worker_counts))
        kind_offset = int(self.np_random.integers(0, 2))
        workers = tuple(
            WorkbenchWorker(
                worker_id=index,
                kind=(index + kind_offset) % 2,
                efficiency=float(self.np_random.uniform(0.75, 1.25)),
            )
            for index in range(worker_count)
        )
        turns: list[WorkbenchTurn] = []
        for _ in range(self.horizon):
            job_count = int(self.np_random.choice(self.job_counts))
            kinds = np.asarray(
                [(index + int(self.np_random.integers(0, 2))) % 2 for index in range(job_count)]
            )
            self.np_random.shuffle(kinds)
            costs = self.np_random.integers(1, 4, size=job_count)
            values = self.np_random.integers(4, 13, size=job_count)
            # At least the first pair conflicts when two jobs exist; later
            # groups vary, so job identity alone cannot solve the task.
            conflicts = np.arange(job_count, dtype=np.int64)
            if job_count >= 2:
                conflicts[0] = conflicts[1] = 0
            if job_count >= 4 and bool(self.np_random.integers(0, 2)):
                conflicts[2] = conflicts[3] = 1
            permutation = self.np_random.permutation(job_count)
            jobs = tuple(
                WorkbenchJob(
                    job_id=slot,
                    kind=int(kinds[source]),
                    value=float(values[source]),
                    capacity_cost=int(costs[source]),
                    conflict_group=int(conflicts[source]),
                )
                for slot, source in enumerate(permutation)
            )
            capacity = int(
                self.np_random.integers(
                    2,
                    min(7, max(3, int(np.sum(costs)))) + 1,
                )
            )
            turns.append(
                WorkbenchTurn(workers=workers, jobs=jobs, capacity=capacity)
            )
        return tuple(turns)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
        super().reset(seed=seed)
        del options
        self._turns = self._generate_turns()
        self._turn_index = 0
        self._episode_utility = 0.0
        self._terminated = False
        self._invalid_action_attempts = 0
        self._oracle_terminal_utility = sum(
            self._oracle_for_turn(turn).utility for turn in self._turns
        )
        observation = self._padded_observation()
        return observation, {
            "oracle_terminal_utility": self._oracle_terminal_utility,
            "oracle_terminal_return": 1.0,
        }

    def _initial_action_mask(self, turn: WorkbenchTurn | None = None) -> np.ndarray:
        active = self.current_turn if turn is None else turn
        mask = np.zeros(
            (self.max_workers, self.max_jobs + 1), dtype=np.int8
        )
        for worker in active.workers:
            mask[worker.worker_id, 0] = 1
            for job in active.jobs:
                if job.capacity_cost <= active.capacity:
                    mask[worker.worker_id, job.job_id + 1] = 1
        for worker_index in range(len(active.workers), self.max_workers):
            mask[worker_index, 0] = 1
        return mask

    def _padded_observation(self) -> Mapping[str, np.ndarray]:
        turn = self.current_turn
        workers = np.zeros((self.max_workers, 3), dtype=np.float32)
        worker_mask = np.zeros(self.max_workers, dtype=np.int8)
        for worker in turn.workers:
            workers[worker.worker_id] = (
                float(worker.kind),
                float(worker.efficiency),
                1.0,
            )
            worker_mask[worker.worker_id] = 1
        jobs = np.zeros((self.max_jobs, 5), dtype=np.float32)
        job_mask = np.zeros(self.max_jobs, dtype=np.int8)
        for job in turn.jobs:
            jobs[job.job_id] = (
                float(job.kind),
                float(job.value),
                float(job.capacity_cost),
                float(job.conflict_group),
                1.0,
            )
            job_mask[job.job_id] = 1
        return {
            "turn": np.asarray([self._turn_index], dtype=np.int32),
            "worker_count": np.asarray([len(turn.workers)], dtype=np.int32),
            "job_count": np.asarray([len(turn.jobs)], dtype=np.int32),
            "capacity": np.asarray([turn.capacity], dtype=np.int32),
            "workers": workers,
            "worker_mask": worker_mask,
            "jobs": jobs,
            "job_mask": job_mask,
            "initial_action_mask": self._initial_action_mask(turn),
        }

    @staticmethod
    def _decode_candidate(candidate_id: str) -> tuple[int, int | None]:
        parts = str(candidate_id).split(":")
        if len(parts) == 3 and parts[0] == "worker" and parts[2] == "pass":
            return int(parts[1]), None
        if (
            len(parts) == 4
            and parts[0] == "worker"
            and parts[2] == "job"
        ):
            return int(parts[1]), int(parts[3])
        raise ValueError(f"invalid workbench candidate ID: {candidate_id!r}")

    def semantic_to_action(
        self, selected_candidate_ids: Sequence[str]
    ) -> np.ndarray:
        selected = tuple(selected_candidate_ids)
        turn = self.current_turn
        if len(selected) != len(turn.workers):
            raise ValueError("one semantic choice is required per active worker")
        action = np.zeros(self.max_workers, dtype=np.int64)
        for expected_worker, candidate_id in zip(turn.workers, selected):
            worker_id, job_id = self._decode_candidate(candidate_id)
            if worker_id != expected_worker.worker_id:
                raise ValueError("semantic choices must follow worker factor order")
            action[worker_id] = 0 if job_id is None else job_id + 1
        return action

    def action_to_semantic(self, action: Sequence[int] | np.ndarray) -> tuple[str, ...]:
        values = np.asarray(action, dtype=np.int64)
        if values.shape != (self.max_workers,):
            raise ValueError("padded joint action has the wrong shape")
        turn = self.current_turn
        if np.any(values[len(turn.workers) :] != 0):
            raise ValueError("inactive workers must PASS")
        selected: list[str] = []
        for worker in turn.workers:
            choice = int(values[worker.worker_id])
            if choice == 0:
                selected.append(_pass_candidate(worker))
            elif 1 <= choice <= len(turn.jobs):
                selected.append(_job_candidate(worker, turn.jobs[choice - 1]))
            else:
                raise ValueError("worker selected an absent job")
        return tuple(selected)

    def _validate_and_utility(
        self,
        turn: WorkbenchTurn,
        selected_candidate_ids: Sequence[str],
    ) -> float:
        selected = tuple(selected_candidate_ids)
        if len(selected) != len(turn.workers):
            raise ValueError("one choice is required per worker")
        jobs_by_id = {job.job_id: job for job in turn.jobs}
        used_jobs: set[int] = set()
        used_conflicts: set[int] = set()
        capacity_used = 0
        utility = 0.0
        for expected_worker, candidate_id in zip(turn.workers, selected):
            worker_id, job_id = self._decode_candidate(candidate_id)
            if worker_id != expected_worker.worker_id:
                raise ValueError("worker factors are out of order")
            if job_id is None:
                continue
            job = jobs_by_id.get(job_id)
            if job is None:
                raise ValueError("selected job is absent")
            if job_id in used_jobs:
                raise ValueError("a job cannot be selected twice")
            if job.conflict_group >= 0 and job.conflict_group in used_conflicts:
                raise ValueError("selected jobs conflict")
            capacity_used += job.capacity_cost
            if capacity_used > turn.capacity:
                raise ValueError("selected jobs exceed shared capacity")
            used_jobs.add(job_id)
            if job.conflict_group >= 0:
                used_conflicts.add(job.conflict_group)
            utility += self.assignment_utility(expected_worker, job)
        return float(utility)

    def is_feasible(self, selected_candidate_ids: Sequence[str]) -> bool:
        try:
            self._validate_and_utility(self.current_turn, selected_candidate_ids)
        except ValueError:
            return False
        return True

    def step(
        self, action: Sequence[int] | np.ndarray
    ) -> tuple[Mapping[str, np.ndarray], float, bool, bool, Mapping[str, Any]]:
        if self._terminated:
            raise RuntimeError("reset is required after termination")
        try:
            selected = self.action_to_semantic(action)
            turn_utility = self._validate_and_utility(self.current_turn, selected)
        except ValueError:
            self._invalid_action_attempts += 1
            raise
        self._episode_utility += turn_utility
        self._turn_index += 1
        self._terminated = self._turn_index >= self.horizon
        reward = (
            self._episode_utility / self._oracle_terminal_utility
            if self._terminated
            else 0.0
        )
        observation = self._padded_observation()
        info = {
            "selected_candidate_ids": selected,
            "turn_utility": turn_utility,
            "episode_utility": self._episode_utility,
            "oracle_terminal_utility": self._oracle_terminal_utility,
            "feasible": True,
            "invalid_action_attempts": self._invalid_action_attempts,
        }
        return observation, float(reward), self._terminated, False, info

    def step_semantic(
        self, selected_candidate_ids: Sequence[str]
    ) -> tuple[Mapping[str, np.ndarray], float, bool, bool, Mapping[str, Any]]:
        return self.step(self.semantic_to_action(selected_candidate_ids))

    def structured_observation(self) -> EntityCandidateObservation:
        turn = self.current_turn
        entities: list[list[float]] = []
        entity_type_ids: list[int] = []
        entity_ids: list[str] = []
        for worker in turn.workers:
            entities.append(
                [
                    1.0,
                    0.0,
                    0.0,
                    float(worker.kind == 0),
                    float(worker.kind == 1),
                    float(worker.efficiency),
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            entity_type_ids.append(0)
            entity_ids.append(_worker_id(worker))
        for job in turn.jobs:
            entities.append(
                [
                    0.0,
                    1.0,
                    0.0,
                    float(job.kind == 0),
                    float(job.kind == 1),
                    0.0,
                    float(job.value / 12.0),
                    float(job.capacity_cost / 3.0),
                    float((job.conflict_group + 1) / 5.0),
                ]
            )
            entity_type_ids.append(1)
            entity_ids.append(_job_id(job))
        resource_index = len(entities)
        entities.append(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, turn.capacity / 6.0, 0.0]
        )
        entity_type_ids.append(2)
        entity_ids.append("resource:capacity")

        candidate_features: list[list[float]] = []
        candidate_ids: list[str] = []
        candidate_references: list[list[int]] = []
        legal_mask: list[bool] = []
        job_entity_offset = len(turn.workers)
        for worker in turn.workers:
            candidate_features.append(
                [
                    1.0,
                    0.0,
                    float(worker.kind == 0),
                    float(worker.kind == 1),
                    float(worker.efficiency),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            candidate_ids.append(_pass_candidate(worker))
            candidate_references.append([worker.worker_id, resource_index])
            legal_mask.append(True)
            for job in turn.jobs:
                utility = self.assignment_utility(worker, job)
                candidate_features.append(
                    [
                        0.0,
                        1.0,
                        float(worker.kind == 0),
                        float(worker.kind == 1),
                        float(worker.efficiency),
                        float(job.kind == 0),
                        float(job.kind == 1),
                        float(job.value / 12.0),
                        float(job.capacity_cost / 3.0),
                        float(worker.kind == job.kind),
                        float(utility / 15.0),
                    ]
                )
                candidate_ids.append(_job_candidate(worker, job))
                candidate_references.append(
                    [worker.worker_id, job_entity_offset + job.job_id]
                )
                legal_mask.append(job.capacity_cost <= turn.capacity)
        return EntityCandidateObservation(
            global_features=np.asarray(
                [
                    self._turn_index / self.horizon,
                    (self.horizon - self._turn_index) / self.horizon,
                    turn.capacity / 6.0,
                    len(turn.workers) / self.max_workers,
                    len(turn.jobs) / self.max_jobs,
                    1.0,
                ],
                dtype=np.float32,
            ),
            entity_features=np.asarray(entities, dtype=np.float32),
            entity_type_ids=np.asarray(entity_type_ids, dtype=np.int64),
            entity_ids=tuple(entity_ids),
            candidate_features=np.asarray(candidate_features, dtype=np.float32),
            candidate_ids=tuple(candidate_ids),
            legal_action_mask=np.asarray(legal_mask, dtype=np.bool_),
            candidate_entity_indices=np.asarray(
                candidate_references, dtype=np.int64
            ),
            metadata={
                "benchmark": "ConstrainedWorkbench-v0",
                "turn": self._turn_index,
                "worker_count": len(turn.workers),
                "job_count": len(turn.jobs),
                "capacity": turn.capacity,
            },
        )

    def action_factors(self) -> tuple[StructuredActionFactor, ...]:
        turn = self.current_turn
        return tuple(
            StructuredActionFactor(
                factor_id=_worker_id(worker),
                candidate_ids=tuple(
                    [_pass_candidate(worker)]
                    + [_job_candidate(worker, job) for job in turn.jobs]
                ),
            )
            for worker in turn.workers
        )

    def legal_mask_update(
        self,
        factor_index: int,
        selected_candidate_ids: tuple[str, ...],
        current_mask: np.ndarray,
    ) -> np.ndarray:
        """Restrict the current/later factors from prior semantic choices."""

        turn = self.current_turn
        factors = self.action_factors()
        if not 0 <= int(factor_index) < len(factors):
            raise ValueError("factor index is out of range")
        if len(selected_candidate_ids) != int(factor_index):
            raise ValueError("selected prefix does not match factor index")
        used_jobs: set[int] = set()
        used_conflicts: set[int] = set()
        capacity_used = 0
        jobs_by_id = {job.job_id: job for job in turn.jobs}
        for expected_worker, candidate_id in zip(
            turn.workers, selected_candidate_ids
        ):
            worker_id, job_id = self._decode_candidate(candidate_id)
            if worker_id != expected_worker.worker_id:
                raise ValueError("selected prefix is out of factor order")
            if job_id is None:
                continue
            job = jobs_by_id[job_id]
            if job_id in used_jobs:
                raise ValueError("selected prefix repeats a job")
            if job.conflict_group >= 0 and job.conflict_group in used_conflicts:
                raise ValueError("selected prefix contains a conflict")
            used_jobs.add(job_id)
            capacity_used += job.capacity_cost
            if capacity_used > turn.capacity:
                raise ValueError("selected prefix exceeds capacity")
            if job.conflict_group >= 0:
                used_conflicts.add(job.conflict_group)

        updated = np.asarray(current_mask, dtype=np.bool_).copy()
        by_id = {
            candidate_id: index
            for index, candidate_id in enumerate(
                self.structured_observation().candidate_ids
            )
        }
        for factor in factors[factor_index:]:
            for candidate_id in factor.candidate_ids:
                _, job_id = self._decode_candidate(candidate_id)
                if job_id is None:
                    continue
                job = jobs_by_id[job_id]
                if (
                    job_id in used_jobs
                    or capacity_used + job.capacity_cost > turn.capacity
                    or (
                        job.conflict_group >= 0
                        and job.conflict_group in used_conflicts
                    )
                ):
                    updated[by_id[candidate_id]] = False
        return updated

    def _oracle_for_turn(self, turn: WorkbenchTurn) -> WorkbenchOracleDecision:
        best_utility = -math.inf
        best_choices: tuple[str, ...] = ()

        def search(
            worker_index: int,
            choices: tuple[str, ...],
            used_jobs: frozenset[int],
            used_conflicts: frozenset[int],
            capacity_used: int,
            utility: float,
        ) -> None:
            nonlocal best_choices, best_utility
            if worker_index == len(turn.workers):
                if utility > best_utility + 1e-12 or (
                    math.isclose(utility, best_utility, abs_tol=1e-12)
                    and (not best_choices or choices < best_choices)
                ):
                    best_utility = utility
                    best_choices = choices
                return
            worker = turn.workers[worker_index]
            search(
                worker_index + 1,
                choices + (_pass_candidate(worker),),
                used_jobs,
                used_conflicts,
                capacity_used,
                utility,
            )
            for job in turn.jobs:
                if job.job_id in used_jobs:
                    continue
                if capacity_used + job.capacity_cost > turn.capacity:
                    continue
                if (
                    job.conflict_group >= 0
                    and job.conflict_group in used_conflicts
                ):
                    continue
                next_conflicts = used_conflicts
                if job.conflict_group >= 0:
                    next_conflicts = used_conflicts | {job.conflict_group}
                search(
                    worker_index + 1,
                    choices + (_job_candidate(worker, job),),
                    used_jobs | {job.job_id},
                    next_conflicts,
                    capacity_used + job.capacity_cost,
                    utility + self.assignment_utility(worker, job),
                )

        search(0, (), frozenset(), frozenset(), 0, 0.0)
        return WorkbenchOracleDecision(best_choices, float(best_utility))

    def oracle_joint_action(self) -> WorkbenchOracleDecision:
        return self._oracle_for_turn(self.current_turn)

    def random_legal_joint_action(
        self, rng: np.random.Generator
    ) -> tuple[str, ...]:
        observation = self.structured_observation()
        current_mask = observation.legal_action_mask.copy()
        by_id = {
            candidate_id: index
            for index, candidate_id in enumerate(observation.candidate_ids)
        }
        selected: list[str] = []
        for factor_index, factor in enumerate(self.action_factors()):
            current_mask = self.legal_mask_update(
                factor_index, tuple(selected), current_mask
            )
            legal = [
                candidate_id
                for candidate_id in factor.candidate_ids
                if bool(current_mask[by_id[candidate_id]])
            ]
            selected.append(str(rng.choice(legal)))
        return tuple(selected)

    def close(self) -> None:
        self._turns = ()
