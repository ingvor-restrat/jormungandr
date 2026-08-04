"""Validated joint trajectories for variable entity/candidate policies.

One record represents one environment turn. Action factors and their selected
semantic candidates are nested inside that record, so adding factors cannot
silently multiply rewards or replay samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jormungandr.structured import (
    EntityCandidateObservation,
    StructuredPolicySpec,
    entity_candidate_observation_from_payload,
    entity_candidate_observation_to_payload,
)


JOINT_TRAJECTORY_STEP_SCHEMA = "jormungandr.structured_joint_step.v1"
JOINT_TRAJECTORY_SEQUENCE_SCHEMA = (
    "jormungandr.structured_joint_trajectory_sequence.v1"
)


@dataclass(frozen=True)
class StructuredActionFactor:
    """One categorical factor over candidates in a structured observation."""

    factor_id: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        factor_id = str(self.factor_id).strip()
        candidate_ids = tuple(str(value).strip() for value in self.candidate_ids)
        if not factor_id:
            raise ValueError("factor_id is required")
        if not candidate_ids or any(not value for value in candidate_ids):
            raise ValueError("each factor needs non-empty candidate IDs")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique within a factor")
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "candidate_ids", candidate_ids)


@dataclass(frozen=True)
class StructuredFactorChoice:
    """One conditional categorical decision within a joint action."""

    factor_id: str
    candidate_ids: tuple[str, ...]
    selected_candidate_id: str
    behavior_log_probability: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        factor_id = str(self.factor_id).strip()
        candidate_ids = tuple(str(value).strip() for value in self.candidate_ids)
        selected = str(self.selected_candidate_id).strip()
        if not factor_id:
            raise ValueError("factor_id is required")
        if not candidate_ids or any(not value for value in candidate_ids):
            raise ValueError("each factor needs non-empty candidate IDs")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique within a factor")
        if selected not in candidate_ids:
            raise ValueError("selected candidate is absent from its factor")
        log_probability = float(self.behavior_log_probability)
        if not math.isfinite(log_probability) or log_probability > 1e-6:
            raise ValueError("behavior log probability must be finite and <= 0")
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "selected_candidate_id", selected)
        object.__setattr__(self, "behavior_log_probability", log_probability)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class StructuredJointActionResult:
    """An exact sequentially masked sample from factor-local policy logits."""

    observation: EntityCandidateObservation
    factors: tuple[StructuredFactorChoice, ...]
    joint_log_probability: float
    behavior_value: float

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(factor.selected_candidate_id for factor in self.factors)


LegalMaskUpdate = Callable[
    [int, tuple[str, ...], np.ndarray],
    Sequence[bool] | np.ndarray,
]


def sample_structured_joint_action(
    observation: EntityCandidateObservation,
    factors: Sequence[StructuredActionFactor],
    candidate_logits: Sequence[float] | np.ndarray,
    *,
    behavior_value: float,
    deterministic: bool = False,
    rng: np.random.Generator | None = None,
    legal_mask_update: LegalMaskUpdate | None = None,
) -> StructuredJointActionResult:
    """Select one candidate per factor with an exact joint log probability.

    Before factor ``i`` is sampled, ``legal_mask_update`` may restrict its or
    later factors using choices ``0..i-1``. It may never enable an action the
    actor declared illegal initially or rewrite a previous factor's mask. This
    makes the final returned observation a sufficient record of every
    conditional distribution used by PPO.
    """

    ordered = tuple(factors)
    if not ordered:
        raise ValueError("at least one structured action factor is required")
    factor_ids = [factor.factor_id for factor in ordered]
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor IDs must be unique")
    grouped = [candidate for factor in ordered for candidate in factor.candidate_ids]
    if len(set(grouped)) != len(grouped) or set(grouped) != set(
        observation.candidate_ids
    ):
        raise ValueError("factor candidates must partition observation candidates")
    logits = np.asarray(candidate_logits, dtype=np.float64)
    if logits.shape != (len(observation.candidate_ids),):
        raise ValueError("candidate logits must align with observation candidates")
    value = float(behavior_value)
    if not math.isfinite(value):
        raise ValueError("behavior value must be finite")

    initial_mask = observation.legal_action_mask.copy()
    current_mask = initial_mask.copy()
    by_id = {
        candidate_id: index
        for index, candidate_id in enumerate(observation.candidate_ids)
    }
    selected: list[str] = []
    choices: list[StructuredFactorChoice] = []
    random_generator = rng if rng is not None else np.random.default_rng()
    prior_indices: set[int] = set()
    for factor_index, factor in enumerate(ordered):
        if legal_mask_update is not None:
            updated = np.asarray(
                legal_mask_update(
                    factor_index, tuple(selected), current_mask.copy()
                ),
                dtype=np.bool_,
            )
            if updated.shape != current_mask.shape:
                raise ValueError("updated legal mask has the wrong shape")
            if np.any(updated & ~initial_mask):
                raise ValueError("constraint updates cannot enable illegal candidates")
            if any(bool(updated[index]) != bool(current_mask[index]) for index in prior_indices):
                raise ValueError("constraint updates cannot rewrite prior-factor masks")
            current_mask = updated.copy()

        indices = np.asarray(
            [by_id[candidate_id] for candidate_id in factor.candidate_ids],
            dtype=np.int64,
        )
        legal_local = current_mask[indices]
        if not bool(legal_local.any()):
            raise ValueError(f"factor {factor.factor_id!r} has no legal candidate")
        legal_indices = indices[legal_local]
        legal_logits = logits[legal_indices]
        if not np.isfinite(legal_logits).all():
            raise ValueError("legal candidate logits must be finite")
        shifted = legal_logits - float(np.max(legal_logits))
        probabilities = np.exp(shifted)
        probabilities /= float(probabilities.sum())
        if deterministic:
            selected_local = int(np.argmax(legal_logits))
        else:
            selected_local = int(
                random_generator.choice(len(legal_indices), p=probabilities)
            )
        selected_index = int(legal_indices[selected_local])
        selected_id = observation.candidate_ids[selected_index]
        log_probability = math.log(float(probabilities[selected_local]))
        choices.append(
            StructuredFactorChoice(
                factor_id=factor.factor_id,
                candidate_ids=factor.candidate_ids,
                selected_candidate_id=selected_id,
                behavior_log_probability=log_probability,
                metadata={
                    "conditional_legal_candidate_ids": [
                        observation.candidate_ids[int(index)]
                        for index in legal_indices
                    ]
                },
            )
        )
        selected.append(selected_id)
        prior_indices.update(int(index) for index in indices)

    conditional_observation = EntityCandidateObservation(
        global_features=observation.global_features,
        entity_features=observation.entity_features,
        entity_type_ids=observation.entity_type_ids,
        entity_ids=observation.entity_ids,
        candidate_features=observation.candidate_features,
        candidate_ids=observation.candidate_ids,
        legal_action_mask=current_mask,
        candidate_entity_indices=observation.candidate_entity_indices,
        metadata=observation.metadata,
    )
    return StructuredJointActionResult(
        observation=conditional_observation,
        factors=tuple(choices),
        joint_log_probability=sum(
            choice.behavior_log_probability for choice in choices
        ),
        behavior_value=value,
    )


@dataclass(frozen=True)
class StructuredJointTrajectoryStep:
    """One environment turn with nested factor choices and one scalar reward."""

    actor_id: str
    episode_id: str
    timestep: int
    policy_version: int
    observation: EntityCandidateObservation
    factors: tuple[StructuredFactorChoice, ...]
    joint_behavior_log_probability: float
    behavior_value: float
    reward: float
    next_observation: EntityCandidateObservation
    terminated: bool
    truncated: bool = False
    split: str = "train"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actor_id = str(self.actor_id).strip()
        episode_id = str(self.episode_id).strip()
        if not actor_id or not episode_id:
            raise ValueError("actor_id and episode_id are required")
        if self.timestep < 0 or self.policy_version < 0:
            raise ValueError("timestep and policy_version must be non-negative")
        if self.terminated and self.truncated:
            raise ValueError("a step cannot be both terminated and truncated")
        split = "validation" if str(self.split) == "val" else str(self.split)
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        factors = tuple(self.factors)
        if not factors:
            raise ValueError("a joint step needs at least one action factor")
        factor_ids = [factor.factor_id for factor in factors]
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("factor IDs must be unique within a joint step")
        grouped_ids = [
            candidate_id
            for factor in factors
            for candidate_id in factor.candidate_ids
        ]
        if len(set(grouped_ids)) != len(grouped_ids):
            raise ValueError("each candidate must belong to exactly one factor")
        if set(grouped_ids) != set(self.observation.candidate_ids):
            raise ValueError(
                "factor candidate IDs must partition the observation candidates"
            )
        candidate_index = {
            candidate_id: index
            for index, candidate_id in enumerate(self.observation.candidate_ids)
        }
        for factor in factors:
            index = candidate_index[factor.selected_candidate_id]
            if not bool(self.observation.legal_action_mask[index]):
                raise ValueError("a selected factor candidate is observation-illegal")
        dimensions = (
            self.observation.global_dim,
            self.observation.entity_dim,
            self.observation.candidate_dim,
        )
        next_dimensions = (
            self.next_observation.global_dim,
            self.next_observation.entity_dim,
            self.next_observation.candidate_dim,
        )
        if dimensions != next_dimensions:
            raise ValueError(
                "observation and next observation feature dimensions must match"
            )
        joint_log_probability = float(self.joint_behavior_log_probability)
        expected_log_probability = sum(
            factor.behavior_log_probability for factor in factors
        )
        if not math.isfinite(joint_log_probability) or not math.isclose(
            joint_log_probability,
            expected_log_probability,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "joint behavior log probability must equal the conditional sum"
            )
        if not all(
            math.isfinite(float(value))
            for value in (self.behavior_value, self.reward)
        ):
            raise ValueError("behavior value and reward must be finite")
        object.__setattr__(self, "actor_id", actor_id)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(
            self, "joint_behavior_log_probability", joint_log_probability
        )
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(factor.selected_candidate_id for factor in self.factors)


def structured_joint_step_to_payload(
    step: StructuredJointTrajectoryStep,
) -> dict[str, Any]:
    return {
        "schema": JOINT_TRAJECTORY_STEP_SCHEMA,
        "actor_id": step.actor_id,
        "episode_id": step.episode_id,
        "timestep": step.timestep,
        "policy_version": step.policy_version,
        "split": step.split,
        "observation": entity_candidate_observation_to_payload(
            step.observation
        ),
        "factors": [
            {
                "factor_id": factor.factor_id,
                "candidate_ids": list(factor.candidate_ids),
                "selected_candidate_id": factor.selected_candidate_id,
                "behavior_log_probability": factor.behavior_log_probability,
                "metadata": dict(factor.metadata),
            }
            for factor in step.factors
        ],
        "joint_behavior_log_probability": step.joint_behavior_log_probability,
        "behavior_value": step.behavior_value,
        "reward": step.reward,
        "next_observation": entity_candidate_observation_to_payload(
            step.next_observation
        ),
        "terminated": step.terminated,
        "truncated": step.truncated,
        "metadata": dict(step.metadata),
    }


def structured_joint_step_from_payload(
    payload: Mapping[str, Any],
    *,
    spec: StructuredPolicySpec | None = None,
) -> StructuredJointTrajectoryStep:
    if not isinstance(payload, Mapping):
        raise ValueError("joint trajectory step must be an object")
    if str(payload.get("schema", "")) != JOINT_TRAJECTORY_STEP_SCHEMA:
        raise ValueError(
            f"joint trajectory schema must be {JOINT_TRAJECTORY_STEP_SCHEMA!r}"
        )
    raw_factors = payload.get("factors")
    if not isinstance(raw_factors, Sequence) or isinstance(
        raw_factors, (str, bytes)
    ):
        raise ValueError("joint trajectory factors must be an array")
    factors = []
    for raw_factor in raw_factors:
        if not isinstance(raw_factor, Mapping):
            raise ValueError("each joint trajectory factor must be an object")
        metadata = raw_factor.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("factor metadata must be an object")
        factors.append(
            StructuredFactorChoice(
                factor_id=str(raw_factor.get("factor_id", "")),
                candidate_ids=tuple(raw_factor.get("candidate_ids", ())),
                selected_candidate_id=str(
                    raw_factor.get("selected_candidate_id", "")
                ),
                behavior_log_probability=float(
                    raw_factor.get("behavior_log_probability", float("nan"))
                ),
                metadata=metadata,
            )
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("joint trajectory metadata must be an object")
    return StructuredJointTrajectoryStep(
        actor_id=str(payload.get("actor_id", "")),
        episode_id=str(payload.get("episode_id", "")),
        timestep=int(payload.get("timestep", -1)),
        policy_version=int(payload.get("policy_version", -1)),
        observation=entity_candidate_observation_from_payload(
            payload.get("observation", {}), spec=spec
        ),
        factors=tuple(factors),
        joint_behavior_log_probability=float(
            payload.get("joint_behavior_log_probability", float("nan"))
        ),
        behavior_value=float(payload.get("behavior_value", float("nan"))),
        reward=float(payload.get("reward", float("nan"))),
        next_observation=entity_candidate_observation_from_payload(
            payload.get("next_observation", {}), spec=spec
        ),
        terminated=bool(payload.get("terminated", False)),
        truncated=bool(payload.get("truncated", False)),
        split=str(payload.get("split", "train")),
        metadata=metadata,
    )


def structured_joint_trajectory_to_sequence_payload(
    trajectory: Sequence[StructuredJointTrajectoryStep],
) -> dict[str, Any]:
    """Encode one trajectory while storing its observation chain once.

    The legacy step-array wire repeats every intermediate structured
    observation twice: once as ``next_observation`` and once as the following
    step's ``observation``.  This sequence schema retains the same validated
    in-memory contract but transmits exactly ``N + 1`` observations for ``N``
    steps.
    """

    steps = validate_structured_joint_trajectory(trajectory)
    observations = [steps[0].observation]
    observations.extend(step.next_observation for step in steps)
    return {
        "schema": JOINT_TRAJECTORY_SEQUENCE_SCHEMA,
        "actor_id": steps[0].actor_id,
        "episode_id": steps[0].episode_id,
        "split": steps[0].split,
        "observations": [
            entity_candidate_observation_to_payload(observation)
            for observation in observations
        ],
        "steps": [
            {
                "timestep": step.timestep,
                "policy_version": step.policy_version,
                "factors": [
                    {
                        "factor_id": factor.factor_id,
                        "candidate_ids": list(factor.candidate_ids),
                        "selected_candidate_id": factor.selected_candidate_id,
                        "behavior_log_probability": (
                            factor.behavior_log_probability
                        ),
                        "metadata": dict(factor.metadata),
                    }
                    for factor in step.factors
                ],
                "joint_behavior_log_probability": (
                    step.joint_behavior_log_probability
                ),
                "behavior_value": step.behavior_value,
                "reward": step.reward,
                "terminated": step.terminated,
                "truncated": step.truncated,
                "metadata": dict(step.metadata),
            }
            for step in steps
        ],
    }


def structured_joint_trajectory_from_sequence_payload(
    payload: Mapping[str, Any],
    *,
    spec: StructuredPolicySpec | None = None,
) -> tuple[StructuredJointTrajectoryStep, ...]:
    """Decode and fully validate a compact observation-chain trajectory."""

    if not isinstance(payload, Mapping):
        raise ValueError("joint trajectory sequence must be an object")
    if str(payload.get("schema", "")) != JOINT_TRAJECTORY_SEQUENCE_SCHEMA:
        raise ValueError(
            "joint trajectory sequence schema must be "
            f"{JOINT_TRAJECTORY_SEQUENCE_SCHEMA!r}"
        )
    actor_id = str(payload.get("actor_id", ""))
    episode_id = str(payload.get("episode_id", ""))
    split = str(payload.get("split", "train"))
    raw_observations = payload.get("observations")
    raw_steps = payload.get("steps")
    if not isinstance(raw_observations, Sequence) or isinstance(
        raw_observations, (str, bytes)
    ):
        raise ValueError("sequence observations must be an array")
    if not isinstance(raw_steps, Sequence) or isinstance(
        raw_steps, (str, bytes)
    ):
        raise ValueError("sequence steps must be an array")
    if not raw_steps or len(raw_observations) != len(raw_steps) + 1:
        raise ValueError("a sequence needs exactly one more observation than step")
    observations = tuple(
        entity_candidate_observation_from_payload(item, spec=spec)
        for item in raw_observations
    )
    steps: list[StructuredJointTrajectoryStep] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise ValueError("each sequence step must be an object")
        raw_factors = raw_step.get("factors")
        if not isinstance(raw_factors, Sequence) or isinstance(
            raw_factors, (str, bytes)
        ):
            raise ValueError("sequence step factors must be an array")
        factors = []
        for raw_factor in raw_factors:
            if not isinstance(raw_factor, Mapping):
                raise ValueError("each sequence factor must be an object")
            factor_metadata = raw_factor.get("metadata", {})
            if not isinstance(factor_metadata, Mapping):
                raise ValueError("sequence factor metadata must be an object")
            factors.append(
                StructuredFactorChoice(
                    factor_id=str(raw_factor.get("factor_id", "")),
                    candidate_ids=tuple(raw_factor.get("candidate_ids", ())),
                    selected_candidate_id=str(
                        raw_factor.get("selected_candidate_id", "")
                    ),
                    behavior_log_probability=float(
                        raw_factor.get(
                            "behavior_log_probability", float("nan")
                        )
                    ),
                    metadata=factor_metadata,
                )
            )
        step_metadata = raw_step.get("metadata", {})
        if not isinstance(step_metadata, Mapping):
            raise ValueError("sequence step metadata must be an object")
        steps.append(
            StructuredJointTrajectoryStep(
                actor_id=actor_id,
                episode_id=episode_id,
                timestep=int(raw_step.get("timestep", -1)),
                policy_version=int(raw_step.get("policy_version", -1)),
                observation=observations[index],
                factors=tuple(factors),
                joint_behavior_log_probability=float(
                    raw_step.get(
                        "joint_behavior_log_probability", float("nan")
                    )
                ),
                behavior_value=float(
                    raw_step.get("behavior_value", float("nan"))
                ),
                reward=float(raw_step.get("reward", float("nan"))),
                next_observation=observations[index + 1],
                terminated=bool(raw_step.get("terminated", False)),
                truncated=bool(raw_step.get("truncated", False)),
                split=split,
                metadata=step_metadata,
            )
        )
    return validate_structured_joint_trajectory(steps)


def _same_observation(
    left: EntityCandidateObservation,
    right: EntityCandidateObservation,
) -> bool:
    return (
        np.array_equal(left.global_features, right.global_features)
        and np.array_equal(left.entity_features, right.entity_features)
        and np.array_equal(left.entity_type_ids, right.entity_type_ids)
        and left.entity_ids == right.entity_ids
        and np.array_equal(left.candidate_features, right.candidate_features)
        and left.candidate_ids == right.candidate_ids
        and np.array_equal(left.legal_action_mask, right.legal_action_mask)
        and np.array_equal(
            left.candidate_entity_indices, right.candidate_entity_indices
        )
    )


def validate_structured_joint_trajectory(
    steps: Sequence[StructuredJointTrajectoryStep],
    *,
    require_complete: bool = True,
) -> tuple[StructuredJointTrajectoryStep, ...]:
    """Validate one actor/episode sequence and return an immutable snapshot."""

    trajectory = tuple(steps)
    if not trajectory:
        raise ValueError("a trajectory must contain at least one step")
    actor_id = trajectory[0].actor_id
    episode_id = trajectory[0].episode_id
    split = trajectory[0].split
    for index, step in enumerate(trajectory):
        if (
            step.actor_id != actor_id
            or step.episode_id != episode_id
            or step.split != split
            or step.timestep != index
        ):
            raise ValueError(
                "trajectory actor, episode, split, and timesteps must be contiguous"
            )
    for index, step in enumerate(trajectory):
        if index < len(trajectory) - 1:
            if step.done:
                raise ValueError("only the final trajectory step may be terminal")
            if not _same_observation(
                step.next_observation, trajectory[index + 1].observation
            ):
                raise ValueError(
                    "a step next observation must match the following observation"
                )
    if require_complete and not trajectory[-1].done:
        raise ValueError("a complete trajectory must end in termination or truncation")
    return trajectory
