"""Central learner runtime for entity/candidate HTTP models.

This is the structured counterpart to the legacy vector/discrete service
profile.  External actors own environments; this manager owns the sole policy,
prioritized replay, background learner updates, versions, and checkpoints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry, canonical_algorithm_name
from jormungandr.core import TBLogger
from jormungandr.structured import (
    StructuredPolicySpec,
    entity_candidate_observation_from_payload,
)
from jormungandr.structured_replay import (
    StructuredPrioritizedReplayBuffer,
    StructuredReplayTransition,
)
from jormungandr.structured_trajectory import (
    structured_joint_trajectory_from_sequence_payload,
    structured_joint_step_from_payload,
    validate_structured_joint_trajectory,
)
from jormungandr.structured_trajectory_store import StructuredTrajectoryBuffer
from jormungandr.structured_supervision import (
    structured_supervision_from_payload,
)
from jormungandr.structured_supervision_store import StructuredSupervisionBuffer


@dataclass(frozen=True)
class StructuredServiceConfig:
    algo: str
    replay_mode: str
    device: str
    batch_size: int
    min_replay: int
    updates_per_tick: int
    tick_interval_s: float
    beta0: float
    beta_steps: int
    replay_ratio: float
    supervision_sampling: str
    checkpoint_every: int
    checkpoint_dir: str
    metric_history_size: int
    min_trajectory_steps: int
    max_trajectory_batch_steps: int
    max_policy_lag: int
    max_updates: int
    seed: int | None
    agent_config: Mapping[str, Any]


@dataclass
class StructuredModelRecord:
    model_id: str
    spec: StructuredPolicySpec
    capacity: int
    validation_capacity: int
    alpha: float
    metadata: dict[str, Any]
    replay: StructuredPrioritizedReplayBuffer
    validation: StructuredPrioritizedReplayBuffer
    trajectories: StructuredTrajectoryBuffer
    validation_trajectories: StructuredTrajectoryBuffer
    supervision: StructuredSupervisionBuffer
    validation_supervision: StructuredSupervisionBuffer
    config: StructuredServiceConfig
    plugin: Any
    agent: Any
    tb_logger: TBLogger
    tensorboard_enabled: bool
    tensorboard_logdir: str
    trainable: bool = True
    checkpoint_source: str = ""
    policy_initialization_source: str = ""
    restored_updates: int = 0
    restored_policy_version: int = 0
    created_ts: float = field(default_factory=time.time)
    updates: int = 0
    policy_version: int = 0
    replay_add_calls: int = 0
    replay_add_items: int = 0
    train_add_items: int = 0
    validation_add_items: int = 0
    inference_calls: int = 0
    score_calls: int = 0
    trajectory_add_calls: int = 0
    trajectory_add_episodes: int = 0
    trajectory_add_steps: int = 0
    trajectory_train_episodes: int = 0
    trajectory_train_steps: int = 0
    supervision_add_calls: int = 0
    supervision_add_items: int = 0
    supervision_train_items: int = 0
    policy_lag_count: int = 0
    policy_lag_sum: float = 0.0
    policy_lag_max: int = 0
    actor_latency_count: int = 0
    actor_latency_sum_ms: float = 0.0
    seen_trajectory_keys: set[tuple[str, str, str]] = field(
        default_factory=set
    )
    last_loss: float = 0.0
    last_td_abs: float = 0.0
    last_update_ts: float = 0.0
    last_error: str = ""
    last_metrics: dict[str, float] = field(default_factory=dict)
    metric_history: deque[dict[str, Any]] = field(default_factory=deque)
    performance_history: deque[dict[str, Any]] = field(default_factory=deque)
    latest_performance_metrics: dict[str, float] = field(default_factory=dict)
    metrics_calls: int = 0
    metrics_points: int = 0
    last_metrics_step: int = 0
    last_metrics_ts: float = 0.0
    last_checkpoint: str = ""
    checkpoint_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    agent_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class StructuredServiceManager:
    """Thread-safe registry of central structured learner models."""

    def __init__(
        self,
        *,
        checkpoint_root: str | Path | None = None,
        tensorboard_root: str | Path | None = None,
    ) -> None:
        self._records: dict[str, StructuredModelRecord] = {}
        self._lock = threading.Lock()
        self._model_build_lock = threading.Lock()
        self._checkpoint_root = (
            Path(checkpoint_root).expanduser().resolve()
            if checkpoint_root is not None and str(checkpoint_root).strip()
            else None
        )
        self._tensorboard_root = (
            Path(tensorboard_root).expanduser().resolve()
            if tensorboard_root is not None and str(tensorboard_root).strip()
            else None
        )

    def _tensorboard_logdir(self, model_id: str, raw: str) -> str:
        value = str(raw or "").strip()
        if value:
            base = Path(value).expanduser()
            if not base.is_absolute() and self._tensorboard_root is not None:
                base = self._tensorboard_root / base
        elif self._tensorboard_root is not None:
            base = self._tensorboard_root
        else:
            base = Path("./runs/jormungandr-structured")
        return str((base / model_id).resolve())

    @staticmethod
    def _resolve_device(raw: str) -> str:
        value = str(raw or "auto").strip().lower()
        if value in {"", "auto"}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if value.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return str(torch.device(value))

    @staticmethod
    def _parse_spec(raw: Mapping[str, Any]) -> StructuredPolicySpec:
        if not isinstance(raw, Mapping):
            raise ValueError("representation must be an object")
        return StructuredPolicySpec(
            global_dim=int(raw.get("global_dim", 0)),
            entity_dim=int(raw.get("entity_dim", 0)),
            candidate_dim=int(raw.get("candidate_dim", 0)),
            entity_type_count=int(raw.get("entity_type_count", 0)),
        )

    def _parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        model_id: str,
    ) -> tuple[StructuredServiceConfig, Any, bool]:
        if not isinstance(raw, Mapping):
            raise ValueError("learner must be an object")
        trainable = bool(raw.get("enabled", True))
        algo = canonical_algorithm_name(
            str(raw.get("algo", "structured_dqn"))
        )
        plugin = algorithm_registry.get(algo)
        if not plugin.supports_representation("entity_candidates"):
            raise ValueError(
                f"algorithm {algo!r} does not support entity_candidates"
            )
        if plugin.build_structured is None:
            raise ValueError(f"algorithm {algo!r} has no structured builder")
        if plugin.replay_mode not in {
            "transition",
            "trajectory",
            "supervision",
        }:
            raise ValueError(
                "structured service plugins must use transition, trajectory, "
                "or supervision mode"
            )
        batch_size = max(1, int(raw.get("batch_size", 256)))
        min_replay = max(batch_size, int(raw.get("min_replay", 2048)))
        plugin_config = raw.get("plugin_config", {})
        if not isinstance(plugin_config, Mapping):
            raise ValueError("learner.plugin_config must be an object")
        runtime_keys = {
            "enabled",
            "algo",
            "device",
            "batch_size",
            "min_replay",
            "updates_per_tick",
            "tick_interval_s",
            "beta0",
            "beta_steps",
            "replay_ratio",
            "supervision_sampling",
            "checkpoint_every",
            "checkpoint_dir",
            "metric_history_size",
            "min_trajectory_steps",
            "max_trajectory_batch_steps",
            "max_policy_lag",
            "max_updates",
            "seed",
            "plugin_config",
        }
        agent_config = {
            str(key): value
            for key, value in raw.items()
            if str(key) not in runtime_keys
        }
        agent_config.update({str(key): value for key, value in plugin_config.items()})
        checkpoint_dir_raw = str(raw.get("checkpoint_dir", "")).strip()
        if checkpoint_dir_raw:
            checkpoint_dir = Path(checkpoint_dir_raw).expanduser()
            if not checkpoint_dir.is_absolute() and self._checkpoint_root is not None:
                checkpoint_dir = self._checkpoint_root / checkpoint_dir
            checkpoint_dir = checkpoint_dir.resolve()
        elif self._checkpoint_root is not None:
            checkpoint_dir = self._checkpoint_root
        else:
            checkpoint_dir = Path("./checkpoints/jormungandr-structured").resolve()
        supervision_sampling = str(
            raw.get("supervision_sampling", "uniform")
        ).strip().lower()
        if supervision_sampling not in {"uniform", "sample_weight"}:
            raise ValueError(
                "supervision_sampling must be uniform or sample_weight"
            )
        config = StructuredServiceConfig(
            algo=algo,
            replay_mode=plugin.replay_mode,
            device=self._resolve_device(str(raw.get("device", "auto"))),
            batch_size=batch_size,
            min_replay=min_replay,
            updates_per_tick=max(1, int(raw.get("updates_per_tick", 1))),
            tick_interval_s=max(0.001, float(raw.get("tick_interval_s", 0.05))),
            beta0=min(1.0, max(0.0, float(raw.get("beta0", 0.4)))),
            beta_steps=max(1, int(raw.get("beta_steps", 100_000))),
            replay_ratio=max(0.0, float(raw.get("replay_ratio", 1.0))),
            supervision_sampling=supervision_sampling,
            checkpoint_every=max(0, int(raw.get("checkpoint_every", 5000))),
            checkpoint_dir=str(checkpoint_dir / model_id),
            metric_history_size=max(
                1, int(raw.get("metric_history_size", 2048))
            ),
            min_trajectory_steps=max(
                1, int(raw.get("min_trajectory_steps", 2048))
            ),
            max_trajectory_batch_steps=max(
                1, int(raw.get("max_trajectory_batch_steps", 8192))
            ),
            max_policy_lag=max(0, int(raw.get("max_policy_lag", 2))),
            max_updates=max(0, int(raw.get("max_updates", 0))),
            seed=(int(raw["seed"]) if raw.get("seed") is not None else None),
            agent_config=agent_config,
        )
        return config, plugin, trainable

    def create_model(
        self,
        *,
        representation: Mapping[str, Any],
        learner: Mapping[str, Any],
        model_id: str | None = None,
        capacity: int = 200_000,
        validation_capacity: int = 20_000,
        alpha: float = 0.6,
        metadata: Mapping[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
        policy_initialization_path: str | Path | None = None,
        tensorboard_enabled: bool = True,
        tensorboard_logdir: str = "",
    ) -> dict[str, Any]:
        if int(capacity) <= 0 or int(validation_capacity) <= 0:
            raise ValueError("replay capacities must be positive")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("replay alpha must be in [0, 1]")
        mid = str(model_id or uuid.uuid4().hex).strip()
        if not mid:
            raise ValueError("model_id cannot be empty")
        spec = self._parse_spec(representation)
        config, plugin, trainable = self._parse_config(learner, model_id=mid)
        # Model initialization is part of experiment provenance.  Isolate an
        # explicit seed from process-global PyTorch state so concurrently
        # hosted models do not silently determine one another's parameters.
        with self._model_build_lock:
            if config.seed is None:
                agent = plugin.build_structured(
                    spec, config.agent_config, config.device
                )
            else:
                cuda_devices: list[int] = []
                if config.device.startswith("cuda") and torch.cuda.is_available():
                    parsed_device = torch.device(config.device)
                    cuda_devices = [
                        parsed_device.index
                        if parsed_device.index is not None
                        else torch.cuda.current_device()
                    ]
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(config.seed)
                    if cuda_devices:
                        torch.cuda.manual_seed_all(config.seed)
                    agent = plugin.build_structured(
                        spec, config.agent_config, config.device
                    )
        if (
            checkpoint_path is not None
            and str(checkpoint_path).strip()
            and policy_initialization_path is not None
            and str(policy_initialization_path).strip()
        ):
            raise ValueError(
                "checkpoint_path and policy_initialization_path are mutually exclusive"
            )
        if config.replay_mode == "trajectory" and not all(
            hasattr(agent, method)
            for method in ("score_results_structured", "update_joint_structured")
        ):
            raise ValueError(
                f"trajectory algorithm {config.algo!r} lacks score/update methods"
            )
        if config.replay_mode == "supervision" and not all(
            hasattr(agent, method)
            for method in (
                "score_results_structured",
                "update_structured_supervision",
                "evaluate_structured_supervision",
            )
        ):
            raise ValueError(
                f"supervision algorithm {config.algo!r} lacks score/update methods"
            )
        source = ""
        restored_updates = 0
        restored_policy_version = 0
        restored_metadata: dict[str, Any] = {}
        initialization_source = ""
        if checkpoint_path is not None and str(checkpoint_path).strip():
            path = Path(checkpoint_path).expanduser().resolve()
            payload = torch.load(path, map_location=config.device, weights_only=False)
            if not isinstance(payload, Mapping):
                raise ValueError("structured checkpoint must contain an object")
            if payload.get("schema") != "jormungandr.structured_checkpoint.v1":
                raise ValueError("unsupported structured checkpoint schema")
            saved_representation = payload.get("representation")
            if not isinstance(saved_representation, Mapping):
                raise ValueError("structured checkpoint has no representation")
            if self._parse_spec(saved_representation) != spec:
                raise ValueError("structured checkpoint representation does not match")
            saved_plugin = payload.get("plugin")
            saved_plugin_name = (
                str(saved_plugin.get("name", ""))
                if isinstance(saved_plugin, Mapping)
                else ""
            )
            if canonical_algorithm_name(saved_plugin_name) != config.algo:
                raise ValueError("structured checkpoint algorithm does not match")
            saved_agent = payload.get("agent")
            if not isinstance(saved_agent, Mapping):
                raise ValueError("structured checkpoint has no agent state")
            agent.load_state_dict(saved_agent)
            source = str(path)
            restored_updates = int(payload.get("updates", 0))
            restored_policy_version = int(payload.get("policy_version", 0))
            saved_metadata = payload.get("metadata")
            if isinstance(saved_metadata, Mapping):
                restored_metadata = dict(saved_metadata)
        elif (
            policy_initialization_path is not None
            and str(policy_initialization_path).strip()
        ):
            if not hasattr(agent, "initialize_policy_from_state"):
                raise ValueError(
                    f"algorithm {config.algo!r} cannot initialize from another policy"
                )
            path = Path(policy_initialization_path).expanduser().resolve()
            payload = torch.load(
                path, map_location=config.device, weights_only=False
            )
            if not isinstance(payload, Mapping) or payload.get("schema") != (
                "jormungandr.structured_checkpoint.v1"
            ):
                raise ValueError("unsupported structured policy initialization")
            saved_representation = payload.get("representation")
            if not isinstance(saved_representation, Mapping) or (
                self._parse_spec(saved_representation) != spec
            ):
                raise ValueError(
                    "structured policy initialization representation does not match"
                )
            saved_agent = payload.get("agent")
            if not isinstance(saved_agent, Mapping):
                raise ValueError("structured policy initialization has no agent state")
            agent.initialize_policy_from_state(saved_agent)
            initialization_source = str(path)
        model_metadata = dict(restored_metadata)
        model_metadata.update(dict(metadata or {}))
        if source:
            model_metadata.update(
                {
                    "checkpoint_source": source,
                    "checkpoint_updates": restored_updates,
                    "checkpoint_policy_version": restored_policy_version,
                }
            )
        if initialization_source:
            model_metadata["policy_initialization_source"] = (
                initialization_source
            )
        tb_logdir = (
            self._tensorboard_logdir(mid, tensorboard_logdir)
            if tensorboard_enabled
            else ""
        )
        record = StructuredModelRecord(
            model_id=mid,
            spec=spec,
            capacity=int(capacity),
            validation_capacity=int(validation_capacity),
            alpha=float(alpha),
            metadata=model_metadata,
            replay=StructuredPrioritizedReplayBuffer(int(capacity), float(alpha)),
            validation=StructuredPrioritizedReplayBuffer(
                int(validation_capacity), 0.0
            ),
            trajectories=StructuredTrajectoryBuffer(int(capacity)),
            validation_trajectories=StructuredTrajectoryBuffer(
                int(validation_capacity)
            ),
            supervision=StructuredSupervisionBuffer(int(capacity)),
            validation_supervision=StructuredSupervisionBuffer(
                int(validation_capacity)
            ),
            config=config,
            plugin=plugin,
            agent=agent,
            tb_logger=TBLogger(
                enabled=bool(tensorboard_enabled), logdir=tb_logdir
            ),
            tensorboard_enabled=bool(tensorboard_enabled),
            tensorboard_logdir=tb_logdir,
            trainable=trainable,
            checkpoint_source=source,
            policy_initialization_source=initialization_source,
            restored_updates=restored_updates,
            restored_policy_version=restored_policy_version,
            policy_version=restored_policy_version,
            metric_history=deque(maxlen=config.metric_history_size),
            performance_history=deque(maxlen=config.metric_history_size),
        )
        with self._lock:
            if mid in self._records:
                raise ValueError(f"model_id already exists: {mid}")
            self._records[mid] = record
        if record.trainable:
            record.thread = threading.Thread(
                target=self._learner_loop,
                args=(record,),
                name=f"jormungandr-structured-{mid}",
                daemon=True,
            )
            record.thread.start()
        return self.get_model(mid)

    def _record(self, model_id: str) -> StructuredModelRecord:
        with self._lock:
            record = self._records.get(str(model_id))
        if record is None:
            raise KeyError(f"unknown structured model_id: {model_id}")
        return record

    @staticmethod
    def _split(raw: Any) -> str:
        value = str(raw or "train").strip().lower()
        if value == "val":
            value = "validation"
        if value not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        return value

    def experience_add(
        self,
        model_id: str,
        items: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not items:
            raise ValueError("items must be a non-empty array")
        record = self._record(model_id)
        if not record.trainable:
            raise ValueError("cannot add experience to a frozen structured model")
        if record.config.replay_mode != "transition":
            raise ValueError(
                "trajectory-mode models require joint trajectory ingress"
            )
        parsed: list[tuple[StructuredReplayTransition, float | None]] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("each experience item must be an object")
            missing = [
                key
                for key in (
                    "actor_id",
                    "episode_id",
                    "timestep",
                    "policy_version",
                    "observation",
                    "candidate_id",
                    "next_observation",
                )
                if item.get(key) is None
            ]
            if missing:
                raise ValueError(
                    "structured experience is missing required fields: "
                    + ", ".join(missing)
                )
            metadata = item.get("meta", {})
            if not isinstance(metadata, Mapping):
                raise ValueError("experience meta must be an object")
            transition = StructuredReplayTransition(
                observation=entity_candidate_observation_from_payload(
                    item["observation"], spec=record.spec
                ),
                candidate_id=str(item["candidate_id"]),
                reward=float(item.get("reward", 0.0)),
                next_observation=entity_candidate_observation_from_payload(
                    item["next_observation"], spec=record.spec
                ),
                done=bool(item.get("done", False)),
                actor_id=str(item["actor_id"]),
                episode_id=str(item["episode_id"]),
                timestep=int(item["timestep"]),
                policy_version=int(item["policy_version"]),
                split=self._split(item.get("split", "train")),
                metadata=dict(metadata),
            )
            priority_raw = item.get("priority")
            priority = None if priority_raw is None else float(priority_raw)
            parsed.append((transition, priority))

        added = {"train": 0, "validation": 0}
        with record.lock:
            for transition, priority in parsed:
                store = (
                    record.replay
                    if transition.split == "train"
                    else record.validation
                )
                store.add(transition, priority=priority)
                added[transition.split] += 1
            record.replay_add_calls += 1
            record.replay_add_items += len(parsed)
            record.train_add_items += added["train"]
            record.validation_add_items += added["validation"]
        return {
            "model_id": model_id,
            "added": len(parsed),
            "added_by_split": added,
            "replay": {
                "size": len(record.replay),
                "capacity": record.capacity,
                "alpha": record.alpha,
            },
            "validation": {
                "size": len(record.validation),
                "capacity": record.validation_capacity,
            },
        }

    def trajectory_add(
        self,
        model_id: str,
        trajectories: Sequence[Sequence[Mapping[str, Any]]],
    ) -> dict[str, Any]:
        """Atomically validate and enqueue complete joint trajectories."""

        if not trajectories:
            raise ValueError("trajectories must be a non-empty array")
        record = self._record(model_id)
        if not record.trainable:
            raise ValueError("cannot add trajectories to a frozen structured model")
        if record.config.replay_mode != "trajectory":
            raise ValueError("transition-mode models require experience ingress")
        parsed = []
        for raw_trajectory in trajectories:
            if not isinstance(raw_trajectory, Sequence) or isinstance(
                raw_trajectory, (str, bytes)
            ):
                raise ValueError("each trajectory must be an array of steps")
            steps = tuple(
                structured_joint_step_from_payload(item, spec=record.spec)
                for item in raw_trajectory
            )
            parsed.append(validate_structured_joint_trajectory(steps))

        return self._enqueue_trajectories(record, parsed)

    def trajectory_sequence_add(
        self,
        model_id: str,
        sequences: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Decode compact observation-chain sequences and enqueue them."""

        if not sequences:
            raise ValueError("trajectory sequences must be a non-empty array")
        record = self._record(model_id)
        if not record.trainable:
            raise ValueError("cannot add trajectories to a frozen structured model")
        if record.config.replay_mode != "trajectory":
            raise ValueError("transition-mode models require experience ingress")
        parsed = [
            structured_joint_trajectory_from_sequence_payload(
                raw_sequence, spec=record.spec
            )
            for raw_sequence in sequences
        ]
        return self._enqueue_trajectories(record, parsed)

    def _enqueue_trajectories(
        self,
        record: StructuredModelRecord,
        parsed: Sequence[Sequence[Any]],
    ) -> dict[str, Any]:
        """Apply common duplicate, policy-lag, and split checks atomically."""

        incoming_keys = [
            (steps[0].actor_id, steps[0].episode_id, steps[0].split)
            for steps in parsed
        ]
        if len(set(incoming_keys)) != len(incoming_keys):
            raise ValueError("duplicate actor/episode/split in trajectory request")
        added_by_split = {"train": 0, "validation": 0}
        steps_by_split = {"train": 0, "validation": 0}
        with record.lock:
            duplicated = record.seen_trajectory_keys.intersection(incoming_keys)
            if duplicated:
                raise ValueError(
                    "duplicate actor/episode/split trajectory was already ingested"
                )
            current_version = int(record.policy_version)
            lags = []
            actor_latencies = []
            for steps in parsed:
                if steps[0].split == "train":
                    for step in steps:
                        lag = current_version - int(step.policy_version)
                        if lag < 0:
                            raise ValueError(
                                "trajectory policy version is newer than the service"
                            )
                        if lag > record.config.max_policy_lag:
                            raise ValueError(
                                "stale trajectory exceeds max_policy_lag: "
                                f"lag={lag}, maximum={record.config.max_policy_lag}"
                            )
                        lags.append(lag)
                for step in steps:
                    latency = step.metadata.get("actor_latency_ms")
                    try:
                        latency_value = float(latency)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(latency_value) and latency_value >= 0.0:
                        actor_latencies.append(latency_value)

            for steps in parsed:
                split = steps[0].split
                store = (
                    record.trajectories
                    if split == "train"
                    else record.validation_trajectories
                )
                store.add(steps)
                record.seen_trajectory_keys.add(
                    (steps[0].actor_id, steps[0].episode_id, split)
                )
                added_by_split[split] += 1
                steps_by_split[split] += len(steps)
            record.trajectory_add_calls += 1
            record.trajectory_add_episodes += len(parsed)
            record.trajectory_add_steps += sum(len(steps) for steps in parsed)
            record.train_add_items += steps_by_split["train"]
            record.validation_add_items += steps_by_split["validation"]
            record.policy_lag_count += len(lags)
            record.policy_lag_sum += float(sum(lags))
            record.policy_lag_max = max(
                record.policy_lag_max, max(lags, default=0)
            )
            record.actor_latency_count += len(actor_latencies)
            record.actor_latency_sum_ms += float(sum(actor_latencies))
            train_steps = record.trajectories.step_count
            validation_steps = record.validation_trajectories.step_count
        return {
            "model_id": record.model_id,
            "added_trajectories": len(parsed),
            "added_steps": sum(len(steps) for steps in parsed),
            "added_by_split": added_by_split,
            "steps_by_split": steps_by_split,
            "trajectory_store": {
                "train_steps": train_steps,
                "validation_steps": validation_steps,
            },
        }

    def score(
        self,
        model_id: str,
        observations: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return candidate-aligned logits for actor-owned joint sampling."""

        if not observations:
            raise ValueError("at least one observation is required")
        record = self._record(model_id)
        decoded = tuple(
            entity_candidate_observation_from_payload(item, spec=record.spec)
            for item in observations
        )
        started = time.perf_counter()
        with record.agent_lock:
            if not hasattr(record.agent, "score_results_structured"):
                raise ValueError(
                    f"algorithm {record.config.algo!r} does not expose policy scores"
                )
            results = record.agent.score_results_structured(decoded)
            with record.lock:
                policy_version = record.policy_version
                updates = record.updates
                record.score_calls += 1
        items = []
        for result in results:
            logits = [
                float(value) if np.isfinite(float(value)) else -1e30
                for value in result.candidate_logits
            ]
            items.append(
                {
                    "candidate_ids": list(result.candidate_ids),
                    "candidate_logits": logits,
                    "behavior_value": float(result.value),
                    **(
                        {
                            "candidate_prefix_keys": [
                                list(vector)
                                for vector in result.candidate_prefix_keys
                            ],
                            "candidate_prefix_values": [
                                list(vector)
                                for vector in result.candidate_prefix_values
                            ],
                            "preference_conditioning": {
                                "mode": "low_rank_additive_v1",
                                "dim": len(result.candidate_prefix_keys[0]),
                            },
                        }
                        if result.candidate_prefix_keys
                        else {
                            "preference_conditioning": {
                                "mode": "prefix_independent",
                                "dim": 0,
                            }
                        }
                    ),
                }
            )
        return {
            "schema": "jormungandr.structured_policy_scores.v1",
            "model_id": model_id,
            "algo": record.config.algo,
            "policy_version": int(policy_version),
            "updates": int(updates),
            "batch_size": len(items),
            "items": items,
            "server_latency_ms": (time.perf_counter() - started) * 1000.0,
            "ts": time.time(),
        }

    def supervision_add(
        self,
        model_id: str,
        items: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Add reward-free structured labels with strict split isolation."""

        if not items:
            raise ValueError("supervision items must be a non-empty array")
        record = self._record(model_id)
        if not record.trainable:
            raise ValueError("cannot add supervision to a frozen structured model")
        if record.config.replay_mode != "supervision":
            raise ValueError("model does not use structured supervision ingress")
        parsed = tuple(
            structured_supervision_from_payload(item, spec=record.spec)
            for item in items
        )
        keys = [
            (
                item.actor_id,
                item.episode_id,
                item.timestep,
                item.factor_id,
                item.split,
            )
            for item in parsed
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate structured supervision in request")
        added = {"train": 0, "validation": 0}
        with record.lock:
            existing = {
                (
                    item.actor_id,
                    item.episode_id,
                    item.timestep,
                    item.factor_id,
                    item.split,
                )
                for store in (
                    record.supervision,
                    record.validation_supervision,
                )
                for item in store.snapshot()
            }
            if existing.intersection(keys):
                raise ValueError("duplicate structured supervision example")
            for item in parsed:
                store = (
                    record.supervision
                    if item.split == "train"
                    else record.validation_supervision
                )
                store.add(item)
                added[item.split] += 1
            record.supervision_add_calls += 1
            record.supervision_add_items += len(parsed)
            record.train_add_items += added["train"]
            record.validation_add_items += added["validation"]
            train_size = len(record.supervision)
            validation_size = len(record.validation_supervision)
        return {
            "model_id": model_id,
            "added": len(parsed),
            "added_by_split": added,
            "supervision": {
                "train_size": train_size,
                "validation_size": validation_size,
            },
        }

    def infer(
        self,
        model_id: str,
        observations: Sequence[Mapping[str, Any]],
        *,
        deterministic: bool,
        epsilon: float,
    ) -> dict[str, Any]:
        if not observations:
            raise ValueError("at least one observation is required")
        record = self._record(model_id)
        if record.config.replay_mode != "transition":
            raise ValueError(
                "trajectory-mode joint policies require policy/score inference"
            )
        decoded = tuple(
            entity_candidate_observation_from_payload(item, spec=record.spec)
            for item in observations
        )
        with record.agent_lock:
            results = record.agent.action_results_structured(
                decoded,
                deterministic=bool(deterministic),
                epsilon=max(0.0, float(epsilon)),
            )
            with record.lock:
                policy_version = record.policy_version
                updates = record.updates
                record.inference_calls += 1
        items = []
        for result in results:
            items.append(
                {
                    "candidate_id": result.candidate_id,
                    "candidate_index": result.candidate_index,
                    "behavior_logp": result.log_probability,
                    "q_value": result.value,
                    "candidate_values": list(result.candidate_values),
                }
            )
        return {
            "model_id": model_id,
            "algo": record.config.algo,
            "policy_version": policy_version,
            "updates": updates,
            "batch_size": len(items),
            "items": items,
            "ts": time.time(),
        }

    def _learner_loop(self, record: StructuredModelRecord) -> None:
        config = record.config
        if config.replay_mode == "supervision":
            self._supervision_learner_loop(record)
            return
        if config.replay_mode == "trajectory":
            self._trajectory_learner_loop(record)
            return
        while not record.stop_event.wait(config.tick_interval_s):
            try:
                for _ in range(config.updates_per_tick):
                    with record.lock:
                        if config.max_updates and record.updates >= config.max_updates:
                            break
                        replay_size = len(record.replay)
                        if replay_size < config.min_replay:
                            break
                        sample_budget = int(
                            record.train_add_items * config.replay_ratio
                        )
                        if (
                            (record.updates + 1) * config.batch_size
                            > sample_budget
                        ):
                            break
                        beta = min(
                            1.0,
                            config.beta0
                            + record.updates / max(1, config.beta_steps),
                        )
                        transitions, indices, weights = record.replay.sample(
                            config.batch_size, beta
                        )
                    with record.agent_lock:
                        result = record.agent.update_structured(
                            transitions, weights
                        )
                        with record.lock:
                            record.replay.update_priorities(
                                indices, result.priorities
                            )
                            record.updates += 1
                            record.policy_version += 1
                            record.last_loss = float(result.loss)
                            record.last_td_abs = float(
                                np.mean(np.abs(result.priorities))
                            )
                            record.last_update_ts = time.time()
                            record.last_metrics = {
                                str(key): float(value)
                                for key, value in result.metrics.items()
                                if np.isfinite(float(value))
                            }
                            record.last_error = ""
                            metric_point = {
                                "update": int(record.updates),
                                "policy_version": int(record.policy_version),
                                "ts": float(record.last_update_ts),
                                "loss": float(record.last_loss),
                                "td_abs_mean": float(record.last_td_abs),
                                "replay_size": int(replay_size),
                                "train_items": int(record.train_add_items),
                                "validation_items": int(
                                    record.validation_add_items
                                ),
                                "metrics": dict(record.last_metrics),
                            }
                            record.metric_history.append(metric_point)
                            record.tb_logger.add(
                                "learner/loss", record.last_loss, record.updates
                            )
                            record.tb_logger.add(
                                "learner/td_abs_mean",
                                record.last_td_abs,
                                record.updates,
                            )
                            record.tb_logger.add(
                                "learner/replay_size",
                                float(replay_size),
                                record.updates,
                            )
                            for metric_name, metric_value in (
                                record.last_metrics.items()
                            ):
                                record.tb_logger.add(
                                    f"algorithms/{config.algo}/{metric_name}",
                                    metric_value,
                                    record.updates,
                                )
                            if record.updates % 50 == 0:
                                record.tb_logger.flush()
                            checkpoint_due = (
                                config.checkpoint_every > 0
                                and record.updates % config.checkpoint_every == 0
                            )
                    if checkpoint_due:
                        self.checkpoint(record.model_id)
            except Exception as exc:
                with record.lock:
                    record.last_error = f"{type(exc).__name__}: {exc}"
                record.stop_event.wait(min(1.0, config.tick_interval_s * 10.0))

    def _supervision_learner_loop(self, record: StructuredModelRecord) -> None:
        """Train reward-free labels while never updating from validation."""

        config = record.config
        while not record.stop_event.wait(config.tick_interval_s):
            try:
                for _ in range(config.updates_per_tick):
                    with record.lock:
                        if config.max_updates and record.updates >= config.max_updates:
                            break
                        train_size = len(record.supervision)
                        if train_size < config.min_replay:
                            break
                        sample_budget = int(
                            record.train_add_items * config.replay_ratio
                        )
                        if (
                            (record.updates + 1) * config.batch_size
                            > sample_budget
                        ):
                            break
                        examples = record.supervision.sample(
                            config.batch_size,
                            rng=np.random.default_rng(
                                43_000_003 + record.updates
                            ),
                            strategy=config.supervision_sampling,
                        )
                        validation = record.validation_supervision.snapshot()
                    with record.agent_lock:
                        result = record.agent.update_structured_supervision(
                            examples
                        )
                        validation_result = (
                            record.agent.evaluate_structured_supervision(
                                validation
                            )
                            if validation
                            else None
                        )
                        with record.lock:
                            record.updates += 1
                            record.policy_version += 1
                            record.supervision_train_items += len(examples)
                            record.last_loss = float(result.loss)
                            record.last_td_abs = 0.0
                            record.last_update_ts = time.time()
                            record.last_metrics = {
                                str(key): float(value)
                                for key, value in result.metrics.items()
                                if np.isfinite(float(value))
                            }
                            if validation_result is not None:
                                record.last_metrics.update(
                                    {
                                        f"validation/{key}": float(value)
                                        for key, value in (
                                            validation_result.metrics.items()
                                        )
                                        if np.isfinite(float(value))
                                    }
                                )
                            record.last_error = ""
                            metric_point = {
                                "update": int(record.updates),
                                "policy_version": int(record.policy_version),
                                "ts": float(record.last_update_ts),
                                "loss": float(record.last_loss),
                                "supervision_batch": len(examples),
                                "train_items": int(record.train_add_items),
                                "validation_items": int(
                                    record.validation_add_items
                                ),
                                "metrics": dict(record.last_metrics),
                            }
                            record.metric_history.append(metric_point)
                            record.tb_logger.add(
                                "learner/loss", record.last_loss, record.updates
                            )
                            for metric_name, metric_value in (
                                record.last_metrics.items()
                            ):
                                record.tb_logger.add(
                                    f"algorithms/{config.algo}/{metric_name}",
                                    metric_value,
                                    record.updates,
                                )
                            record.tb_logger.flush()
                            checkpoint_due = (
                                config.checkpoint_every > 0
                                and record.updates % config.checkpoint_every == 0
                            )
                    if checkpoint_due:
                        self.checkpoint(record.model_id)
            except Exception as exc:
                with record.lock:
                    record.last_error = f"{type(exc).__name__}: {exc}"
                record.stop_event.wait(min(1.0, config.tick_interval_s * 10.0))

    def _trajectory_learner_loop(self, record: StructuredModelRecord) -> None:
        """Consume whole on-policy episodes without transition replay."""

        config = record.config
        while not record.stop_event.wait(config.tick_interval_s):
            try:
                for _ in range(config.updates_per_tick):
                    with record.lock:
                        if config.max_updates and record.updates >= config.max_updates:
                            break
                        queued_steps = record.trajectories.step_count
                        if queued_steps < config.min_trajectory_steps:
                            break
                        trajectories = record.trajectories.pop_at_least(
                            config.min_trajectory_steps,
                            maximum_steps=config.max_trajectory_batch_steps,
                        )
                    if not trajectories:
                        break
                    transition_count = sum(len(item) for item in trajectories)
                    with record.agent_lock:
                        result = record.agent.update_joint_structured(
                            trajectories
                        )
                        with record.lock:
                            record.updates += 1
                            record.policy_version += 1
                            record.trajectory_train_episodes += len(trajectories)
                            record.trajectory_train_steps += transition_count
                            record.last_loss = float(result.loss)
                            record.last_td_abs = 0.0
                            record.last_update_ts = time.time()
                            result_metrics = asdict(result)
                            record.last_metrics = {
                                str(key): float(value)
                                for key, value in result_metrics.items()
                                if key
                                not in {
                                    "transitions",
                                    "episodes",
                                    "epochs",
                                    "minibatches",
                                }
                                and np.isfinite(float(value))
                            }
                            record.last_metrics.update(
                                {
                                    "trajectory_steps": float(transition_count),
                                    "trajectory_episodes": float(len(trajectories)),
                                    "trajectory_queue_steps": float(
                                        record.trajectories.step_count
                                    ),
                                    "policy_lag_mean": (
                                        record.policy_lag_sum
                                        / max(1, record.policy_lag_count)
                                    ),
                                    "policy_lag_max": float(
                                        record.policy_lag_max
                                    ),
                                    "actor_latency_ms_mean": (
                                        record.actor_latency_sum_ms
                                        / max(1, record.actor_latency_count)
                                    ),
                                }
                            )
                            record.last_error = ""
                            metric_point = {
                                "update": int(record.updates),
                                "policy_version": int(record.policy_version),
                                "ts": float(record.last_update_ts),
                                "loss": float(record.last_loss),
                                "trajectory_steps": int(transition_count),
                                "trajectory_episodes": len(trajectories),
                                "trajectory_queue_steps": int(
                                    record.trajectories.step_count
                                ),
                                "train_items": int(record.train_add_items),
                                "validation_items": int(
                                    record.validation_add_items
                                ),
                                "metrics": dict(record.last_metrics),
                            }
                            record.metric_history.append(metric_point)
                            record.tb_logger.add(
                                "learner/loss", record.last_loss, record.updates
                            )
                            record.tb_logger.add(
                                "learner/trajectory_queue_steps",
                                float(record.trajectories.step_count),
                                record.updates,
                            )
                            for metric_name, metric_value in (
                                record.last_metrics.items()
                            ):
                                record.tb_logger.add(
                                    f"algorithms/{config.algo}/{metric_name}",
                                    metric_value,
                                    record.updates,
                                )
                            record.tb_logger.flush()
                            checkpoint_due = (
                                config.checkpoint_every > 0
                                and record.updates % config.checkpoint_every == 0
                            )
                    if checkpoint_due:
                        self.checkpoint(record.model_id)
            except Exception as exc:
                with record.lock:
                    record.last_error = f"{type(exc).__name__}: {exc}"
                record.stop_event.wait(min(1.0, config.tick_interval_s * 10.0))

    def checkpoint(self, model_id: str) -> dict[str, Any]:
        record = self._record(model_id)
        with record.agent_lock:
            with record.lock:
                updates = record.updates
                policy_version = record.policy_version
                payload = {
                    "schema": "jormungandr.structured_checkpoint.v1",
                    "model_id": record.model_id,
                    "representation": asdict(record.spec),
                    "plugin": {
                        "name": record.plugin.name,
                        "version": record.plugin.version,
                    },
                    "config": asdict(record.config),
                    "updates": updates,
                    "policy_version": policy_version,
                    "metadata": dict(record.metadata),
                    "agent": record.agent.state_dict(),
                }
            path = (
                Path(record.config.checkpoint_dir)
                / f"ckpt_u{updates:09d}_v{policy_version:09d}.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            with record.lock:
                record.last_checkpoint = str(path.resolve())
                record.checkpoint_count += 1
        return {
            "model_id": model_id,
            "checkpoint": str(path.resolve()),
            "updates": updates,
            "policy_version": policy_version,
        }

    def log_metrics(
        self,
        model_id: str,
        *,
        step: int,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Attach finite experiment/evaluation scalars to a structured model."""

        record = self._record(model_id)
        resolved: dict[str, float] = {}
        for raw_name, raw_value in metrics.items():
            name = str(raw_name).strip()
            if not name:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                resolved[name] = value
        now = time.time()
        with record.lock:
            record.metrics_calls += 1
            record.metrics_points += len(resolved)
            record.last_metrics_step = int(step)
            record.last_metrics_ts = now
            record.latest_performance_metrics.update(resolved)
            if resolved:
                record.performance_history.append(
                    {
                        "step": int(step),
                        "ts": float(now),
                        "metrics": dict(resolved),
                    }
                )
                for name, value in resolved.items():
                    record.tb_logger.add(name, value, int(step))
                record.tb_logger.flush()
        return {
            "model_id": model_id,
            "step": int(step),
            "logged": len(resolved),
        }

    def get_metrics(self, model_id: str) -> dict[str, Any]:
        record = self._record(model_id)
        with record.lock:
            return {
                "schema": "jormungandr.structured_metrics.v1",
                "model_id": record.model_id,
                "algo": record.config.algo,
                "updates": int(record.updates),
                "policy_version": int(record.policy_version),
                "history": list(record.metric_history),
                "performance_history": list(record.performance_history),
                "latest_performance_metrics": dict(
                    record.latest_performance_metrics
                ),
            }

    def get_model(self, model_id: str) -> dict[str, Any]:
        record = self._record(model_id)
        with record.lock:
            trainable_module = getattr(
                record.agent,
                "policy",
                getattr(record.agent, "q", None),
            )
            trainable_parameters = (
                sum(
                    parameter.numel()
                    for parameter in trainable_module.parameters()
                    if parameter.requires_grad
                )
                if trainable_module is not None
                else 0
            )
            return {
                "model_id": record.model_id,
                "representation": {
                    "mode": "entity_candidates",
                    **asdict(record.spec),
                },
                "algorithm": {
                    "name": record.plugin.name,
                    "version": record.plugin.version,
                    "family": record.plugin.family,
                    "replay_mode": record.config.replay_mode,
                },
                "device": str(record.agent.device),
                "trainable_parameters": int(trainable_parameters),
                "policy_conditioning": {
                    "mode": (
                        "low_rank_additive_v1"
                        if int(getattr(trainable_module, "prefix_dim", 0)) > 0
                        else "prefix_independent"
                    ),
                    "prefix_dim": int(
                        getattr(trainable_module, "prefix_dim", 0)
                    ),
                },
                "tensorboard": {
                    "enabled": record.tensorboard_enabled,
                    "logdir": record.tensorboard_logdir,
                },
                "trainable": record.trainable,
                "checkpoint_source": record.checkpoint_source,
                "policy_initialization_source": (
                    record.policy_initialization_source
                ),
                "restored_updates": record.restored_updates,
                "restored_policy_version": record.restored_policy_version,
                "replay": {
                    "size": len(record.replay),
                    "capacity": record.capacity,
                    "alpha": record.alpha,
                },
                "validation": {
                    "size": (
                        len(record.validation)
                        if record.config.replay_mode == "transition"
                        else (
                            record.validation_trajectories.step_count
                            if record.config.replay_mode == "trajectory"
                            else len(record.validation_supervision)
                        )
                    ),
                    "capacity": record.validation_capacity,
                },
                "trajectories": {
                    "train_episodes_queued": len(record.trajectories),
                    "train_steps_queued": record.trajectories.step_count,
                    "validation_episodes": len(
                        record.validation_trajectories
                    ),
                    "validation_steps": (
                        record.validation_trajectories.step_count
                    ),
                    "capacity_steps": record.capacity,
                    "validation_capacity_steps": record.validation_capacity,
                    "min_update_steps": (
                        record.config.min_trajectory_steps
                    ),
                    "max_policy_lag": record.config.max_policy_lag,
                    "ingress_calls": record.trajectory_add_calls,
                    "ingested_episodes": record.trajectory_add_episodes,
                    "ingested_steps": record.trajectory_add_steps,
                    "trained_episodes": record.trajectory_train_episodes,
                    "trained_steps": record.trajectory_train_steps,
                },
                "supervision": {
                    "train_size": len(record.supervision),
                    "validation_size": len(record.validation_supervision),
                    "capacity": record.capacity,
                    "validation_capacity": record.validation_capacity,
                    "ingress_calls": record.supervision_add_calls,
                    "ingested_items": record.supervision_add_items,
                    "trained_items": record.supervision_train_items,
                    "sampling": record.config.supervision_sampling,
                },
                "updates": record.updates,
                "policy_version": record.policy_version,
                "inference_calls": record.inference_calls,
                "experience": {
                    "calls": record.replay_add_calls,
                    "items": record.replay_add_items,
                    "train_items": record.train_add_items,
                    "validation_items": record.validation_add_items,
                    "replay_ratio": record.config.replay_ratio,
                },
                "policy_lag": {
                    "count": record.policy_lag_count,
                    "mean": (
                        record.policy_lag_sum
                        / max(1, record.policy_lag_count)
                    ),
                    "max": record.policy_lag_max,
                },
                "actor_latency_ms": {
                    "count": record.actor_latency_count,
                    "mean": (
                        record.actor_latency_sum_ms
                        / max(1, record.actor_latency_count)
                    ),
                },
                "score_calls": record.score_calls,
                "last_loss": record.last_loss,
                "last_td_abs": record.last_td_abs,
                "last_update_ts": record.last_update_ts,
                "last_metrics": dict(record.last_metrics),
                "metrics": {
                    "calls": record.metrics_calls,
                    "points": record.metrics_points,
                    "history_size": len(record.metric_history),
                    "performance_history_size": len(
                        record.performance_history
                    ),
                    "last_step": record.last_metrics_step,
                    "last_ts": record.last_metrics_ts,
                    "latest_performance": dict(
                        record.latest_performance_metrics
                    ),
                },
                "last_error": record.last_error,
                "last_checkpoint": record.last_checkpoint,
                "checkpoint_count": record.checkpoint_count,
                "metadata": dict(record.metadata),
                "created_ts": record.created_ts,
            }

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            identifiers = sorted(self._records)
        return [self.get_model(model_id) for model_id in identifiers]

    def delete_model(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.pop(str(model_id), None)
        if record is None:
            raise KeyError(f"unknown structured model_id: {model_id}")
        record.stop_event.set()
        if record.thread is not None:
            record.thread.join(timeout=5.0)
        record.tb_logger.flush()
        record.tb_logger.close()
        return {"model_id": model_id, "deleted": True}

    def close_all(self) -> None:
        with self._lock:
            identifiers = list(self._records)
        for model_id in identifiers:
            try:
                self.delete_model(model_id)
            except Exception:
                pass
