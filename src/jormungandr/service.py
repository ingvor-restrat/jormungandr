from __future__ import annotations

import argparse
import errno
import json
import shlex
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, TypeVar
from urllib.parse import urlparse

import numpy as np

from jormungandr.algorithms import (
    AlgorithmPlugin,
    algorithm_registry,
    canonical_algorithm_name,
    normalize_update_result,
)
from jormungandr.core import PrioritizedReplayBuffer, RunningNormalizer, TBLogger
from jormungandr.episode import CtypesEpisodeFactory, SubprocessEpisodeFactory
from jormungandr.interfaces import EpisodeClient, EpisodeFactory
from jormungandr.selectors import (
    ReplaySelector,
    build_replay_selector,
    select_trajectory_replay,
)

JsonDict = Dict[str, Any]
T = TypeVar("T")


def _is_client_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in {
            errno.EPIPE,
            errno.ECONNRESET,
            errno.ECONNABORTED,
            errno.ETIMEDOUT,
        }
    return False


class JormungandrServiceError(RuntimeError):
    pass


@dataclass
class SessionRecord:
    session_id: str
    model_id: str
    created_ts: float
    driver: str
    driver_config: JsonDict
    episode_config: JsonDict
    metadata: JsonDict
    episode: EpisodeClient
    initialize_count: int = 0
    step_count: int = 0
    agent_cmd_count: int = 0
    last_step_ts: float = 0.0
    last_step_latency_ms: float = 0.0
    step_latency_count: int = 0
    step_latency_sum_ms: float = 0.0
    last_step_terminal: bool = False
    last_remaining_steps: Optional[int] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class LearnerConfig:
    enabled: bool = False
    algo: str = "c51"
    device: str = "auto"
    hidden: int = 256
    aux_hidden: int = 0
    lr: float = 1e-4
    gamma: float = 0.99
    v_min: float = -10.0
    v_max: float = 10.0
    atoms: int = 51
    quantiles: int = 51
    quantile_risk_measure: str = "mean"
    quantile_risk_level: float = 0.1
    target_update: int = 1000
    max_grad: float = 1.0
    batch_size: int = 256
    beta0: float = 0.4
    beta_steps: int = 100_000
    min_replay: int = 2048
    updates_per_tick: int = 1
    tick_interval_s: float = 0.05
    checkpoint_every: int = 5000
    checkpoint_dir: str = "./checkpoints/jormungandr"
    validation_every: int = 100
    validation_batch_size: int = 512
    min_validation: int = 64
    action_values: List[float] = field(default_factory=lambda: [-1.0, 0.0, 1.0])
    aux_enabled: bool = False
    aux_weight: float = 0.0
    aux_classes: int = 3
    aux_kind: str = ""
    aux_label_key: str = "label"
    aux_class_weighting: str = "none"
    aux_label_smoothing: float = 0.0
    # Shared numerical-robustness controls.  They are opt-in when they alter
    # task semantics (noise augmentation or reward clipping).
    huber_delta: float = 1.0
    reward_clip: float = 0.0
    observation_noise_std: float = 0.0
    # Actor-critic and entropy-regularized objectives.
    temperature: float = 0.2
    auto_entropy: bool = True
    target_entropy: Optional[float] = None
    tau: float = 0.005
    cql_alpha: float = 1.0
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatch_size: int = 256
    vtrace_rho_clip: float = 1.0
    vtrace_pg_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    appo_target_worker_clip: float = 2.0
    rollout_length: int = 32
    max_policy_lag: int = 64
    # Offline imitation.
    marwil_beta: float = 1.0
    marwil_c2_rate: float = 1e-8
    marwil_c2_start: float = 100.0
    max_advantage_weight: float = 20.0
    # Compact vector-observation DreamerV3 profile.
    dreamer_latent: int = 128
    imagination_horizon: int = 15
    dreamer_lambda: float = 0.95
    # Replay/rollout selection is independently replaceable from the learner.
    replay_selector: str = "prioritized"
    qubo_pool_factor: float = 4.0
    qubo_utility_weight: float = 1.0
    qubo_diversity_weight: float = 0.35
    qubo_cardinality_penalty: float = 4.0
    qubo_local_search_passes: int = 8
    qubo_reward_utility_weight: float = 0.0
    # External plugins receive arbitrary settings here without service edits.
    plugin_config: JsonDict = field(default_factory=dict)


@dataclass
class LearnerState:
    cfg: LearnerConfig
    agent: Optional[Any] = None
    plugin: Optional[AlgorithmPlugin] = None
    replay_selector: Optional[ReplaySelector] = None
    agent_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    updates: int = 0
    policy_version: int = 0
    last_loss: float = 0.0
    last_td_abs: float = 0.0
    last_beta: float = 0.0
    last_train_ts: float = 0.0
    last_checkpoint: str = ""
    checkpoint_count: int = 0
    last_aux_loss: float = 0.0
    last_aux_acc: float = 0.0
    last_aux_count: int = 0
    validation_runs: int = 0
    validation_policy_version: int = 0
    last_validation_loss: float = 0.0
    last_validation_td_abs: float = 0.0
    last_validation_aux_loss: float = 0.0
    last_validation_aux_acc: float = 0.0
    last_validation_count: int = 0
    last_validation_ts: float = 0.0
    last_error: str = ""
    last_weight_summary: JsonDict = field(default_factory=dict)
    last_metrics: JsonDict = field(default_factory=dict)
    last_selector_metrics: JsonDict = field(default_factory=dict)
    last_selector_audit: JsonDict = field(default_factory=dict)
    metric_history: deque[JsonDict] = field(default_factory=lambda: deque(maxlen=256))


@dataclass
class ModelRecord:
    model_id: str
    obs_dim: int
    capacity: int
    validation_capacity: int
    alpha: float
    created_ts: float
    metadata: JsonDict
    replay: PrioritizedReplayBuffer
    validation: PrioritizedReplayBuffer
    obs_normalizer: RunningNormalizer
    tb_logger: TBLogger
    learner: Optional[LearnerState]
    tensorboard_enabled: bool
    tensorboard_logdir: str
    inference_step: int
    inference_ts: float
    inference_payload: JsonDict
    replay_add_calls: int = 0
    replay_add_items: int = 0
    train_add_items: int = 0
    validation_add_items: int = 0
    replay_sample_calls: int = 0
    replay_priority_update_calls: int = 0
    metrics_calls: int = 0
    metrics_points: int = 0
    sampler_train_episodes: int = 0
    sampler_val_episodes: int = 0
    last_metrics_step: int = 0
    last_metrics_ts: float = 0.0
    latest_logged_metrics: JsonDict = field(default_factory=dict)
    logged_metric_history: deque[JsonDict] = field(
        default_factory=lambda: deque(maxlen=256)
    )
    inference_updates: int = 0
    inference_reads: int = 0
    policy_infer_calls: int = 0
    aux_infer_calls: int = 0
    replay_meta_by_idx: Dict[int, JsonDict] = field(default_factory=dict)
    replay_key_by_idx: Dict[int, str] = field(default_factory=dict)
    replay_idx_by_key: Dict[str, int] = field(default_factory=dict)
    replay_aux_by_idx: Dict[int, JsonDict] = field(default_factory=dict)
    replay_aux_pending_by_key: Dict[str, JsonDict] = field(default_factory=dict)
    validation_meta_by_idx: Dict[int, JsonDict] = field(default_factory=dict)
    validation_key_by_idx: Dict[int, str] = field(default_factory=dict)
    validation_idx_by_key: Dict[str, int] = field(default_factory=dict)
    validation_aux_by_idx: Dict[int, JsonDict] = field(default_factory=dict)
    aux_update_calls: int = 0
    aux_update_items: int = 0
    aux_update_matched: int = 0
    aux_update_pending: int = 0
    session_ids: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


class JormungandrRuntime:
    """Thread-safe Jörmungandr runtime for multi-model training over HTTP."""

    _OBS_NORMALIZER_WARMUP = 32

    def __init__(
        self,
        *,
        tensorboard_root: str = "",
        checkpoint_root: str = "",
    ) -> None:
        self._models: Dict[str, ModelRecord] = {}
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = threading.Lock()
        # Serialize ctypes-backed episode calls to avoid native reentrancy issues.
        self._ctypes_call_lock = threading.Lock()
        self._started_ts = time.time()
        self._tensorboard_root = self._resolve_root(tensorboard_root)
        self._checkpoint_root = self._resolve_root(checkpoint_root)
        self._http_total = 0
        self._http_status: Counter[int] = Counter()
        self._http_methods: Counter[str] = Counter()
        self._http_routes: Counter[str] = Counter()
        self._http_latency_count = 0
        self._http_latency_sum_ms = 0.0
        self._http_latency_max_ms = 0.0
        self._http_latency_recent_ms: deque[float] = deque(maxlen=4096)
        self._http_route_latency_sum_ms: Dict[str, float] = defaultdict(float)
        self._http_route_latency_count: Counter[str] = Counter()

    def _episode_call(self, srec: SessionRecord, fn: Callable[[], T]) -> T:
        if str(srec.driver).lower() == "ctypes":
            with self._ctypes_call_lock:
                return fn()
        return fn()

    @staticmethod
    def _resolve_root(raw: str) -> Optional[Path]:
        txt = str(raw).strip()
        if not txt:
            return None
        return Path(txt).expanduser().resolve()

    @staticmethod
    def _resolve_dir(value: str, *, root: Optional[Path], fallback: str) -> Path:
        txt = str(value or "").strip()
        if txt:
            path = Path(txt).expanduser()
            if not path.is_absolute():
                if root is not None:
                    path = root / path
                else:
                    path = path.resolve()
            return path.resolve()
        if root is not None:
            return root
        return Path(fallback).expanduser().resolve()

    @staticmethod
    def _parse_learner_config(raw: Optional[Mapping[str, Any]]) -> Optional[LearnerConfig]:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise JormungandrServiceError("learner must be an object")
        action_values_raw = raw.get("action_values", [-1.0, 0.0, 1.0])
        if not isinstance(action_values_raw, list) or not action_values_raw:
            raise JormungandrServiceError("learner.action_values must be a non-empty array")
        action_values = [float(x) for x in action_values_raw]
        if not all(np.isfinite(x) for x in action_values):
            raise JormungandrServiceError("learner.action_values must be finite")
        if len(set(action_values)) != len(action_values):
            raise JormungandrServiceError("learner.action_values must be unique")
        algo = canonical_algorithm_name(str(raw.get("algo", "c51"))) or "c51"
        try:
            plugin = algorithm_registry.get(algo)
        except KeyError as exc:
            raise JormungandrServiceError(str(exc)) from exc
        plugin_defaults = dict(plugin.runtime_defaults)
        plugin_config_raw = raw.get("plugin_config", {})
        if not isinstance(plugin_config_raw, Mapping):
            raise JormungandrServiceError("learner.plugin_config must be an object")
        runtime_owned = {
            "enabled",
            "algo",
            "device",
            "action_values",
            "batch_size",
            "beta0",
            "beta_steps",
            "min_replay",
            "updates_per_tick",
            "tick_interval_s",
            "checkpoint_every",
            "checkpoint_dir",
            "validation_every",
            "validation_batch_size",
            "min_validation",
            "rollout_length",
            "max_policy_lag",
            "replay_selector",
            "qubo_pool_factor",
            "qubo_utility_weight",
            "qubo_diversity_weight",
            "qubo_cardinality_penalty",
            "qubo_local_search_passes",
            "qubo_reward_utility_weight",
            "aux_enabled",
            "aux_weight",
            "aux_classes",
            "aux_kind",
            "aux_label_key",
            "aux_class_weighting",
            "aux_label_smoothing",
        }
        misplaced = sorted(runtime_owned.intersection(plugin_config_raw))
        if misplaced:
            raise JormungandrServiceError(
                "runtime-owned learner settings must be top-level, not in "
                f"plugin_config: {', '.join(misplaced)}"
            )
        target_entropy_raw = raw.get("target_entropy")
        target_entropy = (
            None if target_entropy_raw is None else float(target_entropy_raw)
        )
        cfg = LearnerConfig(
            enabled=bool(raw.get("enabled", False)),
            algo=algo,
            device=str(raw.get("device", "auto")).strip() or "auto",
            hidden=max(8, int(raw.get("hidden", 256))),
            aux_hidden=max(0, int(raw.get("aux_hidden", 0))),
            lr=float(raw.get("lr", 1e-4)),
            gamma=float(raw.get("gamma", 0.99)),
            v_min=float(raw.get("v_min", -10.0)),
            v_max=float(raw.get("v_max", 10.0)),
            atoms=max(2, int(raw.get("atoms", 51))),
            quantiles=max(2, int(raw.get("quantiles", 51))),
            quantile_risk_measure=str(
                raw.get("quantile_risk_measure", "mean")
            ).strip().lower().replace("-", "_") or "mean",
            quantile_risk_level=float(raw.get("quantile_risk_level", 0.1)),
            target_update=max(1, int(raw.get("target_update", 1000))),
            max_grad=float(raw.get("max_grad", 1.0)),
            batch_size=max(1, int(raw.get("batch_size", 256))),
            beta0=float(raw.get("beta0", 0.4)),
            beta_steps=max(1, int(raw.get("beta_steps", 100_000))),
            min_replay=max(1, int(raw.get("min_replay", 2048))),
            updates_per_tick=max(1, int(raw.get("updates_per_tick", 1))),
            tick_interval_s=max(0.001, float(raw.get("tick_interval_s", 0.05))),
            checkpoint_every=max(0, int(raw.get("checkpoint_every", 5000))),
            checkpoint_dir=str(raw.get("checkpoint_dir", "./checkpoints/jormungandr")),
            validation_every=max(0, int(raw.get("validation_every", 100))),
            validation_batch_size=max(1, int(raw.get("validation_batch_size", 512))),
            min_validation=max(1, int(raw.get("min_validation", 64))),
            action_values=action_values,
            aux_enabled=bool(raw.get("aux_enabled", False)),
            aux_weight=max(0.0, float(raw.get("aux_weight", 0.0))),
            aux_classes=max(2, int(raw.get("aux_classes", 3))),
            aux_kind=str(raw.get("aux_kind", "")).strip(),
            aux_label_key=str(raw.get("aux_label_key", "label")).strip() or "label",
            aux_class_weighting=str(raw.get("aux_class_weighting", "none")).strip().lower() or "none",
            aux_label_smoothing=min(0.25, max(0.0, float(raw.get("aux_label_smoothing", 0.0)))),
            huber_delta=max(1e-6, float(raw.get("huber_delta", 1.0))),
            reward_clip=max(0.0, float(raw.get("reward_clip", 0.0))),
            observation_noise_std=max(
                0.0, float(raw.get("observation_noise_std", 0.0))
            ),
            temperature=max(1e-5, float(raw.get("temperature", 0.2))),
            auto_entropy=bool(raw.get("auto_entropy", True)),
            target_entropy=target_entropy,
            tau=min(1.0, max(1e-6, float(raw.get("tau", 0.005)))),
            cql_alpha=max(0.0, float(raw.get("cql_alpha", 1.0))),
            clip_ratio=max(0.0, float(raw.get("clip_ratio", 0.2))),
            entropy_coef=max(0.0, float(raw.get("entropy_coef", 0.01))),
            value_coef=max(0.0, float(raw.get("value_coef", 0.5))),
            gae_lambda=min(1.0, max(0.0, float(raw.get("gae_lambda", 0.95)))),
            epochs=max(
                1, int(raw.get("epochs", plugin_defaults.get("epochs", 1)))
            ),
            minibatch_size=max(
                1, int(raw.get("minibatch_size", raw.get("batch_size", 256)))
            ),
            vtrace_rho_clip=max(1.0, float(raw.get("vtrace_rho_clip", 1.0))),
            vtrace_pg_rho_clip=max(
                1.0, float(raw.get("vtrace_pg_rho_clip", 1.0))
            ),
            vtrace_c_clip=max(0.0, float(raw.get("vtrace_c_clip", 1.0))),
            appo_target_worker_clip=max(
                1.0, float(raw.get("appo_target_worker_clip", 2.0))
            ),
            rollout_length=max(1, int(raw.get("rollout_length", 32))),
            max_policy_lag=max(
                0,
                int(
                    raw.get(
                        "max_policy_lag",
                        plugin_defaults.get("max_policy_lag", 64),
                    )
                ),
            ),
            marwil_beta=max(0.0, float(raw.get("marwil_beta", 1.0))),
            marwil_c2_rate=min(
                1.0, max(1e-12, float(raw.get("marwil_c2_rate", 1e-8)))
            ),
            marwil_c2_start=max(1e-8, float(raw.get("marwil_c2_start", 100.0))),
            max_advantage_weight=max(
                1.0, float(raw.get("max_advantage_weight", 20.0))
            ),
            dreamer_latent=max(8, int(raw.get("dreamer_latent", 128))),
            imagination_horizon=max(1, int(raw.get("imagination_horizon", 15))),
            dreamer_lambda=min(
                1.0, max(0.0, float(raw.get("dreamer_lambda", 0.95)))
            ),
            replay_selector=str(raw.get("replay_selector", "prioritized")).strip().lower()
            or "prioritized",
            qubo_pool_factor=max(1.0, float(raw.get("qubo_pool_factor", 4.0))),
            qubo_utility_weight=max(
                0.0, float(raw.get("qubo_utility_weight", 1.0))
            ),
            qubo_diversity_weight=max(
                0.0, float(raw.get("qubo_diversity_weight", 0.35))
            ),
            qubo_cardinality_penalty=max(
                0.0, float(raw.get("qubo_cardinality_penalty", 4.0))
            ),
            qubo_local_search_passes=max(
                0, int(raw.get("qubo_local_search_passes", 8))
            ),
            qubo_reward_utility_weight=max(
                0.0, float(raw.get("qubo_reward_utility_weight", 0.0))
            ),
            plugin_config={str(k): v for k, v in plugin_config_raw.items()},
        )
        if not 0.0 <= cfg.gamma <= 1.0:
            raise JormungandrServiceError("learner.gamma must be in [0, 1]")
        if cfg.v_min >= cfg.v_max:
            raise JormungandrServiceError("learner.v_min must be less than learner.v_max")
        if cfg.quantile_risk_measure not in {"mean", "lower_quantile", "cvar"}:
            raise JormungandrServiceError(
                "learner.quantile_risk_measure must be mean, lower_quantile, or cvar"
            )
        if not 0.0 < cfg.quantile_risk_level <= 1.0:
            raise JormungandrServiceError(
                "learner.quantile_risk_level must be in (0, 1]"
            )
        if cfg.lr <= 0.0 or not np.isfinite(cfg.lr):
            raise JormungandrServiceError("learner.lr must be finite and > 0")
        if not 0.0 <= cfg.beta0 <= 1.0:
            raise JormungandrServiceError("learner.beta0 must be in [0, 1]")
        cfg.min_replay = max(cfg.min_replay, cfg.batch_size)
        if cfg.target_entropy is not None and not np.isfinite(cfg.target_entropy):
            raise JormungandrServiceError("learner.target_entropy must be finite")
        try:
            build_replay_selector(cfg.replay_selector, asdict(cfg))
        except ValueError as exc:
            raise JormungandrServiceError(str(exc)) from exc
        return cfg

    @staticmethod
    def _resolve_learner_device(raw_device: str) -> str:
        req = str(raw_device or "auto").strip().lower()
        try:
            import torch
        except Exception:
            if req in {"", "auto", "cpu"}:
                return "cpu"
            raise JormungandrServiceError(
                f"learner.device={raw_device} requested but torch is unavailable on this host"
            )
        if req in {"", "auto"}:
            if bool(torch.cuda.is_available()):
                return "cuda"
            if hasattr(torch.backends, "mps") and bool(torch.backends.mps.is_available()):
                return "mps"
            return "cpu"
        if req.startswith("cuda"):
            if not bool(torch.cuda.is_available()):
                return "cpu"
            return req
        if req == "mps":
            if hasattr(torch.backends, "mps") and bool(torch.backends.mps.is_available()):
                return "mps"
            return "cpu"
        if req == "cpu":
            return "cpu"
        try:
            _ = torch.device(req)
        except Exception as exc:
            raise JormungandrServiceError(f"invalid learner.device value: {raw_device}") from exc
        return req

    @staticmethod
    def _learner_summary(learner: Optional[LearnerState]) -> JsonDict:
        if learner is None:
            return {"enabled": False}
        return {
            "enabled": bool(learner.cfg.enabled),
            "algo": learner.cfg.algo,
            "device": learner.cfg.device,
            "updates": int(learner.updates),
            "policy_version": int(learner.policy_version),
            "last_loss": float(learner.last_loss),
            "last_td_abs": float(learner.last_td_abs),
            "last_beta": float(learner.last_beta),
            "last_train_ts": float(learner.last_train_ts),
            "checkpoint_count": int(learner.checkpoint_count),
            "last_checkpoint": learner.last_checkpoint,
            "last_aux_loss": float(learner.last_aux_loss),
            "last_aux_acc": float(learner.last_aux_acc),
            "last_aux_count": int(learner.last_aux_count),
            "validation_runs": int(learner.validation_runs),
            "validation_policy_version": int(learner.validation_policy_version),
            "last_validation_loss": float(learner.last_validation_loss),
            "last_validation_td_abs": float(learner.last_validation_td_abs),
            "last_validation_aux_loss": float(learner.last_validation_aux_loss),
            "last_validation_aux_acc": float(learner.last_validation_aux_acc),
            "last_validation_count": int(learner.last_validation_count),
            "last_validation_ts": float(learner.last_validation_ts),
            "last_error": learner.last_error,
            "weight_summary": dict(learner.last_weight_summary),
            "metrics": dict(learner.last_metrics),
            "selector_metrics": dict(learner.last_selector_metrics),
            "plugin": (
                {
                    "name": learner.plugin.name,
                    "version": learner.plugin.version,
                    "checkpoint_id": learner.plugin.checkpoint_id,
                    "family": learner.plugin.family,
                    "backend": learner.plugin.backend,
                    "replay_mode": learner.plugin.replay_mode,
                    "enforce_policy_lag": bool(learner.plugin.enforce_policy_lag),
                    "runtime_defaults": dict(learner.plugin.runtime_defaults),
                    "noise_profile": learner.plugin.noise_profile,
                }
                if learner.plugin is not None
                else {}
            ),
            "config": asdict(learner.cfg),
        }

    @classmethod
    def _normalizer_summary_locked(cls, rec: ModelRecord) -> JsonDict:
        std = rec.obs_normalizer.std()
        return {
            "kind": "running_zscore",
            "count": int(rec.obs_normalizer.count),
            "warmup": int(cls._OBS_NORMALIZER_WARMUP),
            "ready": bool(rec.obs_normalizer.count >= cls._OBS_NORMALIZER_WARMUP),
            "mean_abs": float(np.mean(np.abs(rec.obs_normalizer.mean))) if rec.obs_normalizer.mean.size else 0.0,
            "std_min": float(np.min(std)) if std.size else 0.0,
            "std_max": float(np.max(std)) if std.size else 0.0,
        }

    @staticmethod
    def _sanitize_obs_array(obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if np.isfinite(arr).all():
            return arr
        return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)

    @classmethod
    def _normalize_obs_locked(cls, rec: ModelRecord, obs: np.ndarray) -> np.ndarray:
        obs_np = cls._sanitize_obs_array(obs)
        if rec.obs_normalizer.count < cls._OBS_NORMALIZER_WARMUP:
            return obs_np
        return rec.obs_normalizer.normalize(obs_np).astype(np.float32, copy=False)

    @classmethod
    def _normalize_batch_locked(cls, rec: ModelRecord, batch):
        obs, action, reward, next_obs, done = batch
        if rec.obs_normalizer.count < cls._OBS_NORMALIZER_WARMUP:
            return (
                cls._sanitize_obs_array(obs),
                action,
                reward,
                cls._sanitize_obs_array(next_obs),
                done,
            )
        obs_n = rec.obs_normalizer.normalize(cls._sanitize_obs_array(obs)).astype(np.float32, copy=False)
        next_obs_n = rec.obs_normalizer.normalize(cls._sanitize_obs_array(next_obs)).astype(np.float32, copy=False)
        return obs_n, action, reward, next_obs_n, done

    @staticmethod
    def _parameter_stats_from_tensors(tensors: List[Any]) -> JsonDict:
        arrays: List[np.ndarray] = []
        for tensor in tensors:
            if tensor is None:
                continue
            try:
                arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            except Exception:
                continue
            if arr.size > 0:
                arrays.append(arr)
        if not arrays:
            return {}
        values = np.concatenate(arrays, axis=0)
        counts, edges = np.histogram(values, bins=24)
        centers = ((edges[:-1] + edges[1:]) * 0.5).astype(np.float32)
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "abs_mean": float(np.mean(np.abs(values))),
            "abs_max": float(np.max(np.abs(values))),
            "l2": float(np.linalg.norm(values)),
            "histogram": {
                "bins": [float(x) for x in centers.tolist()],
                "counts": [int(x) for x in counts.astype(np.int64).tolist()],
            },
        }

    @classmethod
    def _agent_weight_summary(cls, agent: Optional[Any]) -> JsonDict:
        if agent is None:
            return {}
        summary: JsonDict = {}
        for module_name in (
            "q",
            "q1",
            "q2",
            "policy",
            "value",
            "encoder",
            "dynamics",
            "aux_head",
        ):
            module = getattr(agent, module_name, None)
            if module is None or not hasattr(module, "parameters"):
                continue
            try:
                summary[module_name] = cls._parameter_stats_from_tensors(
                    list(module.parameters())
                )
            except Exception:
                summary[module_name] = {}
        return summary

    @staticmethod
    def _build_learner_state(obs_dim: int, cfg: Optional[LearnerConfig]) -> Optional[LearnerState]:
        if cfg is None or not cfg.enabled:
            return None
        resolved_device = JormungandrRuntime._resolve_learner_device(cfg.device)
        cfg.device = str(resolved_device)
        plugin = algorithm_registry.get(cfg.algo)
        config = asdict(cfg)
        agent = plugin.build(int(obs_dim), config, str(cfg.device))
        selector = build_replay_selector(cfg.replay_selector, config)
        return LearnerState(
            cfg=cfg,
            agent=agent,
            plugin=plugin,
            replay_selector=selector,
        )

    @classmethod
    def _build_restored_learner_state(
        cls,
        checkpoint_path: str,
        *,
        learner_overrides: Optional[Mapping[str, Any]] = None,
        checkpoint_root: Optional[Path] = None,
    ) -> tuple[Any, LearnerState, Optional[RunningNormalizer], JsonDict]:
        from jormungandr.export import _build_agent_from_checkpoint, _checkpoint_spec_from_payload, _load_checkpoint

        ckpt = _load_checkpoint(checkpoint_path)
        spec = _checkpoint_spec_from_payload(checkpoint_path, ckpt)

        learner_raw = ckpt.get("learner_config")
        merged_cfg: JsonDict = dict(learner_raw) if isinstance(learner_raw, Mapping) else {}
        if isinstance(learner_overrides, Mapping):
            merged_cfg.update({str(k): v for k, v in learner_overrides.items()})
        merged_cfg.update(
            {
                "algo": spec.algo,
                "hidden": int(spec.hidden),
                "aux_hidden": int(spec.aux_hidden),
                "lr": float(spec.lr),
                "gamma": float(spec.gamma),
                "v_min": float(spec.v_min),
                "v_max": float(spec.v_max),
                "atoms": int(spec.atoms),
                "quantiles": int(spec.quantiles),
                "quantile_risk_measure": spec.quantile_risk_measure,
                "quantile_risk_level": float(spec.quantile_risk_level),
                "target_update": int(spec.target_update),
                "max_grad": float(spec.max_grad),
                "action_values": [float(x) for x in spec.action_values],
                "aux_enabled": bool(int(spec.aux_classes) > 0),
                "aux_classes": int(spec.aux_classes),
            }
        )
        cfg = cls._parse_learner_config(merged_cfg)
        if cfg is None:
            raise JormungandrServiceError("checkpoint restore requires learner configuration")
        cfg.device = cls._resolve_learner_device(cfg.device)
        cfg.checkpoint_dir = str(
            cls._resolve_dir(
                cfg.checkpoint_dir,
                root=checkpoint_root,
                fallback="./checkpoints/jormungandr",
            )
        )

        agent = _build_agent_from_checkpoint(
            ckpt,
            spec=spec,
            device=str(cfg.device),
            load_optimizer=True,
        )
        plugin = algorithm_registry.get(cfg.algo)
        learner = LearnerState(
            cfg=cfg,
            agent=agent,
            plugin=plugin,
            replay_selector=build_replay_selector(cfg.replay_selector, asdict(cfg)),
        )
        learner.updates = int(spec.updates)
        learner.policy_version = int(spec.policy_version)
        learner.last_checkpoint = str(Path(checkpoint_path).expanduser().resolve())
        learner.checkpoint_count = 1

        normalizer_state = ckpt.get("obs_normalizer") or ckpt.get("normalizer")
        normalizer: Optional[RunningNormalizer] = None
        if isinstance(normalizer_state, Mapping):
            try:
                normalizer = RunningNormalizer.load(dict(normalizer_state))
            except Exception:
                normalizer = None

        metadata = dict(ckpt.get("metadata") or {}) if isinstance(ckpt.get("metadata"), Mapping) else {}
        if spec.feature_keys and not isinstance(metadata.get("feature_keys"), list):
            metadata["feature_keys"] = list(spec.feature_keys)
        metadata["checkpoint_source"] = str(Path(checkpoint_path).expanduser().resolve())
        metadata["checkpoint_policy_version"] = int(spec.policy_version)
        metadata["checkpoint_updates"] = int(spec.updates)
        return spec, learner, normalizer, metadata

    @staticmethod
    def _prepare_checkpoint(rec: ModelRecord, learner: LearnerState) -> tuple[Path, JsonDict]:
        step = int(learner.updates)
        ver = int(learner.policy_version)
        out_dir = Path(learner.cfg.checkpoint_dir).expanduser().resolve() / rec.model_id
        out_file = out_dir / f"ckpt_u{step:09d}_v{ver:09d}.pt"
        feature_keys: List[str] = []
        raw_fk = rec.metadata.get("feature_keys")
        if isinstance(raw_fk, list):
            feature_keys = [str(x) for x in raw_fk]
        payload: JsonDict = {
            "format": "jormungandr.checkpoint.v1",
            "algo": learner.cfg.algo,
            "model_id": rec.model_id,
            "obs_dim": int(rec.obs_dim),
            "action_values": list(learner.cfg.action_values),
            "learner_config": asdict(learner.cfg),
            "metadata": dict(rec.metadata),
            "feature_keys": feature_keys,
            "agent": learner.agent.state_dict() if learner.agent is not None else {},
            "algorithm_plugin": (
                {
                    "name": learner.plugin.name,
                    "version": learner.plugin.version,
                    "checkpoint_id": learner.plugin.checkpoint_id,
                    "family": learner.plugin.family,
                    "backend": learner.plugin.backend,
                    "replay_mode": learner.plugin.replay_mode,
                    "enforce_policy_lag": bool(
                        learner.plugin.enforce_policy_lag
                    ),
                }
                if learner.plugin is not None
                else {"name": learner.cfg.algo}
            ),
            "replay_selector": {
                "name": learner.cfg.replay_selector,
                "last_metrics": dict(learner.last_selector_metrics),
            },
            "obs_normalizer": rec.obs_normalizer.state_dict(),
            "updates": step,
            "policy_version": ver,
            "experience_schema": "jormungandr.experience.v1",
            "ts": time.time(),
        }
        return out_file, payload

    def _save_checkpoint(self, rec: ModelRecord, learner: LearnerState) -> None:
        import torch

        with rec.lock:
            with learner.agent_lock:
                path, payload = self._prepare_checkpoint(rec, learner)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))
        with rec.lock:
            learner.last_checkpoint = str(path)
            learner.checkpoint_count += 1

    @staticmethod
    def _metadata_for_indices(
        source: Mapping[int, JsonDict], indices: np.ndarray
    ) -> List[JsonDict]:
        return [dict(source.get(int(index), {})) for index in indices.tolist()]

    @staticmethod
    def _order_trajectory_sample(
        batch: Any,
        indices: np.ndarray,
        weights: np.ndarray,
        metadata: List[JsonDict],
    ) -> tuple[Any, np.ndarray, np.ndarray, List[JsonDict]]:
        """Put sampled actor steps in episode/timestep order for GAE/V-trace."""

        if len(indices) <= 1:
            return batch, indices, weights, metadata

        def key(position: int) -> tuple[str, str, int, int]:
            item = metadata[position]
            try:
                timestep = int(item.get("timestep", position))
            except Exception:
                timestep = position
            return (
                str(item.get("actor_id", "")),
                str(item.get("episode_id", f"~{position}")),
                timestep,
                position,
            )

        order = np.asarray(sorted(range(len(indices)), key=key), dtype=np.int64)
        ordered_batch = tuple(np.asarray(component)[order] for component in batch)
        return (
            ordered_batch,
            np.asarray(indices)[order],
            np.asarray(weights)[order],
            [metadata[int(position)] for position in order.tolist()],
        )

    def _run_learner_loop(self, rec: ModelRecord, learner: LearnerState) -> None:
        cfg = learner.cfg
        while not learner.stop_event.wait(float(cfg.tick_interval_s)):
            try:
                for _ in range(int(cfg.updates_per_tick)):
                    if learner.stop_event.is_set():
                        break
                    with rec.lock:
                        replay_size = len(rec.replay)
                    if replay_size < int(cfg.min_replay):
                        break

                    beta = min(1.0, float(cfg.beta0) + float(learner.updates) / float(max(1, cfg.beta_steps)))
                    with rec.lock:
                        replay_size = len(rec.replay)
                        if replay_size < int(cfg.min_replay):
                            break
                        bs = max(1, min(int(cfg.batch_size), replay_size))
                        if (
                            learner.plugin is not None
                            and learner.plugin.replay_mode == "trajectory"
                            and cfg.replay_selector in {"prioritized", "per", "qubo"}
                        ):
                            selected = select_trajectory_replay(
                                rec.replay,
                                rec.replay_meta_by_idx,
                                batch_size=bs,
                                beta=beta,
                                rollout_length=cfg.rollout_length,
                                current_policy_version=learner.policy_version,
                                max_policy_lag=(
                                    cfg.max_policy_lag
                                    if learner.plugin.enforce_policy_lag
                                    else -1
                                ),
                                selector_name=cfg.replay_selector,
                                config=asdict(cfg),
                            )
                            if selected is None:
                                break
                        else:
                            selector = learner.replay_selector or build_replay_selector(
                                cfg.replay_selector, asdict(cfg)
                            )
                            selected = selector.select(rec.replay, bs, beta)
                        batch_raw = selected.batch
                        idxs = np.asarray(selected.indices, dtype=np.int64)
                        weights = np.asarray(selected.weights, dtype=np.float32)
                        sample_metadata = self._metadata_for_indices(
                            rec.replay_meta_by_idx, idxs
                        )
                        if (
                            learner.plugin is not None
                            and learner.plugin.replay_mode == "trajectory"
                            and cfg.replay_selector not in {"prioritized", "per", "qubo"}
                        ):
                            batch_raw, idxs, weights, sample_metadata = self._order_trajectory_sample(
                                batch_raw,
                                idxs,
                                weights,
                                sample_metadata,
                            )
                        batch = self._normalize_batch_locked(rec, batch_raw)
                        aux_obs_rows: List[np.ndarray] = []
                        aux_target_rows: List[int] = []
                        if bool(cfg.aux_enabled):
                            idx_arr = np.asarray(idxs, dtype=np.int64).reshape(-1)
                            obs_arr = np.asarray(batch[0], dtype=np.float32)
                            for j, idx_val in enumerate(idx_arr.tolist()):
                                aux = rec.replay_aux_by_idx.get(int(idx_val))
                                if not isinstance(aux, Mapping):
                                    continue
                                if cfg.aux_kind and str(aux.get("kind", "")) != str(cfg.aux_kind):
                                    continue
                                label_raw = aux.get(str(cfg.aux_label_key))
                                try:
                                    label_i = int(label_raw)
                                except Exception:
                                    continue
                                if label_i < 0 or label_i >= int(cfg.aux_classes):
                                    continue
                                aux_obs_rows.append(obs_arr[j])
                                aux_target_rows.append(label_i)

                    aux_obs_np: Optional[np.ndarray] = None
                    aux_targets_np: Optional[np.ndarray] = None
                    if aux_obs_rows:
                        aux_obs_np = np.asarray(aux_obs_rows, dtype=np.float32)
                        aux_targets_np = np.asarray(aux_target_rows, dtype=np.int64)
                    with learner.agent_lock:
                        update_result = normalize_update_result(
                            learner.agent.update(
                                batch,
                                weights,
                                metadata=sample_metadata,
                                aux_obs=aux_obs_np,
                                aux_targets=aux_targets_np,
                                aux_weight=float(cfg.aux_weight),
                            )
                            if learner.agent is not None
                            else (0.0, np.array([]))
                        )
                        loss = float(update_result.loss)
                        td_err = np.asarray(update_result.priorities, dtype=np.float32)
                        algorithm_metrics = {
                            str(k): float(v)
                            for k, v in update_result.metrics.items()
                            if np.isfinite(float(v))
                        }
                        td_abs = float(np.mean(np.abs(td_err))) if len(td_err) else 0.0
                        aux_loss = learner.agent.last_aux_loss if learner.agent is not None else None
                        aux_acc = learner.agent.last_aux_acc if learner.agent is not None else None
                        aux_count = int(aux_targets_np.shape[0]) if aux_targets_np is not None else 0
                        weight_summary = self._agent_weight_summary(learner.agent)
                        selector_metrics = {
                            str(k): float(v)
                            for k, v in selected.metrics.items()
                            if np.isfinite(float(v))
                        }
                        selector_audit = dict(selected.audit)
                    checkpoint_due = False
                    validation_due = False
                    with rec.lock:
                        rec.replay.update_priorities(
                            np.asarray(idxs, dtype=np.int64),
                            np.asarray(td_err, dtype=np.float32),
                        )
                        learner.updates += 1
                        learner.policy_version += 1
                        learner.last_loss = float(loss)
                        learner.last_td_abs = float(td_abs)
                        learner.last_beta = float(beta)
                        learner.last_train_ts = time.time()
                        learner.last_aux_loss = float(aux_loss) if aux_loss is not None else 0.0
                        learner.last_aux_acc = float(aux_acc) if aux_acc is not None else 0.0
                        learner.last_aux_count = int(aux_count)
                        learner.last_error = ""
                        learner.last_weight_summary = dict(weight_summary)
                        learner.last_metrics = dict(algorithm_metrics)
                        learner.last_selector_metrics = dict(selector_metrics)
                        learner.last_selector_audit = dict(selector_audit)
                        learner.metric_history.append(
                            {
                                "update": int(learner.updates),
                                "policy_version": int(learner.policy_version),
                                "ts": float(learner.last_train_ts),
                                "metrics": dict(algorithm_metrics),
                                "selector": dict(selector_metrics),
                            }
                        )
                        rec.tb_logger.add("learner/loss", learner.last_loss, learner.updates)
                        rec.tb_logger.add("learner/td_abs_mean", learner.last_td_abs, learner.updates)
                        rec.tb_logger.add("learner/replay_size", float(replay_size), learner.updates)
                        rec.tb_logger.add("learner/aux_loss", aux_loss, learner.updates)
                        rec.tb_logger.add("learner/aux_acc", aux_acc, learner.updates)
                        rec.tb_logger.add("learner/aux_count", float(aux_count), learner.updates)
                        rec.tb_logger.add_split("train", "rl/loss", learner.last_loss, learner.updates)
                        rec.tb_logger.add_split("train", "rl/td_abs_mean", learner.last_td_abs, learner.updates)
                        rec.tb_logger.add_split("train", "rl/replay_size", float(replay_size), learner.updates)
                        rec.tb_logger.add_split("train", "aux/loss", aux_loss, learner.updates)
                        rec.tb_logger.add_split("train", "aux/acc", aux_acc, learner.updates)
                        rec.tb_logger.add_split("train", "aux/count", float(aux_count), learner.updates)
                        algorithm_name = cfg.algo
                        for metric_name, metric_value in algorithm_metrics.items():
                            rec.tb_logger.add(
                                f"algorithms/{algorithm_name}/{metric_name}",
                                metric_value,
                                learner.updates,
                            )
                            rec.tb_logger.add_split(
                                "train",
                                f"algorithms/{algorithm_name}/{metric_name}",
                                metric_value,
                                learner.updates,
                            )
                        for metric_name, metric_value in selector_metrics.items():
                            rec.tb_logger.add(
                                f"selectors/{cfg.replay_selector}/{metric_name}",
                                metric_value,
                                learner.updates,
                            )
                        if learner.updates % 25 == 0:
                            self._log_learner_batch_histograms(
                                rec.tb_logger,
                                step=learner.updates,
                                batch_raw=batch_raw,
                                weights=weights,
                                td_err=td_err,
                                aux_targets=aux_targets_np,
                            )
                        if learner.updates % 250 == 0 and learner.agent is not None:
                            self._log_agent_weight_histograms(
                                rec.tb_logger,
                                learner.agent,
                                step=learner.updates,
                            )
                        if learner.updates % 50 == 0:
                            rec.tb_logger.flush()
                        if cfg.checkpoint_every > 0 and learner.updates % int(cfg.checkpoint_every) == 0:
                            checkpoint_due = True
                        if (
                            cfg.validation_every > 0
                            and learner.updates % int(cfg.validation_every) == 0
                            and len(rec.validation) >= int(cfg.min_validation)
                        ):
                            validation_due = True
                    if validation_due and learner.agent is not None:
                        with rec.lock:
                            validation_size = len(rec.validation)
                            validation_bs = max(
                                1,
                                min(int(cfg.validation_batch_size), validation_size),
                            )
                            validation_selection = None
                            if (
                                learner.plugin is not None
                                and learner.plugin.replay_mode == "trajectory"
                            ):
                                validation_selection = select_trajectory_replay(
                                    rec.validation,
                                    rec.validation_meta_by_idx,
                                    batch_size=validation_bs,
                                    beta=0.0,
                                    rollout_length=cfg.rollout_length,
                                    current_policy_version=learner.policy_version,
                                    max_policy_lag=-1,
                                    selector_name="prioritized",
                                    config=asdict(cfg),
                                )
                            if validation_selection is not None:
                                validation_raw = validation_selection.batch
                                validation_idxs = np.asarray(
                                    validation_selection.indices,
                                    dtype=np.int64,
                                )
                                validation_metadata = self._metadata_for_indices(
                                    rec.validation_meta_by_idx,
                                    validation_idxs,
                                )
                            else:
                                validation_raw, validation_idxs, _ = rec.validation.sample(
                                    validation_bs,
                                    beta=0.0,
                                )
                                validation_idxs = np.asarray(
                                    validation_idxs, dtype=np.int64
                                )
                                validation_metadata = self._metadata_for_indices(
                                    rec.validation_meta_by_idx,
                                    validation_idxs,
                                )
                                if (
                                    learner.plugin is not None
                                    and learner.plugin.replay_mode == "trajectory"
                                ):
                                    (
                                        validation_raw,
                                        validation_idxs,
                                        _validation_weights,
                                        validation_metadata,
                                    ) = self._order_trajectory_sample(
                                        validation_raw,
                                        validation_idxs,
                                        np.ones(
                                            len(validation_idxs),
                                            dtype=np.float32,
                                        ),
                                        validation_metadata,
                                    )
                            validation_batch = self._normalize_batch_locked(
                                rec,
                                validation_raw,
                            )
                            validation_aux_obs: List[np.ndarray] = []
                            validation_aux_targets: List[int] = []
                            if bool(cfg.aux_enabled):
                                validation_idx_arr = np.asarray(
                                    validation_idxs,
                                    dtype=np.int64,
                                ).reshape(-1)
                                validation_obs_arr = np.asarray(
                                    validation_batch[0],
                                    dtype=np.float32,
                                )
                                for j, idx_val in enumerate(validation_idx_arr.tolist()):
                                    aux = rec.validation_aux_by_idx.get(int(idx_val))
                                    if not isinstance(aux, Mapping):
                                        continue
                                    if cfg.aux_kind and str(aux.get("kind", "")) != str(
                                        cfg.aux_kind
                                    ):
                                        continue
                                    try:
                                        label_i = int(aux.get(str(cfg.aux_label_key)))
                                    except Exception:
                                        continue
                                    if 0 <= label_i < int(cfg.aux_classes):
                                        validation_aux_obs.append(validation_obs_arr[j])
                                        validation_aux_targets.append(label_i)
                            validation_policy_version = int(learner.policy_version)

                        with learner.agent_lock:
                            validation_metrics = learner.agent.evaluate_batch(
                                validation_batch,
                                metadata=validation_metadata,
                                aux_obs=(
                                    np.asarray(validation_aux_obs, dtype=np.float32)
                                    if validation_aux_obs
                                    else None
                                ),
                                aux_targets=(
                                    np.asarray(validation_aux_targets, dtype=np.int64)
                                    if validation_aux_targets
                                    else None
                                ),
                            )
                        with rec.lock:
                            learner.validation_runs += 1
                            learner.validation_policy_version = validation_policy_version
                            learner.last_validation_loss = float(
                                validation_metrics.get("loss", 0.0)
                            )
                            learner.last_validation_td_abs = float(
                                validation_metrics.get("td_abs_mean", 0.0)
                            )
                            learner.last_validation_aux_loss = float(
                                validation_metrics.get("aux_loss", 0.0)
                            )
                            learner.last_validation_aux_acc = float(
                                validation_metrics.get("aux_acc", 0.0)
                            )
                            learner.last_validation_count = int(
                                validation_metrics.get("count", 0.0)
                            )
                            learner.last_validation_ts = time.time()
                            rec.tb_logger.add_split(
                                "validation",
                                "rl/loss",
                                learner.last_validation_loss,
                                validation_policy_version,
                            )
                            rec.tb_logger.add_split(
                                "validation",
                                "rl/td_abs_mean",
                                learner.last_validation_td_abs,
                                validation_policy_version,
                            )
                            rec.tb_logger.add_split(
                                "validation",
                                "aux/loss",
                                validation_metrics.get("aux_loss"),
                                validation_policy_version,
                            )
                            rec.tb_logger.add_split(
                                "validation",
                                "aux/acc",
                                validation_metrics.get("aux_acc"),
                                validation_policy_version,
                            )
                            for metric_name, metric_value in validation_metrics.items():
                                try:
                                    finite_value = float(metric_value)
                                except Exception:
                                    continue
                                if not np.isfinite(finite_value):
                                    continue
                                rec.tb_logger.add_split(
                                    "validation",
                                    f"algorithms/{cfg.algo}/{metric_name}",
                                    finite_value,
                                    validation_policy_version,
                                )
                    if checkpoint_due:
                        self._save_checkpoint(rec, learner)
            except Exception as exc:
                with rec.lock:
                    learner.last_error = str(exc)
                time.sleep(0.25)

    @staticmethod
    def _log_hist_pair(tb_logger: TBLogger, tag: str, values: Any, step: int, *, split_tag: Optional[str] = None) -> None:
        try:
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
        except Exception:
            return
        if arr.size == 0:
            return
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        tb_logger.add_hist(tag, arr, step)
        if split_tag:
            tb_logger.add_split_hist("train", split_tag, arr, step)

    def _log_learner_batch_histograms(
        self,
        tb_logger: TBLogger,
        *,
        step: int,
        batch_raw: Any,
        weights: Any,
        td_err: Any,
        aux_targets: Optional[np.ndarray],
    ) -> None:
        self._log_hist_pair(tb_logger, "learner/td_error", td_err, step, split_tag="rl/td_error")
        self._log_hist_pair(tb_logger, "learner/sample_weight", weights, step, split_tag="rl/sample_weight")
        try:
            actions = batch_raw[1]
            rewards = batch_raw[2]
            done = batch_raw[4]
        except Exception:
            return
        self._log_hist_pair(tb_logger, "learner/batch_action", actions, step, split_tag="rl/batch_action")
        self._log_hist_pair(tb_logger, "learner/batch_reward", rewards, step, split_tag="rl/batch_reward")
        self._log_hist_pair(tb_logger, "learner/batch_done", done, step, split_tag="rl/batch_done")
        if aux_targets is not None:
            self._log_hist_pair(tb_logger, "learner/aux_target", aux_targets, step, split_tag="aux/target")

    def _log_agent_weight_histograms(self, tb_logger: TBLogger, agent: Any, *, step: int) -> None:
        seen: set[int] = set()
        for module_name in (
            "q",
            "q1",
            "q2",
            "policy",
            "value",
            "encoder",
            "dynamics",
            "reward_head",
            "aux_head",
        ):
            module = getattr(agent, module_name, None)
            if module is None or not hasattr(module, "named_parameters"):
                continue
            if id(module) in seen:
                continue
            seen.add(id(module))
            for param_name, param in module.named_parameters():
                try:
                    arr = param.detach().cpu().numpy()
                except Exception:
                    continue
                tag_name = str(param_name).replace(".", "/")
                self._log_hist_pair(
                    tb_logger,
                    f"weights/{module_name}/{tag_name}",
                    arr,
                    step,
                    split_tag=f"weights/{module_name}/{tag_name}",
                )

    def _start_model_learner(self, rec: ModelRecord) -> None:
        learner = rec.learner
        if learner is None or not learner.cfg.enabled:
            return
        if learner.thread is not None and learner.thread.is_alive():
            return
        learner.stop_event.clear()
        learner.thread = threading.Thread(
            target=self._run_learner_loop,
            args=(rec, learner),
            name=f"jormungandr-learner-{rec.model_id}",
            daemon=True,
        )
        learner.thread.start()

    @staticmethod
    def _stop_model_learner(rec: ModelRecord) -> None:
        learner = rec.learner
        if learner is None:
            return
        learner.stop_event.set()
        th = learner.thread
        if th is not None and th.is_alive():
            th.join(timeout=3.0)

    @staticmethod
    def _canonical_route(path: str) -> str:
        parts = [p for p in urlparse(path).path.split("/") if p]
        if len(parts) >= 3 and parts[:2] == ["v1", "models"]:
            parts[2] = "{model_id}"
        return "/" + "/".join(parts) if parts else "/"

    @staticmethod
    def _quantile(sorted_values: List[float], q: float) -> float:
        if not sorted_values:
            return 0.0
        q = min(1.0, max(0.0, float(q)))
        idx = int(round((len(sorted_values) - 1) * q))
        return float(sorted_values[idx])

    def record_http(self, method: str, path: str, status: int, *, latency_ms: Optional[float] = None) -> None:
        key = f"{method.upper()} {self._canonical_route(path)}"
        with self._lock:
            self._http_total += 1
            self._http_status[int(status)] += 1
            self._http_methods[method.upper()] += 1
            self._http_routes[key] += 1
            if latency_ms is not None and latency_ms >= 0.0:
                lat = float(latency_ms)
                self._http_latency_count += 1
                self._http_latency_sum_ms += lat
                self._http_latency_max_ms = max(self._http_latency_max_ms, lat)
                self._http_latency_recent_ms.append(lat)
                self._http_route_latency_sum_ms[key] += lat
                self._http_route_latency_count[key] += 1

    def snapshot_stats(self, *, max_sessions: int = 8, max_endpoints: int = 6) -> JsonDict:
        with self._lock:
            model_ids = list(self._models.keys())
            sessions = list(self._sessions.values())
            http_total = int(self._http_total)
            http_status = {str(k): int(v) for k, v in self._http_status.items()}
            http_methods = {str(k): int(v) for k, v in self._http_methods.items()}
            top_endpoints: List[JsonDict] = []
            for key, count in self._http_routes.most_common(max(1, int(max_endpoints))):
                lat_cnt = int(self._http_route_latency_count.get(key, 0))
                lat_avg = 0.0
                if lat_cnt > 0:
                    lat_avg = float(self._http_route_latency_sum_ms.get(key, 0.0)) / float(lat_cnt)
                top_endpoints.append(
                    {
                        "key": key,
                        "count": int(count),
                        "avg_latency_ms": lat_avg,
                    }
                )
            started_ts = float(self._started_ts)
            latency_count = int(self._http_latency_count)
            latency_avg_ms = float(self._http_latency_sum_ms) / float(latency_count) if latency_count > 0 else 0.0
            latency_max_ms = float(self._http_latency_max_ms)
            lat_recent = list(self._http_latency_recent_ms)
            lat_recent.sort()
            latency_p50_ms = self._quantile(lat_recent, 0.50)
            latency_p95_ms = self._quantile(lat_recent, 0.95)

        models: List[JsonDict] = []
        total_replay_size = 0
        total_replay_capacity = 0
        total_validation_size = 0
        total_validation_capacity = 0
        total_inference_updates = 0
        total_inference_reads = 0
        total_learner_updates = 0
        total_learner_checkpoints = 0
        total_policy_infer_calls = 0
        total_aux_infer_calls = 0
        total_sampler_train_episodes = 0
        total_sampler_val_episodes = 0
        total_aux_update_items = 0
        total_aux_update_matched = 0
        total_aux_update_pending = 0
        for model_id in model_ids:
            try:
                rec = self.get_model_record(model_id)
            except KeyError:
                continue
            with rec.lock:
                replay_size = int(len(rec.replay))
                replay_capacity = int(rec.capacity)
                validation_size = int(len(rec.validation))
                validation_capacity = int(rec.validation_capacity)
                learner_updates = 0
                policy_version = 0
                learner_enabled = False
                learner_last_loss = 0.0
                learner_last_aux_loss = 0.0
                learner_last_aux_acc = 0.0
                learner_last_aux_count = 0
                learner_last_error = ""
                learner_checkpoint_count = 0
                learner_device = "-"
                if rec.learner is not None:
                    learner_updates = int(rec.learner.updates)
                    policy_version = int(rec.learner.policy_version)
                    learner_enabled = bool(rec.learner.cfg.enabled)
                    learner_last_loss = float(rec.learner.last_loss)
                    learner_last_aux_loss = float(rec.learner.last_aux_loss)
                    learner_last_aux_acc = float(rec.learner.last_aux_acc)
                    learner_last_aux_count = int(rec.learner.last_aux_count)
                    learner_last_error = rec.learner.last_error
                    learner_checkpoint_count = int(rec.learner.checkpoint_count)
                    learner_device = str(rec.learner.cfg.device)
                models.append(
                    {
                        "model_id": rec.model_id,
                        "sessions": len(rec.session_ids),
                        "replay_size": replay_size,
                        "replay_capacity": replay_capacity,
                        "validation_size": validation_size,
                        "validation_capacity": validation_capacity,
                        "replay_add_calls": int(rec.replay_add_calls),
                        "replay_add_items": int(rec.replay_add_items),
                        "train_add_items": int(rec.train_add_items),
                        "validation_add_items": int(rec.validation_add_items),
                        "replay_sample_calls": int(rec.replay_sample_calls),
                        "replay_priority_update_calls": int(rec.replay_priority_update_calls),
                        "metrics_calls": int(rec.metrics_calls),
                        "metrics_points": int(rec.metrics_points),
                        "sampler_train_episodes": int(rec.sampler_train_episodes),
                        "sampler_val_episodes": int(rec.sampler_val_episodes),
                        "last_metrics_step": int(rec.last_metrics_step),
                        "inference_updates": int(rec.inference_updates),
                        "inference_reads": int(rec.inference_reads),
                        "policy_infer_calls": int(rec.policy_infer_calls),
                        "aux_infer_calls": int(rec.aux_infer_calls),
                        "aux_update_calls": int(rec.aux_update_calls),
                        "aux_update_items": int(rec.aux_update_items),
                        "aux_update_matched": int(rec.aux_update_matched),
                        "aux_update_pending": int(rec.aux_update_pending),
                        "learner_enabled": learner_enabled,
                        "learner_updates": learner_updates,
                        "policy_version": policy_version,
                        "learner_checkpoints": learner_checkpoint_count,
                        "learner_device": learner_device,
                        "learner_last_loss": learner_last_loss,
                        "learner_last_aux_loss": learner_last_aux_loss,
                        "learner_last_aux_acc": learner_last_aux_acc,
                        "learner_last_aux_count": learner_last_aux_count,
                        "validation_runs": (
                            int(rec.learner.validation_runs)
                            if rec.learner is not None
                            else 0
                        ),
                        "validation_policy_version": (
                            int(rec.learner.validation_policy_version)
                            if rec.learner is not None
                            else 0
                        ),
                        "learner_last_validation_loss": (
                            float(rec.learner.last_validation_loss)
                            if rec.learner is not None
                            else 0.0
                        ),
                        "learner_last_error": learner_last_error,
                    }
                )
                total_replay_size += replay_size
                total_replay_capacity += replay_capacity
                total_validation_size += validation_size
                total_validation_capacity += validation_capacity
                total_inference_updates += int(rec.inference_updates)
                total_inference_reads += int(rec.inference_reads)
                total_learner_updates += int(learner_updates)
                total_learner_checkpoints += int(learner_checkpoint_count)
                total_policy_infer_calls += int(rec.policy_infer_calls)
                total_aux_infer_calls += int(rec.aux_infer_calls)
                total_sampler_train_episodes += int(rec.sampler_train_episodes)
                total_sampler_val_episodes += int(rec.sampler_val_episodes)
                total_aux_update_items += int(rec.aux_update_items)
                total_aux_update_matched += int(rec.aux_update_matched)
                total_aux_update_pending += int(rec.aux_update_pending)
        models.sort(key=lambda r: str(r.get("model_id", "")))

        now = time.time()
        session_rows: List[JsonDict] = []
        for srec in sessions:
            with srec.lock:
                age_s = now - srec.created_ts
                last_step_age_s = None
                if srec.last_step_ts > 0:
                    last_step_age_s = max(0.0, now - srec.last_step_ts)
                avg_step_latency_ms = 0.0
                if srec.step_latency_count > 0:
                    avg_step_latency_ms = float(srec.step_latency_sum_ms) / float(srec.step_latency_count)
                session_rows.append(
                    {
                        "session_id": srec.session_id,
                        "model_id": srec.model_id,
                        "age_s": age_s,
                        "initialize_count": int(srec.initialize_count),
                        "step_count": int(srec.step_count),
                        "agent_cmd_count": int(srec.agent_cmd_count),
                        "last_step_age_s": last_step_age_s,
                        "last_step_latency_ms": float(srec.last_step_latency_ms),
                        "avg_step_latency_ms": avg_step_latency_ms,
                        "terminal": bool(srec.last_step_terminal),
                        "remaining_steps": srec.last_remaining_steps,
                    }
                )
        session_rows.sort(
            key=lambda r: (
                -float(r.get("step_count", 0)),
                float(r.get("last_step_age_s") or 1e18),
            )
        )

        return {
            "ts": now,
            "started_ts": started_ts,
            "uptime_s": max(0.0, now - started_ts),
            "http": {
                "total": http_total,
                "status": http_status,
                "methods": http_methods,
                "latency": {
                    "count": latency_count,
                    "avg_ms": latency_avg_ms,
                    "p50_ms": latency_p50_ms,
                    "p95_ms": latency_p95_ms,
                    "max_ms": latency_max_ms,
                },
                "top_endpoints": top_endpoints,
            },
            "totals": {
                "replay_size": int(total_replay_size),
                "replay_capacity": int(total_replay_capacity),
                "validation_size": int(total_validation_size),
                "validation_capacity": int(total_validation_capacity),
                "inference_updates": int(total_inference_updates),
                "inference_reads": int(total_inference_reads),
                "learner_updates": int(total_learner_updates),
                "learner_checkpoints": int(total_learner_checkpoints),
                "policy_infer_calls": int(total_policy_infer_calls),
                "aux_infer_calls": int(total_aux_infer_calls),
                "sampler_train_episodes": int(total_sampler_train_episodes),
                "sampler_val_episodes": int(total_sampler_val_episodes),
                "aux_update_items": int(total_aux_update_items),
                "aux_update_matched": int(total_aux_update_matched),
                "aux_update_pending": int(total_aux_update_pending),
            },
            "models": models,
            "sessions": session_rows[: max(1, int(max_sessions))],
            "num_sessions_total": len(sessions),
        }

    # -----------------
    # model lifecycle
    # -----------------
    def create_model(
        self,
        *,
        obs_dim: int,
        model_id: Optional[str] = None,
        capacity: int = 200_000,
        validation_capacity: int = 20_000,
        alpha: float = 0.6,
        metadata: Optional[JsonDict] = None,
        learner: Optional[Mapping[str, Any]] = None,
        tensorboard_enabled: bool = True,
        tensorboard_logdir: str = "",
        checkpoint_path: str = "",
    ) -> JsonDict:
        restore_checkpoint = str(checkpoint_path or "").strip()
        restored_spec = None
        restored_learner: Optional[LearnerState] = None
        restored_normalizer: Optional[RunningNormalizer] = None
        restored_metadata: JsonDict = {}

        if restore_checkpoint:
            restored_spec, restored_learner, restored_normalizer, restored_metadata = self._build_restored_learner_state(
                restore_checkpoint,
                learner_overrides=learner,
                checkpoint_root=self._checkpoint_root,
            )
            if obs_dim > 0 and int(obs_dim) != int(restored_spec.obs_dim):
                raise JormungandrServiceError(
                    f"checkpoint obs_dim={restored_spec.obs_dim} does not match requested obs_dim={obs_dim}"
                )
            obs_dim = int(restored_spec.obs_dim)

        if obs_dim <= 0:
            raise JormungandrServiceError("obs_dim must be > 0")
        if capacity <= 0:
            raise JormungandrServiceError("replay capacity must be > 0")
        if validation_capacity <= 0:
            raise JormungandrServiceError("validation capacity must be > 0")
        if not 0.0 <= alpha <= 1.0:
            raise JormungandrServiceError("replay alpha must be in [0, 1]")

        mid = model_id or (str(getattr(restored_spec, "model_id", "")).strip() if restored_spec is not None else "") or uuid.uuid4().hex
        if not mid:
            raise JormungandrServiceError("model_id cannot be empty")

        tb_logdir = ""
        if tensorboard_enabled:
            base = self._resolve_dir(
                tensorboard_logdir,
                root=self._tensorboard_root,
                fallback="./runs/jormungandr",
            )
            tb_logdir = str(base / mid)

        replay = PrioritizedReplayBuffer(capacity=capacity, obs_dim=obs_dim, alpha=alpha)
        validation = PrioritizedReplayBuffer(
            capacity=int(validation_capacity),
            obs_dim=obs_dim,
            alpha=0.0,
        )
        tb = TBLogger(enabled=tensorboard_enabled, logdir=tb_logdir)
        learner_cfg = None if restore_checkpoint else self._parse_learner_config(learner)
        if learner_cfg is not None:
            learner_cfg.checkpoint_dir = str(
                self._resolve_dir(
                    learner_cfg.checkpoint_dir,
                    root=self._checkpoint_root,
                    fallback="./checkpoints/jormungandr",
                )
            )
        learner_state = restored_learner if restore_checkpoint else self._build_learner_state(obs_dim=obs_dim, cfg=learner_cfg)
        merged_metadata = {**restored_metadata, **dict(metadata or {})}

        rec = ModelRecord(
            model_id=mid,
            obs_dim=int(obs_dim),
            capacity=int(capacity),
            validation_capacity=int(validation_capacity),
            alpha=float(alpha),
            created_ts=time.time(),
            metadata=merged_metadata,
            replay=replay,
            validation=validation,
            obs_normalizer=restored_normalizer or RunningNormalizer(dim=int(obs_dim)),
            tb_logger=tb,
            learner=learner_state,
            tensorboard_enabled=bool(tensorboard_enabled),
            tensorboard_logdir=tb_logdir,
            inference_step=0,
            inference_ts=0.0,
            inference_payload={},
        )
        rec.tb_logger.add_custom_scalars(
            {
                "Sampler Compare": {
                    "candidate_reward_mean": [
                        "Multiline",
                        [
                            "sampler_compare/candidate_reward_mean/train",
                            "sampler_compare/candidate_reward_mean/val",
                            "sampler_compare/candidate_reward_mean/test",
                        ],
                    ],
                    "candidate_score": [
                        "Multiline",
                        [
                            "sampler_compare/candidate_score/train",
                            "sampler_compare/candidate_score/val",
                            "sampler_compare/candidate_score/test",
                        ],
                    ],
                    "running_reward_mean": [
                        "Multiline",
                        [
                            "sampler_compare/running_reward_mean/train",
                            "sampler_compare/running_reward_mean/val",
                            "sampler_compare/running_reward_mean/test",
                        ],
                    ],
                    "aux_eval_acc": [
                        "Multiline",
                        [
                            "sampler_compare/candidate_aux_eval_acc/train",
                            "sampler_compare/candidate_aux_eval_acc/val",
                            "sampler_compare/candidate_aux_eval_acc/test",
                        ],
                    ],
                    "aux_eval_nll": [
                        "Multiline",
                        [
                            "sampler_compare/candidate_aux_eval_nll/train",
                            "sampler_compare/candidate_aux_eval_nll/val",
                            "sampler_compare/candidate_aux_eval_nll/test",
                        ],
                    ],
                    "fill_ratio": [
                        "Multiline",
                        [
                            "sampler_compare/candidate_fill_ratio/train",
                            "sampler_compare/candidate_fill_ratio/val",
                            "sampler_compare/candidate_fill_ratio/test",
                        ],
                    ],
                }
            }
        )

        with self._lock:
            if mid in self._models:
                raise JormungandrServiceError(f"model_id already exists: {mid}")
            self._models[mid] = rec
        self._start_model_learner(rec)

        return self.get_model(mid)

    def get_model_record(self, model_id: str) -> ModelRecord:
        with self._lock:
            rec = self._models.get(model_id)
        if rec is None:
            raise KeyError(f"unknown model_id: {model_id}")
        return rec

    def get_model(self, model_id: str) -> JsonDict:
        rec = self.get_model_record(model_id)
        with rec.lock:
            return {
                "model_id": rec.model_id,
                "obs_dim": rec.obs_dim,
                "capacity": rec.capacity,
                "alpha": rec.alpha,
                "created_ts": rec.created_ts,
                "metadata": rec.metadata,
                "tensorboard": {
                    "enabled": rec.tensorboard_enabled,
                    "logdir": rec.tensorboard_logdir,
                },
                "replay": {
                    "size": int(len(rec.replay)),
                    "capacity": rec.capacity,
                    "alpha": rec.alpha,
                },
                "validation": {
                    "size": int(len(rec.validation)),
                    "capacity": rec.validation_capacity,
                },
                "experience": {
                    "train_items": int(rec.train_add_items),
                    "validation_items": int(rec.validation_add_items),
                },
                "inference": {
                    "step": rec.inference_step,
                    "ts": rec.inference_ts,
                    "has_payload": bool(rec.inference_payload),
                },
                "policy_infer_calls": int(rec.policy_infer_calls),
                "aux_infer_calls": int(rec.aux_infer_calls),
                "performance_metrics": dict(rec.latest_logged_metrics),
                "obs_normalizer": self._normalizer_summary_locked(rec),
                "num_sessions": len(rec.session_ids),
                "learner": self._learner_summary(rec.learner),
            }

    def list_models(self) -> List[JsonDict]:
        with self._lock:
            ids = list(self._models.keys())
        return [self.get_model(mid) for mid in ids]

    @staticmethod
    def list_algorithm_plugins() -> List[JsonDict]:
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "checkpoint_id": plugin.checkpoint_id,
                "family": plugin.family,
                "backend": plugin.backend,
                "replay_mode": plugin.replay_mode,
                "enforce_policy_lag": bool(plugin.enforce_policy_lag),
                "runtime_defaults": dict(plugin.runtime_defaults),
                "default_export_module": plugin.default_export_module,
                "description": plugin.description,
                "noise_profile": plugin.noise_profile,
                "aliases": list(plugin.aliases),
            }
            for plugin in algorithm_registry.plugins()
        ]

    def compare_models(self) -> JsonDict:
        """Return aligned latest learner metrics for internal experiment tools."""

        rows: List[JsonDict] = []
        with self._lock:
            records = list(self._models.values())
        for rec in records:
            with rec.lock:
                learner = rec.learner
                if learner is None:
                    continue
                rows.append(
                    {
                        "model_id": rec.model_id,
                        "algo": learner.cfg.algo,
                        "plugin_version": (
                            learner.plugin.version if learner.plugin is not None else ""
                        ),
                        "updates": int(learner.updates),
                        "policy_version": int(learner.policy_version),
                        "replay_size": int(len(rec.replay)),
                        "validation_size": int(len(rec.validation)),
                        "metrics": dict(learner.last_metrics),
                        "performance_metrics": dict(rec.latest_logged_metrics),
                        "validation": {
                            "loss": float(learner.last_validation_loss),
                            "td_abs_mean": float(learner.last_validation_td_abs),
                            "runs": int(learner.validation_runs),
                        },
                        "selector": {
                            "name": learner.cfg.replay_selector,
                            "metrics": dict(learner.last_selector_metrics),
                        },
                    }
                )
        algorithm_metric_names = sorted(
            {
                str(name)
                for row in rows
                for name in (row.get("metrics") or {}).keys()
            }
        )
        performance_metric_names = sorted(
            {
                str(name)
                for row in rows
                for name in (row.get("performance_metrics") or {}).keys()
            }
        )
        return {
            "models": rows,
            "metric_names": algorithm_metric_names,
            "algorithm_metric_names": algorithm_metric_names,
            "performance_metric_names": performance_metric_names,
        }

    def get_training_metrics(self, model_id: str) -> JsonDict:
        rec = self.get_model_record(model_id)
        with rec.lock:
            learner = rec.learner
            if learner is None:
                return {"model_id": model_id, "history": []}
            return {
                "model_id": model_id,
                "algo": learner.cfg.algo,
                "history": list(learner.metric_history),
                "performance_history": list(rec.logged_metric_history),
                "latest_performance_metrics": dict(rec.latest_logged_metrics),
                "last_selector_audit": dict(learner.last_selector_audit),
            }

    def delete_model(self, model_id: str) -> JsonDict:
        rec = self.get_model_record(model_id)

        with rec.lock:
            session_ids = list(rec.session_ids)
        for sid in session_ids:
            with self._lock:
                srec = self._sessions.pop(sid, None)
            if srec is None:
                continue
            with srec.lock:
                try:
                    self._episode_call(srec, srec.episode.close)
                except Exception:
                    pass
            with rec.lock:
                rec.session_ids.discard(sid)

        self._stop_model_learner(rec)

        with self._lock:
            removed = self._models.pop(model_id, None)
        if removed is None:
            raise KeyError(f"unknown model_id: {model_id}")

        rec.tb_logger.flush()
        rec.tb_logger.close()
        return {"model_id": model_id, "deleted": True}

    # -----------------
    # split-aware experience
    # -----------------
    @staticmethod
    def _canonical_experience_split(value: Any) -> str:
        split = str(value or "train").strip().lower()
        if split == "val":
            split = "validation"
        if split not in {"train", "validation"}:
            raise JormungandrServiceError("split must be 'train' or 'validation'")
        return split

    @classmethod
    def _experience_step_key(
        cls,
        split: Any,
        actor_id: Any,
        episode_id: Any,
        timestep: Any,
    ) -> Optional[str]:
        if episode_id is None or timestep is None:
            return None
        split_name = cls._canonical_experience_split(split)
        actor = str(actor_id or "").strip()
        eid = str(episode_id).strip()
        if not eid:
            return None
        try:
            ts = int(timestep)
        except Exception:
            return None
        return f"{split_name}:{actor}:{eid}:{ts}"

    @classmethod
    def _replay_step_key(cls, episode_id: Any, timestep: Any) -> Optional[str]:
        """Return the compatibility key used by pre-Jörmungandr clients."""
        return cls._experience_step_key("train", "", episode_id, timestep)

    @staticmethod
    def _normalize_aux_payload(item: Mapping[str, Any]) -> Optional[JsonDict]:
        aux = item.get("aux")
        if isinstance(aux, Mapping):
            return dict(aux)
        if "aux_label" in item:
            out: JsonDict = {"label": int(item.get("aux_label"))}
            if "aux_score" in item:
                out["score"] = float(item.get("aux_score"))
            return out
        return None

    @staticmethod
    def _action_index_locked(rec: ModelRecord, item: Mapping[str, Any]) -> float:
        learner = rec.learner
        action_values = list(learner.cfg.action_values) if learner is not None else []
        if "action_idx" in item:
            idx = int(item["action_idx"])
            if action_values and not 0 <= idx < len(action_values):
                raise JormungandrServiceError(
                    f"action_idx must be in [0, {len(action_values) - 1}]"
                )
            return float(idx)

        if "action" not in item:
            raise JormungandrServiceError("experience requires action_idx or action")
        action = float(item["action"])
        if not np.isfinite(action):
            raise JormungandrServiceError("action must be finite")
        if not action_values:
            return action

        matches = [
            idx
            for idx, value in enumerate(action_values)
            if bool(np.isclose(action, float(value), rtol=0.0, atol=1e-7))
        ]
        if len(matches) != 1:
            raise JormungandrServiceError(
                "action does not match exactly one configured action value; send action_idx"
            )
        return float(matches[0])

    @staticmethod
    def _legal_action_mask_locked(
        rec: ModelRecord,
        raw: Any,
        *,
        field_name: str,
    ) -> Optional[List[bool]]:
        if raw is None:
            return None
        learner = rec.learner
        action_values = list(learner.cfg.action_values) if learner is not None else []
        if not action_values:
            raise JormungandrServiceError(
                f"{field_name} requires a learner with a fixed action vocabulary"
            )
        if not isinstance(raw, (list, tuple)):
            raise JormungandrServiceError(f"{field_name} must be a Boolean array")
        mask = [bool(value) for value in raw]
        if len(mask) != len(action_values):
            raise JormungandrServiceError(
                f"{field_name} must contain {len(action_values)} entries"
            )
        if not any(mask):
            raise JormungandrServiceError(f"{field_name} must admit at least one action")
        return mask

    def experience_add(
        self,
        model_id: str,
        items: List[JsonDict],
        *,
        require_identity: bool = True,
    ) -> JsonDict:
        rec = self.get_model_record(model_id)
        if not isinstance(items, list) or not items:
            raise JormungandrServiceError("items must be a non-empty array")

        added_by_split = {"train": 0, "validation": 0}
        with rec.lock:
            for item in items:
                if not isinstance(item, Mapping):
                    raise JormungandrServiceError("each experience item must be an object")
                split = self._canonical_experience_split(item.get("split", "train"))
                if require_identity:
                    missing = [
                        key
                        for key in ("actor_id", "episode_id", "timestep", "policy_version")
                        if item.get(key) is None
                    ]
                    if missing:
                        raise JormungandrServiceError(
                            f"experience item is missing required fields: {', '.join(missing)}"
                        )
                obs = self._sanitize_obs_array(np.asarray(item.get("obs", []), dtype=np.float32))
                next_obs = self._sanitize_obs_array(np.asarray(item.get("next_obs", []), dtype=np.float32))
                if obs.shape != (rec.obs_dim,) or next_obs.shape != (rec.obs_dim,):
                    raise JormungandrServiceError(f"obs and next_obs must have shape [{rec.obs_dim}]")
                action = self._action_index_locked(rec, item)
                action_mask = self._legal_action_mask_locked(
                    rec,
                    item.get("action_mask"),
                    field_name="action_mask",
                )
                next_action_mask = self._legal_action_mask_locked(
                    rec,
                    item.get("next_action_mask"),
                    field_name="next_action_mask",
                )
                if action_mask is not None and not action_mask[int(action)]:
                    raise JormungandrServiceError(
                        "experience action_idx is illegal under action_mask"
                    )
                reward = float(item.get("reward", 0.0))
                if not np.isfinite(reward):
                    raise JormungandrServiceError("reward must be finite")
                done = float(bool(item.get("done", False)))

                if split == "train":
                    store = rec.replay
                    meta_by_idx = rec.replay_meta_by_idx
                    key_by_idx = rec.replay_key_by_idx
                    idx_by_key = rec.replay_idx_by_key
                    aux_by_idx = rec.replay_aux_by_idx
                else:
                    store = rec.validation
                    meta_by_idx = rec.validation_meta_by_idx
                    key_by_idx = rec.validation_key_by_idx
                    idx_by_key = rec.validation_idx_by_key
                    aux_by_idx = rec.validation_aux_by_idx

                idx = store.pos
                old_key = key_by_idx.pop(idx, None)
                if old_key is not None and idx_by_key.get(old_key) == idx:
                    del idx_by_key[old_key]
                meta_by_idx.pop(idx, None)
                aux_by_idx.pop(idx, None)

                # Validation data never influences learner preprocessing state.
                if split == "train":
                    rec.obs_normalizer.update(obs)
                    rec.obs_normalizer.update(next_obs)
                store.add(obs, action, reward, next_obs, done)

                priority = item.get("priority")
                if split == "train" and priority is not None:
                    p = float(priority)
                    if np.isfinite(p) and p > 0.0:
                        store.update_priorities(
                            np.array([idx]),
                            np.array([p], dtype=np.float32),
                        )

                meta: JsonDict = {"split": split}
                actor_id = item.get("actor_id")
                if actor_id is not None:
                    meta["actor_id"] = str(actor_id)
                session_id = item.get("session_id")
                if session_id is not None:
                    meta["session_id"] = str(session_id)
                episode_id = item.get("episode_id")
                if episode_id is not None:
                    meta["episode_id"] = str(episode_id)
                timestep = item.get("timestep")
                if timestep is not None:
                    try:
                        meta["timestep"] = int(timestep)
                    except Exception:
                        pass
                ts_ns = item.get("ts_ns")
                if ts_ns is not None:
                    try:
                        meta["ts_ns"] = int(ts_ns)
                    except Exception:
                        pass
                policy_version = item.get("policy_version")
                if policy_version is not None:
                    try:
                        meta["policy_version"] = int(policy_version)
                    except Exception:
                        pass
                behavior_logp_raw = item.get(
                    "behavior_logp", item.get("log_probability")
                )
                if behavior_logp_raw is not None:
                    behavior_logp = float(behavior_logp_raw)
                    if not np.isfinite(behavior_logp):
                        raise JormungandrServiceError("behavior_logp must be finite")
                    meta["behavior_logp"] = behavior_logp
                behavior_value_raw = item.get(
                    "behavior_value", item.get("value")
                )
                if behavior_value_raw is not None:
                    behavior_value = float(behavior_value_raw)
                    if not np.isfinite(behavior_value):
                        raise JormungandrServiceError("behavior_value must be finite")
                    meta["behavior_value"] = behavior_value
                if action_mask is not None:
                    meta["action_mask"] = list(action_mask)
                if next_action_mask is not None:
                    meta["next_action_mask"] = list(next_action_mask)
                extra_meta = item.get("meta")
                if isinstance(extra_meta, Mapping):
                    for k, v in extra_meta.items():
                        key_text = str(k)
                        if key_text in {
                            "split",
                            "actor_id",
                            "episode_id",
                            "timestep",
                            "policy_version",
                            "behavior_logp",
                            "behavior_value",
                            "action_mask",
                            "next_action_mask",
                        }:
                            continue
                        meta[key_text] = v
                meta_by_idx[idx] = meta

                step_key = self._experience_step_key(
                    split,
                    meta.get("actor_id"),
                    meta.get("episode_id"),
                    meta.get("timestep"),
                )
                if step_key is not None:
                    key_by_idx[idx] = step_key
                    idx_by_key[step_key] = idx
                    pending_aux = rec.replay_aux_pending_by_key.pop(step_key, None)
                    if pending_aux is not None:
                        aux_by_idx[idx] = dict(pending_aux)

                aux_payload = self._normalize_aux_payload(item)
                if aux_payload is not None:
                    aux_by_idx[idx] = aux_payload

                added_by_split[split] += 1
            rec.replay_add_calls += 1
            added = sum(added_by_split.values())
            rec.replay_add_items += int(added)
            rec.train_add_items += int(added_by_split["train"])
            rec.validation_add_items += int(added_by_split["validation"])

        return {
            "model_id": model_id,
            "added": added,
            "added_by_split": dict(added_by_split),
            "replay": {"size": int(len(rec.replay)), "capacity": rec.capacity, "alpha": rec.alpha},
            "validation": {
                "size": int(len(rec.validation)),
                "capacity": rec.validation_capacity,
            },
        }

    def replay_add(self, model_id: str, items: List[JsonDict]) -> JsonDict:
        """Compatibility alias for the original unversioned replay ingress."""
        return self.experience_add(model_id, items, require_identity=False)

    def replay_sample(self, model_id: str, batch_size: int, beta: float) -> JsonDict:
        rec = self.get_model_record(model_id)
        with rec.lock:
            rec.replay_sample_calls += 1
            if len(rec.replay) == 0:
                return {
                    "model_id": model_id,
                    "batch": [],
                    "idxs": [],
                    "weights": [],
                }
            bs = max(1, min(int(batch_size), len(rec.replay)))
            b = float(beta)
            batch, idxs, weights = rec.replay.sample(bs, b)
            obs, action, reward, next_obs, done = batch
            action_values = (
                list(rec.learner.cfg.action_values)
                if rec.learner is not None
                else []
            )
            idx_arr = np.asarray(idxs, dtype=np.int64)
            meta_subset: Dict[int, JsonDict] = {}
            aux_subset: Dict[int, JsonDict] = {}
            for j in idx_arr.tolist():
                jj = int(j)
                mm = rec.replay_meta_by_idx.get(jj)
                if mm is not None:
                    meta_subset[jj] = dict(mm)
                aa = rec.replay_aux_by_idx.get(jj)
                if aa is not None:
                    aux_subset[jj] = dict(aa)

        rows = []
        for i in range(obs.shape[0]):
            idx_i = int(idx_arr[i])
            action_idx = int(action[i][0])
            rows.append(
                {
                    "obs": obs[i].tolist(),
                    "action_idx": action_idx,
                    "action": (
                        float(action_values[action_idx])
                        if 0 <= action_idx < len(action_values)
                        else float(action[i][0])
                    ),
                    "reward": float(reward[i][0]),
                    "next_obs": next_obs[i].tolist(),
                    "done": bool(done[i][0] >= 0.5),
                }
            )
            meta = meta_subset.get(idx_i)
            if meta is not None:
                rows[-1]["meta"] = meta
                if "split" in meta:
                    rows[-1]["split"] = meta.get("split")
                if "actor_id" in meta:
                    rows[-1]["actor_id"] = meta.get("actor_id")
                if "episode_id" in meta:
                    rows[-1]["episode_id"] = meta.get("episode_id")
                if "timestep" in meta:
                    rows[-1]["timestep"] = meta.get("timestep")
                if "ts_ns" in meta:
                    rows[-1]["ts_ns"] = meta.get("ts_ns")
                if "policy_version" in meta:
                    rows[-1]["policy_version"] = meta.get("policy_version")
            aux = aux_subset.get(idx_i)
            if aux is not None:
                rows[-1]["aux"] = aux

        return {
            "model_id": model_id,
            "batch": rows,
            "idxs": [int(x) for x in np.asarray(idxs).tolist()],
            "weights": [float(x) for x in np.asarray(weights).tolist()],
        }

    def experience_aux_update(self, model_id: str, updates: List[JsonDict]) -> JsonDict:
        rec = self.get_model_record(model_id)
        if not isinstance(updates, list) or not updates:
            raise JormungandrServiceError("updates must be a non-empty array")

        matched = 0
        pending = 0
        with rec.lock:
            for upd in updates:
                if not isinstance(upd, Mapping):
                    continue
                key = upd.get("key")
                if key is None:
                    key = self._experience_step_key(
                        upd.get("split", "train"),
                        upd.get("actor_id"),
                        upd.get("episode_id"),
                        upd.get("timestep"),
                    )
                key_txt = str(key).strip() if key is not None else ""
                if not key_txt:
                    continue
                aux = self._normalize_aux_payload(upd)
                if aux is None:
                    continue

                split = self._canonical_experience_split(upd.get("split", "train"))
                if split == "train":
                    idx_by_key = rec.replay_idx_by_key
                    aux_by_idx = rec.replay_aux_by_idx
                else:
                    idx_by_key = rec.validation_idx_by_key
                    aux_by_idx = rec.validation_aux_by_idx
                idx = idx_by_key.get(key_txt)
                if idx is not None:
                    aux_by_idx[int(idx)] = dict(aux)
                    matched += 1
                else:
                    rec.replay_aux_pending_by_key[key_txt] = dict(aux)
                    pending += 1

            rec.aux_update_calls += 1
            rec.aux_update_items += int(len(updates))
            rec.aux_update_matched += int(matched)
            rec.aux_update_pending += int(pending)

        return {
            "model_id": model_id,
            "updates": int(len(updates)),
            "matched": int(matched),
            "pending": int(pending),
        }

    def replay_aux_update(self, model_id: str, updates: List[JsonDict]) -> JsonDict:
        """Compatibility alias for delayed auxiliary labels."""
        return self.experience_aux_update(model_id, updates)

    def replay_update_priorities(self, model_id: str, idxs: List[int], priorities: List[float]) -> JsonDict:
        rec = self.get_model_record(model_id)
        if not idxs or not priorities or len(idxs) != len(priorities):
            raise JormungandrServiceError("idxs and priorities must be non-empty arrays with equal length")

        with rec.lock:
            rec.replay_priority_update_calls += 1
            rec.replay.update_priorities(
                np.asarray(idxs, dtype=np.int64),
                np.asarray(priorities, dtype=np.float32),
            )

        return {
            "model_id": model_id,
            "updated": len(idxs),
            "replay": {"size": int(len(rec.replay)), "capacity": rec.capacity, "alpha": rec.alpha},
        }

    # -----------------
    # metrics
    # -----------------
    def log_metrics(self, model_id: str, step: int, metrics: Mapping[str, Any]) -> JsonDict:
        rec = self.get_model_record(model_id)
        n = 0
        logged_scalars: JsonDict = {}
        has_train_episode = False
        has_val_episode = False
        for k in metrics.keys():
            key = str(k)
            if key.startswith("sampler/train/episode_"):
                has_train_episode = True
            elif key.startswith("sampler/val/episode_") or key.startswith(
                "sampler/validation/episode_"
            ):
                has_val_episode = True
        with rec.lock:
            rec.metrics_calls += 1
            for k, v in metrics.items():
                try:
                    tag = str(k)
                    split_tag = self._split_metric_tag(tag)
                    hist_arr = self._metric_hist_array(v)
                    if hist_arr is not None:
                        if split_tag is not None:
                            split, normalized_tag = split_tag
                            rec.tb_logger.add_split_hist(split, normalized_tag, hist_arr, int(step))
                        else:
                            rec.tb_logger.add_hist(tag, hist_arr, int(step))
                        n += 1
                        continue
                    value = float(v)
                    if not np.isfinite(value):
                        continue
                    if split_tag is not None:
                        split, normalized_tag = split_tag
                        rec.tb_logger.add_split(split, normalized_tag, value, int(step))
                        logged_scalars[f"{split}/{normalized_tag}"] = value
                    else:
                        rec.tb_logger.add(tag, value, int(step))
                        logged_scalars[tag] = value
                    n += 1
                except Exception:
                    continue
            rec.metrics_points += int(n)
            if has_train_episode:
                rec.sampler_train_episodes += 1
            if has_val_episode:
                rec.sampler_val_episodes += 1
            rec.last_metrics_step = int(step)
            rec.last_metrics_ts = time.time()
            rec.latest_logged_metrics.update(logged_scalars)
            if logged_scalars:
                rec.logged_metric_history.append(
                    {
                        "step": int(step),
                        "ts": float(rec.last_metrics_ts),
                        "metrics": dict(logged_scalars),
                    }
                )
            rec.tb_logger.flush()
        return {"model_id": model_id, "step": int(step), "logged": n}

    @staticmethod
    def _metric_hist_array(value: Any) -> Optional[np.ndarray]:
        if not isinstance(value, (list, tuple)):
            return None
        if not value:
            return None
        vals: List[float] = []
        for x in value:
            try:
                fx = float(x)
            except Exception:
                continue
            if np.isfinite(fx):
                vals.append(float(fx))
        if not vals:
            return None
        return np.asarray(vals, dtype=np.float32)

    @staticmethod
    def _split_metric_tag(tag: str) -> Optional[tuple[str, str]]:
        parts = str(tag).split("/")
        if len(parts) == 3 and parts[0] == "sampler_compare":
            metric = parts[1]
            split = parts[2]
            if split in {"train", "val", "validation", "test"}:
                split = "validation" if split == "val" else split
                if "aux_eval_acc" in metric:
                    return split, "aux/eval_acc"
                if "aux_eval_nll" in metric:
                    return split, "aux/eval_nll"
                if "reward_mean" in metric:
                    return split, "rl/reward_mean"
                if "reward_sum" in metric:
                    return split, "rl/reward_sum"
                if "score" in metric:
                    return split, "rl/score"
                if "ic_1step" in metric:
                    return split, "rl/ic_1step"
                if "fill_ratio" in metric:
                    return split, "execution/fill_ratio"
                return split, f"sampler/{metric}"
        if len(parts) >= 3 and parts[0] == "sampler" and parts[1] in {
            "train",
            "val",
            "validation",
            "test",
        }:
            split = "validation" if parts[1] == "val" else parts[1]
            metric = "/".join(parts[2:])
            return split, f"sampler/{metric}"
        return None

    # -----------------
    # inference snapshot
    # -----------------
    def publish_inference(
        self,
        model_id: str,
        *,
        inference: Mapping[str, Any],
        step: Optional[int] = None,
    ) -> JsonDict:
        rec = self.get_model_record(model_id)
        if not isinstance(inference, Mapping):
            raise JormungandrServiceError("inference must be an object")
        with rec.lock:
            if step is not None:
                rec.inference_step = int(step)
            rec.inference_ts = time.time()
            rec.inference_payload = dict(inference)
            rec.inference_updates += 1
            out = {
                "model_id": model_id,
                "step": rec.inference_step,
                "ts": rec.inference_ts,
                "inference": rec.inference_payload,
            }
        return out

    def get_inference(self, model_id: str) -> JsonDict:
        rec = self.get_model_record(model_id)
        with rec.lock:
            rec.inference_reads += 1
            return {
                "model_id": model_id,
                "step": rec.inference_step,
                "ts": rec.inference_ts,
                "inference": dict(rec.inference_payload),
            }

    # -----------------
    # learner policy
    # -----------------
    def get_policy(self, model_id: str) -> JsonDict:
        rec = self.get_model_record(model_id)
        with rec.lock:
            return {
                "model_id": model_id,
                "obs_dim": int(rec.obs_dim),
                "obs_normalizer": self._normalizer_summary_locked(rec),
                "learner": self._learner_summary(rec.learner),
            }

    @staticmethod
    def _finite_float(v: Any) -> float:
        try:
            x = float(v)
        except Exception:
            return 0.0
        if not np.isfinite(x):
            return 0.0
        return float(x)

    @staticmethod
    def _feature_keys_locked(rec: ModelRecord, features: Mapping[str, Any]) -> List[str]:
        raw_fk = rec.metadata.get("feature_keys")
        keys: List[str] = []
        if isinstance(raw_fk, list):
            keys = [str(k) for k in raw_fk]
        incoming = sorted(str(k) for k in features.keys())
        if not keys:
            keys = incoming[: int(rec.obs_dim)]
        else:
            seen = set(keys)
            for k in incoming:
                if len(keys) >= int(rec.obs_dim):
                    break
                if k in seen:
                    continue
                keys.append(k)
                seen.add(k)
        if len(keys) > int(rec.obs_dim):
            keys = keys[: int(rec.obs_dim)]
        rec.metadata["feature_keys"] = list(keys)
        return keys

    def _obs_from_features_locked(self, rec: ModelRecord, features: Mapping[str, Any]) -> tuple[np.ndarray, List[str]]:
        keys = self._feature_keys_locked(rec, features)
        obs_np = np.zeros((int(rec.obs_dim),), dtype=np.float32)
        for i, k in enumerate(keys):
            if i >= int(rec.obs_dim):
                break
            obs_np[i] = np.float32(self._finite_float(features.get(k)))
        return obs_np, keys

    @staticmethod
    def _aux_score_from_probs(probs: np.ndarray) -> float:
        p = np.asarray(probs, dtype=np.float64).reshape(-1)
        if p.size >= 3:
            return float(p[2] - p[0])
        if p.size == 2:
            return float(p[1] - p[0])
        if p.size == 1:
            return float(p[0])
        return 0.0

    def aux_infer(
        self,
        model_id: str,
        *,
        obs: Optional[List[Any]] = None,
        features: Optional[Mapping[str, Any]] = None,
        deterministic: bool = True,
    ) -> JsonDict:
        _ = bool(deterministic)  # reserved for future stochastic aux heads
        rec = self.get_model_record(model_id)

        with rec.lock:
            learner = rec.learner
            if learner is None or not learner.cfg.enabled or learner.agent is None:
                raise JormungandrServiceError("learner is not enabled for this model")
            if getattr(learner.agent, "aux_head", None) is None:
                raise JormungandrServiceError("aux head is not enabled for this model")

            feature_keys: List[str] = []
            if obs is not None:
                obs_np = self._sanitize_obs_array(np.asarray(obs, dtype=np.float32))
                if obs_np.shape != (rec.obs_dim,):
                    raise JormungandrServiceError(f"obs must have shape [{rec.obs_dim}]")
                raw_fk = rec.metadata.get("feature_keys")
                if isinstance(raw_fk, list):
                    feature_keys = [str(k) for k in raw_fk]
            else:
                if not isinstance(features, Mapping):
                    raise JormungandrServiceError("features must be an object when obs is not provided")
                obs_np, feature_keys = self._obs_from_features_locked(rec, features)
            obs_np = self._normalize_obs_locked(rec, obs_np)

        import torch

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=learner.agent.device).unsqueeze(0)
        with learner.agent_lock:
            with torch.no_grad():
                logits_t = learner.agent.aux_head(obs_t)
                probs_t = torch.softmax(logits_t, dim=-1)
                logits = logits_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
                probs = probs_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
                pred_class = int(np.argmax(probs)) if probs.size > 0 else 0
                score = self._aux_score_from_probs(probs)

        with rec.lock:
            rec.aux_infer_calls += 1
            out: JsonDict = {
                "model_id": model_id,
                "ts": time.time(),
                "obs_dim": int(rec.obs_dim),
                "feature_keys": feature_keys,
                "aux": {
                    "pred_class": int(pred_class),
                    "score": float(score),
                    "probs": [float(x) for x in probs.tolist()],
                    "logits": [float(x) for x in logits.tolist()],
                    "outputs": [
                        {
                            "name": "aux_main",
                            "type": "classification",
                            "pred_class": int(pred_class),
                            "score": float(score),
                            "probs": [float(x) for x in probs.tolist()],
                            "logits": [float(x) for x in logits.tolist()],
                        }
                    ],
                },
            }
            # Backward-compatible top-level aliases.
            out["aux_pred"] = int(pred_class)
            out["aux_probs"] = [float(x) for x in probs.tolist()]
            out["aux_score"] = float(score)
            return out

    def policy_infer(
        self,
        model_id: str,
        *,
        obs: List[Any],
        deterministic: bool = True,
        epsilon: float = 0.0,
        action_mask: Optional[List[Any]] = None,
    ) -> JsonDict:
        rec = self.get_model_record(model_id)
        obs_np = self._sanitize_obs_array(np.asarray(obs, dtype=np.float32))
        if obs_np.shape != (rec.obs_dim,):
            raise JormungandrServiceError(f"obs must have shape [{rec.obs_dim}]")

        with rec.lock:
            learner = rec.learner
            if learner is None or not learner.cfg.enabled or learner.agent is None:
                raise JormungandrServiceError("learner is not enabled for this model")
            obs_model_np = self._normalize_obs_locked(rec, obs_np)
            validated_mask = self._legal_action_mask_locked(
                rec, action_mask, field_name="action_mask"
            )
            import torch

            obs_t = torch.tensor(obs_model_np, dtype=torch.float32, device=learner.agent.device).unsqueeze(0)
            with learner.agent_lock:
                result = learner.agent.action_result(
                    obs_model_np,
                    epsilon=max(0.0, float(epsilon)),
                    deterministic=bool(deterministic),
                    action_mask=validated_mask,
                )
                with torch.no_grad():
                    aux_probs = None
                    aux_pred = None
                    if getattr(learner.agent, "aux_head", None) is not None:
                        logits = learner.agent.aux_head(obs_t)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
                        aux_probs = [float(x) for x in probs.tolist()]
                        aux_pred = int(np.argmax(probs))
            rec.policy_infer_calls += 1
            out: JsonDict = {
                "model_id": model_id,
                "ts": time.time(),
                "algo": learner.cfg.algo,
                "action": float(result.action),
                "action_idx": int(result.action_idx),
                "action_values": list(learner.cfg.action_values),
                "policy_version": int(learner.policy_version),
                "updates": int(learner.updates),
            }
            out.update(dict(result.extras))
            if result.log_probability is not None:
                out["log_probability"] = float(result.log_probability)
                out["behavior_logp"] = float(result.log_probability)
            if result.value is not None:
                out["value"] = float(result.value)
                out["behavior_value"] = float(result.value)
            if validated_mask is not None:
                out["action_mask"] = list(validated_mask)
            if aux_probs is not None:
                out["aux_probs"] = aux_probs
                out["aux_pred"] = int(aux_pred) if aux_pred is not None else None
            return out

    def policy_infer_batch(
        self,
        model_id: str,
        *,
        obs_batch: List[Any],
        deterministic: bool = True,
        epsilon: float = 0.0,
        action_masks: Optional[List[Any]] = None,
    ) -> JsonDict:
        rec = self.get_model_record(model_id)
        obs_np = self._sanitize_obs_array(np.asarray(obs_batch, dtype=np.float32))
        if obs_np.ndim != 2 or obs_np.shape[1] != rec.obs_dim:
            raise JormungandrServiceError(f"obs_batch must have shape [N, {rec.obs_dim}]")
        if obs_np.shape[0] < 1:
            raise JormungandrServiceError("obs_batch must contain at least one observation")

        with rec.lock:
            learner = rec.learner
            if learner is None or not learner.cfg.enabled or learner.agent is None:
                raise JormungandrServiceError("learner is not enabled for this model")
            obs_model_np = np.stack([self._normalize_obs_locked(rec, row) for row in obs_np], axis=0)
            validated_masks = None
            if action_masks is not None:
                if not isinstance(action_masks, list) or len(action_masks) != obs_np.shape[0]:
                    raise JormungandrServiceError(
                        "action_masks must contain one mask per observation"
                    )
                validated_masks = np.asarray(
                    [
                        self._legal_action_mask_locked(
                            rec, mask, field_name="action_masks[]"
                        )
                        for mask in action_masks
                    ],
                    dtype=np.bool_,
                )
            import torch

            obs_t = torch.tensor(obs_model_np, dtype=torch.float32, device=learner.agent.device)
            with learner.agent_lock:
                results = learner.agent.inference_batch(
                    obs_model_np,
                    deterministic=bool(deterministic),
                    epsilon=max(0.0, float(epsilon)),
                    action_masks=validated_masks,
                )
                with torch.no_grad():
                    aux_probs_arr = None
                    aux_preds = None
                    if getattr(learner.agent, "aux_head", None) is not None:
                        logits = learner.agent.aux_head(obs_t)
                        probs_t = torch.softmax(logits, dim=-1)
                        aux_probs_arr = probs_t.detach().cpu().numpy()
                        aux_preds = np.argmax(aux_probs_arr, axis=1).astype(int)

            items: List[JsonDict] = []
            action_values = list(learner.cfg.action_values)
            for i, result in enumerate(results):
                item: JsonDict = {
                    "action": float(result.action),
                    "action_idx": int(result.action_idx),
                }
                item.update(dict(result.extras))
                if result.log_probability is not None:
                    item["log_probability"] = float(result.log_probability)
                    item["behavior_logp"] = float(result.log_probability)
                if result.value is not None:
                    item["value"] = float(result.value)
                    item["behavior_value"] = float(result.value)
                if validated_masks is not None:
                    item["action_mask"] = [bool(x) for x in validated_masks[i].tolist()]
                if aux_probs_arr is not None:
                    item["aux_probs"] = [float(x) for x in aux_probs_arr[i].tolist()]
                    item["aux_pred"] = int(aux_preds[i]) if aux_preds is not None else None
                items.append(item)

            rec.policy_infer_calls += 1
            return {
                "model_id": model_id,
                "ts": time.time(),
                "algo": learner.cfg.algo,
                "items": items,
                "batch_size": int(len(items)),
                "action_values": action_values,
                "policy_version": int(learner.policy_version),
                "updates": int(learner.updates),
            }

    def force_policy_checkpoint(self, model_id: str) -> JsonDict:
        rec = self.get_model_record(model_id)
        with rec.lock:
            learner = rec.learner
            if learner is None or not learner.cfg.enabled or learner.agent is None:
                raise JormungandrServiceError("learner is not enabled for this model")
        self._save_checkpoint(rec, learner)
        with rec.lock:
            return {
                "model_id": model_id,
                "checkpoint": learner.last_checkpoint,
                "checkpoint_count": int(learner.checkpoint_count),
                "updates": int(learner.updates),
                "policy_version": int(learner.policy_version),
            }

    # -----------------
    # session lifecycle
    # -----------------
    @staticmethod
    def _build_factory(driver: str, driver_config: Mapping[str, Any]) -> EpisodeFactory:
        if driver == "ctypes":
            lib_path = str(driver_config.get("lib_path", "")).strip()
            if not lib_path:
                raise JormungandrServiceError("driver_config.lib_path is required for driver=ctypes")
            api_prefix = str(driver_config.get("api_prefix", "episode"))
            return CtypesEpisodeFactory(lib_path=lib_path, api_prefix=api_prefix)

        if driver == "subprocess-json":
            command = driver_config.get("command")
            if isinstance(command, str):
                cmd = shlex.split(command)
            elif isinstance(command, list):
                cmd = [str(x) for x in command]
            else:
                cmd = []
            if not cmd:
                raise JormungandrServiceError("driver_config.command is required for driver=subprocess-json")
            cwd = driver_config.get("cwd")
            env = driver_config.get("env")
            if env is not None and not isinstance(env, dict):
                raise JormungandrServiceError("driver_config.env must be an object")
            return SubprocessEpisodeFactory(command=cmd, cwd=str(cwd) if cwd else None, env=env)

        raise JormungandrServiceError(f"unsupported driver: {driver}")

    def create_session(
        self,
        model_id: str,
        *,
        driver: str,
        driver_config: Mapping[str, Any],
        episode_config: Mapping[str, Any],
        metadata: Optional[JsonDict] = None,
        session_id: Optional[str] = None,
    ) -> JsonDict:
        rec = self.get_model_record(model_id)
        sid = session_id or uuid.uuid4().hex

        factory = self._build_factory(driver, driver_config)
        if str(driver).lower() == "ctypes":
            with self._ctypes_call_lock:
                episode = factory.create(dict(episode_config))
        else:
            episode = factory.create(dict(episode_config))

        srec = SessionRecord(
            session_id=sid,
            model_id=model_id,
            created_ts=time.time(),
            driver=driver,
            driver_config=dict(driver_config),
            episode_config=dict(episode_config),
            metadata=dict(metadata or {}),
            episode=episode,
        )

        with self._lock:
            if sid in self._sessions:
                raise JormungandrServiceError(f"session_id already exists: {sid}")
            self._sessions[sid] = srec

        with rec.lock:
            rec.session_ids.add(sid)

        return self.get_session(model_id, sid)

    def get_session_record(self, model_id: str, session_id: str) -> SessionRecord:
        _ = self.get_model_record(model_id)
        with self._lock:
            srec = self._sessions.get(session_id)
        if srec is None or srec.model_id != model_id:
            raise KeyError(f"unknown session_id for model {model_id}: {session_id}")
        return srec

    def get_session(self, model_id: str, session_id: str) -> JsonDict:
        srec = self.get_session_record(model_id, session_id)
        with srec.lock:
            remaining = self._episode_call(srec, srec.episode.remaining_steps)
            is_terminal = self._episode_call(srec, srec.episode.is_terminal)
            return {
                "session_id": srec.session_id,
                "model_id": srec.model_id,
                "created_ts": srec.created_ts,
                "driver": srec.driver,
                "metadata": srec.metadata,
                "is_terminal": bool(is_terminal),
                "remaining_steps": int(remaining) if remaining is not None else None,
            }

    def list_sessions(self, model_id: str) -> List[JsonDict]:
        rec = self.get_model_record(model_id)
        with rec.lock:
            ids = list(rec.session_ids)
        return [self.get_session(model_id, sid) for sid in ids]

    def initialize_session(self, model_id: str, session_id: str, params: Optional[Mapping[str, Any]]) -> JsonDict:
        srec = self.get_session_record(model_id, session_id)
        with srec.lock:
            self._episode_call(srec, lambda: srec.episode.initialize(dict(params or {})))
            srec.initialize_count += 1
        return {"model_id": model_id, "session_id": session_id, "initialized": True}

    def step_session(self, model_id: str, session_id: str, params: Optional[Mapping[str, Any]]) -> JsonDict:
        srec = self.get_session_record(model_id, session_id)
        with srec.lock:
            t0 = time.perf_counter()
            out = self._episode_call(srec, lambda: srec.episode.step(dict(params or {})))
            lat_ms = (time.perf_counter() - t0) * 1000.0
            srec.step_count += 1
            srec.last_step_ts = time.time()
            srec.last_step_latency_ms = float(lat_ms)
            srec.step_latency_count += 1
            srec.step_latency_sum_ms += float(lat_ms)
            try:
                rem = self._episode_call(srec, srec.episode.remaining_steps)
                srec.last_remaining_steps = int(rem) if rem is not None else None
            except Exception:
                srec.last_remaining_steps = None
            try:
                srec.last_step_terminal = bool(self._episode_call(srec, srec.episode.is_terminal))
            except Exception:
                srec.last_step_terminal = False
        return {"model_id": model_id, "session_id": session_id, "step": out}

    def agent_cmd(
        self,
        model_id: str,
        session_id: str,
        cmd: Mapping[str, Any],
        *,
        seconds: float = 0.0,
        return_events: bool = False,
    ) -> JsonDict:
        srec = self.get_session_record(model_id, session_id)
        payload = {
            "mode": "seconds",
            "seconds": max(0.0, float(seconds)),
            "return_events": bool(return_events),
            "return_state": True,
            "update": {"agent_cmd": dict(cmd)},
        }
        with srec.lock:
            t0 = time.perf_counter()
            out = self._episode_call(srec, lambda: srec.episode.step(payload))
            lat_ms = (time.perf_counter() - t0) * 1000.0
            srec.agent_cmd_count += 1
            srec.step_count += 1
            srec.last_step_ts = time.time()
            srec.last_step_latency_ms = float(lat_ms)
            srec.step_latency_count += 1
            srec.step_latency_sum_ms += float(lat_ms)
            try:
                rem = self._episode_call(srec, srec.episode.remaining_steps)
                srec.last_remaining_steps = int(rem) if rem is not None else None
            except Exception:
                srec.last_remaining_steps = None
            try:
                srec.last_step_terminal = bool(self._episode_call(srec, srec.episode.is_terminal))
            except Exception:
                srec.last_step_terminal = False

        state = out.get("state") if isinstance(out, dict) else None
        if isinstance(state, dict):
            result = state.get("agent_cmd_result", {}) or {}
        elif isinstance(state, list):
            result = {}
            for item in state:
                if isinstance(item, dict) and "agent_cmd_result" in item:
                    result = item.get("agent_cmd_result", {}) or {}
                    break
        else:
            result = {}

        return {
            "model_id": model_id,
            "session_id": session_id,
            "step": out,
            "agent_cmd_result": result,
        }

    def close_session(self, model_id: str, session_id: str) -> JsonDict:
        rec = self.get_model_record(model_id)

        with self._lock:
            srec = self._sessions.pop(session_id, None)
        if srec is None or srec.model_id != model_id:
            raise KeyError(f"unknown session_id for model {model_id}: {session_id}")

        with srec.lock:
            try:
                self._episode_call(srec, srec.episode.close)
            except Exception:
                pass

        with rec.lock:
            rec.session_ids.discard(session_id)

        return {"model_id": model_id, "session_id": session_id, "closed": True}

    def close_all(self) -> None:
        with self._lock:
            mids = list(self._models.keys())
        for mid in mids:
            try:
                self.delete_model(mid)
            except Exception:
                pass


class JormungandrHttpHandler(BaseHTTPRequestHandler):
    server_version = "JormungandrRuntime/0.2"

    @property
    def runtime(self) -> JormungandrRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def _path_parts(self) -> List[str]:
        p = urlparse(self.path).path
        return [x for x in p.split("/") if x]

    def _read_json(self) -> JsonDict:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        if not raw:
            return {}
        try:
            obj = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}")
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise ValueError("request body must be a JSON object")
        return obj

    def _write(self, status: int, payload: JsonDict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        started = getattr(self, "_request_started_ts", None)
        latency_ms = None
        if isinstance(started, (int, float)) and started > 0:
            latency_ms = max(0.0, (time.time() - float(started)) * 1000.0)

        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            if _is_client_disconnect(exc):
                try:
                    # 499 = client closed request (common when caller times out/cancels).
                    self.runtime.record_http(
                        self.command or "UNKNOWN",
                        self.path,
                        499,
                        latency_ms=latency_ms,
                    )
                except Exception:
                    pass
                return
            raise

        try:
            self.runtime.record_http(
                self.command or "UNKNOWN",
                self.path,
                int(status),
                latency_ms=latency_ms,
            )
        except Exception:
            pass

    def _ok(self, data: JsonDict, status: int = 200) -> None:
        self._write(status, {"ok": True, **data})

    def _err(self, status: int, message: str, detail: Optional[str] = None) -> None:
        out: JsonDict = {"ok": False, "error": message}
        if detail:
            out["detail"] = detail
        self._write(status, out)

    def do_GET(self) -> None:  # noqa: N802
        self._request_started_ts = time.time()
        try:
            parts = self._path_parts()
            if parts == ["health"]:
                self._ok({"service": "jormungandr", "ts": time.time()})
                return

            if parts == ["v1", "runtime", "stats"]:
                self._ok({"stats": self.runtime.snapshot_stats()})
                return

            if parts == ["v1", "algorithms"]:
                self._ok({"algorithms": self.runtime.list_algorithm_plugins()})
                return

            if parts == ["v1", "models"]:
                self._ok({"models": self.runtime.list_models()})
                return

            if parts == ["v1", "models", "compare"]:
                self._ok(self.runtime.compare_models())
                return

            if len(parts) == 3 and parts[:2] == ["v1", "models"]:
                self._ok({"model": self.runtime.get_model(parts[2])})
                return

            if len(parts) == 4 and parts[:2] == ["v1", "models"] and parts[3] == "inference":
                self._ok(self.runtime.get_inference(parts[2]))
                return

            if len(parts) == 4 and parts[:2] == ["v1", "models"] and parts[3] == "policy":
                self._ok(self.runtime.get_policy(parts[2]))
                return

            if len(parts) == 4 and parts[:2] == ["v1", "models"] and parts[3] == "metrics":
                self._ok(self.runtime.get_training_metrics(parts[2]))
                return

            if len(parts) == 4 and parts[:2] == ["v1", "models"] and parts[3] == "sessions":
                self._ok({"sessions": self.runtime.list_sessions(parts[2])})
                return

            if len(parts) == 5 and parts[:2] == ["v1", "models"] and parts[3] == "sessions":
                self._ok({"session": self.runtime.get_session(parts[2], parts[4])})
                return

            self._err(404, "not found")
        except KeyError as exc:
            self._err(404, str(exc))
        except Exception as exc:
            self._err(500, "internal error", detail=str(exc))

    def do_POST(self) -> None:  # noqa: N802
        self._request_started_ts = time.time()
        try:
            parts = self._path_parts()
            body = self._read_json()

            if parts == ["v1", "models"]:
                replay = body.get("replay", {})
                validation = body.get("validation", {})
                tb = body.get("tensorboard", {})
                learner = body.get("learner", None)
                if replay is not None and not isinstance(replay, dict):
                    raise JormungandrServiceError("replay must be an object")
                if validation is not None and not isinstance(validation, dict):
                    raise JormungandrServiceError("validation must be an object")
                if tb is not None and not isinstance(tb, dict):
                    raise JormungandrServiceError("tensorboard must be an object")
                if learner is not None and not isinstance(learner, dict):
                    raise JormungandrServiceError("learner must be an object")

                model = self.runtime.create_model(
                    obs_dim=int(body.get("obs_dim", 0)),
                    model_id=body.get("model_id"),
                    capacity=int((replay or {}).get("capacity", 200_000)),
                    validation_capacity=int((validation or {}).get("capacity", 20_000)),
                    alpha=float((replay or {}).get("alpha", 0.6)),
                    metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                    learner=learner,
                    tensorboard_enabled=bool((tb or {}).get("enabled", True)),
                    tensorboard_logdir=str((tb or {}).get("logdir", "")),
                    checkpoint_path=str(body.get("checkpoint_path", "")),
                )
                self._ok({"model": model}, status=201)
                return

            if len(parts) < 4 or parts[:2] != ["v1", "models"]:
                self._err(404, "not found")
                return

            model_id = parts[2]
            action = parts[3]

            if action == "metrics":
                metrics = body.get("metrics")
                if not isinstance(metrics, dict):
                    raise JormungandrServiceError("metrics must be an object")
                step = int(body.get("step", 0))
                out = self.runtime.log_metrics(model_id, step, metrics)
                self._ok(out)
                return

            if action == "inference":
                inference = body.get("inference")
                if not isinstance(inference, dict):
                    raise JormungandrServiceError("inference must be an object")
                step_raw = body.get("step")
                step = int(step_raw) if step_raw is not None else None
                out = self.runtime.publish_inference(model_id, inference=inference, step=step)
                self._ok(out)
                return

            if action == "policy":
                if len(parts) == 4:
                    self._ok(self.runtime.get_policy(model_id))
                    return
                if len(parts) < 5:
                    self._err(404, "not found")
                    return
                sub = parts[4]
                if sub == "infer":
                    obs_batch = body.get("obs_batch")
                    if obs_batch is not None:
                        if not isinstance(obs_batch, list):
                            raise JormungandrServiceError("obs_batch must be an array")
                        deterministic = bool(body.get("deterministic", True))
                        epsilon = float(body.get("epsilon", 0.0))
                        self._ok(
                            self.runtime.policy_infer_batch(
                                model_id,
                                obs_batch=obs_batch,
                                deterministic=deterministic,
                                epsilon=epsilon,
                                action_masks=body.get("action_masks"),
                            )
                        )
                        return
                    obs = body.get("obs")
                    if not isinstance(obs, list):
                        raise JormungandrServiceError("obs must be an array")
                    deterministic = bool(body.get("deterministic", True))
                    epsilon = float(body.get("epsilon", 0.0))
                    self._ok(
                        self.runtime.policy_infer(
                            model_id,
                            obs=obs,
                            deterministic=deterministic,
                            epsilon=epsilon,
                            action_mask=body.get("action_mask"),
                        )
                    )
                    return
                if sub == "checkpoint":
                    self._ok(self.runtime.force_policy_checkpoint(model_id))
                    return
                self._err(404, "not found")
                return

            if action == "aux":
                if len(parts) < 5:
                    self._err(404, "not found")
                    return
                sub = parts[4]
                if sub == "infer":
                    obs = body.get("obs")
                    if obs is not None and not isinstance(obs, list):
                        raise JormungandrServiceError("obs must be an array when provided")
                    features = body.get("features")
                    if features is not None and not isinstance(features, dict):
                        raise JormungandrServiceError("features must be an object when provided")
                    deterministic = bool(body.get("deterministic", True))
                    self._ok(
                        self.runtime.aux_infer(
                            model_id,
                            obs=obs,
                            features=features,
                            deterministic=deterministic,
                        )
                    )
                    return
                self._err(404, "not found")
                return

            if action == "sessions":
                if len(parts) == 4:
                    driver = str(body.get("driver", "")).strip()
                    dcfg = body.get("driver_config")
                    ecfg = body.get("episode_config")
                    if not isinstance(dcfg, dict):
                        raise JormungandrServiceError("driver_config must be an object")
                    if not isinstance(ecfg, dict):
                        raise JormungandrServiceError("episode_config must be an object")
                    out = self.runtime.create_session(
                        model_id,
                        driver=driver,
                        driver_config=dcfg,
                        episode_config=ecfg,
                        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                        session_id=body.get("session_id"),
                    )
                    self._ok({"session": out}, status=201)
                    return

                if len(parts) < 6:
                    self._err(404, "not found")
                    return

                session_id = parts[4]
                sub = parts[5]
                if sub == "initialize":
                    init = body.get("init") if isinstance(body.get("init"), dict) else body
                    self._ok(self.runtime.initialize_session(model_id, session_id, init))
                    return
                if sub == "step":
                    step = body.get("step") if isinstance(body.get("step"), dict) else body
                    self._ok(self.runtime.step_session(model_id, session_id, step))
                    return
                if sub == "agent_cmd":
                    cmd = body.get("cmd")
                    if not isinstance(cmd, dict):
                        raise JormungandrServiceError("cmd must be an object")
                    seconds = float(body.get("seconds", 0.0))
                    return_events = bool(body.get("return_events", False))
                    self._ok(
                        self.runtime.agent_cmd(
                            model_id,
                            session_id,
                            cmd,
                            seconds=seconds,
                            return_events=return_events,
                        )
                    )
                    return

                self._err(404, "not found")
                return

            if action == "experience":
                if len(parts) < 5:
                    self._err(404, "not found")
                    return
                sub = parts[4]
                if sub == "add":
                    schema = str(body.get("schema", "")).strip()
                    if schema != "jormungandr.experience.v1":
                        raise JormungandrServiceError(
                            "schema must be 'jormungandr.experience.v1'"
                        )
                    items = body.get("items")
                    if not isinstance(items, list):
                        raise JormungandrServiceError("items must be an array")
                    self._ok(
                        self.runtime.experience_add(
                            model_id,
                            items,
                            require_identity=True,
                        )
                    )
                    return
                if sub == "aux_update":
                    schema = str(body.get("schema", "")).strip()
                    if schema != "jormungandr.aux_update.v1":
                        raise JormungandrServiceError(
                            "schema must be 'jormungandr.aux_update.v1'"
                        )
                    updates = body.get("updates")
                    if not isinstance(updates, list):
                        raise JormungandrServiceError("updates must be an array")
                    self._ok(self.runtime.experience_aux_update(model_id, updates))
                    return
                self._err(404, "not found")
                return

            if action == "replay":
                if len(parts) < 5:
                    self._err(404, "not found")
                    return
                sub = parts[4]
                if sub == "add":
                    items = body.get("items")
                    if not isinstance(items, list):
                        raise JormungandrServiceError("items must be an array")
                    self._ok(self.runtime.replay_add(model_id, items))
                    return
                if sub == "sample":
                    batch_size = int(body.get("batch_size", 256))
                    beta = float(body.get("beta", 0.4))
                    self._ok(self.runtime.replay_sample(model_id, batch_size, beta))
                    return
                if sub == "update_priorities":
                    idxs = body.get("idxs")
                    priorities = body.get("priorities")
                    if not isinstance(idxs, list) or not isinstance(priorities, list):
                        raise JormungandrServiceError("idxs and priorities must be arrays")
                    self._ok(self.runtime.replay_update_priorities(model_id, idxs, priorities))
                    return
                if sub == "aux_update":
                    updates = body.get("updates")
                    if not isinstance(updates, list):
                        raise JormungandrServiceError("updates must be an array")
                    self._ok(self.runtime.replay_aux_update(model_id, updates))
                    return
                self._err(404, "not found")
                return

            self._err(404, "not found")
        except KeyError as exc:
            self._err(404, str(exc))
        except (JormungandrServiceError, ValueError) as exc:
            self._err(400, str(exc))
        except Exception as exc:
            self._err(500, "internal error", detail=str(exc))

    def do_DELETE(self) -> None:  # noqa: N802
        self._request_started_ts = time.time()
        try:
            parts = self._path_parts()
            if len(parts) == 3 and parts[:2] == ["v1", "models"]:
                self._ok(self.runtime.delete_model(parts[2]))
                return

            if len(parts) == 5 and parts[:2] == ["v1", "models"] and parts[3] == "sessions":
                self._ok(self.runtime.close_session(parts[2], parts[4]))
                return

            self._err(404, "not found")
        except KeyError as exc:
            self._err(404, str(exc))
        except Exception as exc:
            self._err(500, "internal error", detail=str(exc))

    def log_message(self, fmt: str, *args: Any) -> None:
        if not bool(getattr(self.server, "access_log", False)):  # type: ignore[attr-defined]
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{ts}] {self.address_string()} {fmt % args}")


class JormungandrHttpServer(ThreadingHTTPServer):
    def __init__(
        self,
        host: str,
        port: int,
        runtime: Optional[JormungandrRuntime] = None,
        *,
        access_log: bool = False,
        tensorboard_root: str = "",
        checkpoint_root: str = "",
    ) -> None:
        super().__init__((host, port), JormungandrHttpHandler)
        self.runtime = runtime or JormungandrRuntime(
            tensorboard_root=tensorboard_root,
            checkpoint_root=checkpoint_root,
        )
        self.access_log = bool(access_log)


def _count_status_bucket(status_counts: Mapping[str, int], lo: int, hi: int) -> int:
    total = 0
    for code, value in status_counts.items():
        try:
            c = int(code)
        except Exception:
            continue
        if lo <= c < hi:
            total += int(value)
    return total


def _build_dashboard_lines(
    snap: Mapping[str, Any],
    *,
    prev_total: int,
    prev_ts: float,
) -> tuple[List[str], int, float]:
    now = float(snap.get("ts", time.time()))
    total = int((snap.get("http") or {}).get("total", 0))
    dt = max(1e-6, now - prev_ts)
    dreq = max(0, total - prev_total)
    rps = dreq / dt

    status = (snap.get("http") or {}).get("status", {})
    if not isinstance(status, dict):
        status = {}
    s2xx = _count_status_bucket(status, 200, 300)
    s4xx = _count_status_bucket(status, 400, 500)
    s5xx = _count_status_bucket(status, 500, 600)
    models = snap.get("models") or []
    uptime_s = float(snap.get("uptime_s", 0.0))
    latency = (snap.get("http") or {}).get("latency") or {}
    totals = snap.get("totals") or {}

    lines: List[str] = []
    lines.append(
        f"[jormungandr-top] up={uptime_s:8.1f}s req={total} (+{dreq} / {dt:.1f}s, {rps:5.1f} rps) "
        f"lat(avg/p95)={float(latency.get('avg_ms', 0.0)):6.2f}/{float(latency.get('p95_ms', 0.0)):6.2f}ms "
        f"2xx={s2xx} 4xx={s4xx} 5xx={s5xx}"
    )
    lines.append(
        f"totals: models={len(models)} "
        f"replay={int(totals.get('replay_size', 0))}/{int(totals.get('replay_capacity', 0))} "
        f"validation={int(totals.get('validation_size', 0))}/{int(totals.get('validation_capacity', 0))} "
        f"inference(pub/read)={int(totals.get('inference_updates', 0))}/{int(totals.get('inference_reads', 0))} "
        f"learn_updates={int(totals.get('learner_updates', 0))} "
        f"checkpoints={int(totals.get('learner_checkpoints', 0))} "
        f"policy_infer={int(totals.get('policy_infer_calls', 0))} "
        f"aux_infer={int(totals.get('aux_infer_calls', 0))} "
        f"aux_update={int(totals.get('aux_update_matched', 0))}/{int(totals.get('aux_update_items', 0))} "
        f"train_ep={int(totals.get('sampler_train_episodes', 0))} "
        f"val_ep={int(totals.get('sampler_val_episodes', 0))}"
    )

    lines.append("models:")
    lines.append(
        "  "
        f"{'model':<20} {'train/val':>20} {'%':>5} {'add':>6} {'smp':>5} "
        f"{'met':>9} {'ep(tv)':>9} {'inf':>9} {'upd':>6} {'ver':>5} {'ckpt':>5} {'pinf':>6} {'dev':>6} {'loss':>8}"
    )
    for m in models:
        if not isinstance(m, dict):
            continue
        replay_size = int(m.get("replay_size", 0))
        replay_cap = max(1, int(m.get("replay_capacity", 1)))
        validation_size = int(m.get("validation_size", 0))
        validation_cap = max(1, int(m.get("validation_capacity", 1)))
        replay_pct = 100.0 * float(replay_size) / float(replay_cap)
        metrics_txt = f"{int(m.get('metrics_calls', 0))}/{int(m.get('metrics_points', 0))}"
        ep_txt = f"{int(m.get('sampler_train_episodes', 0))}/{int(m.get('sampler_val_episodes', 0))}"
        inf_txt = f"{int(m.get('inference_updates', 0))}/{int(m.get('inference_reads', 0))}"
        lines.append(
            "  "
            f"{str(m.get('model_id', ''))[:20]:<20} "
            f"{f'{replay_size}/{replay_cap}|{validation_size}/{validation_cap}':>20} "
            f"{replay_pct:5.1f} "
            f"{int(m.get('replay_add_items', 0)):6d} "
            f"{int(m.get('replay_sample_calls', 0)):5d} "
            f"{metrics_txt:>9} "
            f"{ep_txt:>9} "
            f"{inf_txt:>9} "
            f"{int(m.get('learner_updates', 0)):6d} "
            f"{int(m.get('policy_version', 0)):5d} "
            f"{int(m.get('learner_checkpoints', 0)):5d} "
            f"{int(m.get('policy_infer_calls', 0)):6d} "
            f"{str(m.get('learner_device', '-'))[:6]:>6} "
            f"{float(m.get('learner_last_loss', 0.0)):8.4f}"
        )

    top_eps = (snap.get("http") or {}).get("top_endpoints", [])
    if isinstance(top_eps, list) and top_eps:
        lines.append("http endpoints:")
        for e in top_eps:
            if not isinstance(e, dict):
                continue
            lines.append(
                f"  {int(e.get('count', 0)):6d}  avg={float(e.get('avg_latency_ms', 0.0)):7.2f}ms  {str(e.get('key', ''))}"
            )
    lines.append("press q to quit dashboard")
    return lines, total, now


class JormungandrTopPrinter:
    def __init__(
        self,
        runtime: JormungandrRuntime,
        *,
        interval_s: float = 2.0,
        max_endpoints: int = 6,
        clear_screen: bool = False,
    ) -> None:
        self.runtime = runtime
        self.interval_s = max(0.5, float(interval_s))
        self.max_endpoints = max(1, int(max_endpoints))
        self.clear_screen = bool(clear_screen)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jormungandr-top-printer", daemon=True)
        self._prev_total = 0
        self._prev_ts = time.time()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            snap = self.runtime.snapshot_stats(max_endpoints=self.max_endpoints)
            lines, self._prev_total, self._prev_ts = _build_dashboard_lines(
                snap,
                prev_total=self._prev_total,
                prev_ts=self._prev_ts,
            )
            if self.clear_screen:
                print("\033[2J\033[H", end="")
            for ln in lines:
                print(ln)
            print("", flush=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class JormungandrTopCursesDashboard:
    def __init__(
        self,
        runtime: JormungandrRuntime,
        *,
        interval_s: float = 1.0,
        max_endpoints: int = 8,
        theme: str = "auto",
    ) -> None:
        self.runtime = runtime
        self.interval_s = max(0.2, float(interval_s))
        self.max_endpoints = max(1, int(max_endpoints))
        self.theme = str(theme)
        self._prev_total = 0
        self._prev_ts = time.time()
        self._stop = False
        self._attr_normal = 0
        self._attr_header = 0

    def _setup_colors(self, stdscr: Any) -> None:
        import curses

        self._attr_normal = curses.A_NORMAL
        self._attr_header = curses.A_BOLD
        if not curses.has_colors():
            stdscr.bkgd(" ", self._attr_normal)
            return

        try:
            curses.start_color()
        except Exception:
            stdscr.bkgd(" ", self._attr_normal)
            return
        try:
            curses.use_default_colors()
        except Exception:
            pass

        fg = -1
        bg = -1
        if self.theme == "light":
            fg = curses.COLOR_BLACK
            bg = curses.COLOR_WHITE
        elif self.theme == "dark":
            fg = curses.COLOR_WHITE
            bg = curses.COLOR_BLACK

        try:
            curses.init_pair(1, fg, bg)
            self._attr_normal = curses.color_pair(1)
            self._attr_header = curses.color_pair(1) | curses.A_BOLD
        except Exception:
            self._attr_normal = curses.A_NORMAL
            self._attr_header = curses.A_BOLD
        stdscr.bkgd(" ", self._attr_normal)

    def _draw(self, stdscr: Any, lines: List[str]) -> None:
        h, w = stdscr.getmaxyx()
        stdscr.erase()
        max_rows = max(0, h - 1)
        for i, ln in enumerate(lines[:max_rows]):
            attr = self._attr_normal
            if i == 0 or ln in ("models:", "http endpoints:"):
                attr = self._attr_header
            stdscr.addnstr(i, 0, ln, max(0, w - 1), attr)
        stdscr.noutrefresh()

    def run(self) -> None:
        import curses

        def _main(stdscr: Any) -> None:
            stdscr.nodelay(True)
            stdscr.timeout(int(self.interval_s * 1000.0))
            try:
                curses.curs_set(0)
            except Exception:
                pass
            self._setup_colors(stdscr)
            while not self._stop:
                snap = self.runtime.snapshot_stats(max_endpoints=self.max_endpoints)
                lines, self._prev_total, self._prev_ts = _build_dashboard_lines(
                    snap,
                    prev_total=self._prev_total,
                    prev_ts=self._prev_ts,
                )
                self._draw(stdscr, lines)
                curses.doupdate()
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    self._stop = True

        curses.wrapper(_main)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Jörmungandr multi-model runtime service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument(
        "--tensorboard-root",
        default="",
        help=(
            "Optional root directory for tensorboard outputs. "
            "If set, relative tensorboard logdirs are resolved under this root."
        ),
    )
    ap.add_argument(
        "--checkpoint-root",
        default="",
        help=(
            "Optional root directory for policy checkpoints. "
            "If set, relative learner checkpoint dirs are resolved under this root."
        ),
    )
    ap.add_argument(
        "--access-log",
        action="store_true",
        help="Enable per-request access logs (disabled by default to avoid log flooding).",
    )
    ap.add_argument(
        "--top",
        action="store_true",
        help="Print periodic top-style runtime stats (throughput, models, HTTP).",
    )
    ap.add_argument(
        "--top-mode",
        choices=["auto", "print", "curses"],
        default="auto",
        help="Dashboard mode for --top.",
    )
    ap.add_argument(
        "--top-curses",
        action="store_true",
        help="Shortcut for --top-mode curses.",
    )
    ap.add_argument("--top-interval", type=float, default=2.0, help="Refresh interval for --top output.")
    ap.add_argument("--top-endpoints", type=int, default=6, help="How many HTTP endpoint counters to show in --top mode.")
    ap.add_argument(
        "--top-theme",
        choices=["auto", "light", "dark"],
        default="auto",
        help="Curses dashboard color theme.",
    )
    ap.add_argument("--top-clear", action="store_true", help="Clear terminal before each --top refresh.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.top_curses):
        top_mode = "curses"
    else:
        requested_mode = str(args.top_mode)
        if requested_mode == "auto":
            top_mode = "curses" if (sys.stdout.isatty() and sys.stdin.isatty()) else "print"
        else:
            top_mode = requested_mode
    server = JormungandrHttpServer(
        args.host,
        args.port,
        access_log=bool(args.access_log),
        tensorboard_root=str(args.tensorboard_root),
        checkpoint_root=str(args.checkpoint_root),
    )
    top_printer: Optional[JormungandrTopPrinter] = None
    if args.top and top_mode == "print":
        top_printer = JormungandrTopPrinter(
            server.runtime,
            interval_s=float(args.top_interval),
            max_endpoints=int(args.top_endpoints),
            clear_screen=bool(args.top_clear),
        )
        top_printer.start()
    if args.top and top_mode == "curses" and not sys.stdout.isatty():
        print("[jormungandr] --top-mode curses requested but stdout is not a TTY; using --top-mode print")
        top_mode = "print"
        if top_printer is None:
            top_printer = JormungandrTopPrinter(
                server.runtime,
                interval_s=float(args.top_interval),
                max_endpoints=int(args.top_endpoints),
                clear_screen=bool(args.top_clear),
            )
            top_printer.start()

    print(
        f"[jormungandr] runtime service listening at http://{args.host}:{args.port} "
        f"(access_log={'on' if args.access_log else 'off'}, top={'on' if args.top else 'off'}, top_mode={top_mode}, "
        f"tb_root={str(args.tensorboard_root or '-').strip() or '-'}, "
        f"ckpt_root={str(args.checkpoint_root or '-').strip() or '-'})"
    )

    if args.top and top_mode == "curses":
        srv_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
        srv_thread.start()
        dash = JormungandrTopCursesDashboard(
            server.runtime,
            interval_s=float(args.top_interval),
            max_endpoints=int(args.top_endpoints),
            theme=str(args.top_theme),
        )
        try:
            dash.run()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            srv_thread.join(timeout=2.0)
            server.runtime.close_all()
            server.server_close()
        return

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if top_printer is not None:
            top_printer.stop()
        server.runtime.close_all()
        server.server_close()


if __name__ == "__main__":
    main()
