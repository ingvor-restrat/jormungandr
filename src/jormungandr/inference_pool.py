"""Replicated, dynamically batched inference with atomic policy activation.

The pool deliberately owns neither an RL algorithm nor a transport protocol.
Callers provide an agent factory and a batch inference function; the pool owns
replica placement, request scheduling, hot weight refresh, and telemetry.
"""

from __future__ import annotations

import threading
import time
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Hashable, Mapping, Optional, Sequence


class InferencePoolError(RuntimeError):
    """Raised when a serving replica or queued inference request fails."""


@dataclass(frozen=True)
class InferencePoolConfig:
    """Process-local inference resources shared by one policy model."""

    devices: tuple[str, ...] = ()
    # Zero selects hardware-aware automatic placement. Explicit positive values
    # remain available for reproducible benchmarks and constrained hosts.
    replicas: int = 0
    max_auto_replicas: int = 2
    max_batch_size: int = 64
    batch_wait_ms: float = 2.0
    request_timeout_s: float = 30.0
    auto_tune: bool = True
    tune_interval_s: float = 15.0
    tune_batch_wait_ms: tuple[float, ...] = (0.5, 2.0, 5.0, 10.0)

    def __post_init__(self) -> None:
        if int(self.replicas) < 0:
            raise ValueError("inference replicas must be >= 0 (zero means auto)")
        if int(self.max_auto_replicas) <= 0:
            raise ValueError("inference max_auto_replicas must be > 0")
        if int(self.max_batch_size) <= 0:
            raise ValueError("inference max_batch_size must be > 0")
        if float(self.batch_wait_ms) < 0.0:
            raise ValueError("inference batch_wait_ms must be >= 0")
        if float(self.request_timeout_s) <= 0.0:
            raise ValueError("inference request_timeout_s must be > 0")
        if float(self.tune_interval_s) <= 0.0:
            raise ValueError("inference tune_interval_s must be > 0")
        if not self.tune_batch_wait_ms or any(
            float(value) < 0.0 for value in self.tune_batch_wait_ms
        ):
            raise ValueError("inference tune_batch_wait_ms must be nonempty and non-negative")


@dataclass(frozen=True)
class InferenceResult:
    value: Any
    replica_id: int
    device: str
    policy_version: int
    activated_ts: float


@dataclass
class _Request:
    key: Hashable
    payload: Any
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[InferenceResult] = None
    error: Optional[BaseException] = None


@dataclass
class _Replica:
    replica_id: int
    device: str
    active_agent: Any
    standby_agent: Any
    policy_version: int
    metadata: Mapping[str, Any]
    activated_ts: float
    requests: int = 0
    items: int = 0
    batches: int = 0
    busy: bool = False
    last_batch_size: int = 0
    last_inference_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


AgentFactory = Callable[[str, Mapping[str, Any]], Any]
BatchInfer = Callable[[Any, Sequence[Any], Mapping[str, Any]], Sequence[Any]]


def available_cuda_devices() -> list[dict[str, Any]]:
    """Return CUDA resources visible to the current PyTorch process."""

    try:
        import torch

        if not torch.cuda.is_available():
            return []
        rows: list[dict[str, Any]] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            rows.append(
                {
                    "device": f"cuda:{index}",
                    "name": str(props.name),
                    "total_memory_bytes": int(total_bytes),
                    "free_memory_bytes": int(free_bytes),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        return rows
    except Exception:
        return []


def validate_devices(devices: Sequence[str]) -> tuple[str, ...]:
    """Validate and canonicalize explicit CPU/CUDA serving devices."""

    import torch

    out: list[str] = []
    for raw in devices:
        device = str(raw).strip().lower()
        if not device or device == "auto":
            continue
        if device == "cpu":
            out.append(device)
            continue
        if device == "cuda":
            device = "cuda:0"
        if not device.startswith("cuda:"):
            raise ValueError(
                f"unsupported inference device {raw!r}; use cpu or cuda:N"
            )
        try:
            index = int(device.split(":", 1)[1])
        except Exception as exc:
            raise ValueError(f"invalid inference device {raw!r}") from exc
        if not torch.cuda.is_available() or index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"inference device {device} is not available")
        canonical = f"cuda:{index}"
        if canonical not in out:
            out.append(canonical)
    return tuple(out)


def resolve_devices(
    devices: Sequence[str],
    *,
    fallback_device: str = "auto",
) -> tuple[str, ...]:
    """Resolve explicit placement or select the learner-capable local device.

    Auto placement never assumes a CUDA ordinal or even that CUDA exists.  It
    follows an already resolved learner device when supplied, otherwise it
    selects the first visible CUDA device and falls back to CPU.
    """

    explicit = validate_devices(devices)
    if explicit:
        return explicit
    fallback = str(fallback_device or "auto").strip().lower()
    if fallback in {"", "auto"}:
        visible = available_cuda_devices()
        return (str(visible[0]["device"]),) if visible else ("cpu",)
    return validate_devices((fallback,))


def auto_replica_count(
    devices: Sequence[str],
    *,
    max_replicas: int,
) -> int:
    """Return a conservative capability-derived serving replica count."""

    maximum = max(1, int(max_replicas))
    cuda_devices = [value for value in devices if str(value).startswith("cuda:")]
    if cuda_devices:
        # Two candidates per visible accelerator let the online tuner measure
        # whether overlapping small batches help. The configured maximum is
        # the admission bound and may be set to one on memory-constrained hosts.
        return min(maximum, max(1, 2 * len(cuda_devices)))
    cpu_parallelism = max(1, (os.cpu_count() or 1) // 2)
    return min(maximum, cpu_parallelism)


class ReplicatedInferencePool:
    """A common dynamic-batch queue backed by coherent serving replicas."""

    def __init__(
        self,
        config: InferencePoolConfig,
        *,
        agent_factory: AgentFactory,
        infer_batch: BatchInfer,
    ) -> None:
        self.config = config
        self._agent_factory = agent_factory
        self._infer_batch = infer_batch
        self._replicas: list[_Replica] = []
        self._queue: Deque[_Request] = deque()
        self._condition = threading.Condition()
        self._refresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._refreshes = 0
        self._last_refresh_ms = 0.0
        self._active_replicas = 0
        self._runtime_batch_wait_ms = float(config.batch_wait_ms)
        self._tune_candidates: list[tuple[float, int]] = []
        self._tune_scores: dict[tuple[float, int], dict[str, float | int]] = {}
        self._tune_current = (float(config.batch_wait_ms), 1)
        self._tune_round = 0
        self._tune_thread: Optional[threading.Thread] = None

    @property
    def replicas(self) -> int:
        return len(self._replicas)

    def start(
        self,
        state: Mapping[str, Any],
        *,
        policy_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
        fallback_device: str = "cpu",
    ) -> None:
        if self._replicas:
            raise InferencePoolError("inference pool is already started")
        devices = resolve_devices(
            self.config.devices,
            fallback_device=str(fallback_device),
        )
        if not devices:
            raise InferencePoolError("no inference devices are configured")
        replica_count = int(self.config.replicas) or auto_replica_count(
            devices,
            max_replicas=int(self.config.max_auto_replicas),
        )
        activated_ts = time.time()
        snapshot_metadata = dict(metadata or {})
        for replica_id in range(replica_count):
            device = devices[replica_id % len(devices)]
            active = self._agent_factory(device, state)
            standby = self._agent_factory(device, state)
            self._set_eval(active)
            self._set_eval(standby)
            self._replicas.append(
                _Replica(
                    replica_id=replica_id,
                    device=device,
                    active_agent=active,
                    standby_agent=standby,
                    policy_version=int(policy_version),
                    metadata=dict(snapshot_metadata),
                    activated_ts=activated_ts,
                )
            )
        self._stop.clear()
        self._active_replicas = 1 if self.config.auto_tune else replica_count
        waits = sorted(
            {
                float(self.config.batch_wait_ms),
                *(float(value) for value in self.config.tune_batch_wait_ms),
            }
        )
        self._tune_candidates = [
            (wait_ms, active)
            for wait_ms in waits
            for active in range(1, replica_count + 1)
        ]
        self._tune_current = (
            float(self._runtime_batch_wait_ms),
            int(self._active_replicas),
        )
        for replica in self._replicas:
            thread = threading.Thread(
                target=self._worker,
                args=(replica,),
                name=f"inference-replica-{replica.replica_id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        if self.config.auto_tune and len(self._tune_candidates) > 1:
            self._tune_thread = threading.Thread(
                target=self._tune_worker,
                name="inference-auto-tuner",
                daemon=True,
            )
            self._tune_thread.start()

    def _take_batch(self, replica_id: int) -> list[_Request]:
        with self._condition:
            while (
                (not self._queue or replica_id >= self._active_replicas)
                and not self._stop.is_set()
            ):
                self._condition.wait(timeout=0.25)
            if self._stop.is_set():
                return []
            first = self._queue.popleft()
            batch = [first]
            deadline = time.monotonic() + self._runtime_batch_wait_ms / 1000.0
            while len(batch) < int(self.config.max_batch_size):
                matching_index = next(
                    (i for i, request in enumerate(self._queue) if request.key == first.key),
                    None,
                )
                if matching_index is not None:
                    request = self._queue[matching_index]
                    del self._queue[matching_index]
                    batch.append(request)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            return batch

    def _worker(self, replica: _Replica) -> None:
        while not self._stop.is_set():
            batch = self._take_batch(replica.replica_id)
            if not batch:
                continue
            started = time.perf_counter()
            try:
                with replica.lock:
                    replica.busy = True
                    values = list(
                        self._infer_batch(
                            replica.active_agent,
                            [request.payload for request in batch],
                            replica.metadata,
                        )
                    )
                    if len(values) != len(batch):
                        raise InferencePoolError(
                            "batch inference returned a different number of results"
                        )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    replica.requests += len(batch)
                    replica.items += len(batch)
                    replica.batches += 1
                    replica.last_batch_size = len(batch)
                    replica.last_inference_ms = elapsed_ms
                    results = [
                        InferenceResult(
                            value=value,
                            replica_id=replica.replica_id,
                            device=replica.device,
                            policy_version=replica.policy_version,
                            activated_ts=replica.activated_ts,
                        )
                        for value in values
                    ]
            except BaseException as exc:
                results = []
                for request in batch:
                    request.error = exc
            finally:
                replica.busy = False
            for index, request in enumerate(batch):
                if request.error is None:
                    request.result = results[index]
                request.done.set()

    def configure_runtime(
        self,
        *,
        active_replicas: int,
        batch_wait_ms: float,
    ) -> None:
        """Atomically change the active serving subset and batching delay."""

        if not self._replicas:
            raise InferencePoolError("inference pool is not running")
        active = int(active_replicas)
        wait_ms = float(batch_wait_ms)
        if active < 1 or active > len(self._replicas):
            raise ValueError("active_replicas must be within the allocated fleet")
        if wait_ms < 0.0:
            raise ValueError("batch_wait_ms must be non-negative")
        with self._condition:
            self._active_replicas = active
            self._runtime_batch_wait_ms = wait_ms
            self._tune_current = (wait_ms, active)
            self._condition.notify_all()

    @staticmethod
    def best_tune_candidate(
        candidates: Sequence[tuple[float, int]],
        scores: Mapping[tuple[float, int], Mapping[str, float | int]],
        *,
        throughput_tolerance: float = 0.05,
    ) -> tuple[float, int]:
        """Prefer lower delay and fewer replicas when throughput is within 5%."""

        if not candidates:
            raise ValueError("inference tune candidate list is empty")
        best_rate = max(
            float(scores[item]["ewma_items_per_s"]) for item in candidates
        )
        acceptable_rate = best_rate * max(0.0, 1.0 - throughput_tolerance)
        acceptable = [
            item
            for item in candidates
            if float(scores[item]["ewma_items_per_s"]) >= acceptable_rate
        ]
        return min(acceptable, key=lambda item: (float(item[0]), int(item[1])))

    def _total_items(self) -> int:
        return sum(int(replica.items) for replica in self._replicas)

    def _tune_worker(self) -> None:
        last_ts = time.monotonic()
        last_items = self._total_items()
        while not self._stop.wait(float(self.config.tune_interval_s)):
            now = time.monotonic()
            total_items = self._total_items()
            elapsed = max(1e-9, now - last_ts)
            item_delta = max(0, total_items - last_items)
            had_previous_traffic = last_items > 0
            last_ts = now
            last_items = total_items
            if item_delta <= 0 or not had_previous_traffic:
                continue
            rate = float(item_delta) / elapsed
            candidate = self._tune_current
            stats = self._tune_scores.setdefault(
                candidate,
                {"samples": 0, "ewma_items_per_s": 0.0, "last_items_per_s": 0.0},
            )
            samples = int(stats["samples"])
            stats["samples"] = samples + 1
            stats["last_items_per_s"] = rate
            stats["ewma_items_per_s"] = (
                rate
                if samples == 0
                else 0.7 * float(stats["ewma_items_per_s"]) + 0.3 * rate
            )
            self._tune_round += 1
            untried = [
                item
                for item in self._tune_candidates
                if int(self._tune_scores.get(item, {}).get("samples", 0)) == 0
            ]
            if untried:
                next_candidate = untried[0]
            elif self._tune_round % 10 == 0:
                index = (self._tune_candidates.index(candidate) + 1) % len(
                    self._tune_candidates
                )
                next_candidate = self._tune_candidates[index]
            else:
                next_candidate = self.best_tune_candidate(
                    self._tune_candidates,
                    self._tune_scores,
                )
            self.configure_runtime(
                batch_wait_ms=next_candidate[0],
                active_replicas=next_candidate[1],
            )

    def submit(
        self,
        payload: Any,
        *,
        key: Hashable = "default",
        timeout_s: Optional[float] = None,
    ) -> InferenceResult:
        return self.submit_many([payload], key=key, timeout_s=timeout_s)[0]

    def submit_many(
        self,
        payloads: Sequence[Any],
        *,
        key: Hashable = "default",
        timeout_s: Optional[float] = None,
    ) -> list[InferenceResult]:
        if not self._replicas or self._stop.is_set():
            raise InferencePoolError("inference pool is not running")
        requests = [_Request(key=key, payload=payload) for payload in payloads]
        if not requests:
            return []
        with self._condition:
            self._queue.extend(requests)
            self._condition.notify_all()
        timeout = (
            float(self.config.request_timeout_s)
            if timeout_s is None
            else float(timeout_s)
        )
        deadline = time.monotonic() + timeout
        out: list[InferenceResult] = []
        for request in requests:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not request.done.wait(timeout=remaining):
                raise InferencePoolError("timed out waiting for inference replica")
            if request.error is not None:
                raise InferencePoolError(str(request.error)) from request.error
            if request.result is None:
                raise InferencePoolError("inference request completed without a result")
            out.append(request.result)
        return out

    def refresh(
        self,
        state: Mapping[str, Any],
        *,
        policy_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Load standby replicas, then expose the new version fleet-wide."""

        if not self._replicas:
            raise InferencePoolError("inference pool is not running")
        started = time.perf_counter()
        snapshot_metadata = dict(metadata or {})
        with self._refresh_lock:
            for replica in self._replicas:
                replica.standby_agent.load_state_dict(state)
                self._set_eval(replica.standby_agent)
            self._synchronize_cuda_devices()
            acquired: list[threading.Lock] = []
            try:
                for replica in self._replicas:
                    replica.lock.acquire()
                    acquired.append(replica.lock)
                activated_ts = time.time()
                for replica in self._replicas:
                    replica.active_agent, replica.standby_agent = (
                        replica.standby_agent,
                        replica.active_agent,
                    )
                    replica.policy_version = int(policy_version)
                    replica.metadata = dict(snapshot_metadata)
                    replica.activated_ts = activated_ts
                self._refreshes += 1
                self._last_refresh_ms = (time.perf_counter() - started) * 1000.0
            finally:
                for lock in reversed(acquired):
                    lock.release()

    @staticmethod
    def _set_eval(agent: Any) -> None:
        try:
            import torch

            for value in vars(agent).values():
                if isinstance(value, torch.nn.Module):
                    value.eval()
        except Exception:
            pass

    def _synchronize_cuda_devices(self) -> None:
        devices = sorted(
            {replica.device for replica in self._replicas if replica.device.startswith("cuda:")}
        )
        if not devices:
            return
        import torch

        for device in devices:
            torch.cuda.synchronize(device)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            queued = len(self._queue)
        return {
            "schema": "jormungandr.inference_pool.v1",
            "replica_count": len(self._replicas),
            "active_replica_count": int(self._active_replicas),
            "devices": sorted({replica.device for replica in self._replicas}),
            "queued_requests": int(queued),
            "refreshes": int(self._refreshes),
            "last_refresh_ms": float(self._last_refresh_ms),
            "scheduler": "shared_dynamic_batch_queue",
            "weight_transport": "pytorch_device_copy",
            "activation": "double_buffered_atomic",
            "max_batch_size": int(self.config.max_batch_size),
            "batch_wait_ms": float(self._runtime_batch_wait_ms),
            "auto_tune": {
                "enabled": bool(self.config.auto_tune),
                "round": int(self._tune_round),
                "current": {
                    "batch_wait_ms": float(self._tune_current[0]),
                    "active_replicas": int(self._tune_current[1]),
                },
                "candidates": [
                    {
                        "batch_wait_ms": float(candidate[0]),
                        "active_replicas": int(candidate[1]),
                        **dict(self._tune_scores.get(candidate, {})),
                    }
                    for candidate in self._tune_candidates
                ],
            },
            "replicas": [
                {
                    "replica_id": replica.replica_id,
                    "device": replica.device,
                    "policy_version": replica.policy_version,
                    "activated_ts": replica.activated_ts,
                    "requests": replica.requests,
                    "items": replica.items,
                    "batches": replica.batches,
                    "busy": bool(replica.busy),
                    "last_batch_size": replica.last_batch_size,
                    "last_inference_ms": replica.last_inference_ms,
                }
                for replica in self._replicas
            ],
        }

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        with self._condition:
            pending = list(self._queue)
            self._queue.clear()
            self._condition.notify_all()
        error = InferencePoolError("inference pool stopped")
        for request in pending:
            request.error = error
            request.done.set()
        for thread in self._threads:
            thread.join(timeout=3.0)
        self._threads.clear()
        if self._tune_thread is not None:
            self._tune_thread.join(timeout=3.0)
            self._tune_thread = None


__all__ = [
    "InferencePoolConfig",
    "InferencePoolError",
    "InferenceResult",
    "ReplicatedInferencePool",
    "available_cuda_devices",
    "auto_replica_count",
    "resolve_devices",
    "validate_devices",
]
