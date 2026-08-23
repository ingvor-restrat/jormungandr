"""Replicated, dynamically batched inference with atomic policy activation.

The pool deliberately owns neither an RL algorithm nor a transport protocol.
Callers provide an agent factory and a batch inference function; the pool owns
replica placement, request scheduling, hot weight refresh, and telemetry.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Hashable, Mapping, Optional, Sequence


class InferencePoolError(RuntimeError):
    """Raised when a serving replica or queued inference request fails."""


@dataclass(frozen=True)
class InferencePoolConfig:
    """Process-local inference resources shared by one policy model."""

    devices: tuple[str, ...] = ()
    replicas: int = 1
    max_batch_size: int = 64
    batch_wait_ms: float = 2.0
    request_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if int(self.replicas) <= 0:
            raise ValueError("inference replicas must be > 0")
        if int(self.max_batch_size) <= 0:
            raise ValueError("inference max_batch_size must be > 0")
        if float(self.batch_wait_ms) < 0.0:
            raise ValueError("inference batch_wait_ms must be >= 0")
        if float(self.request_timeout_s) <= 0.0:
            raise ValueError("inference request_timeout_s must be > 0")


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
        if not device:
            continue
        if device == "cpu":
            out.append(device)
            continue
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
        out.append(f"cuda:{index}")
    return tuple(out)


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
        devices = self.config.devices or (str(fallback_device),)
        devices = validate_devices(devices)
        if not devices:
            raise InferencePoolError("no inference devices are configured")
        activated_ts = time.time()
        snapshot_metadata = dict(metadata or {})
        for replica_id in range(int(self.config.replicas)):
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
        for replica in self._replicas:
            thread = threading.Thread(
                target=self._worker,
                args=(replica,),
                name=f"inference-replica-{replica.replica_id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _take_batch(self) -> list[_Request]:
        with self._condition:
            while not self._queue and not self._stop.is_set():
                self._condition.wait(timeout=0.25)
            if self._stop.is_set():
                return []
            first = self._queue.popleft()
            batch = [first]
            deadline = time.monotonic() + float(self.config.batch_wait_ms) / 1000.0
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
            batch = self._take_batch()
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
            "devices": sorted({replica.device for replica in self._replicas}),
            "queued_requests": int(queued),
            "refreshes": int(self._refreshes),
            "last_refresh_ms": float(self._last_refresh_ms),
            "scheduler": "shared_dynamic_batch_queue",
            "weight_transport": "pytorch_device_copy",
            "activation": "double_buffered_atomic",
            "max_batch_size": int(self.config.max_batch_size),
            "batch_wait_ms": float(self.config.batch_wait_ms),
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


__all__ = [
    "InferencePoolConfig",
    "InferencePoolError",
    "InferenceResult",
    "ReplicatedInferencePool",
    "available_cuda_devices",
    "validate_devices",
]
