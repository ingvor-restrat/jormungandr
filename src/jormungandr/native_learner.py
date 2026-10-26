"""Optional native learners with resident state and bounded JSONL transport.

Backend commands are administrator configuration. HTTP callers select a
registered backend; requests never select an executable or checkpoint path.
This module has no Torch dependency and leaves autodiff tensors in the worker.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping
import uuid

MAX_MESSAGE_BYTES = 16 * 1024 * 1024
SCHEMA = "volt.native_learner.v2"


class NativeWorkerError(RuntimeError):
    """Worker failed; an interrupted update may have an unknown outcome."""


@dataclass(frozen=True)
class NativeBackend:
    command: tuple[str, ...]
    protocol: str = SCHEMA
    timeout_seconds: float = 60.0


def configured_native_backends() -> dict[str, NativeBackend]:
    executable = os.environ.get("VOLT_LEARNER_BIN", "").strip()
    if not executable:
        return {}
    path = Path(executable).expanduser().resolve()
    if not path.is_file():
        raise ValueError("VOLT_LEARNER_BIN must name an installed native learner")
    return {"volt_burn": NativeBackend((str(path),))}


class NativeWorker:
    def __init__(self, backend: NativeBackend):
        if not backend.command or not 0 < backend.timeout_seconds <= 3600:
            raise ValueError("invalid native backend configuration")
        self.backend = backend
        self._lock = threading.Lock()
        self._stderr = tempfile.TemporaryFile()
        self._closed = False
        try:
            self.process = subprocess.Popen(
                backend.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr,
            )
        except Exception:
            self._stderr.close()
            raise

    def request(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        payload = dict(body)
        payload["id"] = request_id
        wire = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(wire) > MAX_MESSAGE_BYTES:
            raise ValueError("native request exceeds 16 MiB limit")
        with self._lock:
            if self._closed or self.process.poll() is not None:
                raise NativeWorkerError("native learner is unavailable; restore its last checkpoint")
            replies: queue.Queue[Any] = queue.Queue(maxsize=1)

            def exchange() -> None:
                try:
                    assert self.process.stdin is not None and self.process.stdout is not None
                    # Raw pipes may perform partial writes and reads.
                    view = memoryview(wire)
                    while view:
                        written = self.process.stdin.write(view)
                        if not written:
                            raise EOFError("native stdin closed")
                        view = view[written:]
                    self.process.stdin.flush()
                    response = self.process.stdout.readline(MAX_MESSAGE_BYTES + 1)
                    if not response or len(response) > MAX_MESSAGE_BYTES or not response.endswith(b"\n"):
                        raise NativeWorkerError("missing, incomplete or oversized native response")
                    replies.put(json.loads(response))
                except Exception as exc:
                    replies.put(exc)

            thread = threading.Thread(target=exchange, daemon=True, name="jormungandr-native-request")
            thread.start()
            try:
                response = replies.get(timeout=self.backend.timeout_seconds)
            except queue.Empty as exc:
                self._stop()
                raise NativeWorkerError("native request timed out; update outcome is unknown; restore a checkpoint") from exc
            if isinstance(response, Exception):
                self._stop()
                raise NativeWorkerError(f"native protocol failure: {response}") from response
            if not isinstance(response, dict) or response.get("id") != request_id or response.get("schema") != self.backend.protocol or type(response.get("ok")) is not bool:
                self._stop()
                raise NativeWorkerError("native response identity/schema mismatch")
            if not response["ok"]:
                raise ValueError(str(response.get("error", "native command rejected")))
            if not isinstance(response.get("result"), dict):
                self._stop()
                raise NativeWorkerError("native result must be an object")
            return response["result"]

    def _stop(self) -> None:
        self._closed = True
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        for pipe in [self.process.stdin, self.process.stdout]:
            if pipe is not None:
                pipe.close()
        self._stderr.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._stop()


class NativeLearnerManager:
    """Independent native-model namespace beside existing Jormungandr learners."""
    OPERATIONS = frozenset({"register_dataset", "upload", "drop_batch", "update", "evaluate", "infer", "checkpoint", "status"})

    def __init__(self, backends: Mapping[str, NativeBackend] | None = None, *, max_models: int = 4):
        self.backends = dict(configured_native_backends() if backends is None else backends)
        self.max_models = max_models
        self._models: dict[str, tuple[str, NativeWorker]] = {}
        self._metrics: dict[str, deque] = {}
        self._lock = threading.RLock()
        self._closed = False

    def capabilities(self) -> dict[str, Any]:
        return {"backends": [{"name": name, "protocol": spec.protocol} for name, spec in sorted(self.backends.items())],
                "operations": sorted(self.OPERATIONS), "max_models": self.max_models,
                "max_message_bytes": MAX_MESSAGE_BYTES}

    def create(self, body: Mapping[str, Any]) -> dict[str, Any]:
        if set(body) - {"model_id", "backend", "config", "checkpoint"}:
            raise ValueError("unsupported native model fields")
        model_id = str(body.get("model_id") or uuid.uuid4().hex)
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", model_id) is None:
            raise ValueError("invalid native model_id")
        backend_name = str(body.get("backend", "volt_burn"))
        if backend_name not in self.backends:
            raise ValueError("native backend is not configured by the service administrator")
        if "checkpoint" in body and "config" in body:
            raise ValueError("supply config or checkpoint, rather than both")
        payload = {"op": "restore", "checkpoint": body["checkpoint"]} if "checkpoint" in body else {"op": "configure", "config": body.get("config", {})}
        with self._lock:
            if self._closed or model_id in self._models or len(self._models) >= self.max_models:
                raise ValueError("native model already exists or model capacity is exhausted")
            worker = NativeWorker(self.backends[backend_name])
            try:
                result = worker.request(payload)
            except Exception:
                worker.close()
                raise
            self._models[model_id] = (backend_name, worker)
            self._metrics[model_id] = deque(maxlen=256)
            return {"model_id": model_id, "backend": backend_name, **result}

    def command(self, model_id: str, operation: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if operation not in self.OPERATIONS:
            raise ValueError("unsupported native operation")
        if body is not None and ({"id", "op"} & set(body)):
            raise ValueError("native transport identity and operation are service-owned")
        with self._lock:
            _, worker = self._models[model_id]
        start = time.monotonic()
        result = worker.request({**dict(body or {}), "op": operation})
        if operation in {"update", "evaluate"}:
            record = {"operation": operation, "elapsed_ms": (time.monotonic()-start)*1000,
                      **{k: result[k] for k in ["version", "loss", "loss_before_update", "gradient_l2_norm", "split", "batch_sha256", "paths"] if k in result}}
            with self._lock:
                if model_id in self._metrics:
                    self._metrics[model_id].append(record)
        return result

    def list_models(self) -> list[str]:
        with self._lock:
            return sorted(self._models)

    def metrics(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            return {"model_id": model_id, "history": list(self._metrics[model_id])}

    def delete(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            _, worker = self._models.pop(model_id)
            self._metrics.pop(model_id)
        worker.close()
        return {"deleted": model_id}

    def close_all(self) -> None:
        with self._lock:
            self._closed = True
            models = list(self._models)
        for model in models:
            self.delete(model)
