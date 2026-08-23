"""Measure Jormungandr fixed-vector inference throughput on the local host.

The benchmark reports observed rates and latency distributions. It deliberately
sets no universal pass threshold because results depend on the model, devices,
thread settings, and host load.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.request import Request, urlopen

import numpy as np
import torch

from jormungandr.generated_sbe import (
    decode_policy_infer_response,
    encode_policy_infer_request,
)
from jormungandr.service import JormungandrHttpServer, JormungandrRuntime


@dataclass(frozen=True)
class ThroughputConfig:
    devices: tuple[str, ...] = ("cpu",)
    replicas: int = 1
    obs_dim: int = 256
    hidden: int = 512
    actions: int = 32
    max_batch_size: int = 256
    batch_wait_ms: float = 1.0
    concurrency: int = 128
    direct_requests: int = 4096
    http_requests: int = 1024
    warmup_requests: int = 256
    explicit_batch_size: int = 256
    seed: int = 20260823

    def __post_init__(self) -> None:
        positive = {
            "replicas": self.replicas,
            "obs_dim": self.obs_dim,
            "hidden": self.hidden,
            "actions": self.actions,
            "max_batch_size": self.max_batch_size,
            "concurrency": self.concurrency,
            "direct_requests": self.direct_requests,
            "explicit_batch_size": self.explicit_batch_size,
        }
        invalid = [name for name, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"positive benchmark settings required: {', '.join(invalid)}")
        if int(self.http_requests) < 0 or int(self.warmup_requests) < 0:
            raise ValueError("request counts must be non-negative")
        if float(self.batch_wait_ms) < 0.0:
            raise ValueError("batch_wait_ms must be non-negative")
        if not self.devices:
            raise ValueError("at least one inference device is required")


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def _measure(
    name: str,
    tasks: Iterable[Any],
    work: Callable[[Any], int],
    *,
    concurrency: int,
) -> dict[str, Any]:
    task_values = list(tasks)

    def timed(task: Any) -> tuple[int, float, str]:
        started = time.perf_counter()
        try:
            items = int(work(task))
            error = ""
        except Exception as exc:
            items = 0
            error = f"{type(exc).__name__}: {exc}"
        return items, (time.perf_counter() - started) * 1000.0, error

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        rows = list(executor.map(timed, task_values))
    elapsed = max(1e-9, time.perf_counter() - started)
    items = sum(row[0] for row in rows)
    latencies = [row[1] for row in rows]
    errors = [row[2] for row in rows if row[2]]
    return {
        "name": name,
        "operations": len(rows),
        "items": int(items),
        "elapsed_s": float(elapsed),
        "operations_per_s": float(len(rows) / elapsed),
        "items_per_s": float(items / elapsed),
        "errors": len(errors),
        "error_samples": list(dict.fromkeys(errors))[:5],
        "operation_latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else 0.0,
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
            "p99": _percentile(latencies, 99.0),
            "max": max(latencies, default=0.0),
        },
    }


def _exercise(
    count: int,
    work: Callable[[int], int],
    *,
    concurrency: int,
) -> None:
    if count <= 0:
        return
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        list(executor.map(work, range(int(count))))


def _pool_totals(runtime: JormungandrRuntime, model_id: str) -> dict[str, int]:
    pool = runtime.get_model(model_id)["inference"]["pool"]
    replicas = pool.get("replicas", [])
    return {
        "items": sum(int(row.get("items", 0)) for row in replicas),
        "batches": sum(int(row.get("batches", 0)) for row in replicas),
    }


def _attach_pool_delta(
    result: dict[str, Any],
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> None:
    items = max(0, int(after["items"]) - int(before["items"]))
    batches = max(0, int(after["batches"]) - int(before["batches"]))
    result["formed_batches"] = batches
    result["mean_formed_batch_size"] = (
        float(items / batches) if batches else 0.0
    )


def _environment() -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free_memory, total_memory = torch.cuda.mem_get_info(index)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "memory_bytes": int(total_memory),
                "free_memory_bytes": int(free_memory),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = []
    return {
        "captured_unix_s": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_average_1m_5m_15m": load_average,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpus,
    }


def run_benchmark(
    config: ThroughputConfig,
    *,
    include_http: bool = True,
) -> dict[str, Any]:
    """Run the benchmark and return a JSON-serializable report."""

    environment = _environment()
    rng = np.random.default_rng(int(config.seed))
    observation_count = max(
        8,
        min(1024, config.direct_requests, max(1, config.http_requests)),
    )
    observations = rng.normal(
        0.0, 1.0, size=(observation_count, config.obs_dim)
    ).astype(np.float32)
    model_id = "inference-throughput"
    runtime = JormungandrRuntime(
        inference_devices=config.devices,
        inference_replicas=config.replicas,
        inference_max_batch_size=config.max_batch_size,
        inference_batch_wait_ms=config.batch_wait_ms,
        inference_auto_tune=False,
    )
    runtime.create_model(
        obs_dim=config.obs_dim,
        model_id=model_id,
        tensorboard_enabled=False,
        learner={
            "enabled": True,
            "algo": "dqn",
            "device": config.devices[0],
            "hidden": config.hidden,
            "action_values": [float(index) for index in range(config.actions)],
            "min_replay": 1_000_000,
            "checkpoint_every": 0,
        },
    )

    def direct_one(index: int) -> int:
        result = runtime.policy_infer(
            model_id,
            obs=observations[index % observation_count],
            deterministic=True,
        )
        if not 0 <= int(result["action_idx"]) < config.actions:
            raise RuntimeError("inference returned an invalid action index")
        return 1

    results: list[dict[str, Any]] = []
    server: JormungandrHttpServer | None = None
    server_thread: threading.Thread | None = None
    try:
        _exercise(
            config.warmup_requests,
            direct_one,
            concurrency=min(config.concurrency, config.warmup_requests or 1),
        )
        before = _pool_totals(runtime, model_id)
        direct = _measure(
            "runtime_concurrent_single",
            range(config.direct_requests),
            direct_one,
            concurrency=config.concurrency,
        )
        _attach_pool_delta(direct, before, _pool_totals(runtime, model_id))
        results.append(direct)

        batch_size = min(config.explicit_batch_size, config.direct_requests)
        batch_sizes = [batch_size] * (config.direct_requests // batch_size)
        remainder = config.direct_requests % batch_size
        if remainder:
            batch_sizes.append(remainder)

        def direct_batch(task: tuple[int, int]) -> int:
            start, size = task
            batch = [
                observations[(start + offset) % observation_count]
                for offset in range(size)
            ]
            result = runtime.policy_infer_batch(
                model_id,
                obs_batch=batch,
                deterministic=True,
            )
            if int(result["batch_size"]) != size:
                raise RuntimeError("batch inference returned the wrong item count")
            return size

        batch_tasks: list[tuple[int, int]] = []
        offset = 0
        for size in batch_sizes:
            batch_tasks.append((offset, size))
            offset += size
        before = _pool_totals(runtime, model_id)
        explicit = _measure(
            "runtime_explicit_batch",
            batch_tasks,
            direct_batch,
            concurrency=min(config.concurrency, len(batch_tasks)),
        )
        _attach_pool_delta(explicit, before, _pool_totals(runtime, model_id))
        results.append(explicit)

        if include_http and config.http_requests > 0:
            server = JormungandrHttpServer("127.0.0.1", 0, runtime=runtime)
            server_thread = threading.Thread(
                target=server.serve_forever,
                name="throughput-http-server",
                daemon=True,
            )
            server_thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            json_url = f"{base_url}/v1/models/{model_id}/policy/infer"
            sbe_url = f"{base_url}/v1/models/{model_id}/policy/infer-sbe"

            def http_json(index: int) -> int:
                body = json.dumps(
                    {
                        "obs": observations[index % observation_count].tolist(),
                        "deterministic": True,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                request = Request(
                    json_url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=30.0) as response:
                    payload = json.load(response)
                if not payload.get("ok"):
                    raise RuntimeError("JSON inference failed")
                return 1

            def http_sbe(index: int) -> int:
                body = encode_policy_infer_request(
                    observations[index % observation_count],
                    deterministic=True,
                    epsilon=0.0,
                    request_id=index,
                )
                request = Request(
                    sbe_url,
                    data=body,
                    headers={
                        "Content-Type": "application/vnd.fixprotocol.sbe"
                    },
                    method="POST",
                )
                with urlopen(request, timeout=30.0) as response:
                    payload = decode_policy_infer_response(response.read())
                if int(payload["request_id"]) != index:
                    raise RuntimeError("SBE response request ID mismatch")
                return 1

            http_warmup = min(config.warmup_requests, config.http_requests)
            _exercise(
                http_warmup,
                http_json,
                concurrency=min(config.concurrency, http_warmup or 1),
            )
            before = _pool_totals(runtime, model_id)
            json_result = _measure(
                "http_json_single",
                range(config.http_requests),
                http_json,
                concurrency=config.concurrency,
            )
            _attach_pool_delta(
                json_result, before, _pool_totals(runtime, model_id)
            )
            results.append(json_result)

            _exercise(
                http_warmup,
                http_sbe,
                concurrency=min(config.concurrency, http_warmup or 1),
            )
            before = _pool_totals(runtime, model_id)
            sbe_result = _measure(
                "http_sbe_single",
                range(config.http_requests),
                http_sbe,
                concurrency=config.concurrency,
            )
            _attach_pool_delta(
                sbe_result, before, _pool_totals(runtime, model_id)
            )
            results.append(sbe_result)

        model = runtime.get_model(model_id)
        return {
            "schema": "jormungandr.inference_throughput.v1",
            "environment": environment,
            "config": {**asdict(config), "devices": list(config.devices)},
            "results": results,
            "inference_pool": model["inference"]["pool"],
        }
    finally:
        if server is not None:
            server.shutdown()
        if server_thread is not None:
            server_thread.join(timeout=5.0)
        if server is not None:
            server.server_close()
        runtime.close_all()


def _print_report(report: Mapping[str, Any]) -> None:
    env = report["environment"]
    config = report["config"]
    print(
        f"host: cpu={env['cpu_count']} torch_threads={env['torch_threads']} "
        f"devices={','.join(config['devices'])} replicas={config['replicas']}"
    )
    print(
        f"model: obs={config['obs_dim']} hidden={config['hidden']} "
        f"actions={config['actions']} max_batch={config['max_batch_size']}"
    )
    for row in report["results"]:
        latency = row["operation_latency_ms"]
        print(
            f"{row['name']:<28} {row['items_per_s']:>11,.0f} items/s  "
            f"p50={latency['p50']:>7.2f} ms p95={latency['p95']:>7.2f} ms  "
            f"formed_batch={row['mean_formed_batch_size']:>6.1f}  "
            f"errors={row['errors']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--obs-dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--actions", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--batch-wait-ms", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--direct-requests", type=int, default=4096)
    parser.add_argument("--http-requests", type=int, default=1024)
    parser.add_argument("--warmup-requests", type=int, default=256)
    parser.add_argument("--explicit-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--torch-interop-threads", type=int, default=0)
    parser.add_argument("--skip-http", action="store_true")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_interop_threads)
    devices = tuple(
        value.strip() for value in args.devices.split(",") if value.strip()
    )
    config = ThroughputConfig(
        devices=devices,
        replicas=args.replicas,
        obs_dim=args.obs_dim,
        hidden=args.hidden,
        actions=args.actions,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        concurrency=args.concurrency,
        direct_requests=args.direct_requests,
        http_requests=args.http_requests,
        warmup_requests=args.warmup_requests,
        explicit_batch_size=args.explicit_batch_size,
        seed=args.seed,
    )
    report = run_benchmark(config, include_http=not args.skip_http)
    _print_report(report)
    if args.json_output:
        path = Path(args.json_output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
