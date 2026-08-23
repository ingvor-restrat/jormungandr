from examples.benchmark_inference_throughput import (
    ThroughputConfig,
    run_benchmark,
)
from jormungandr.service import JormungandrHttpServer


def test_inference_throughput_benchmark_exercises_runtime_and_transports() -> None:
    assert JormungandrHttpServer.request_queue_size >= 128
    report = run_benchmark(
        ThroughputConfig(
            devices=("cpu",),
            replicas=1,
            obs_dim=8,
            hidden=8,
            actions=3,
            max_batch_size=8,
            batch_wait_ms=0.25,
            concurrency=8,
            direct_requests=32,
            http_requests=16,
            warmup_requests=8,
            explicit_batch_size=8,
            seed=17,
        )
    )

    assert report["schema"] == "jormungandr.inference_throughput.v1"
    assert report["environment"]["cpu_count"] >= 1
    rows = {row["name"]: row for row in report["results"]}
    assert set(rows) == {
        "runtime_concurrent_single",
        "runtime_explicit_batch",
        "http_json_single",
        "http_sbe_single",
    }
    assert rows["runtime_concurrent_single"]["items"] == 32
    assert rows["runtime_explicit_batch"]["items"] == 32
    assert rows["http_json_single"]["items"] == 16
    assert rows["http_sbe_single"]["items"] == 16
    for row in rows.values():
        assert row["items_per_s"] > 0.0
        assert row["errors"] == 0
        assert row["operation_latency_ms"]["p95"] >= 0.0
        assert row["formed_batches"] >= 1
        assert row["mean_formed_batch_size"] >= 1.0
