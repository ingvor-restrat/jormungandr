# Inference throughput benchmark

Jörmungandr includes a local benchmark for the fixed-vector inference paths:

- concurrent single-item calls through `JormungandrRuntime`;
- explicit batches through `JormungandrRuntime`;
- single-item JSON requests over HTTP;
- single-item SBE requests over HTTP.

Each path uses the same DQN, observations, devices, replica count, batching
limit, and caller concurrency. The report contains successful items per second,
operation-latency percentiles, batches formed by the inference pool, mean batch
size, and transport errors.

## Run it

CPU example:

```bash
PYTHONPATH=src python examples/benchmark_inference_throughput.py \
  --devices cpu --replicas 1 \
  --obs-dim 256 --hidden 512 --actions 32 \
  --max-batch-size 256 --batch-wait-ms 1 \
  --concurrency 128 --direct-requests 4096 --http-requests 1024 \
  --warmup-requests 256 --explicit-batch-size 256 \
  --torch-threads 16 --torch-interop-threads 4 \
  --json-output /tmp/jormungandr-throughput-cpu.json
```

One GPU:

```bash
PYTHONPATH=src python examples/benchmark_inference_throughput.py \
  --devices cuda:0 --replicas 1 \
  --obs-dim 256 --hidden 512 --actions 32 \
  --max-batch-size 256 --batch-wait-ms 1 \
  --concurrency 128 --direct-requests 4096 --http-requests 1024 \
  --warmup-requests 256 --explicit-batch-size 256 \
  --torch-threads 16 --torch-interop-threads 4 \
  --json-output /tmp/jormungandr-throughput-gpu.json
```

Two explicitly assigned GPUs use `--devices cuda:0,cuda:1 --replicas 2`.
`--skip-http` measures the runtime paths alone.

The JSON report records CPU count and load average, PyTorch thread settings,
GPU model and compute capability, and free GPU memory observed before model
creation. These fields make contention visible when comparing runs.

## Reading the results

`runtime_concurrent_single` measures calls that arrive independently and are
combined by the shared queue. Its `mean_formed_batch_size` shows how much
batching occurred at the requested concurrency.

`runtime_explicit_batch` measures actors that submit arrays directly. This is
the clearest measurement of model execution and result construction at large
batch sizes.

`http_json_single` and `http_sbe_single` include request encoding, a local TCP
connection, server dispatch, inference, response encoding, and client decoding.
They measure the complete supplied HTTP path. Each HTTP request contains one
observation; the server may combine concurrent requests into model batches.

Rates depend on model dimensions, batch size, device occupancy, NUMA placement,
PyTorch thread settings, and background load. Compare runs made with the same
command and inspect the recorded environment before treating a change as a
regression.

## Test-suite coverage

`tests/test_inference_throughput.py` runs a small CPU workload through all four
paths. It verifies completed item counts, zero transport errors, pool batching,
and valid timing data. The normal test suite uses no fixed items-per-second
threshold because its hosts vary.

A performance gate can run the benchmark command on a dedicated runner and
compare selected rates with that runner's stored baseline. Keeping the baseline
with the runner preserves sensitivity while allowing the package tests to run
on CPUs, GPUs, developer workstations, and shared CI hosts.
