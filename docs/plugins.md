# Algorithm, Replay, and Search Plugin Contracts

Algorithm choice is resolved by a registry rather than a service conditional.
Built-ins and installed packages use the same `AlgorithmPlugin` record:

```python
AlgorithmPlugin(
    name="example",
    version="1.0.0",
    family="off-policy value",
    build=build_agent,
    default_export_module="q",
    replay_mode="transition",
    enforce_policy_lag=False,
    runtime_defaults={},
    backend="python-torch",
)
```

`build_agent(obs_dim, config, device)` returns an object that provides:

- `action_result` and `inference_batch`;
- `update` and `evaluate_batch`;
- `state_dict` and `load_state_dict`;
- `action_values`, `device`, and `last_metrics`; and
- a Torch module named by `default_export_module`, or an explicit
  `export_module` method.

Updates return `UpdateResult(loss, priorities, metrics)`. Priorities align
one-for-one with the sampled rows so the shared replay service can update
them. A trajectory plugin receives contiguous episode fragments and their
behavior log probabilities, behavior values, masks, versions, and provenance
as metadata.

`enforce_policy_lag` is true for PPO, IMPALA, and APPO. PPO defaults to zero
lag so a fragment is consumed only at the behavior version; the asynchronous
profiles default to a bounded lag of 64 versions. Offline MARWIL also uses
trajectory fragments, but does not discard demonstrations as the learned
policy version advances.

## External discovery

An installed Python distribution can expose a plugin without modifying
Jörmungandr:

```toml
[project.entry-points."jormungandr.algorithms"]
my_algorithm = "my_package.plugin:PLUGIN"
```

The loaded value may be an `AlgorithmPlugin` or a zero-argument callable that
returns one. Names and aliases are canonicalized. A broken optional entry
point is isolated from the built-in registry. The service endpoint
`GET /v1/algorithms` reports the discovered name, version, family, backend,
replay mode, export module, description, and noise profile.

Algorithm-specific settings belong under `learner.plugin_config`; stable
settings shared by built-ins may also be promoted to `LearnerConfig`. Runtime
controls such as the action vocabulary, replay selector, trajectory age,
checkpoint cadence, and auxiliary-routing fields remain top-level so the
service and plugin cannot interpret different boundaries.
Checkpoint payloads retain the resolved algorithm name, plugin version,
backend, full learner configuration, optimizer state, and preprocessing state.
The outer format remains `jormungandr.checkpoint.v1`, preserving existing C51
checkpoints while allowing new plugin state dictionaries inside it. New
checkpoints refuse silent restore through a different plugin name or version;
a changed plugin must either retain its semantic checkpoint version or provide
an explicit migration.

Replay selection is independently replaceable through the
`jormungandr.rollout_selectors` entry-point group. A selector receives the
shared replay buffer and returns the chosen batch, physical indices,
importance weights, scalar metrics, and an optional audit record. The built-in
`prioritized` and `qubo` selectors use this contract.

Search-frontier pruning is a separate boundary again. Installed packages can
publish a pruner without coupling it to replay or a policy implementation:

```toml
[project.entry-points."jormungandr.frontier_pruners"]
my_pruner = "my_package.search:build_pruner"
```

A pruner receives `SearchNode` records containing a stable key, cheap utility
estimate, similarity embedding, and caller-owned payload. It returns exactly
the requested number of nodes plus metrics and a binary decision audit. The
built-in `utility` baseline implements ordinary top-k beam search; `qubo`
reuses the binary utility/diversity solver. `bounded_beam_search` accounts for
generated, expanded, retained, and pruned nodes and reports selection time.
Volt can therefore retain ownership of option-graph state and path evaluation
while importing only the bounded search and pruner contract.

## A future Rust boundary

Rust should enter at measured hot spots, not replace Torch policy code merely
because an algorithm has a loop. The likely order is:

1. replay indexing, contiguous-fragment assembly, and QUBO frontier/local
   search;
2. graph collation and feature preprocessing;
3. GAE, V-trace, distribution projection, or other CPU kernels shown by a
   profiler to dominate; and
4. transport and shared-memory batches when JSON becomes material.

A PyO3/maturin extension can implement the existing NumPy/plain-mapping
boundary. Torch remains responsible for modules, autograd, optimizers, and
device execution. Passing contiguous arrays or DLPack-compatible tensors
across that boundary avoids a second model runtime and preserves TorchScript
exports.

Native plugins should retain:

- a semantic plugin name and version independent of crate version;
- the same `UpdateResult` metric and priority contract;
- explicit dtype, shape, byte-order, and ownership rules;
- deterministic seeds and a reference Python implementation for parity; and
- checkpoint migrations that preserve tensor names or declare a format
  conversion.

The `backend` field already distinguishes `python-torch` from a future value
such as `rust-pyo3-torch`. No Rust implementation is implied by the current
release.
