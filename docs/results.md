# Reproducible Results

These functional results were recorded for Jormungandr 0.1.0 on 25 July 2026.
They validate framework behavior and packaging; they are not learning-quality
or performance benchmark claims.

## Test suite

```bash
python -m pytest
```

Result:

| Suite | Tests | Result |
| --- | ---: | --- |
| replay and C51 correctness | 3 | passed |
| split-aware runtime, HTTP, and validation | 4 | passed |
| checkpoints and inference bundles | 1 | passed |
| generic trainer scheduling and environment | 2 | passed |
| OU spread process, reference, and scheduling | 3 | passed |
| **Total** | **13** | **passed** |

The tests cover zero-priority replay safety, probability preservation in the
C51 projection, held-out evaluation without parameter changes, action-value
to action-index mapping, training-only observation normalization, required
actor provenance, interleaved split routing, auxiliary validation, and
policy-versioned validation metrics. A loopback HTTP test exercises model
creation and interleaved experience ingestion through the public endpoint.
The artifact test saves and inspects a versioned checkpoint, scripts the C51
and auxiliary heads, and validates the manifest and generated C++ namespace.
The trainer tests verify deterministic environment behavior and the declared
interleaving schedule. The OU tests additionally verify finite accounting,
same-seed path identity, and positive expected reward for the stated
same-path reference.

## Syntax and package checks

```bash
python -m compileall -q src
python -m build
```

The package builds both a source distribution and wheel from
`pyproject.toml`. CI runs compilation, tests, and both package builds.

## Interleaved smoke run

The bounded runtime test creates one C51 model with two training and two
validation transitions in the same request:

```text
train -> validation -> train -> validation
```

Observed state:

```json
{
  "training_replay_size": 2,
  "validation_store_size": 2,
  "normalizer_observation_count": 4,
  "learner_updates": 1,
  "validation_runs": 1
}
```

The normalizer count is four because each of the two training transitions
contributes `obs` and `next_obs`. Validation observations contribute zero.

## Generic trainer

Command:

```bash
python examples/train_synthetic_control.py \
  --output-dir /tmp/jormungandr-synthetic-example
```

The default two-actor run completed eight training and two validation episodes
of 24 steps each. It produced:

```json
{
  "training_replay_size": 192,
  "validation_store_size": 48,
  "minimum_learner_updates": 10,
  "validation_runs": "at least 1"
}
```

The run also emitted a versioned manifest, episode JSON Lines, summary, and
checkpoint beneath the selected output directory. Reward values are synthetic
smoke results, not an algorithm comparison or convergence claim.

## OU spread recording

Command:

```bash
python docs/markup/record.py
```

The committed recording was generated from 28 training and seven interleaved
validation episodes, each 48 steps long, with two concurrent actors. Its final
frame reported:

```json
{
  "training_replay_size": 1344,
  "validation_store_size": 336,
  "learner_updates": 344,
  "validation_runs": 31,
  "train_aux_accuracy": 0.984,
  "validation_aux_accuracy": 0.992,
  "validation_reward_mean": 4.285,
  "same_path_reference_reward_mean": 4.230
}
```

The final deterministic-inference policy probe selected `LONG` at `-2z` and
`-1z`, `FLAT` at zero, and `SHORT` at `+1z` and `+2z`. Synthetic paths and
split assignment are fixed by seed. Exact learner update counts and learned
statistics can vary slightly because actors and the learner run concurrently.
These values demonstrate a functioning training and validation path; they are
not trading-performance or algorithm-comparison claims.

## Algorithm scope

C51 is exercised through ingestion, replay, training, held-out validation,
inference, checkpointing, and export. PPO and DDPG are included as development
building blocks and are not counted as service-supported algorithms.
