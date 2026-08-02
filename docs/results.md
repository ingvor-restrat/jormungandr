# Reproducible Results

These functional results were recorded for Jörmungandr 0.2.0 on 2 August 2026.
They validate framework behavior and packaging; they are not learning-quality
or performance benchmark claims.

## Test suite

```bash
python -m pytest
```

Result:

| Suite | Tests | Result |
| --- | ---: | --- |
| replay, masks, graph trajectories, and C51 correctness | 9 | passed |
| algorithm, QUBO, quantile, and plugin lifecycle | 19 | passed |
| split-aware runtime, HTTP, validation, and internal comparison | 5 | passed |
| checkpoints and inference bundles | 1 | passed |
| generic trainer scheduling and environment | 2 | passed |
| OU spread process, reference, and scheduling | 3 | passed |
| branch-frontier pruning and oracle benchmark contract | 5 | passed |
| **Total** | **44** | **passed** |

The tests cover zero-priority replay safety, probability preservation in the
C51 projection, legal masking, graph-reference GAE, held-out evaluation
without parameter changes, action-value
to action-index mapping, training-only observation normalization, required
actor provenance, interleaved split routing, auxiliary validation, and
policy-versioned validation metrics. A loopback HTTP test exercises model
creation and interleaved experience ingestion through the public endpoint.
The plugin matrix performs a finite update, masked inference, and state-dict
round trip for every built-in algorithm. Dedicated tests cover QR-DQN return
quantiles and CVaR scoring, finite categorical-SAC masked entropy, exact QUBO
cardinality, rollout-fragment decisions, and policy-lag exclusion. The
MARWIL test verifies contiguous discounted returns and its deliberate
exemption from online policy aging. The
artifact test saves and inspects a versioned checkpoint, scripts the C51
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

C51, QR-DQN, DQN, CQL, maximum-entropy soft Q, BC, MARWIL, PPO, IMPALA,
APPO, categorical SAC, and the compact vector DreamerV3 profile were each
exercised through the threaded service with eight legal-mask-aware
transitions. Every learner completed an update, checkpointed, exported its
default TorchScript module, restored into a new runtime, and produced masked
inference. C51 exported `[N, 3, 51]` logits; the configured QR-DQN exported
`[N, 3, 13]` quantile values; policy and scalar-Q modules exported `[N, 3]`.

This is a lifecycle matrix, not an equivalence test against reference
benchmarks and not evidence that one algorithm learns better than another.
The DreamerV3 result applies only to the documented compact vector profile.
Replay QUBO tests establish objective construction, cardinality, audit output,
and rollout integration. The separate controlled search benchmark below tests
frontier efficiency; neither result establishes a Volt learning advantage.

## Online transition-agent convergence

The first learning comparison is deliberately restricted to algorithms that
can consume the same online one-step transition stream: DQN, C51, QR-DQN,
maximum-entropy Soft-Q, and categorical SAC. Five independently initialized
runs per algorithm each receive 64 fixed-schedule training episodes of 32
steps. After a 128-transition warm-up, every run performs one update per new
transition, yielding 1,921 updates. Evaluation uses the same 16 held-out path
seeds at every checkpoint and never changes the model.

Command:

```bash
PYTHONPATH=src python examples/compare_ou_algorithms.py \
  --runs 5 --train-episodes 64 --eval-interval 4 \
  --eval-episodes 16 --horizon 32 \
  --json-output docs/latex/figures/ou_algorithm_convergence.json \
  --plot-output docs/latex/figures/ou_algorithm_convergence.pdf \
  --convergence-gif-output docs/markup/algorithm-convergence.gif \
  --playback-gif-output docs/markup/algorithm-playback.gif
```

Recorded on 2 August 2026:

| algorithm | clean return | normal 95% CI | sensor-noise return | normal 95% CI |
| --- | ---: | ---: | ---: | ---: |
| DQN | 3.4413 | [3.1439, 3.7387] | 3.6837 | [3.5181, 3.8493] |
| C51 | 3.2101 | [3.0546, 3.3655] | 3.1983 | [3.0047, 3.3919] |
| QR-DQN | 3.4382 | [3.1950, 3.6814] | 3.5558 | [3.3141, 3.7974] |
| MaxEnt Soft-Q | 3.6087 | [3.3839, 3.8334] | 3.7893 | [3.5476, 4.0310] |
| categorical SAC | 3.5508 | [3.2913, 3.8104] | 3.9155 | [3.7581, 4.0728] |

The deterministic threshold reference returned 2.8367 on the same paths. Most
agents crossed that level by the episode-eight checkpoint; C51 improved more
slowly. MaxEnt Soft-Q has the largest final clean point estimate and SAC the
largest sensor-noise point estimate, but their intervals overlap several other
agents. Noise also improves some point estimates in this threshold-like
environment, so this study does not establish general noise robustness or an
algorithm ranking.

The [animated curves and same-path playback](markup/README.md) are explanatory
views of the committed JSON. PPO/APPO/IMPALA require behavior-policy
trajectories; BC/MARWIL/CQL require a declared offline dataset; DreamerV3 needs
a sequence and model-compute budget. Those cohorts remain separate by design.

## QUBO branch-search benchmark

Command:

```bash
PYTHONPATH=src python examples/benchmark_qubo_branch_search.py \
  --trials 500 --widths 4,6,8,10,12,16 --seed 20260802
```

The environment is a seeded four-way, depth-six synthetic tree with 4,096
terminal paths. A noisy proxy is correlated within the first branch; the exact
subtree optimum plus that noise defines the proxy, and the exact terminal
optimum is retained for scoring. Both selectors receive the same trees and
beam budget.

At width eight, utility-only search recovered the exact optimum on 72.2% of
trials with mean regret 0.1435. QUBO recovered 73.2% with mean regret 0.1305.
The paired mean-regret difference was 0.0130 in QUBO's favor, with a normal
95% confidence interval from 0.0022 to 0.0239. Selection averaged 0.09 ms for
utility-only search and 2.87 ms for QUBO.

QUBO width eight was near utility width ten in aggregate quality while
generating 148 rather than 180 candidates and evaluating eight rather than ten
terminal paths. Its additional selection time implied a terminal-only
break-even cost of approximately 1.38 ms per avoided terminal evaluation.
This is a positive controlled result, but it is deliberately not described as
general QUBO superiority: the tree and proxy are synthetic, the built-in
solver is classical and heuristic, and no Volt path was evaluated.
