# Reproducible Results

These functional results were recorded for Jörmungandr 0.2.0 through 4 August
2026.
They validate framework behavior and packaging; they are not learning-quality
or performance benchmark claims.

## Test suite

```bash
python -m pytest
```

Result:

| Suite | Tests | Result |
| --- | ---: | --- |
| core algorithm, replay, and artifact lifecycle | 32 | passed |
| runtime, actors, monitor, trainer, and OU example | 13 | passed |
| search, joint-action composition, and constrained environment | 13 | passed |
| structured representation, transition/PPO service, export, and parity | 15 | passed |
| structured joint trajectory and multiprocess service | 13 | passed |
| structured reward-free supervision and behavior cloning | 6 | passed |
| **Total** | **92** | **passed** |

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

## Independent CartPole PPO reference

The trajectory implementation diagnostic compares Jörmungandr's real PPO
plugin with Stable-Baselines3 2.9.0 in the pinned CPU environment at
[`benchmarks/requirements-sb3-reference.txt`](../benchmarks/requirements-sb3-reference.txt).
Both implementations have exactly 9,155 trainable parameters: separate
two-layer, width-64 policy and value MLPs. They share the major PPO controls,
1,024-step rollouts, and a 49,152-interaction budget. Deterministic evaluation
uses the same twenty held-out reset seeds every 4,096 interactions.

Command:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTHONPATH="$PWD/src" \
  /path/to/sb3-reference/bin/python examples/benchmark_gym_ppo.py \
  --runs 3 --total-timesteps 49152 --rollout-steps 1024 \
  --evaluation-every-timesteps 4096 --evaluation-episodes 20 \
  --json-output docs/latex/figures/cartpole_ppo_reference.json \
  --plot-output docs/latex/figures/cartpole_ppo_reference.pdf
```

Recorded on 4 August 2026:

| implementation | first solved run 0 | run 1 | run 2 | final held-out means |
| --- | ---: | ---: | ---: | ---: |
| Jörmungandr PPO | 20,480 | 20,480 | 20,480 | 500 / 500 / 500 |
| Stable-Baselines3 PPO | 8,192 | 24,576 | 12,288 | 500 / 500 / 500 |

The built-in PPO therefore solves the reference control task at matched model
capacity and budget. SB3 often reaches the threshold earlier, and both methods
have non-monotone intermediate evaluations. Jörmungandr's final
behavior-log-probability and behavior-value fallback rates are zero in every
run, confirming that PPO consumed recorded behavior-policy data rather than
silently recomputing it. This controls for a gross vector-PPO implementation
bug; it does not validate an application's reward, structured representation,
joint-action trajectory, or opponent protocol.

## Structured CartPole representation parity

The S0 control reuses the flat Jörmungandr CartPole cohort above and changes
only its interface and compatible encoder. Four scalar coordinates become
typed entities with stable IDs, and left/right become state-local candidates.
The structured transformer has 20,674 parameters. Seeds, reward, PPO controls,
49,152-turn budget, evaluation interval, and twenty held-out resets are fixed.

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTHONPATH="$PWD/src:$PWD" \
  /path/to/sb3-reference/bin/python \
  examples/benchmark_structured_cartpole_ppo.py \
  --flat-reference docs/latex/figures/cartpole_ppo_reference.json \
  --json-output docs/latex/figures/structured_cartpole_parity.json \
  --plot-output docs/latex/figures/structured_cartpole_parity.pdf
```

| representation | first solved run 0 | run 1 | run 2 | final held-out means |
| --- | ---: | ---: | ---: | ---: |
| flat MLP | 20,480 | 20,480 | 20,480 | 500 / 500 / 500 |
| entity/candidate transformer | 8,192 | 32,768 | 12,288 | 500 / 500 / 500 |

The declared S0 gate passes. Structured final median is 500, and its median
first-solved checkpoint is 12,288, below twice the flat 20,480 reference.
Semantic logits and values are invariant to entity/candidate permutations
within the declared `1e-5` tolerance. Wire round trip, trajectory storage,
checkpoint restore, and the versioned structured inference bundle preserve
candidate identity. This localizes any later Kaggriculture failure away from a
gross entity/candidate representation defect.

## Structured joint-trajectory service contract

The J0 functional gate uses two spawned actor processes and one central
`structured_ppo` service model. The actors submit different factor counts, but
four environment turns produce four reward-bearing trajectory steps. The
service rejects stale, duplicate, future, or out-of-order provenance, keeps
validation isolated, updates the common model, and persists PPO loss, KL,
entropy, value loss, explained variance, gradient norm, actor latency, and
policy lag. A checkpoint restored into a clean frozen model reproduces policy
logits and value. This is a lifecycle/formulation test, not yet an
environment-learning benchmark.

The same contract now has two equivalent HTTP wires. The compact sequence
round trip stores `N + 1` observations for `N` steps, reconstructs shared
adjacent observations, and is smaller than the legacy step-array JSON. The
loopback service test sends gzip-compressed requests and admits a compact
validation trajectory through the public endpoint without routing it into
training.

Structured PPO 1.2 adds pre-GAE signal diagnostics to every update. The unit
control with two terminal returns, `1.0` and `-0.5`, reports mean `0.25`,
standard deviation `0.75`, range `[-0.5, 1.0]`, two unique returns, mean length
three, and a one-third nonzero-reward fraction. The exact-joint control with
one two-step episode reports return `1.0`, zero spread, one unique return, and
a one-half nonzero-reward fraction. These metrics are descriptive; they do not
modify advantages or the PPO objective.

## Structured behavior-cloning contract

The BC0 gate trains `structured_bc` on eight weighted semantic labels covering
two factor families and four independent candidate permutations. The tiny
corpus reaches 100% validation accuracy. Reordering candidate rows while
retaining IDs preserves the semantic logits within `1e-5`; validation
evaluation leaves every policy parameter unchanged. The supervision record has
no reward field, requires its factor candidates to be currently legal, rejects
duplicate provenance, and reports overall plus source/factor accuracy, NLL,
entropy, calibration error, and gradient norm.

A central service test sends interleaved training and validation labels into
separate bounded buffers, performs updates only from the training split, and
checkpoints the learned entity/candidate transformer. A frozen
`structured_ppo` model initialized from that checkpoint produces identical
candidate logits while beginning at update and policy version zero. BC0 passes;
this validates the generic imitation and initialization contract, not expert
quality or downstream return.

## Constrained joint-action learning gate

`ConstrainedWorkbench-v0` is a Gymnasium-compatible generic assignment task
with no application concepts. Each three-turn episode observes two to four
typed workers and two to four state-local jobs. Every worker selects one job or
PASS; a job can be used once, jobs consume shared capacity, and conflict groups
are mutually exclusive. Step utility is hidden from the learner: reward is zero
until termination, when accumulated utility is divided by the exact enumerated
episode optimum. The oracle therefore returns 1.0 by construction.

All learned arms use the same 12,770-parameter width-32, one-layer
entity/candidate transformer. BC receives 256 oracle training episodes and 500
updates. Random-start PPO and BC-initialized PPO each receive 12,288 environment
turns per run. Results below average three training seeds on the same 64 held-out
episode seeds.

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTHONPATH="$PWD/src:$PWD" \
  /path/to/sb3-reference/bin/python \
  examples/benchmark_constrained_workbench.py \
  --runs 3 --ppo-total-turns 12288 \
  --bc-updates 500 --bc-train-episodes 256 \
  --bc-validation-episodes 96 --evaluation-episodes 64 \
  --json-output docs/latex/figures/constrained_workbench_j1.json \
  --plot-output docs/latex/figures/constrained_workbench_j1.pdf
```

| policy | return before PPO | return at 12,288 turns |
| --- | ---: | ---: |
| exact oracle | 1.0000 | 1.0000 |
| random legal | 0.4689 | 0.4689 |
| structured BC | 0.9080 | — |
| joint PPO, random initialization | 0.5050 | 0.8750 |
| structured BC then joint PPO | 0.9080 | 0.9227 |

BC held-out accuracy is 0.8228 for worker-kind 0 and 0.8159 for worker-kind 1;
their action-frequency baselines are 0.4875 and 0.4755. All three tiny
deterministic corpora reach 100% accuracy. The mean paired final PPO improvement
over random legal play is 0.4061. A hierarchical paired bootstrap resampling
both independent training runs and common evaluation seeds gives a 95% interval
of [0.3581, 0.4533]. BC-to-PPO median return is 0.9337 of oracle, above the
predeclared 0.80 threshold.

No oracle, random, BC, or PPO evaluation deployed an infeasible action. Moving
from two to four workers changes nested factor count but retains exactly three
trajectory steps and one nonzero terminal reward. Every J1 condition therefore
passes. This is a deliberately small exact-oracle diagnostic, not evidence of
performance on a larger application.

## Masked Taxi constraint control

Taxi-v4 compares masked Jörmungandr PPO, the identical PPO with masks disabled,
SB3-contrib 2.9.0 MaskablePPO, and a masked tabular Q-learning reference. The
two neural masked implementations have exactly 72,903 trainable parameters.
All participants receive 131,072 interactions over three seeds, with fifty
fixed held-out episodes every 16,384 interactions.

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTHONPATH="$PWD/src" \
  /path/to/sb3-reference/bin/python \
  examples/benchmark_masked_taxi_ppo.py \
  --runs 3 --total-timesteps 131072 --rollout-steps 2048 \
  --evaluation-every-timesteps 16384 --evaluation-episodes 50 \
  --json-output docs/latex/figures/masked_taxi_ppo_reference.json \
  --plot-output docs/latex/figures/masked_taxi_ppo_reference.pdf
```

Recorded on 4 August 2026:

| implementation | run 0 success / return | run 1 | run 2 | invalid choices |
| --- | ---: | ---: | ---: | ---: |
| Jörmungandr PPO, masked | .96 / 0.18 | 1.00 / 8.20 | .92 / −8.34 | 0 |
| Jörmungandr PPO, unmasked | .00 / −200 | .00 / −200 | .00 / −200 | 462,511 |
| SB3-contrib MaskablePPO | 1.00 / 8.44 | 1.00 / 8.24 | 1.00 / 8.44 | 0 |
| masked tabular Q-learning | 1.00 / 8.44 | 1.00 / 8.44 | 1.00 / 8.44 | 0 |

Masked Jörmungandr's mean success AUC is 0.7042 versus zero for its unmasked
twin, so masking materially improves learning efficiency and the declared
zero-invalid-action gate passes. Its final seed variance is nevertheless worse
than both independent masked references, and mean success peaked at 97.3% at
65,536 steps before regressing. The result supports making system constraints
part of the available action space; it also supports checkpoint selection and
continued PPO stability diagnostics.

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
