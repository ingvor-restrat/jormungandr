# Jörmungandr

Distributed actor, central learner reinforcement learning with split-aware
experience ingestion and online inference.

Jörmungandr coordinates many external actors against one versioned model.
Actors own environment execution and send observations, actions, rewards,
outcomes, and provenance to the service. Jörmungandr owns training replay,
held-out validation, learner updates, checkpoints, metrics, and inference.

Jörmungandr is part of the open-source analytical architecture behind
strategynet.ai.

The publishable Python distribution is named `jormungandr-rl`; its import name
is `jormungandr`.

The current release provides:

- batched experience ingress from multiple concurrent actors;
- explicit interleaving of `train` and `validation` samples;
- prioritized training replay with annealed importance weights;
- independently discoverable algorithm and rollout-selector plugins;
- C51, QR-DQN, DQN, discrete CQL, maximum-entropy soft Q, BC, MARWIL,
  PPO, IMPALA, APPO/IMPACT, categorical SAC, and a compact vector DreamerV3
  profile;
- contiguous, policy-lag-aware trajectory sampling for on-policy and
  asynchronous learners;
- auditable QUBO utility/diversity selection over transitions or rollout
  fragments;
- bounded beam search with utility and QUBO frontier-pruner plugins;
- held-out validation that never changes weights or normalization state;
- online single-observation and batched policy inference;
- an optional auxiliary classification head with delayed-label attachment;
- running observation normalization learned only from training data;
- model lifecycle, metrics, runtime statistics, and checkpoints over HTTP;
- TorchScript bundles with manifests and generated C++ model specifications;
  and
- TensorBoard namespaces and internal APIs for comparing algorithm runs.

All built-ins share the service lifecycle through a stable plugin contract.
The profiles currently use fixed discrete action values. The DreamerV3
profile is a compact vector-control implementation, not full image/RSSM
benchmark parity.

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m pytest
jormungandr-service --host 127.0.0.1 --port 8811 --top
```

In another terminal, run the distributed actor example:

```bash
python examples/distributed_actor.py
```

The example creates one model and submits a batch whose training and validation
transitions are interleaved in arrival order.

For a complete learner run with two concurrent actors:

```bash
python examples/train_synthetic_control.py
```

The [generic trainer walkthrough](docs/example-trainer.md) explains its
environment interface, split scheduler, delayed auxiliary labels, run
artifacts, and the future Volt adapter boundary.

## See distributed learning

![A live distributed OU spread training monitor](docs/markup/ou-spread.gif)

The financial research example makes the actor/learner loop and held-out
statistics visible without private data:

```bash
python examples/train_ou_spread.py
```

Two actors train on fixed-seed synthetic mean-reverting spread paths. The live
screen reports reward, a declared same-path reference, drawdown, turnover,
training and validation stores, learner versions, auxiliary accuracy, and the
policy selected at five spread z-scores. Training and validation episodes are
interleaved in the schedule while remaining isolated in the service.

Read the [OU spread walkthrough](docs/ou-spread-example.md) for the process,
reward, limitations, Volt boundary, and recording workflow.

## Compare compatible agents and frontier selection

![Five online transition agents converging on clean and noisy held-out paths](docs/markup/algorithm-convergence.gif)

The seeded comparison holds the synthetic environment, transition budget, and
held-out path cohort fixed for DQN, C51, QR-DQN, maximum-entropy Soft-Q, and
categorical SAC. Offline and trajectory algorithms are intentionally evaluated
in separate cohorts. The complete protocol and numeric intervals are in
[reproducible results](docs/results.md).

![QUBO applied to successive expanded branch frontiers](docs/markup/qubo-frontier.gif)

At each search depth, Jörmungandr expands the retained beam, constructs one
binary optimization over the whole candidate frontier, and keeps candidates
whose decision is one. The [QUBO study](docs/qubo-rollout-selection.md)
separates this explanatory trace from the paired 500-tree empirical result.

## Experience contract

Every public experience item declares its split and actor provenance:

```json
{
  "split": "train",
  "actor_id": "actor-07",
  "episode_id": "episode-20260725-001",
  "timestep": 42,
  "policy_version": 18,
  "obs": [0.1, -0.2, 0.3],
  "action_idx": 2,
  "reward": 0.75,
  "next_obs": [0.2, -0.1, 0.4],
  "done": false,
  "aux": {"kind": "direction", "label": 2}
}
```

Send an array of items to:

```text
POST /v1/models/{model_id}/experience/add
```

The request envelope declares
`"schema": "jormungandr.experience.v1"`.

Training items update the training-only normalizer and enter prioritized
replay. Validation items enter a separate bounded store and are evaluated
against a declared policy version without gradient or optimizer changes.
`val` is accepted as an input alias and canonicalized to `validation`.

For the full wire contract, read [Experience and inference protocol](docs/protocol.md).

## In-process API

```python
from jormungandr import JormungandrRuntime

runtime = JormungandrRuntime()
runtime.create_model(
    model_id="example",
    obs_dim=3,
    learner={
        "enabled": True,
        "algo": "qrdqn",
        "action_values": [-1.0, 0.0, 1.0],
        "quantiles": 51,
        "quantile_risk_measure": "mean",
    },
    tensorboard_enabled=False,
)
```

The HTTP service uses the same runtime implementation.

## Architecture boundary

Jörmungandr is a learner and inference framework. It does not define an
environment, reward function, market simulator, data vendor, portfolio
policy, or execution system. Generic subprocess and C-ABI episode adapters
exist for compatibility, but external actors publishing versioned experience
are the primary architecture.

The bundled HTTP server is intended for trusted service networks and local
development. Authentication, TLS termination, admission control, and
internet-edge rate limiting belong in the deployment layer.

Continue with:

- [How Jörmungandr fits together](docs/architecture.md)
- [Algorithm choices, noise, and quantile return models](docs/algorithms.md)
- [Algorithm and selector plugin contract](docs/plugins.md)
- [QUBO rollout-selection experiment](docs/qubo-rollout-selection.md)
- [Guided tour](docs/guide.md)
- [Generic example trainer](docs/example-trainer.md)
- [Synthetic OU spread example](docs/ou-spread-example.md)
- [Experience and inference protocol](docs/protocol.md)
- [Reproducible results](docs/results.md)
- [Jörmungandr research paper](docs/latex/jormungandr_learning_systems.pdf)

Licensed under Apache-2.0.
