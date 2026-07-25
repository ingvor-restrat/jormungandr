# Jormungandr

Distributed actor, central learner reinforcement learning with split-aware
experience ingestion and online inference.

Jormungandr coordinates many external actors against one versioned model.
Actors own environment execution and send observations, actions, rewards,
outcomes, and provenance to the service. Jormungandr owns training replay,
held-out validation, learner updates, checkpoints, metrics, and inference.

Jormungandr is part of the open-source analytical architecture behind
strategynet.ai.

The publishable Python distribution is named `jormungandr-rl`; its import name
is `jormungandr`.

The initial release provides:

- batched experience ingress from multiple concurrent actors;
- explicit interleaving of `train` and `validation` samples;
- prioritized training replay with annealed importance weights;
- a central C51 distributional Q learner with versioned policies;
- held-out validation that never changes weights or normalization state;
- online single-observation and batched policy inference;
- an optional auxiliary classification head with delayed-label attachment;
- running observation normalization learned only from training data;
- model lifecycle, metrics, runtime statistics, and checkpoints over HTTP; and
- TorchScript bundles with manifests and generated C++ model specifications.

PPO and DDPG building blocks are included for continued development. C51 is
the only algorithm wired through the complete service lifecycle in version
0.1.

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
        "action_values": [-1.0, 0.0, 1.0],
    },
    tensorboard_enabled=False,
)
```

The HTTP service uses the same runtime implementation.

## Architecture boundary

Jormungandr is a learner and inference framework. It does not define an
environment, reward function, market simulator, data vendor, portfolio
policy, or execution system. Generic subprocess and C-ABI episode adapters
exist for compatibility, but external actors publishing versioned experience
are the primary architecture.

The bundled HTTP server is intended for trusted service networks and local
development. Authentication, TLS termination, admission control, and
internet-edge rate limiting belong in the deployment layer.

Continue with:

- [How Jormungandr fits together](docs/architecture.md)
- [Guided tour](docs/guide.md)
- [Generic example trainer](docs/example-trainer.md)
- [Synthetic OU spread example](docs/ou-spread-example.md)
- [Experience and inference protocol](docs/protocol.md)
- [Reproducible results](docs/results.md)

Licensed under Apache-2.0.
