# Generic Example Trainer

The synthetic trainer is a complete actor-to-learner example without a
financial simulator, market data, order-book replay, or proprietary strategy
logic.

Run it from the repository root:

```bash
python examples/train_synthetic_control.py
```

To retain the resolved manifest, episode summaries, model summary, and final
checkpoint:

```bash
python examples/train_synthetic_control.py \
  --output-dir /tmp/jormungandr-synthetic-example
```

The default run launches two actor threads, eight training episodes, and two
validation episodes against a local Jörmungandr HTTP service.

## What it demonstrates

The example owns the pieces that belong to a trainer:

- an environment adapter;
- observation and action definitions;
- a reward function;
- an auxiliary target;
- a seeded episode schedule;
- actor concurrency;
- train/validation assignment;
- experiment metadata and episode summaries.

Jörmungandr owns:

- model and policy versions;
- prioritized training replay;
- the isolated validation store;
- training-only observation normalization;
- selected algorithm and target/value-network updates;
- auxiliary fitting and validation;
- inference and checkpoints.

The actors communicate exclusively through the public HTTP API. Running the
service in the same process makes the example self-contained; replacing its
ephemeral URL with a remote service does not change the actor protocol.

## Environment contract

The example defines a deliberately small protocol:

```python
class EpisodicEnvironment(Protocol):
    observation_dim: int
    action_values: list[float]

    def reset(self, seed: int) -> list[float]: ...
    def step(self, action: float) -> EnvironmentStep: ...
```

`SyntheticTrackingEnvironment` asks an agent to move a point along a smooth
target path. Its observation is:

```text
position
velocity
target - position
remaining episode fraction
```

The three actions decrease, preserve, or increase acceleration. Reward is a
bounded tracking objective:

```text
1 - tracking_error² - 0.05 velocity²
```

The auxiliary target classifies whether the future synthetic target moves
down, stays approximately flat, or moves up. Labels are submitted after their
transitions to exercise Jörmungandr's delayed-label join.

Nothing about this environment is part of Jörmungandr. It can be replaced by
any adapter that implements the same episode boundary.

## Interleaved splits

The default job schedule is:

```text
train
train
train
train
validation
train
train
train
train
validation
```

Workers consume this shared queue concurrently. Every transition still carries
its immutable split, actor, episode, timestep, and policy version, so completion
order cannot change its training or validation meaning.

Validation episodes use deterministic actions. Training episodes use
epsilon-greedy exploration. Both are sent to the same experience endpoint;
Jörmungandr enforces the split boundary.

## Outputs

With `--output-dir`, the example writes:

```text
run_manifest.json
episode_results.jsonl
summary.json
checkpoints/
```

The run manifest contains only resolved experiment configuration and public
schema names. Episode results retain actor, split, episode identity, reward,
error, and observed policy-version range.

The reported train and validation rewards are functional smoke results. This
small run is not intended to establish convergence or compare algorithms.

For a presentation-oriented example with interpretable reward, drawdown,
turnover, a declared reference, and a live learned policy map, continue with
the [synthetic OU spread trainer](ou-spread-example.md).

## Adapter boundary for Volt

A future Volt trainer should preserve the same ownership boundary.
Jörmungandr should not import option-market policy or redefine Volt's
contracts. A Rust actor can instead:

1. construct a canonical Volt portfolio and versioned scenario;
2. derive a fixed observation vector from the portfolio and market view;
3. request an action from Jörmungandr;
4. translate the action index into a typed Volt candidate action;
5. execute the transition through Volt's deterministic lifecycle/scenario
   logic;
6. compute the research reward outside Volt's authoritative contract model;
7. publish the resulting transition with scenario lineage and policy version.

Split assignment should happen at the scenario or path level before an episode
starts. Transitions from one scenario must not be divided across training and
validation, because that would leak the same path into both stores.

Volt's current candidate grammar includes actions such as adding, removing, or
resizing a leg, waiting, and exercising. Volt's runtime graph retains every
action slot and its composed legal mask. Jörmungandr's v1 HTTP inference and
experience contracts apply masks to a fixed discrete vocabulary. The generic
in-process policy layer now also supports state-dependent action descriptors
and candidate scoring through `jormungandr.structured`. A service integration
still needs one explicit choice:

- define a bounded, fixed set of option-strategy action templates and masks;
  or
- use the structured entity/candidate contract and extend replay, inference,
  and export transport around it.

The fixed-template route remains the smaller compatibility experiment.
Dynamic candidate scoring is the general architecture and is a public generic
contract rather than a Volt-specific actor convention.

Variable-size portfolio graphs present a similar boundary. Offline Volt
research trains both Deep Sets and typed graph encoders. Deep Sets is the
default for the first sequential experiment; configuration can select the
graph encoder without changing the policy/value output contract. The first
fixed-vector adapter can freeze the selected encoder, append declared runtime and
account summaries, and send the resulting vector through the existing
experience contract. This supports staged entry, hold/close, and bounded
hedge/resize experiments with a fixed action vocabulary.

Roll and replacement actions depend on the contracts available in each state.
Aligned legal masks are transported now; dynamic descriptors and typed entity
batches exist in-process, while the v1 service still needs a structured wire
codec and replay store. End-to-end graph-encoder updates can use the same actor provenance
and split rules, but require a graph-shaped inference and replay contract
rather than hiding variable action semantics inside a fixed observation
vector.

For dynamic candidate scoring, a compatible model emits a matrix of action
logits, an aligned legal mask, and one state value per observation.
`jormungandr.policy.masked_actor_critic_loss` supplies the common learner-side
objective. The encoder may be Deep Sets, a typed GNN, or a later attention
model; encoder choice does not change the financial action identifiers.

Variable observations can be retained by reference with
`GraphTrajectoryBuffer`. Each step records the external graph identifier,
legal mask, chosen slot, reward, terminal flag, behavior log probability,
value estimate, and policy version. Finishing the buffer computes
generalized-advantage and return targets while the domain adapter remains
responsible for resolving graph identifiers to tensors. This in-memory
facility does not change the fixed-vector HTTP experience schema.
