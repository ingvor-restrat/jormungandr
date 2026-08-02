# Architecture

Jörmungandr separates environment execution, learning, validation, and model
serving. Many actors may generate experience concurrently, but one model
record is the authority for replay state, learner state, policy versions, and
artifacts.

```text
 actor 0 ─┐
 actor 1 ─┼──> versioned experience ingress
 actor N ─┘                |
                    split + validate
                       /       \
                      v         v
          prioritized training  held-out validation
                 replay               store
                    |                   |
                    v                   |
        selector -> algorithm plugin <--+
                    |
          policy version + checkpoint
             /                    \
            v                      v
   online inference       TorchScript bundle
```

## Actor boundary

An actor owns its environment loop. It chooses when an episode begins, steps
the environment, computes or receives rewards, and publishes transitions.
Jörmungandr does not call actor environments in the primary distributed path.

Every transition identifies:

- `split`: `train` or `validation`;
- `actor_id`: the producing worker;
- `episode_id` and `timestep`: the transition identity; and
- `policy_version`: the model version used to choose the action.

For stochastic policy learners, experience also retains `behavior_logp` and
`behavior_value`. `action_mask` and `next_action_mask` preserve point-in-time
legality. This provenance makes interleaved streams inspectable and permits
policy-lag controls without coupling the learner to actor scheduling.

## Split boundary

Training and validation items may alternate within one request. Arrival order
does not alter their meaning.

Training items:

- update the running observation statistics;
- enter prioritized replay;
- may change replay priority after a learner update; and
- contribute gradients to the learner and auxiliary objective.

Validation items:

- enter a separate bounded store;
- are normalized using training statistics;
- never change preprocessing state, priorities, parameters, or optimizer
  state; and
- are periodically evaluated against a recorded policy version.

This is an isolation contract, not merely a metrics naming convention.

## Learner plane

The service resolves an `AlgorithmPlugin` from the model configuration. The
plugin owns its networks, objective, optimizer, update metrics, action rule,
and serializable state. The runtime owns data validation, replay or trajectory
selection, normalization, split isolation, concurrency, versions,
checkpoints, and metrics. C51 has no privileged service branch.

Built-ins cover scalar and distributional value learning, offline imitation,
on-policy and asynchronous actor--critic learning, maximum-entropy learning,
and a compact model-based profile. Their exact scope is listed in
[Algorithm Plugins, Noise, and Return Risk](algorithms.md).

Transition-mode plugins draw from ordinary prioritized replay or the QUBO
candidate selector. Trajectory-mode plugins receive contiguous fragments that
do not cross actor, episode, terminal, or timestep boundaries. Online
trajectory plugins exclude stale fragments using `policy_version` and
`max_policy_lag`; MARWIL intentionally keeps historical demonstrations. PPO
consumes GAE targets; IMPALA and APPO consume behavior probabilities through
V-trace; APPO also retains a target policy and bounded circular replay.

The replay buffer stores action indices. Experience may supply an explicit
`action_idx` or a unique value from the model's `action_values`. Public
responses include both forms. Every legal-action mask must match that fixed
vocabulary and admit at least one action. Masks are applied to action
selection, behavior probabilities, policy objectives, and next-state value
backups as required by the selected plugin.

## Auxiliary objective

The optional auxiliary classifier is an independent MLP over the normalized
observation. Its cross-entropy loss is combined with the selected algorithm
objective using a configurable weight. The original C51 classifier retains
batch-balanced class weights and label smoothing; newer generic plugins share
the core unweighted auxiliary head.

Labels may arrive with a transition or later through `experience/aux_update`.
Delayed labels join by split, actor, episode, and timestep. Validation
auxiliary metrics are calculated without gradient updates.

## Inference plane

Policy inference accepts one observation or a batch plus aligned legal masks.
Results contain:

- selected action value and action index;
- the plugin's policy logits/probabilities or value summaries;
- learner update and policy version;
- auxiliary class probabilities when configured.

Feature-map auxiliary inference uses a stable `feature_keys` order stored in
model metadata. Checkpoints preserve that order and the observation
normalizer.

## Artifact plane

A checkpoint contains the model configuration, plugin identity and version,
policy and target parameters, optimizer state, training-only normalizer,
feature keys, learner update, selector identity, and policy version. The outer
format remains `jormungandr.checkpoint.v1`, so existing C51 artifacts and the
new plugin payloads use one lifecycle envelope.

The exporter produces:

```text
manifest.json
policy.ts.pt
jormungandr_model_spec.hpp
```

New manifests use `jormungandr.inference_bundle.v1`. The reader retains
support for the earlier unversioned service checkpoint shape so existing
internal artifacts can be migrated.

Replay contents and validation contents are not checkpointed. A restored
learner resumes model, optimizer, version, and preprocessing state but actors
must refill its stores.

## Selection plane

Replay selection is a separate plugin boundary. The ordinary selector returns
prioritized samples and annealed importance weights. The QUBO selector forms a
bounded candidate pool and minimizes a utility, similarity, and cardinality
objective. For trajectory algorithms, one binary variable represents one
contiguous rollout fragment. The selected decisions and candidate identities
are retained for audit. See [QUBO Rollout Selection](qubo-rollout-selection.md).

The same binary solver is exposed through an independent frontier-pruner
plugin. `bounded_beam_search` asks an environment-owned expansion callback for
cheaply described children and retains a fixed-width subset before expensive
simulation or terminal evaluation. The utility baseline and QUBO pruner share
the accounting contract, including generated, expanded, retained, and pruned
node counts plus selector time. This is the intended Volt option-graph
integration point; Jörmungandr never owns the contracts or graph state.

## Concurrency

Models have independent locks and learner threads. Replay selection and
metadata snapshots occur under the model lock. Each learner also has a model
lock that serializes parameter updates, held-out evaluation, inference, and
checkpoint snapshots without blocking experience ingestion for the duration
of tensor computation. C-ABI episode adapters are serialized because native
libraries may not be reentrant.

The in-memory stores are process-local. Cross-process durability,
high-throughput binary transports, distributed replay, and multi-learner
parameter synchronization are future layers rather than implied behavior.

## Volt integration stages

A Volt actor retains option contracts, legal actions, market paths, lifecycle
transitions, and reward construction. Jörmungandr integration expands in
bounded stages:

1. entry selection from a fixed compiler-generated action set;
2. repeated hold-or-close decisions;
3. fixed-bucket hedge and resize actions;
4. state-dependent roll and replacement actions; and
5. full graph control over positions, inventory, account state, and compiled
   actions.

The first three stages can send a frozen supervised set or graph embedding
through the fixed-vector interface and select any compatible algorithm. Deep Sets is the current
default because it is smaller and materially faster in the generated Volt
benchmark; the typed graph remains selectable under the same experiment
contract. Volt emits a multi-expiry runtime graph with action descriptors and
account-refined masks. The service transports fixed-slot masks today. Dynamic
descriptors still require a graph/action batch codec. The fifth stage
additionally requires graph-shaped service replay and inference plus a learner
that can update the selected encoder with policy and value heads.
Scenario-level split assignment and immutable actor provenance apply unchanged
at every stage.

`jormungandr.policy` defines the encoder-independent part of that boundary.
It validates and applies an aligned legal-action mask, selects only admitted
actions, and computes a masked discrete actor--critic loss from policy logits
and state values. `GraphTrajectoryBuffer` stores ordered graph references,
chosen action slots, masks, rewards, values, terminal flags, and policy
versions, then computes generalized-advantage targets. It does not prescribe
how a set, graph, or attention model produces policy tensors, and it does not
copy graph tensors into the generic package. A Volt in-process trainer can
resolve those references and backpropagate through its Deep Sets or GNN now.
Durable graph replay and graph-native service transport remain to be
implemented.

## Native evolution

Algorithm and selector boundaries exchange arrays, mappings, metrics, and
state dictionaries. This permits a future PyO3 extension to move measured CPU
hot spots—replay assembly, QUBO search, graph collation, or return kernels—to
Rust without moving Torch policy modules or changing the experience protocol.
The staged native boundary is described in [the plugin contract](plugins.md).
