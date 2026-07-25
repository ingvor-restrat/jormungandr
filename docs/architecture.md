# Architecture

Jormungandr separates environment execution, learning, validation, and model
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
              central learner <--------+
                    |
          policy version + checkpoint
             /                    \
            v                      v
   online inference       TorchScript bundle
```

## Actor boundary

An actor owns its environment loop. It chooses when an episode begins, steps
the environment, computes or receives rewards, and publishes transitions.
Jormungandr does not call actor environments in the primary distributed path.

Every transition identifies:

- `split`: `train` or `validation`;
- `actor_id`: the producing worker;
- `episode_id` and `timestep`: the transition identity; and
- `policy_version`: the model version used to choose the action.

This provenance makes interleaved streams inspectable and permits future
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

The service currently instantiates C51: a categorical distributional
Q-network over a fixed support. Training uses:

- a two-hidden-layer ReLU network;
- prioritized replay with configurable alpha;
- importance weights with beta annealed toward one;
- a hard-updated target network;
- epsilon-greedy inference over declared discrete action values; and
- gradient clipping.

The replay buffer stores action indices. Experience may supply an explicit
`action_idx` or a unique value from the model's `action_values`. Public
responses include both forms.

PPO and DDPG implementations remain lower-level building blocks. They are not
advertised as service algorithms until their rollout and concurrency
contracts are implemented end to end.

## Auxiliary objective

The optional auxiliary classifier is an independent MLP over the normalized
observation. Its cross-entropy loss is combined with the C51 objective using a
configurable weight. Batch-balanced class weights and label smoothing are
available.

Labels may arrive with a transition or later through `experience/aux_update`.
Delayed labels join by split, actor, episode, and timestep. Validation
auxiliary metrics are calculated without gradient updates.

## Inference plane

Policy inference accepts one observation or a batch. Results contain:

- selected action value and action index;
- expected Q values for each action;
- learner update and policy version;
- auxiliary class probabilities when configured.

Feature-map auxiliary inference uses a stable `feature_keys` order stored in
model metadata. Checkpoints preserve that order and the observation
normalizer.

## Artifact plane

A checkpoint contains the model configuration, policy and target parameters,
optimizer state, training-only normalizer, feature keys, learner update, and
policy version. New checkpoints use `jormungandr.checkpoint.v1`.

The exporter produces:

```text
manifest.json
policy.ts.pt
jormungandr_model_spec.hpp
```

New manifests use `jormungandr.inference_bundle.v1`. The reader retains
support for the earlier unversioned service checkpoint shape so existing
internal artifacts can be migrated.

Replay contents and validation contents are not checkpointed in version 0.1.

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
