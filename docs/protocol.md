# Experience and Inference Protocol

The fixed vector/discrete HTTP API is rooted at `/v1`; the variable
entity/candidate API is rooted at `/v2`. JSON responses contain `ok: true` on
success and `ok: false` with an `error` string on failure.

The v2 profile is separate rather than an ambiguous extension of v1 arrays.
It retains actor, episode, timestep, split, and policy-version provenance while
replacing the global action vocabulary with actor-owned semantic candidate
IDs.

## Variable entity/candidate v2

Create a central structured model with `POST /v2/models`. The representation
declares fixed feature widths but entity and candidate counts remain variable:

```json
{
  "model_id": "structured-example",
  "representation": {
    "global_dim": 12,
    "entity_dim": 24,
    "candidate_dim": 41,
    "entity_type_count": 8
  },
  "replay": {"capacity": 200000, "alpha": 0.6},
  "learner": {
    "enabled": true,
    "algo": "structured_dqn",
    "device": "auto",
    "batch_size": 256,
    "min_replay": 2048,
    "replay_ratio": 1.0
  }
}
```

Actors call `POST /v2/models/{id}/policy/infer` with `observation` or an
`observations` batch. Each observation uses schema
`jormungandr.entity_candidates.v1` and contains global features, entity
features/type IDs/semantic IDs, and candidate features/semantic IDs/legal
mask. The response returns the selected `candidate_id`, its local batching
index, behavior log probability, Q value, aligned candidate values, and the
common policy version.

Transitions enter `POST /v2/models/{id}/experience/add` under schema
`jormungandr.structured_experience.v1`. Required fields are `split`,
`actor_id`, `episode_id`, `timestep`, `policy_version`, `observation`,
`candidate_id`, `reward`, `next_observation`, and `done`. The service verifies
the semantic candidate against the recorded local set before admitting it to
prioritized replay. `replay_ratio` bounds learner samples per ingested
environment transition, preventing a fast learner thread from silently
changing the experimental update budget.

Inspect a model with `GET /v2/models/{id}` and force a service-owned checkpoint
with `POST /v2/models/{id}/policy/checkpoint`.

### Joint trajectory mode

Create a trajectory model by setting `learner.algo` to `structured_ppo` and
declaring `min_trajectory_steps` plus `max_policy_lag`. Actors first call
`POST /v2/models/{id}/policy/score`. Its response aligns `candidate_logits`
with semantic `candidate_ids`, returns one `behavior_value`, and names the
exact `policy_version`.

The actor samples one candidate per factor in order. Environment-owned
constraint updates may restrict the current or later candidates. The request
to `POST /v2/models/{id}/trajectories/add` uses schema
`jormungandr.structured_trajectories.v1` and contains an array of complete
trajectories. Every `jormungandr.structured_joint_step.v1` record includes:

- actor, episode, timestep, split, and policy version;
- one entity/candidate observation and next observation;
- factor IDs, their candidate IDs, selected ID, and conditional log probability;
- the exact sum as `joint_behavior_log_probability`;
- one centralized behavior value and one scalar reward; and
- termination/truncation plus optional actor-owned audit metadata.

The service rejects missing, duplicated, out-of-order, future, or excessively
stale train trajectories before they update the model. Validation episodes are
stored separately and do not produce gradients. Current ingress requires
complete episodes; partial fragments need an explicit bootstrap contract.
Each learner-history point exposes the raw batch's episode-return mean,
standard deviation, minimum, maximum, unique-value count, mean episode length,
length range, and nonzero-reward fraction. It also exposes `gae_decay` and the
minimum, mean, and maximum oldest-delta weight
`(gamma * lambda) ** (episode_length - 1)`. These fields are computed before
GAE and are therefore direct checks for both an all-identical sparse-reward
batch and a terminal signal whose direct reach is negligible at early steps.

For a pretrained structured policy with a fresh critic, the following optional
learner fields protect the handoff:

```json
{
  "value_backbone_gradient_scale": 0.0,
  "policy_ratio_guard_min": 0.1,
  "policy_ratio_guard_max": 4.0,
  "policy_ratio_guard_backoff_factor": 0.5,
  "policy_ratio_guard_max_backtracks": 6
}
```

The value head still receives its full loss; the first field scales only the
critic gradient entering the shared policy representation. When ratio bounds
are nonzero, the learner treats the complete PPO proposal as a transaction.
It measures every selected behavior-action ratio over the full batch, restores
policy, Adam, and RNG state on violation, and retries at the declared backoff.
Metrics distinguish the within-optimization ratio range from
`post_update_importance_ratio_*`, and report acceptance, attempts, backtracks,
and the effective learning rate. Omitting the fields preserves the historical
fully shared, unguarded update.

### Structured supervision mode

Create a reward-free supervised model with `learner.algo` set to
`structured_bc`. Submit examples to
`POST /v2/models/{id}/supervision/add` using batch schema
`jormungandr.structured_supervision_batch.v1`. Each item uses schema
`jormungandr.structured_supervision.v1` and contains actor/episode/timestep
provenance, an entity/candidate observation, `factor_id`, the factor's current
legal `candidate_ids`, a semantic `target_candidate_id`, and a `train` or
`validation` split. Optional `sample_weight`, `source_group`, and
`factor_group` fields support application-owned balancing and diagnostics.
`target_group` is the reporting class. The optional `balance_group` is an
independent, finer training stratum; when omitted it defaults to
`target_group` for wire compatibility. There is deliberately no reward field.

The learner field `supervision_sampling` is `uniform` by default.
`sample_weight` draws training records with replacement in proportion to their
positive declared weights and presents unit-weight copies to the learner. This
is importance resampling of the same normalized weighted objective, not an
additional multiplication by the weights. The model response reports the
active choice as `supervision.sampling`.

Duplicate factor labels are rejected. Validation labels enter a distinct
bounded store and never update parameters or optimizer state. Learner metrics
include weighted accuracy, NLL, entropy, calibration error, gradient norm, and
per-source/per-factor summaries.

After checkpointing structured BC, create a `structured_ppo` model with
`policy_initialization_path` set to that checkpoint. Representation and model
configuration must match. This copies only the compatible policy state: the
new PPO learner begins at update and policy version zero with fresh optimizer,
value-learning, and trajectory state. `checkpoint_path`, by contrast, restores
the complete same-algorithm learner state and is mutually exclusive with
`policy_initialization_path`.

## Fixed vector/discrete v1

## Create a model

```http
POST /v1/models
Content-Type: application/json
```

```json
{
  "model_id": "c51-example",
  "obs_dim": 3,
  "replay": {"capacity": 200000, "alpha": 0.6},
  "validation": {"capacity": 20000},
  "learner": {
    "enabled": true,
    "algo": "c51",
    "action_values": [-1.0, 0.0, 1.0],
    "batch_size": 256,
    "min_replay": 2048,
    "validation_every": 100,
    "validation_batch_size": 512,
    "min_validation": 64
  }
}
```

The model identifier is unique within one runtime process.

## Publish experience

```http
POST /v1/models/{model_id}/experience/add
Content-Type: application/json
```

```json
{
  "schema": "jormungandr.experience.v1",
  "items": [
    {
      "split": "train",
      "actor_id": "actor-a",
      "episode_id": "episode-1",
      "timestep": 7,
      "policy_version": 12,
      "obs": [0.1, 0.2, 0.3],
      "action_idx": 1,
      "reward": 0.25,
      "next_obs": [0.2, 0.3, 0.4],
      "done": false
    },
    {
      "split": "validation",
      "actor_id": "actor-b",
      "episode_id": "episode-9",
      "timestep": 3,
      "policy_version": 12,
      "obs": [-0.1, 0.0, 0.1],
      "action": -1.0,
      "reward": -0.5,
      "next_obs": [0.0, 0.1, 0.2],
      "done": false
    }
  ]
}
```

Required identity fields are `split`, `actor_id`, `episode_id`, `timestep`,
and `policy_version`. `obs` and `next_obs` must match the model observation
dimension. Rewards and action values must be finite.

`action_idx` is preferred. `action` is accepted when it uniquely matches one
configured action value. The learner always stores the resulting action index.

Optional fields:

| Field | Meaning |
| --- | --- |
| `ts_ns` | Actor observation time in integer nanoseconds. |
| `priority` | Initial positive priority for a training item. |
| `behavior_logp` | Log probability assigned by the behavior policy; required for exact PPO/V-trace correction. |
| `behavior_value` | State-value estimate recorded when the action was selected; used by GAE. |
| `action_mask` | Boolean mask aligned with `action_values` at the current state. |
| `next_action_mask` | Boolean mask aligned with `action_values` at the next state. |
| `aux` | Auxiliary label object, normally containing `kind` and `label`. |
| `meta` | Application metadata retained with the stored item. |
| `session_id` | Compatibility identifier for a locally managed episode. |

## Attach delayed auxiliary labels

```http
POST /v1/models/{model_id}/experience/aux_update
```

```json
{
  "schema": "jormungandr.aux_update.v1",
  "updates": [
    {
      "split": "validation",
      "actor_id": "actor-b",
      "episode_id": "episode-9",
      "timestep": 3,
      "aux": {"kind": "direction", "label": 2}
    }
  ]
}
```

An update that arrives before its transition is retained as pending and
attached when the matching transition arrives.

## Inference

Single observation:

```http
POST /v1/models/{model_id}/policy/infer
```

```json
{
  "obs": [0.1, 0.2, 0.3],
  "deterministic": false,
  "epsilon": 0.02,
  "action_mask": [true, false, true]
}
```

Batched observations use `obs_batch`:

```json
{
  "obs_batch": [
    [0.1, 0.2, 0.3],
    [0.3, 0.2, 0.1]
  ],
  "deterministic": true,
  "action_masks": [
    [true, false, true],
    [true, true, false]
  ]
}
```

Set `deterministic` to false and provide `epsilon` to enable epsilon-greedy
or epsilon-mixture selection. Every item returns `action` and `action_idx`.
Value learners additionally return `q_values`; policy learners return
`policy_logits` and `policy_probs`; QR-DQN also returns `quantiles` and
`risk_values`. Stochastic responses include `behavior_logp`, and
actor--critic responses include `behavior_value`. Return those fields with the
resulting transition. The response also declares `policy_version` and
`updates`.

Illegal action slots have zero policy probability and cannot be sampled. Raw
finite logits or Q summaries remain in the response for diagnostics; the
Boolean mask is the authority for legality.

Auxiliary-only inference:

```http
POST /v1/models/{model_id}/aux/infer
```

The body may contain `obs` or a `features` object. Feature objects are mapped
using the stable model `feature_keys`.

## Introspection and artifacts

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process health. |
| `GET` | `/v1/runtime/stats` | HTTP, model, split, and learner counters. |
| `GET` | `/v1/algorithms` | List installed algorithm plugins and their versions. |
| `GET` | `/v1/models` | List model records. |
| `GET` | `/v1/models/compare` | Align latest metrics across model identifiers. |
| `GET` | `/v1/models/{id}` | Inspect one model. |
| `GET` | `/v1/models/{id}/policy` | Inspect learner and validation state. |
| `GET` | `/v1/models/{id}/metrics` | Read recent plugin/selector history and selection audit. |
| `POST` | `/v1/models/{id}/policy/checkpoint` | Force a checkpoint. |
| `DELETE` | `/v1/models/{id}` | Stop and remove a model. |

The older `/replay/*` routes remain compatibility aliases. New actors should
use `/experience/*`, whose identity fields and split semantics are enforced.

## Checkpoint restore

Create a model with `checkpoint_path` to restore its algorithm, network and
target state, optimizer, normalizer, update counter, policy version, action
vocabulary, and plugin configuration. An explicit `obs_dim` must either be
zero or match the checkpoint. Replay and validation items are not embedded in
the checkpoint and begin empty. A checkpoint that records an algorithm-plugin
version is not silently loaded through a different installed version; migrate
the state explicitly or install the compatible plugin.
