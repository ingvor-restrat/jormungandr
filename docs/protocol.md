# Experience and Inference Protocol

The HTTP API is rooted at `/v1`. JSON responses contain `ok: true` on success
and `ok: false` with an `error` string on failure.

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
{"obs": [0.1, 0.2, 0.3], "deterministic": true}
```

Batched observations use `obs_batch`:

```json
{
  "obs_batch": [
    [0.1, 0.2, 0.3],
    [0.3, 0.2, 0.1]
  ],
  "deterministic": true
}
```

Set `deterministic` to false and provide `epsilon` to enable epsilon-greedy
selection. Every item returns `action`, `action_idx`, and `q_values`. The
response also declares `policy_version` and `updates`.

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
| `GET` | `/v1/models` | List model records. |
| `GET` | `/v1/models/{id}` | Inspect one model. |
| `GET` | `/v1/models/{id}/policy` | Inspect learner and validation state. |
| `POST` | `/v1/models/{id}/policy/checkpoint` | Force a checkpoint. |
| `DELETE` | `/v1/models/{id}` | Stop and remove a model. |

The older `/replay/*` routes remain compatibility aliases. New actors should
use `/experience/*`, whose identity fields and split semantics are enforced.
