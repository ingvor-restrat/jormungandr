# A Guided Tour of Jormungandr

Jormungandr is easiest to understand as a boundary between actors and one
authoritative learner.

## Start the service

```bash
jormungandr-service \
  --host 127.0.0.1 \
  --port 8811 \
  --checkpoint-root /tmp/jormungandr-checkpoints \
  --top
```

The service starts empty. Models are created through the API because each
model declares its observation size, discrete action vocabulary, replay
limits, learner configuration, and validation cadence.

## Create one C51 learner

```bash
curl -sS http://127.0.0.1:8811/v1/models \
  -H 'content-type: application/json' \
  -d '{
    "model_id": "guide",
    "obs_dim": 4,
    "replay": {"capacity": 10000},
    "validation": {"capacity": 1000},
    "tensorboard": {"enabled": false},
    "learner": {
      "enabled": true,
      "action_values": [-1.0, 0.0, 1.0],
      "min_replay": 256,
      "batch_size": 64,
      "validation_every": 25,
      "min_validation": 32
    }
  }'
```

`action_values` translate the learner's categorical action index into the
domain value returned to actors.

## Interleave train and validation

Actor scheduling does not need separate network streams. One request may
contain:

```text
train, train, validation, train, validation, ...
```

Jormungandr validates and routes every item independently. The model response
reports both resulting store sizes and the number added to each split.

The important invariant is:

```text
validation -> no normalizer update -> no replay priority -> no gradient
```

Periodic validation uses the same fixed learner parameters for the whole
batch and records that policy version with the metrics.

## Ask for an action

```bash
curl -sS http://127.0.0.1:8811/v1/models/guide/policy/infer \
  -H 'content-type: application/json' \
  -d '{"obs":[0.1, 0.2, -0.1, 0.0], "deterministic":true}'
```

C51 emits a probability distribution over fixed return atoms for each action.
The service converts each distribution to an expected Q value, selects an
index, and returns both the index and configured action value.

## Use an auxiliary target

Enable the auxiliary classifier in the learner:

```json
{
  "aux_enabled": true,
  "aux_weight": 0.1,
  "aux_classes": 3,
  "aux_kind": "direction",
  "aux_label_key": "label",
  "aux_class_weighting": "balanced",
  "aux_label_smoothing": 0.02
}
```

Labels can be embedded in experience or attached later. A label affects the
training objective only when its transition belongs to the training split.
The same classifier is evaluated independently on labeled validation items.

## Export a model

Force or locate a checkpoint, then run:

```bash
jormungandr-export inspect --checkpoint /path/to/checkpoint.pt
jormungandr-export export-bundle \
  --checkpoint /path/to/checkpoint.pt \
  --bundle-dir /tmp/jormungandr-bundle \
  --module heads
```

`heads` exports C51 logits and auxiliary logits together. Observation
normalization is embedded in the TorchScript module when a trained normalizer
is present.

Only load trusted Torch checkpoints. PyTorch checkpoint loading can execute
pickle payloads.
