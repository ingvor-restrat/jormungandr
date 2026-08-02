# A Guided Tour of Jörmungandr

Jörmungandr is easiest to understand as a boundary between actors and one
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

## Create one learner

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
      "algo": "qrdqn",
      "action_values": [-1.0, 0.0, 1.0],
      "quantiles": 51,
      "quantile_risk_measure": "mean",
      "min_replay": 256,
      "batch_size": 64,
      "validation_every": 25,
      "min_validation": 32
    }
  }'
```

`action_values` translate the learner's categorical action index into the
domain value returned to actors. Use `GET /v1/algorithms` to inspect installed
plugins. C51 remains available with `"algo":"c51"`; the complete built-in
matrix and its scope are documented in
[Algorithm Plugins, Noise, and Return Risk](algorithms.md).

## Interleave train and validation

Actor scheduling does not need separate network streams. One request may
contain:

```text
train, train, validation, train, validation, ...
```

Jörmungandr validates and routes every item independently. The model response
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
  -d '{
    "obs":[0.1, 0.2, -0.1, 0.0],
    "deterministic":true,
    "action_mask":[true, false, true]
  }'
```

QR-DQN emits learned return quantiles for every action. The service reduces
them using the checkpointed risk rule, applies the point-in-time legal mask,
and returns the selected index and configured action value. Its response also
contains the full `quantiles`, their means in `q_values`, and the decision
statistics in `risk_values`.

Stochastic PPO, IMPALA, and APPO actors should return the inference response's
`behavior_logp` and `behavior_value` with the transition. All algorithms
should return `action_mask` and `next_action_mask` when legality is
state-dependent.

## Select rollout batches with QUBO

Set `"replay_selector":"qubo"` in the learner configuration. Off-policy
algorithms select transition candidates. Trajectory-mode plugins first
construct contiguous episode fragments and make each fragment one QUBO yes/no
variable; PPO, IMPALA, and APPO additionally enforce behavior-policy
freshness. The internal metric history retains the latest decision audit.
Start with the ordinary prioritized selector as the control; the QUBO
formulation and fair experiment gate are in
[QUBO Rollout Selection](qubo-rollout-selection.md).

The same binary optimizer can prune an external search frontier before costly
rollout evaluation:

```python
from jormungandr import SearchNode, bounded_beam_search, build_frontier_pruner

result = bounded_beam_search(
    root,
    expand,
    beam_width=8,
    max_depth=6,
    pruner=build_frontier_pruner(
        "qubo",
        {"qubo_diversity_weight": 0.12},
    ),
)
```

`expand` remains environment-owned and returns `SearchNode` children with a
cheap utility and embedding. Evaluate only `result.frontier` with the costly
simulator. The selected keys and per-level counts make the pruning decision
auditable; see the controlled branch benchmark in the QUBO document.

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
  --module auto
```

`auto` exports the plugin's declared policy or value module. `heads` exports
C51 logits and auxiliary logits together. Observation normalization is
embedded in the TorchScript module when a trained normalizer is present.

## Compare models

Run multiple model identifiers against the same predeclared actor and
validation schedule, then query:

```text
GET /v1/models/compare
GET /v1/models/{model_id}/metrics
```

TensorBoard records each plugin under `algorithms/{algo}/...` and each selector
under `selectors/{selector}/...`. Rank models on held-out environment and risk
metrics at equal experience budgets; objective losses are not comparable
across algorithm families.

Only load trusted Torch checkpoints. PyTorch checkpoint loading can execute
pickle payloads.
