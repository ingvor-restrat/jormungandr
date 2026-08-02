# Algorithm Plugins, Noise, and Return Risk

Jörmungandr is an RL runtime, not a C51 model. C51 remains available as one
algorithm plugin, while the replay, split isolation, normalization, legal
action masks, checkpoints, inference, TensorBoard output, and internal metric
history are shared infrastructure.

The built-in service algorithms all use a fixed discrete action vocabulary.
This matches the first Volt control stages and makes legality auditable. It is
not a claim that every paper's original continuous-control or image benchmark
has been reproduced.

## Built-in profiles

| Plugin | Data and update | Implemented profile | Use it when |
| --- | --- | --- | --- |
| `c51` | off-policy transitions | categorical return distribution on a fixed support | preserving the original behavior or comparing categorical distributional values |
| `qrdqn` | off-policy transitions | Double QR-DQN with quantile-Huber loss | return tails and downside-aware action scoring matter |
| `dqn` | off-policy transitions | double, dueling DQN with Huber TD loss | a small, fast scalar-value baseline is needed |
| `cql` | offline/off-policy transitions | discrete CQL(H) penalty over Double DQN | logged data may contain actions the learned policy should not extrapolate beyond |
| `maxent` | off-policy transitions | discrete soft Q-learning with a Boltzmann policy | a direct maximum-entropy value baseline is wanted |
| `bc` | demonstrations | categorical behavior cloning | establishing whether the demonstrations alone explain performance |
| `marwil` | offline trajectories | normalized, clipped advantage-weighted imitation | rewards should reweight demonstrations without a full online loop |
| `ppo` | contiguous trajectories | clipped surrogate, GAE, multiple minibatch epochs | a conventional on-policy actor--critic baseline is wanted |
| `impala` | contiguous trajectories | V-trace corrected asynchronous actor--critic | actors and learner are decoupled and policy lag is material |
| `appo` | contiguous trajectories | V-trace plus IMPACT-style target-policy clipping | asynchronous sampling also needs proximal update control |
| `sac` | off-policy transitions | categorical SAC, twin critics, soft targets, automatic temperature | entropy-regularized actor--critic behavior is preferred |
| `dreamerv3` | trajectories/transitions | compact vector latent dynamics, symlog scaling, imagination, slow value target | model-based vector-control research is being staged |

The `dreamerv3` plugin is deliberately named a compact vector-observation
profile. It is not pixel DreamerV3 parity: it does not reproduce the full
recurrent state-space model, categorical latents, image decoder, or benchmark
training recipe from the reference implementation. That boundary is recorded
in the plugin and checkpoint metadata instead of hiding a smaller model behind
the paper's name.

Primary references are [C51](https://arxiv.org/abs/1707.06887),
[QR-DQN](https://arxiv.org/abs/1710.10044),
[DQN](https://www.nature.com/articles/nature14236),
[Double DQN](https://arxiv.org/abs/1509.06461),
[CQL](https://arxiv.org/abs/2006.04779),
[PPO](https://arxiv.org/abs/1707.06347),
[IMPALA](https://arxiv.org/abs/1802.01561),
[IMPACT/APPO](https://arxiv.org/abs/1912.00167),
[SAC](https://arxiv.org/abs/1801.01290), and
[DreamerV3](https://arxiv.org/abs/2301.04104). MARWIL follows the
exponentially advantage-weighted batch-policy formulation described in
[Exponentially Weighted Imitation Learning for Batched Historical Data](https://proceedings.neurips.cc/paper/2018/hash/4aec1b3435c52abbdf8334ea0e7141e0-Abstract.html).

## Noise is not one problem

An algorithm described as "robust" may be addressing a different failure
mode from the one present in the experiment. Jörmungandr records them
separately.

| Noise or uncertainty | Relevant controls | What they do not establish |
| --- | --- | --- |
| reward outliers and heavy-tailed returns | Huber losses, optional reward clipping, C51/QR-DQN distributions, tail metrics | robustness to corrupted observations |
| aleatoric outcome variation | C51 or QR-DQN; compare mean, quantiles, and lower-tail CVaR | calibrated epistemic uncertainty |
| observation or feature noise | training-only normalization and opt-in Gaussian augmentation; add domain-specific perturbation tests | adversarial guarantees |
| Q-value overestimation | Double-DQN action selection or SAC twin critics | robustness to dataset shift |
| offline action-distribution shift | CQL, with BC and MARWIL controls | correction for asynchronous policy lag |
| actor/learner policy lag | IMPALA V-trace or APPO target/importance clipping and `max_policy_lag` | market-noise resistance |
| multimodal behavior under disturbances | entropy-regularized `maxent` or SAC | universal robust-MDP optimality |
| learned-model error | held-out multi-step error, short imagination horizons, Dreamer slow targets | correctness outside the world model's support |

`reward_clip` and `observation_noise_std` are opt-in because both alter the
learning problem. Their values are checkpointed. Validation remains clean by
default; a noise study should create named evaluation cohorts rather than
silently perturb the held-out store.

The maximum-entropy result motivating this work proves a lower-bound
relationship for specified reward and dynamics perturbation sets, rather than
a universal robustness theorem. The implementation therefore describes
`maxent` and SAC as useful robustness hypotheses to test, not guaranteed
solutions. See Eysenbach and Levine,
[Maximum Entropy RL (Provably) Solves Some Robust RL Problems](https://arxiv.org/abs/2103.06257).

For noisy Volt outcomes, the recommended comparison is:

1. DQN or PPO as the scalar baseline;
2. QR-DQN with mean action scoring;
3. the same QR-DQN checkpoint scored by a predeclared lower quantile or CVaR;
4. categorical SAC or `maxent` at fixed temperatures; and
5. seed, perturbation, path-cohort, and chronological tests that keep the
   reward and selection rule fixed.

Changing from mean to CVaR after seeing test results is model selection on the
test set. The risk measure and level belong in the run manifest before the
held-out paths are evaluated.

PPO defaults to `max_policy_lag: 0`; IMPALA and APPO default to 64. Offline
MARWIL uses contiguous returns but does not enforce learner-version age on its
demonstrations.

## Quantile policies and probabilistic Torch

In most actor--critic language, quantile regression belongs to the critic:
it estimates the conditional distribution of return. The policy then chooses
an action using a declared functional of that distribution. Jörmungandr's
`qrdqn` plugin supports:

```json
{
  "algo": "qrdqn",
  "quantiles": 51,
  "quantile_risk_measure": "cvar",
  "quantile_risk_level": 0.1
}
```

`mean` recovers the usual expected-return decision rule. `lower_quantile`
uses the selected lower percentile. `cvar` averages the lowest learned
quantiles. Inference returns `quantiles`, `q_values`, and `risk_values`; the
checkpoint and exported manifest retain the quantile count and decision rule.

PyTorch is a good fit for this path. QR-DQN needs a quantile-Huber objective,
which is implemented directly with differentiable tensor operations. Policies
that sample categorical or continuous distributions can use
[`torch.distributions`](https://docs.pytorch.org/docs/stable/distributions.html)
and its score-function or pathwise-gradient machinery. A separate
probabilistic-programming framework is not required merely to learn return
quantiles.

Tree models remain useful controls:

- CatBoost supports `Quantile` and a multi-output `MultiQuantile` objective;
  its current documentation lists `MultiQuantile` optimization on CPU but not
  GPU. It is the more convenient of the two for one model emitting several
  quantiles.
- LightGBM supports the `quantile` objective with one `alpha` setting, so a
  quantile grid normally means one fitted model per level. It is a strong,
  fast numeric-feature baseline.

See the official [CatBoost regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression)
and [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).
Neither tree library backpropagates through a Deep Sets or GNN encoder. Use a
frozen Volt embedding for those baselines; use Torch when the encoder and RL
heads must train end to end.

## Volt set and graph models

The library-level boundary already accepts any Torch encoder that emits
aligned policy logits and a state value. `masked_actor_critic_loss` applies
the legal mask and returns policy, value, entropy, and total loss terms.
`GraphTrajectoryBuffer` retains external graph references and computes GAE
targets without copying domain graphs into the generic library. This permits
Volt's Deep Sets and typed GNN to train end to end in an in-process research
driver while Volt remains the authority for graph construction, action
identity, legality, and financial transitions.

There are three distinct deployment stages:

1. **Available through the service now:** freeze a Deep Sets/GNN encoder and
   send its fixed-size embedding through any built-in algorithm plugin.
2. **Available through the library now:** resolve `GraphTrajectoryBuffer`
   references in a Volt training process and optimize the encoder with the
   common masked loss.
3. **Not yet a public service contract:** send variable-sized graph batches
   through HTTP, retain them in durable replay, and run a generic service
   plugin over graph-native batches.

The third stage should add an explicit graph/batch codec and artifact
identity. Flattening a changing action graph into an undocumented vector would
make replay and checkpoints impossible to audit.

## Comparing runs

Every plugin emits normalized `loss` and priority information plus its own
metrics. TensorBoard writes algorithm metrics under:

```text
algorithms/{algorithm}/{metric}
train/algorithms/{algorithm}/{metric}
validation/algorithms/{algorithm}/{metric}
selectors/{selector}/{metric}
```

The internal comparison API exposes the latest aligned rows at
`GET /v1/models/compare`; per-model history and the latest selector decision
are available at `GET /v1/models/{id}/metrics`. Compare held-out reward and
domain risk measures at equal environment steps and evaluation paths. Raw
losses from different objectives are diagnostic and are not directly ranked.
