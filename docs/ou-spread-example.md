# Synthetic OU Spread Example

The OU spread trainer gives Jörmungandr a financial research shape while
remaining entirely public and synthetic. It launches one local learner service
and multiple actor threads:

```bash
python examples/train_ou_spread.py
```

The terminal shows actor progress, interleaved split assignment, replay sizes,
learner and policy versions, training and held-out statistics, auxiliary-head
accuracy, and a live policy probe across five spread z-scores.

This is an illustration of framework behavior. It does not use market data,
model an executable instrument, or make a claim about deployable returns.

## Process and actions

The spread \(x_t\) follows a discrete Ornstein–Uhlenbeck process:

```text
x[t+1] = x[t] - kappa x[t] + sigma epsilon[t]
epsilon[t] ~ Normal(0, 1)
```

The default parameters are `kappa = 0.18` and `sigma = 0.35`. The observation
uses the corresponding stationary standard deviation and contains:

```text
spread z-score
last spread change in z units
current position
remaining episode fraction
```

The fixed action vocabulary is `SHORT`, `FLAT`, and `LONG`. An action selects
the position held over the next spread change.

## Reward and reference

Per-step reward is synthetic spread P&L in stationary-standard-deviation units,
less turnover cost and a small position penalty:

```text
reward =
    position * (x[t+1] - x[t]) / stationary_std
    - 0.035 * abs(position - previous_position)
    - 0.004 * abs(position)
```

The dashboard also evaluates a declared, non-learning reference on the same
path. It is short above `+0.65z`, long below `-0.65z`, and flat otherwise.
This makes the displayed comparison reproducible and interpretable. It is a
teaching benchmark, not an investable strategy.

Episode statistics include:

- total reward for the sampled policy;
- the same-path reference reward and their difference;
- maximum drawdown of cumulative synthetic reward;
- turnover and the fraction of positive episodes; and
- recent reward traces for both splits.

Training episodes include epsilon-greedy exploration. Validation episodes use
deterministic actions, so their reward is the cleaner policy-quality signal.

## Interleaved holdout

Validation paths are seeded separately and inserted throughout the episode
schedule:

```text
TTTTV TTTTV TTTTV ...
```

Actors can complete work concurrently, but the split is fixed before an episode
starts and carried on every transition. Training samples enter prioritized
replay and update the training-only normalizer. Validation samples enter a
separate store and are evaluated without optimizer or normalization updates.

The auxiliary head predicts the sign of the OU conditional drift: `DOWN`,
`FLAT`, or `UP`. Labels are submitted after each transition batch, exercising
the public delayed-label join.

## Retained artifacts

Pass an output directory to keep the run manifest, episode results, summary,
and checkpoint:

```bash
python examples/train_ou_spread.py \
  --output-dir /tmp/jormungandr-ou-spread
```

Use `--json` for a bounded machine-readable run instead of the live terminal.
The summary includes the final deterministic policy probe and all stated
episode metrics.

## What changes for Volt

The example demonstrates the trainer and service boundary, not the eventual
Volt strategy. A Volt actor can replace the synthetic process with canonical
portfolios and versioned scenarios while retaining actor identity, split
assignment, policy version, metrics, and delayed targets.

The first integration should still use a declared, fixed action vocabulary.
Option-strategy actions might be templates such as wait, resize risk, roll a
leg, or close a structure. Contract selection, lifecycle rules, payoff
semantics, and scenario generation remain Volt responsibilities. If actions
must be state-dependent exact contracts, Jörmungandr will need an explicit
candidate-scoring and masking contract rather than hidden strategy logic.

The recording workflow is documented in
[OU spread recording](markup/README.md).
