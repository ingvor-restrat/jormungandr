# Jormungandr live OU spread learner

![A live distributed OU spread training monitor](ou-spread.gif)

Run the real terminal example:

```bash
python examples/train_ou_spread.py
```

The display follows two actors as they send synthetic OU spread experiences to
one central C51 learner. It exposes training and held-out replay sizes, policy
versions, losses, the delayed-label auxiliary head, reward and drawdown
statistics, and deterministic inference at spread z-scores from `-2` to `+2`.

All paths are generated from fixed seeds. No market data, order-book replay, or
private strategy code is used. The reference convergence rule and all reward
costs are stated in the
[example walkthrough](../ou-spread-example.md).

## Bounded runs

For machine-readable results without the live screen:

```bash
python examples/train_ou_spread.py \
  --json \
  --train-episodes 8 \
  --validation-episodes 2 \
  --horizon 24
```

For ANSI-free terminal output:

```bash
python examples/train_ou_spread.py --no-color
```

## Rebuild the recording

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[record]"
.venv/bin/python docs/markup/record.py
```

The recorder invokes the actual example in `--record` mode and converts its
form-feed-delimited terminal frames with Pillow. The final frame is held long
enough to inspect the learned contrarian policy.
