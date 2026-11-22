# Auxiliary objectives

Jörmungandr supports optional scalar objectives sharing a learner's representation.
The core owns loss composition and update mechanics; applications own heads,
supervision, context construction, and causal validity. No auxiliary task is enabled
by default. This is not an automatic context-label generator or a new RL algorithm.

```python
from jormungandr.auxiliary import AuxiliaryLoss, compose_losses, context_selection_loss

context_loss = context_selection_loss(
    option_logits, positive_indices, valid=valid_pairs,
    weights=importance_weights, margin_clip=5.0,
)
loss, metrics = compose_losses(
    task_loss, [AuxiliaryLoss("context", context_loss, coefficient=0.1)]
)
loss.backward()
```

`option_logits` has shape `(batch, 2)`; `positive_indices` is an int64 vector
containing 0 or 1. The optional mask is boolean. The loss is
`softplus(-clamp(correct_logit - other_logit, -clip, clip))`. Hard clipping also
zeros gradients outside either boundary. Masked rows contribute zero; the
**denominator remains the entire batch size**, including masked rows. Optional
importance weights multiply each row without renormalization. This convention
preserves gradients under uneven microbatch accumulation. Inputs must be finite,
including masked logits. Coefficients must be finite and nonnegative.

## Native structured and language DQN

Subclass `StructuredDQNAgent` (or `LLMDQNAgent`) and override
`_auxiliary_losses(self, transitions, weights=None)` to return `AuxiliaryLoss`
terms. Each term must be averaged over the **whole supplied chunk**. The update
weights chunks by their fraction of the complete batch. Replay importance weights
are passed to the hook; the application decides whether its auxiliary objective
needs that correction.

Register extra heads and shared encoders inside the Q module returned by
`_build_q`, before optimizer and target creation. This includes them in optimizer
parameters, gradient clipping, target synchronization, and checkpoints. Restore
with the same subclass/configuration; a checkpoint cannot reconstruct arbitrary
application code. See `tests/test_auxiliary.py` for a small working subclass and
checkpoint/microbatch tests.

An overridden hook requires training-split transitions, checked before forward or
hook execution. Applications must additionally validate the provenance of labels
and external datasets: the split check cannot detect future information hidden
inside metadata. Synthetic negative contexts belong in the auxiliary objective,
not in replay as invented environment transitions.

Metrics include `auxiliary/<name>`, `auxiliary_weighted/<name>`, and
`auxiliary_total`; the reported total loss includes the weighted auxiliary sum.
Replay priorities remain based on TD errors. Language DQN retains the original
candidate catalog when a hook is installed, even in gamma-zero evidence mode,
because the hook may need candidates omitted by the reward-only optimization.

The generic composer can be used in other learners, but native integration in
this change is limited to structured/language DQN; PPO/GRPO integration is not
implemented. The numeric research example uses the composer in a small standalone
learner to isolate the scientific question. See
[the pilot and replication plan](context_auxiliary_research.md).
