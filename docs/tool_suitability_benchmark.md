# Independent tool suitability

`examples/tool_suitability_benchmark.py` extends the simulated language/tool
experiment with independent binary suitability decisions. The same optional
LoRA backbone and native candidate-scoring policy score `no` versus `yes` for
one request and one tool description. No framework algorithm changes are needed.

The adapter excludes opaque tool identifiers and the rest of the catalog from
the language input. Catalog order and ID renaming therefore cannot change a
request/description score. Each tool with a softmax `yes` score at least 0.5
is eligible. This fixed threshold is not fitted on evaluation data; scores are
not claimed to be calibrated confidence. Zero eligible tools returns an explicit
`abstain` orchestration decision. It does not send an empty action mask to the
numeric policy, invent a tool result, or append feedback.

## Reproduce

Install the package with its `llm` extras. First generate numeric bundles using
`examples.language_tool_benchmark` as described in
[the original benchmark](language_tool_benchmark.md). Then run:

```sh
PYTHONPATH=src:. python -m examples.tool_suitability_benchmark \
  --numeric-root runs/language-tools \
  --output runs/suitability \
  --seeds 2111,2129,2131 \
  --device cuda
```

The numeric root must contain `numeric-SEED` inference bundles for each requested
seed. Use the same seeds when generating them. CPU is supported, but a GPU is
recommended for the language training. Defaults pin the same SmolLM2-135M revision
as the original example. Run directories must be new to avoid overwriting results.

## Protocol

- Training uses 136 request/description pairs in minibatches of eight, with
  positive and negative labels, explicit exclusions, and unsupported proofreading
  requests. Set the training budget with `--epochs` and `--learning-rate`; each
  epoch has 17 optimizer updates. The previous benchmark's test wording is now
  development data.
- The new semantic holdout contains 36 pairs: nine requests and four descriptions,
  including paraphrases, exclusions, and requests unsupported by either capability.
  Report pair accuracy, positive/negative recall, Brier score, and a declared
  keyword-parser baseline. These are small template sets, not independent samples
  of arbitrary language.
- Thirty controlled four-tool catalogs cover zero through four suitable tools.
  Report exact eligible-set accuracy for each count. Repeated capabilities are
  intentional. Order/renaming invariance is an adapter contract, not evidence that
  the underlying LLM itself learned invariance.
- Composition reuses frozen native DQN bundles, new simulated worlds (seed offset
  100000), and the new held-out request/description wording. It compares the learned
  gate, the parser gate, and ground-truth suitability with the same numeric actor.
  All methods see their own causal histories. The actor's training still used
  perfect suitability labels; robustness to interpreter errors is not trained here.
- Abstention receives zero task reward, incurs no tool cost, and adds no history.
  Composition worlds always contain suitable tools; unnecessary abstention is
  visible in its rate, compatibility, and best-tool accuracy. This is a modeling
  choice, not a general-purpose economic value for abstention.

Each run saves configuration, the benchmark source snapshot/hash, per-seed
language inference bundles, semantic logits/scores/labels, composition decisions,
world arrays, numeric artifact hashes, and results. Evaluation asserts that both
actors' weights and update counters remain unchanged. Source snapshots cover this
entry point; retain the repository revision for its imported helpers.

This remains supervised language adaptation plus a frozen numeric bandit policy.
It does not train multi-step planning, execute real APIs, generate arguments,
interpret tool outputs, or implement a prompt-injection defense. Descriptions are
trusted simulated data. More tools increase pair-scoring inference cost, though
identical public request/description pairs are deduplicated within each scoring
call. A lexical baseline can be entirely adequate for this two-capability task;
learned-model value must be demonstrated rather than assumed.

## Recorded local pilot

The 32-epoch recipe uses minibatches of eight at learning rate 0.0001: 544
optimizer updates and 4,352 example presentations per seed. Seed 2111 was used
for development; the recipe was then fixed for seeds 2129 and 2131. The latter
share the wording suite, so they test training variability, not new language
domains. Results are recorded in [`benchmarks/tool_suitability.json`](../benchmarks/tool_suitability.json).

| Seed | Role | Training accuracy | Held-out pair accuracy | Valid-tool recall | Learned-gate reward | Parser/oracle reward |
|---|---|---:|---:|---:|---:|---:|
| 2111 | development | 96.3% | 80.6% | 58.3% | 0.361 | 0.563 |
| 2129 | fixed_recipe | 100.0% | 72.2% | 16.7% | 0.102 | 0.549 |
| 2131 | fixed_recipe | 100.0% | 69.4% | 16.7% | 0.127 | 0.553 |

The parser scores 100% on these 36 pairs and produces exactly the oracle numeric
rollouts under the same random draws. The learned gate does not meet that baseline.
A reject-everything gate would score 66.7% pair accuracy, which is why valid-tool
recall, exact-set accuracy, and abstention rate must accompany accuracy.

Two earlier development probes rejected every held-out tool: single-example
updates for four epochs had unstable loss; minibatches for eight epochs at
0.00003 stabilized loss but underfit. Longer training improved fit, but the
remaining failure is generalization. These probes informed the recipe and are
not independent confirmation results.

This example is useful as a regression/evaluation harness, not as evidence that
the learned gate is preferable to a parser. The original language/tool benchmark
is retained unchanged. A larger and more varied training corpus or a stronger
language baseline needs evaluation before promoting a learned gate; a larger GPU
is not demonstrated to be necessary.
