# Controlled language-gate comparison

The [independent suitability pilot](tool_suitability_benchmark.md) changed both the
language decision format and the test suite. Its failure did not isolate whether
ranking, binary judgment, or wording coverage was responsible. This example
makes those comparisons explicit. It adds no domain-specific runtime behavior.

## Design

This retrains both representations under a common contract; it does not simply
rescore the previous checkpoints. Unsupported requests are absent from training
in both formats because fixed-two ranking cannot represent an empty target set.
Their secondary evaluation is therefore an out-of-training-distribution diagnostic.

Development has four cells: ranking/narrow, ranking/broad, binary/narrow, and
binary/broad. They use the same pretrained backbone revision, LoRA configuration,
initial trainable weights, optimizer, and 544 updates. Each update consumes the
same two labeled catalogs: four tools per catalog, exactly two suitable. Ranking
has two observations with uniform targets over suitable tools; binary has eight
request/tool observations with yes/no targets. Losses are averaged over each
batch. Thus catalog exposure and optimizer steps are matched, but token counts,
compute cost, and loss scales are not. Ranking's irreducible target entropy is
log(2); binary cross-entropy can approach zero.

Narrow uses the existing training request and description templates. Broad retains
them and adds paraphrases and different exclusion constructions. No development
or reserved evaluation strings are copied into either training condition. Broad
coverage gets no additional optimizer updates; insufficient fit under that fixed
budget is a possible outcome, not evidence that diversity is inherently harmful.
The training fit diagnostic scores the first 64 training catalogs, not the entire
training set.

**The primary comparison is identical for both formats:** score the same catalogs
with exactly two suitable tools, select the top two scores, and report exact-set
accuracy and selected-tool precision. For binary models, this deliberately ignores
the absolute yes/no threshold. It separates ordering quality from threshold
behavior. Independent binary scores are logit margins; ranking scores are native
candidate logits.

Secondary diagnostics test catalogs with zero through four suitable tools,
unsupported proofreading requests, and the binary model's native zero-margin
threshold. Fixed-two ranking cannot express these sets or abstain, so it receives
no fabricated native score. These diagnostics do not select the recipe. Invariance
checks reverse candidate order and rename opaque IDs on the same primary cases.
For binary input, IDs and catalog order are absent by construction; forced top-two
selection can still be sensitive to exact score ties at its selection boundary.

Development uses previously examined wording and one training seed. A declared
rule selects the highest primary exact-set accuracy; ties prefer narrow coverage,
then ranking. Confirmation trains that recipe from scratch with two new seeds and
uses a separate wording suite frozen by hash before development. It does not
retune the recipe on confirmation results. The numeric actor's artifact hash is
held fixed across both stages. World random streams differ between development
and confirmation, and policies each receive their own causal history.

The suites are intentionally small: development has 12 primary catalogs covering
six request strings; confirmation has 16 covering eight. Additional variable-size
and unsupported cases bring totals to 66 and 88. Repeated catalogs and seeds are
not independent language samples. This is a controlled diagnostic pilot, not a
statistical claim about arbitrary requests or tools. Confirmation tests the
selected recipe; it does not independently repeat the entire factorial comparison.
The unchanged lexical parser is reported on both suites and is not adjusted after
seeing the reserved wording results.

## Run

Use the repository's optional `llm` dependencies. First obtain a frozen numeric
bundle from [the language/tool benchmark](language_tool_benchmark.md). Then:

```sh
PYTHONPATH=src:. python -m examples.tool_gate_comparison \
  --numeric-root runs/language-tools --numeric-seed 2111 \
  --output runs/gate-development --device cuda

PYTHONPATH=src:. python -m examples.tool_gate_comparison \
  --numeric-root runs/language-tools --numeric-seed 2111 \
  --stage confirmation --selection runs/gate-development/selection.json \
  --seeds 2129,2131 --output runs/gate-confirmation --device cuda
```

CPU is supported. The default backbone is the same pinned SmolLM2-135M base model
used in the earlier experiments. Run directories must not already exist.
Confirmation checks the model/revision, training budget, unused training seeds,
reserved-suite hash, and frozen numeric artifact. Keep the development protocol,
selection, and results together when moving them to another machine.

Artifacts include the protocol, source snapshot, training-catalog hashes, initial
trainable-weight hashes, loss histories, raw semantic scores and labels,
per-policy composition traces, world arrays, and language inference bundles.
Retain the repository revision for imported helpers. All evaluation paths verify
that weights and update counters stay frozen. Numeric composition compares the
shared top-two decision, unchanged parser, and oracle eligibility; binary also
reports its native threshold gate. No real API execution, argument generation,
long-horizon credit assignment, or end-to-end language RL is tested here.

## Recorded controlled pilot

All four development cells used seed 2111 and identical initial weights. Training
catalog hashes match between formats within each coverage condition. Narrow
training covers 14 request strings and eight descriptions; broad covers 26 and
14, respectively. All cells receive 1,088 catalog presentations and 544 updates.

| Format | Coverage | Training sample exact pair | Development exact pair | Native exact pair | Top-two composition reward |
|---|---|---:|---:|---:|---:|
| ranking | narrow | 96.9% | 75.0% | Not represented | 0.336 |
| ranking | broad | 92.2% | 83.3% | Not represented | 0.460 |
| binary | narrow | 100.0% | 50.0% | 25.0% | 0.388 |
| binary | broad | 90.6% | 50.0% | 50.0% | 0.366 |

The predeclared rule selected **ranking / broad**.
The unchanged parser matches oracle eligibility on development, with mean reward
0.579. No development score is described as a blind test result.

| Confirmation seed | Exact pair | Workflow suitability | Top-two reward | Parser reward | Oracle reward |
|---|---:|---:|---:|---:|---:|
| 2129 | 93.8% | 71.9% | 0.376 | -0.035 | 0.570 |
| 2131 | 56.2% | 58.6% | 0.265 | -0.035 | 0.571 |

The same frozen numeric bundle is used throughout. Confirmation changes the
language training seed and evaluates the reserved wording and new simulated
worlds. Its wording digest matches the commitment saved before development.
Full metrics and control hashes are in
[`benchmarks/tool_gate_comparison.json`](../benchmarks/tool_gate_comparison.json).

Exact-pair chance performance for uniformly selecting two of four tools is
1/6 (16.7%), not 50%. These are exact-set metrics, not the earlier independent
yes/no pair-accuracy metric. Differences between cells are conditional on this
small suite, model, training seed, and update budget; the confirmation stage does
not independently establish which representation is universally better.

### Interpretation and post-hoc checks

The development comparison supports two conditional findings: ranking transfers
better than the binary format at this budget, and broader wording helps ranking.
For binary, broader wording improves native eligible-set accuracy across all
cardinalities from 37.9% to 59.1%, but its forced-two primary accuracy remains 50%.
The representation factor includes prompt serialization and objective, not merely
the spelling of labels. These results concern a 135M base backbone with LoRA and a
fresh scalar scoring head, not instruction-tuned LLMs in general.

The selected recipe is **not robust enough to promote**. Training seeds vary both
initialization and sampled training catalogs. Confirmation exact-pair accuracy
ranges from 56.2% to 93.8%; workflow suitability ranges from 58.6% to 71.9%, and
both models remain substantially below oracle numeric reward. Reordering changes
5/16 and 4/16 primary decisions; opaque ID renaming changes 4/16 and 3/16.

A read-only post-hoc audit of saved workflow traces gives a less optimistic gate
metric than the small primary suite: correct eligible pairs on 217/384 (56.5%) and
149/384 (38.8%) requests. These are repeated requests across twelve simulated
catalog worlds, not 384 independent language examples. Catalog/description
combinations therefore need broader coverage in future evaluation. For example,
the first model gets both small-suite Korean and Dutch catalogs right, but only
3/48 and 5/50 of those requests' eligible sets right in the workflow catalogs.
A four-case CPU batching probe produces identical scores and selections when
scored together or individually; this narrow check does not rule out all possible
numerical issues. These post-hoc checks were not used to train or select a recipe.

The unchanged parser also fails to generalize: its reserved primary exact-set
accuracy is 0%, despite 100% on development. This exposes the limits of its literal
vocabulary and exclusion grammar. It does not establish superiority over stronger
rule-based or instruction-model baselines. The useful outcome is a controlled
comparison and a documented failure boundary, not a production-ready tool router.
