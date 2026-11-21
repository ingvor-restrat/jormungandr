# Language interpretation plus numeric tool adaptation

This generic example tests complementary capabilities instead of asking a
language model to rediscover a numeric routing rule from text. A language policy
predicts which tools suit the current request; a native structured DQN policy
chooses among them using recent operational outcomes. The composition lives in
`examples/language_tool_benchmark.py`, using existing Jormungandr policy APIs.
No application-specific behavior is added to the runtime.

## Task and information boundaries

A simulated catalog contains four opaque tool IDs: two translation tools and two
summarization tools. The request asks for one capability. Tools have public text
descriptions, public costs and latency, and hidden, changing success probabilities.
Actual text translation or summarization is not executed; the environment returns
simulated operational success/failure. This is a routing benchmark, not an
assessment of generated text quality or arbitrary tool execution.

Capability assignments are shuffled across catalogs and candidate order is
shuffled every request. Requests are generated independently of recent outcomes.
Two different requests can have identical numeric observations, so a numeric-only
policy cannot infer the current intent from its input. All tools remain callable
at evaluation: semantic correctness is not supplied through the actor's legal
mask. These are simulated task-suitability choices, not authorization decisions.

The language stage sees only the request and descriptions. Numeric features are
zeroed and raw history is omitted. It ranks the candidates, and its top two form
a predicted suitable set. Knowing that exactly two tools are suitable is an
explicit structural assumption of this small benchmark, not a general eligibility
parser. The numeric actor receives recent mean operational reward, normalized
count and age per tool, plus cost and latency. Its window covers eight requests
across all four tools.

The task reward is `compatible * operational_success - cost - 0.1 * latency`.
Numeric history records operational reward regardless of whether the selected
tool fulfilled the linguistic request. This separates tool reliability from
interpretation mistakes. Neither learner observes hidden success probabilities
or unchosen outcomes. Expected regret and best-tool accuracy use hidden expected
task utility only in the evaluator.

With four tools, fully random selection has expected compatibility 50% and
best-compatible-tool accuracy 25%. Perfect request interpretation followed by
uniform choice between the two suitable tools gives 100% compatibility and 50%
best-tool accuracy. These scores must not be compared directly with the earlier
two-route benchmark, where random best-route accuracy was 50%.

## Training and controls

The language scorer uses the same pinned SmolLM2-135M backbone and rank-4 LoRA,
with a fresh scalar action head. It receives 128 labeled examples arranged in
paired requests: the same catalog and order with opposite requested capabilities.
Targets are uniform over the two suitable tools. Four epochs give 512
single-example cross-entropy updates at learning rate 0.0001. The language forward
chunk size is five sequences (four candidates plus a value sequence); this is
not five independent training observations.

Native DQN receives 384 operational reward interactions from twelve fresh
training worlds and 384 uniform replay updates of batch size 32. Its numeric
transformer has width 32, four heads and one layer, with learning rate 0.001.
During numeric training, ground-truth suitability labels restrict collection to
compatible tools, ensuring both capability classes receive experience. This is
privileged semantic supervision during training, not autonomous discovery of
request meaning. Hidden reliability is never provided. Episodes terminate after
each call and gamma is zero, so long-horizon credit assignment is not tested.

Evaluation freezes both learners, resets history between trials and compares:

- **Numeric-only:** DQN over every callable tool, with no request interpretation.
- **Language-only:** uniform selection within the language-predicted suitable set.
- **Combined:** epsilon-0.1 DQN within that same predicted suitable set.
- **Language + rolling:** the same language prediction followed by the simple
  history-based rolling rule.
- **Oracle numeric / oracle rolling:** the same downstream selectors supplied
  with correct suitability labels, isolating interpretation errors.
- **Random:** uniform selection among all four tools.
- **Combined without history:** the same language predictions with numeric
  history features removed.

These controls intentionally hold eligibility predictions fixed when comparing
downstream selectors. Language-only is an ablation of the numeric selector, not
an unrestricted generative agent. The oracle controls are diagnostic references,
not deployable learned interpreters.

Language scores are precomputed once per frozen model and current request, then
reused across policy comparisons. This is valid because the language stage is
stateless and does not receive operational history; no future request or score
is passed to the numeric selector. Different selectors execute separate causal
rollouts and therefore build different histories. Shared precomputed potential
outcomes and action-sampling streams permit paired comparisons.

## Run and reproduce

From a source checkout with optional language dependencies installed:

```bash
PYTHONPATH=src python -m examples.language_tool_benchmark \
  --output runs/language-tools \
  --seeds 2111,2129,2131 --device cuda --dtype bfloat16
```

CPU execution is available with `--device cpu --dtype float32`. The first model
load may download pinned weights; subsequent runs can use the cache offline.
Existing output configurations are never overwritten. Each run saves source
snapshots/hashes, dependency versions, numeric training traces, evaluator-only
world audits, complete evaluation traces and separate inference bundles for the
language and numeric policies. Preserve the actor feature contract, including
the eight-request window, when loading these bundles.

Training and evaluation request strings and tool-description templates are
disjoint. Evaluation still uses the same two capability categories. Twelve
held-out catalogs per seed give 36 worlds and 1,152 requests, covering abrupt and
gradual reliability changes. Seed 2003 with four evaluation catalogs was used for
development; the confirmation cohort uses 2111, 2129 and 2131 without checkpoint
selection or retraining on their outcomes. The same held-out wording templates
are used in development and confirmation; this is not an independent test of new
linguistic categories.

Fast tests assert that request text cannot leak into numeric arrays or callable
masks, language input excludes history and hidden quality, labels are balanced,
permuted tools remain correctly aligned, and incompatible tools receive the
correct task penalty. The optional offline language suite runs the entire
composition with a tiny locally generated backbone. These plumbing tests do not
assert that the random tiny model understands the requests.

## Scope

This tests whether a small learned interpreter supplies useful missing information
to a history-dependent numeric controller. It does not establish broad language
understanding, a need for neural interpretation over a hand-written parser, or
performance on unfamiliar tool categories. A two-category domain can often be
handled by simple rules; this experiment is a controlled capability check before
moving to richer tool descriptions and real application requests.

## Recorded confirmation

The [public artifact](../benchmarks/language_tools.json) records the development
run and the three fresh-seed runs. No hyperparameters or checkpoints were selected
using the confirmation results. The combined policy improves mean task reward
over each single-component policy and language-plus-rolling in all three seeds.

| Policy | Suitable-tool selection | Best-tool accuracy | Mean task reward | Expected regret |
|---|---:|---:|---:|---:|
| Random | 48.2% | 25.1% | 0.147 | 0.585 |
| Numeric-only | 49.3% | 38.1% | 0.251 | 0.494 |
| Language-only | 88.5% | 43.8% | 0.341 | 0.394 |
| Language + rolling | 88.4% | 56.5% | 0.428 | 0.299 |
| **Language + native DQN** | **87.6%** | **62.3%** | **0.479** | **0.260** |
| Combined without history | 89.5% | 44.4% | 0.330 | 0.390 |
| Oracle suitability + DQN | 100% | 74.7% | 0.577 | 0.158 |
| Oracle suitability + rolling | 100% | 65.7% | 0.516 | 0.218 |

The language models fit the training suitable-pair labels perfectly, but identify
both suitable tools correctly on only 78.0% of held-out requests. Some predicted
sets therefore contain one suitable and one unsuitable tool. Downstream policies
can have different suitability rates even with identical predicted sets because
they select differently within those sets. Their histories also diverge.

Combined reward by seed is 0.528, 0.476 and 0.432, versus 0.442, 0.444 and 0.399
for language-plus-rolling. This small pilot supports complementary roles for a
learned request interpreter and a history-dependent numeric selector. It does not
establish statistical superiority across arbitrary tool catalogs.

Removing history no longer produces fully random behavior: the language stage
still supplies request information. Best-tool accuracy falls from 62.3% to 44.4%,
near language-only selection within the predicted suitable set. Conversely,
numeric-only selection uses operational feedback but chooses a suitable
capability only about half the time. The oracle-suitability gap identifies
interpretation/generalization as a concrete next target, rather than evidence
that the numeric learning machinery is broken.
