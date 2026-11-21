# Learning sanity checks and independent controls

A finite loss and changed weights do not demonstrate learning. Jormungandr has
small behavioral tests with known answers, plus an optional experiment that
separates reward learning from learning a representation of numeric feedback.
Generic task logic stays in `examples/`; no finance or host-specific configuration
is required.

## Fast offline tests

```bash
python -m pytest -q tests/test_learning_sanity.py tests/test_learning_diagnostic.py
```

The first suite uses a balanced, fully observable two-action contextual bandit.
The observation explicitly marks the rewarding action. It checks that:

- A small independent PyTorch classifier learns both classes perfectly. Removing
  its informative cue reduces balanced accuracy to exactly 50%.
- Jormungandr DQN learns both actions' rewards with a clear value margin.
- Jormungandr PPO learns to assign over 90% probability to the correct action in
  both contexts using sampled actions and rewards.

The second suite checks an independent PyTorch reward regressor, reconstructs
teacher labels from visible inputs, and exercises the CPU diagnostic end to end.
The independent learner uses a shared per-route two-layer MLP, Adam, uniform
replay and a chosen-action reward loss. It does not use Jormungandr's policy
network, collation, PPO/DQN update routines, or runtime. It does share the task
observations, evaluator and scenario generator; it is an independent learner,
not a completely independent environment implementation.

With the optional `llm` dependencies installed, the existing offline language
suite additionally tests fitting candidates explicitly described as `good` or
`bad`. Its tiny random one-layer backbone and small tokenizer are generated
locally, with no model downloads. An initial test requiring that same tiny model
to match a global text cue to a candidate failed at 128 updates; the candidate
cue test intentionally checks a simpler learning contract. Passing it does not
establish language understanding. The pretrained experiment below separately
checks matching a global text cue to an action.

```bash
USE_TF=0 HF_HUB_OFFLINE=1 python -m pytest -q tests/test_llm_ppo.py
```

These tests run in the existing CI jobs. They detect basic learning regressions;
they do not prove that every algorithm, model or hyperparameter setting is correct.

## Reward learning versus rule imitation

Run the numeric experiment from a source checkout:

```bash
PYTHONPATH=src python -m examples.learning_diagnostic \
  --output runs/learning-numeric --skip-llm
```

Include the pinned pretrained SmolLM2-135M language scorer:

```bash
PYTHONPATH=src python -m examples.learning_diagnostic \
  --output runs/learning-all --device cuda --dtype bfloat16
```

The default seeds are 401, 503 and 607, separate from the earlier pilots.
Each seed has two fixed training worlds and four held-out worlds (two abrupt
changes, two gradual changes). All routes expose the same causal features:
recent mean reward, count and age, plus public cost and latency. No learner gets
hidden route reliability or unchosen outcomes. The language model receives the
summaries as serialized text; numeric models receive numbers directly.

The comparisons are:

- **Independent reward learner:** plain PyTorch, a shared MLP per route, observed
  rewards only. This bypasses Jormungandr's learning code.
- **Jormungandr DQN:** the existing structured Double-DQN implementation and its
  numeric transformer, also using rewards only.
- **Numeric imitation:** the numeric policy directly learns the rolling
  estimator's action distribution through cross-entropy.
- **LLM imitation:** the same language scoring architecture, LoRA rank and
  learning rate used in the earlier pilot, trained through cross-entropy instead
  of PPO. This bypasses the PPO loss and advantage computation.
- **LLM text cue:** a separately initialized language policy learns two explicit
  prompts identifying the correct route. This is a two-example fitting check,
  not a generalization or reasoning benchmark.

Reward learners collect 384 interactions per seed and perform 384 replay updates
of 32 examples each. Behavior remains frozen during each collection round. They
use epsilon 0.2 during collection and 0.1 during evaluation. Since every request
terminates and gamma is zero, their targets are immediate observed rewards:
DQN's bootstrap and target-network machinery are not exercised by this task.
This cannot validate temporal credit assignment on a multi-step problem.

Imitation learns from 64 teacher-generated observations over six epochs, making
384 single-example optimizer updates. Labels depend only on past observed
feedback: try unobserved routes, otherwise prefer higher recent mean reward with
10% exploration. They are not oracle labels. Separate teacher trajectories on
held-out worlds measure agreement on previously unseen inputs; closed-loop
rollouts measure task reward after the learned policy starts choosing its own
actions. These are different measurements. The text-cue control uses 64 updates.

Interaction counts, replay use, architectures, initialization and optimizer
settings differ across methods. This is a diagnostic ladder, not a controlled
claim that one RL algorithm is superior. No checkpoint is selected or tuned
using held-out scores. Evaluation checks parameter hashes and update counters;
removing history also removes every history-derived feature. Configuration,
source snapshots/hashes, world audits, traces, metrics and checkpoints are saved.
An existing output configuration is not overwritten.

## Interpreting chance and failures

Best-route accuracy is the fraction of actions that select the route with the
highest **expected utility**. It is not the rate of successful tool calls. A
uniform random policy has expected accuracy 50% with two actions and a unique
best action. A measured value below 50% on a small sample does not establish
systematic anti-learning. Time steps within a trial are dependent, so treating
all steps as independent observations would give misleading uncertainty estimates.

Neither a near-chance LLM nor a passing unit test establishes the cause. Useful
comparisons are whether an independent learner succeeds, whether removing inputs
hurts, whether the scorer can fit explicit cues, and whether direct supervised
labels solve the same input problem. Report reward, expected regret, per-seed
variation and held-out behavior alongside accuracy.

The earlier LLM results were poor as adaptive policies. A seed reaching 90.6%
training accuracy and 42.2% held-out accuracy is evidence of a generalization
failure in that run, not proof of an implementation bug or proof of memorization.
The successful rolling baseline already shows that past feedback is informative.

## Established reference points

[PyTorch's DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
uses CartPole. It is a useful independent reference for a multi-step discrete
control learner. [Stable-Baselines3's RL guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
recommends environment checks, separate evaluation, multiple runs and attention
to observation preprocessing. The contextual bandit here is a smaller local
sanity test, not a claim to implement either project's standard benchmark suite.

For generated-language RL, [TRL's GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer)
provides a Qwen2.5-0.5B-Instruct math example with a verifiable accuracy reward.
That is a different experiment: an instruction model generates answers using its
language head. Our pilot attaches a fresh scalar action head to a 135M base
model and scores candidate actions. Its results should not be read as a benchmark
of standard TRL training or of open-weight LLMs generally. Explicit cues, small
supervised fitting checks, and independently checked rewards are appropriate
first diagnostics before attempting a larger language-RL reproduction.

## Recorded independent-control results

The [three-seed artifact](../benchmarks/learning_diagnostic.json) records this
pilot on seeds 401, 503 and 607. All values below concern paired held-out worlds.

| Learner | Best-route accuracy | Mean reward | Expected regret |
|---|---:|---:|---:|
| Independent PyTorch reward learner | 76.8% | 0.612 | 0.137 |
| Jormungandr DQN | 78.4% | 0.631 | 0.119 |
| Numeric rule imitation | 66.7% | 0.534 | 0.213 |
| LLM rule imitation | 51.0% | 0.443 | 0.310 |
| Rolling estimator | 67.7% | 0.549 | 0.196 |

Removing history reduces DQN to 49.2% and the independent learner to 56.5%.
All three pretrained text-cue controls reach 100% agreement on the two fitted
prompts. These results demonstrate informative numeric inputs and successful
reward learning both inside and outside Jormungandr. They do not rule out all
bugs in the language path, nor do they demonstrate long-horizon DQN correctness.
The language scorer's poor imitation performance shows that PPO alone cannot
explain its failure on the feedback task. Representation, optimization and the
small training budget remain relevant explanations.

The follow-up [hybrid adaptation pipeline](hybrid_adaptation.md) transfers the
reward-trained numeric skill into an actor that also supports language features,
and evaluates each branch independently.

## Ordered history and action selection

The original numeric summaries discard reward order: two opposite recent trends
can produce identical mean/count/age vectors. A regression test makes that
aliasing explicit. `examples/history_benchmark.py` appends newest-first
reward/observed pairs for the same four-request window, preserving which route
was sampled at each position. Missing observations have an explicit zero flag.
No hidden outcome or future reliability is added.

```bash
PYTHONPATH=src python -m examples.history_benchmark --output runs/history-order
```

The [recorded three-seed pilot](../benchmarks/history_order.json), on seeds 1409,
1511 and 1601, holds DQN hyperparameters and interaction/update budgets fixed.
Only input width differs. Both policies are evaluated with epsilon 0.1, greedily,
and without history; the ordered policy is also evaluated with only the ordered
columns removed. Greedy evaluation uses new rollouts because changed actions
change the feedback available on subsequent requests.

| Input | Epsilon-0.1 accuracy | Greedy accuracy | Epsilon-0.1 reward | Epsilon-0.1 regret |
|---|---:|---:|---:|---:|
| Summary | 75.0% | 76.9% | 0.607 | 0.147 |
| Summary + ordered events | 75.2% | 78.2% | 0.605 | 0.147 |
| Rolling baseline | 68.1% | — | 0.547 | 0.196 |

This does not demonstrate a reliable gain from supplying order. Removing the
ordered columns from the trained ordered policy actually yields 76.3% sampled
accuracy, versus 75.2% with them present. Removing all history drops the models
to 50.8% and 50.1%. The representational information loss is real, but we have
not shown it to be the dominant performance bottleneck at this training budget.
More features also require learning how to use them; the mean/count/age
representation remains the supported default in the confirmed hybrid recipe.

## Training-world coverage at a fixed budget

`examples/coverage_benchmark.py` compares six collection rounds over the same two
worlds against two fresh worlds in each round (12 unique training worlds).
Both modes start from identical weights and use the same action-sampling streams,
384 reward interactions, 384 replay updates of batch size 32, DQN settings, causal
summaries and paired held-out scenarios. The first round is identical, including
its losses. A regression test checks this pairing and both budgets.

```bash
PYTHONPATH=src python -m examples.coverage_benchmark --output runs/coverage
```

Fresh training worlds vary reliability, costs, latency, initial route, switch time
and sampled outcomes within the same abrupt-change family. This is not extra
training on the held-out gradual-change family. The three-seed cohort is 1709,
1801 and 1901; each seed has 12 held-out worlds. Evaluation is frozen and includes
greedy and no-history rollouts. Source snapshots, world audits, traces, checkpoint
bundles and per-seed paired differences are written to the output directory.

The [recorded coverage pilot](../benchmarks/training_coverage.json) gives:

| Training coverage | Epsilon-0.1 accuracy | Mean reward | Expected regret | Greedy accuracy |
|---|---:|---:|---:|---:|
| Two worlds repeated | 76.3% | 0.598 | 0.137 | 79.9% |
| Twelve fresh worlds | 77.1% | 0.609 | 0.130 | 79.9% |
| Rolling baseline | 65.6% | 0.532 | 0.211 | — |

The sampled accuracy differences by seed are -0.3, +1.3 and +1.3 percentage
points. Broader coverage produces modest aggregate improvements in reward and
regret, not a demonstrated large accuracy gain. Greedy accuracy is unchanged in
aggregate, though its reward and regret improve slightly. Removing history gives
48.2% and 50.5%, respectively. Three seeds do not establish statistical superiority.
These absolute scores use a different cohort from the hybrid confirmation and
should not be interpreted as a regression from its 79.6% result.

The confirmed hybrid recipe remains unchanged. This experiment supports testing
broader coverage in subsequent training, while the next distinct capability test
should require language interpretation in addition to numeric adaptation. That
would measure whether the language branch adds useful behavior instead of merely
accompanying a competent numeric controller.
