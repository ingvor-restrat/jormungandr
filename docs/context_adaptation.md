# Learning to use causal feedback

This example isolates adaptation from language classification. Two services both
accept a job, but only one currently fulfills it. The successful service is hidden
and may swap once without warning. An actor receives feedback only after its own
call. It retains that experience across calls, while model weights remain frozen
throughout evaluation. This is a minimal partially observed control task, not a
production router or a full reproduction of the research paper.

The experiment is inspired by [RL + Transformer = A General-Purpose Problem
Solver](https://arxiv.org/pdf/2501.14176). That work uses Llama 3.1 8B Instruct,
IA3 adapters, DQN, and multi-episode Frozen Lake histories. Here the backbone is
SmolLM2-135M, the adapter is LoRA, and the environment is a much simpler service
selection task. The paper's broader planning and transfer claims are not claims
of this benchmark.

## Native learner

`llm_dqn` is an optional structured learner plugin. It shares native Double-DQN
replay, action identity alignment, importance weighting, Huber TD loss, and
terminal masking with `structured_dqn`. It supplies a language Q network,
Polyak target updates (`llm_target_tau`, default 0.1), and adapter-only checkpoints.
The unused state-value head is frozen. Numeric branches are rejected so language
learning cannot be silently attributed to a separate numeric actor.

The online and target networks have separate weights. This prototype duplicates
the frozen backbone in memory; larger models need an appropriate memory budget.
Checkpoints save both online and delayed target adapters/heads, together with
optimizer state for training checkpoints. Inference bundles omit optimizer state.
Loading validates base-model identity and tensor schemas. Optional Transformers
and PEFT dependencies are imported only when constructing the language model.

Replay ingress validates the complete batch before storage, including held-out
contexts. Learner updates additionally require the training split. Both require
matching actor provenance, the
same trial on both sides of a transition, and no current/future feedback in the
current observation from that episode. The next observation may contain the
chosen action's outcome. Actors remain responsible for ordering events across
different episode IDs; episode names alone cannot establish their temporal order.

The existing structured runtime supports the plugin. Its defaults are batch size
4 and minimum replay 32; explicit caller settings take precedence. Configure:

```json
{
  "algo": "llm_dqn",
  "enabled": true,
  "device": "cuda",
  "batch_size": 4,
  "min_replay": 32,
  "plugin_config": {
    "llm_model": "HuggingFaceTB/SmolLM2-135M",
    "llm_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "llm_max_tokens": 1536,
    "llm_forward_batch_size": 12,
    "gamma": 0.9,
    "lr": 0.0001,
    "llm_target_tau": 0.1
  }
}
```

## Information and episode contracts

Each training session has twelve calls. Only the final transition has `done=True`:
individual calls are inner attempts, not boundaries that discard the session's
future value. A new session starts with empty memory. All feature arrays are zero;
the LLM receives public descriptions and causal experience text, never the hidden
successful-service index or an oracle eligibility mask.

Worlds come in matched pairs with identical public inputs and opposite successful
services. Consequently, a deterministic history-free policy must average 50%.
The first call is a common forced probe. Half the sessions remain stable; half
swap successful services at an unseen time. Immediate feedback identifies the
current successful service in this deterministic two-service problem. Therefore,
a simple win-stay/lose-switch policy is an exact reference after one observation.
This setup does not demonstrate learning a sophisticated exploration strategy.

Two history representations use the same evidence:

- Raw: the experience timeline is supplied in actor-owned context.
- Grouped: additionally attach each recorded outcome to the corresponding
  candidate's description, using its action identity. No recommended action,
  unobserved reward, or hidden service state is inserted. This is deterministic
  state preprocessing, not a separately trained numeric policy.

With the default one-event window, grouping does not summarize or discard any
retained information. Removing or corrupting history affects both the timeline
and grouped descriptions. Successful API execution is distinct from fulfilling
the task; the reward measures the latter.

## Reproduce

Install the optional `llm` dependencies and run from the repository root:

```sh
PYTHONPATH=src:. python -m examples.context_adaptation_benchmark \
  --output runs/context-development --device cuda --group-history

PYTHONPATH=src:. python -m examples.context_adaptation_benchmark \
  --output runs/context-confirmation-3203 --device cuda \
  --group-history --stage confirmation --seed 3203
```

Repeat confirmation with seed 3301. Confirmation uses separate world-generation
streams, reserved request paraphrases, and additional renaming trials with unseen
service identifiers. The learning recipe is 128 training sessions, 768 optimizer
updates, minibatches of four, learning rate 0.0001, discount 0.9, and target Polyak
factor 0.1. All collected training actions are random; training uses their actual
rewards, not optimal-action labels. The gamma-zero control is available with
`--gamma 0`; omit `--group-history` for the raw-timeline control.

Two independent controls are included:

```sh
PYTHONPATH=src:. python -m examples.context_numeric_control \
  --output runs/context-numeric-control
PYTHONPATH=src:. python -m examples.context_instruction_baseline \
  --output runs/context-instruction --device cuda --group-history
```

The numeric control is a small, independently implemented reward regressor with
explicit action/history binding. It receives the same chosen-action rewards and
sample budget, but a stronger structured representation. The instruction control
uses the native language-model head of a pinned SmolLM2-135M-Instruct checkpoint,
restricting the response to single-token A/B choices. It uses the same public
information in a chat prompt, with no fine-tuning. It is a useful baseline, not
an architecture-matched ablation of the base model's value-head training.

## Evaluation

Report first-call success, success after the first call in stable sessions, and
success after one feedback observation following a change. Also report total
reward, which includes unavoidable initial uncertainty and the first change
error. For these balanced twelve-call sessions the memory reference averages
11/12 = 91.7%, despite being perfect after adaptation.

Frozen-policy controls remove feedback, invert task feedback, rename services,
and reverse candidate order. The success criterion requires at least 90% stable
success and 85% post-change success for the original, renamed, and reordered
conditions, plus exactly chance behavior without history. Raw traces, learning
curves, model bundles, configuration, and the benchmark source are saved per run.
Retain the repository revision for imported helpers and the learner plugin.

This experiment can establish that the language Q policy uses recorded feedback
and can recover from a hidden change. It cannot establish broad language
understanding, general tool orchestration, long-horizon planning, or robustness to
noisy/misleading real-world feedback. Grouped history is a consequential design
choice and must be reported alongside any performance numbers.

## Recorded results

[Machine-readable results](../benchmarks/context_adaptation.json) retain failed
controls as well as successful runs. Development used seed 3109. Confirmation
used fresh training seeds 3203 and 3301 with reserved wording and identifiers.
Both confirmation source snapshots and generated worlds match the hashes frozen
before training; the training recipe was unchanged.

| Policy / representation | Stable after first | After one change observation |
| --- | ---: | ---: |
| Language DQN, raw history, gamma 0.9 | 50% | 50% |
| Language reward learner, raw history, gamma 0 | 50% | 50% |
| Language reward learner, grouped history, gamma 0 | 100% | 100% |
| Language DQN, grouped history, gamma 0.9 | 100% | 100% |
| Independent numeric control | 100% | 100% |
| Untuned 135M instruction baseline, raw or grouped | 50% | 50% |

Both confirmation seeds reproduce 100% stable and post-change success, including
renaming and order reversal. Removing history gives 50%; consistently reversing
feedback gives 0% on these two metrics. Total reward is 91.7%, matching the memory
reference, because initial probing and unannounced changes still incur errors.
All evaluation weights remain frozen.

The decisive change was explicitly associating previously observed outcomes with
candidate identities. Changing the discount alone did not fix learning. This
establishes feedback-dependent adaptation with this representation and training
budget; it does not establish that the raw-history model could never learn, that
LLMs are necessary for this task, or that the same result holds with many services,
noisy rewards, longer histories, or broader language variation. The instruction
baseline is specific to its small model and restricted A/B prompt.
