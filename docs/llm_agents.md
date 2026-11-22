# Experimental language-model agents

Jormungandr's optional `llm_ppo` plugin places a pretrained Hugging Face language
backbone and LoRA adapters inside the existing structured PPO lifecycle. Multiple
actors can share that learner while retaining their own histories. Other agents,
tools, and evaluators can remain fixed while this designated policy learns.

This first implementation **scores a bounded set of actor-approved actions**.
It does not generate conversational answers, executable programs, or unrestricted
MCP arguments. A newly initialized scalar action head and value head read the
language backbone's representations; PPO trains those heads and the adapters.
The pretrained base weights stay frozen. This is not equivalent to instruction
following through an existing language-generation head.

## Install and run

Use a Python environment with a PyTorch build suitable for the target device:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,llm]'
python -m pytest tests/test_llm_context.py tests/test_llm_ppo.py
```

The tests create a tiny random Llama checkpoint and tokenizer locally. They do
not download model weights and run on CPU. They verify training mechanics, not
the quality of a pretrained model. The core installation does not import or
require Transformers/PEFT until an LLM agent is constructed.

Run the public synthetic prototype using a pinned small pretrained checkpoint:

```bash
python examples/llm_agent_prototype.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --device cuda --dtype bfloat16 \
  --actors 2 --rounds 2 --episodes 4 \
  --output runs/llm-agent-prototype
```

Use `--device cpu --dtype float32` on a CPU host. A local Hugging Face model
directory can replace the model ID; it requires no revision argument. Remote
models require an immutable commit hash. The loader disables remote code.
The model must support `AutoModel`, return `last_hidden_state`, and expose a
`hidden_size`; the tested families here are Llama and SmolLM2. This is not a
promise that every Hub architecture is compatible.

The example uses Jormungandr's public in-process runtime APIs, the same methods
behind the HTTP service. Two actors select between two synthetic tool routes.
A fixed executor and scorer supply outcomes under a hidden switching condition.
It is multiple actors sharing one trained policy, not a demonstration of several
independently learning agents or emergent team coordination. One action is
selected per agent turn. The example:

1. Collects complete episodes under one policy version per training batch.
2. Sends them through structured trajectory ingress for PPO updates.
3. Sends separate validation episodes to the held-out store.
4. Saves a checkpoint and reloads a frozen model through the runtime.
5. Checks restored scores and runs evaluation with and without retained history.

`transitions.jsonl` records observations, history snapshots, selected actions,
rewards, actor/episode identities, splits, and behavior policy versions.
`summary.json` records update metrics, checkpoint verification, and the small
history comparison. These are **smoke diagnostics**, not evidence of successful
meta-learning, superiority to a baseline, or financial performance. Run many
held-out trials and seeds before interpreting reward differences.

## Service configuration

The ordinary `/v2/models` request can select the plugin:

```json
{
  "model_id": "language-researcher",
  "representation": {
    "global_dim": 1,
    "entity_dim": 1,
    "candidate_dim": 1,
    "entity_type_count": 1
  },
  "learner": {
    "enabled": true,
    "algo": "llm_ppo",
    "device": "cuda",
    "min_trajectory_steps": 8,
    "max_trajectory_batch_steps": 8,
    "max_policy_lag": 0,
    "plugin_config": {
      "llm_model": "HuggingFaceTB/SmolLM2-135M",
      "llm_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
      "llm_dtype": "bfloat16",
      "llm_max_tokens": 512,
      "llm_forward_batch_size": 2,
      "llm_lora_rank": 4,
      "llm_lora_alpha": 16,
      "epochs": 1,
      "minibatch_size": 1,
      "lr": 0.0001,
      "value_backbone_gradient_scale": 0.0
    }
  }
}
```

Actors request `/v2/models/{id}/policy/score`, sample using
`sample_structured_joint_action`, execute the selected allowed operation, and
submit `/v2/models/{id}/trajectories/add`. The prototype accepts one factor per
turn. Many different actors/roles may still share the policy, or separate model
IDs can own different policies. The actor supplies role/task text in the
observation. Actor names and trial identifiers are audit metadata rather than
features revealing the correct action.

Numeric observations retain their existing schema. Add language inputs through
`observation.metadata["llm"]`, preferably using `AgentContext.metadata()`:

```python
from jormungandr.llm_context import AgentContext, Experience

context = AgentContext("researcher-1", "experiment-17", "train", max_events=8)
metadata = context.metadata(
    "Choose the next experiment after reviewing the observed results.",
    {"test-a": "Run test A.", "test-b": "Run test B."},
)
# Attach metadata to the current EntityCandidateObservation, then act.
# Only after executing the action and receiving its outcome:
context = context.append(Experience(
    episode_id="episode-1", timestep=0, observation="experiment request",
    action="test-a", outcome="test completed; hypothesis unsupported",
    reward=1.0, policy_version=0,
))
```

Descriptions must exactly cover the current candidate IDs. Numeric state,
entities, candidate descriptors, references, and legal masks are included in the
serialized policy input as well. Illegal actions remain masked by the policy
and the existing actor sampler. The actor owns all actual tool execution and
reward attribution; no external services are invoked by the learner.

## Memory, causality, and evaluation

`AgentContext` is immutable and actor-owned. Append returns a new object and
does not mutate previously submitted snapshots. Retain it across episode resets
when appropriate. Construct a new context at trial boundaries and separately
for each actor and split. Sharing observations between agents must be explicit;
there is no global conversation memory or hidden service session cache.

Only completed past events belong in the current policy input. The learner
rejects validation context in an update, actor/split mismatches in joint
trajectories, and same-episode current/future feedback. Actors must still enforce
causal ordering across episodes, point-in-time data access, and honest rewards;
the framework cannot infer whether arbitrary supplied text contains hindsight.

The same stored snapshot is used for behavior scoring and PPO updates. Context
is trimmed by removing whole oldest events until the token budget fits. Current
state and candidate descriptions are never silently truncated; oversized
current inputs are rejected. This context truncation is separate from episode
termination and RL return bootstrapping.

Validation may accumulate its own inference-time history while model weights
remain fixed. Validation feedback never enters the training trajectory queue.
Evaluation trials must not share memories. Use paired change schedules and
multiple seeds to compare a memory-free policy, a history-conditioned frozen
policy, simple adaptive baselines, and policies that update weights online.

## Checkpoints and portability

Checkpoints retain adapters, action/value heads, optimizer state, counters,
schema, and base-model identity. They do not duplicate frozen base weights or
store actor memory. Base-model restoration checks the pinned revision, adapter
settings, and serialization/token-budget contract. Local model directories are
fingerprinted, including tokenizer and weight files.

The existing structured inference bundle exporter works with this plugin and
omits optimizer state. Such a bundle still depends on the matching base model
and tokenizer: it is not a standalone language-model distribution. Hub IDs and
immutable revisions are the simplest portable configuration. If using a local
directory on another host, mount the same artifact at the configured path;
both path and fingerprint are currently checked. Pre-cache Hub files for
offline deployment. A numeric TorchScript export is not provided for this
text/tokenizer policy.

The example's CLI takes all deployment choices explicitly. No SSH hosts,
credentials, private repositories, or machine-specific filesystem paths are
required. Training on one compatible GPU and inference on another uses the
same checkpoint format.

## Scope and next steps

- This is PPO over candidate choices, not autoregressive token-level PPO/GRPO
  and not a reproduction of the paper's DQN/IA3 objective.
- It uses ordinary LoRA with a frozen floating-point base. QLoRA, sharding,
  multi-GPU training, KV-cache serving, and long-context optimization are not
  implemented. Start with a small checkpoint and short histories.
- Each candidate requires a language-model forward pass, plus a state-value
  pass. `llm_forward_batch_size` bounds each forward batch, but autograd retains
  activations across those passes during training. Limit candidate counts,
  history length, and PPO minibatch size to control memory.
- The backbone uses evaluation mode even during gradient-enabled updates to
  avoid dropout changing behavior probabilities. LoRA and head gradients are
  still computed. The critic can be prevented from changing shared features
  with `value_backbone_gradient_scale=0`.
- Learned joint-prefix interactions and transactional full-model ratio-guard
  snapshots are explicitly rejected by this prototype.
- Next research work is a benchmark with held-out environment families and
  change schedules, followed by history-conditioned DQN or generated-tool-call
  training if the use case requires them. Merely adding history does not prove
  that the model has learned a useful adaptation algorithm.

References: [ICRL paper](https://arxiv.org/abs/2501.14176),
[PEFT](https://huggingface.co/docs/peft/index), and
[TRL](https://huggingface.co/docs/trl/index).

## Controlled adaptation experiment

The next-stage [adaptation benchmark](adaptation_benchmark.md) compares frozen
language policies with numeric PPO, a rolling estimator, and history ablations
on paired held-out switching-tool scenarios. All task logic remains in the
example. `LanguageCandidatePolicy.context_diagnostics(observation)` reports
requested/retained history events and maximum sequence tokens without changing
model weights or memory.

## Optional numeric actor branch

Set `llm_numeric_branch: true` to combine language scores with a small structured
numeric actor. Each legal candidate's logit is the sum of the two learned scores.
The numeric branch consumes the existing global, entity, type, candidate and
candidate/entity-link tensors directly, preserving numbers without tokenization.
It uses the generic entity/candidate transformer; no task-specific summaries or
finance logic are built into the runtime.

`llm_numeric_model_dim` defaults to 32 and must be positive and divisible by four.
The branch has one transformer layer, four heads and zero dropout.
`llm_numeric_lr` defaults to 0.001, independently of the language branch's `lr`.
The language value head remains the critic; the numeric branch's unused value
head is frozen. Thus `value_backbone_gradient_scale` retains its existing meaning.
This is an actor feature-fusion option, not a new RL objective.

Checkpoints include the numeric branch and its architecture identity. Hybrid and
language-only checkpoints are deliberately incompatible. Existing language-only
checkpoints remain loadable with the feature disabled. Export bundles preserve
the branch and reconstruct both optimizer groups when a learner is restored.

For controlled inference diagnostics, the policy's `forward` accepts
`language_scale=0.0` or `numeric_scale=0.0` to ablate either contribution without
changing weights. Both default to one. Masks are reapplied after fusion, including
when both scales are zero. These ablations currently still compute both branches;
they measure behavior, not the latency of removing a branch from deployment.

See the [learning sanity tests](learning_diagnostics.md) and the portable
`examples/hybrid_adaptation.py` experiment. The experiment trains the hybrid by
supervised imitation of a causal rolling rule, with PPO available for subsequent
reward training. It should not be described as evidence that PPO itself has
solved the task or that the language branch supplies the numeric capability.

A fresh hybrid learner can also import an existing numeric actor with
`hybrid.initialize_numeric_actor_from_state(numeric_agent.state_dict())`.
Compatible structured DQN (`q`) or numeric policy (`policy`) weights are accepted;
feature schema, parameter names and tensor shapes must match. This initializes
only the numeric branch, leaving language parameters unchanged. It refuses an
already updated learner and does not import the source optimizer or counters.

`llm_numeric_logit_scale` is a finite positive multiplier (default one) applied
to numeric logits before fusion. It is useful when transferring reward-valued
Q scores into a categorical actor, where their ordering is useful but their
scale differs from policy logits. Non-default scales are part of checkpoint
identity. A scale of ten is used in the DQN-transfer example; it is an experiment
setting, not a universal calibration. The source DQN remains a valid standalone
policy when language inputs are unnecessary.

## Language Double-DQN

The optional `llm_dqn` plugin uses the same language candidate representation with
native transition replay, Double-DQN targets, and adapter-only checkpoints. See
[causal feedback adaptation](context_adaptation.md) for configuration, provenance
contracts, independent controls, and measured limits. Numeric branches are not
supported by this plugin.
