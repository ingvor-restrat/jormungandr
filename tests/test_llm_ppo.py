"""Offline integration tests: real Transformers/PEFT, tiny random local weights."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaModel, PreTrainedTokenizerFast

from jormungandr.algorithms import algorithm_registry
from jormungandr.algorithms.structured_ppo import StructuredTransition
from jormungandr.llm_context import AgentContext, Experience
from jormungandr.structured import (
    EntityCandidateObservation,
    StructuredPolicySpec,
    entity_candidate_observation_to_payload,
)
from jormungandr.structured_export import (
    export_structured_policy_bundle,
    load_structured_policy_bundle,
)
from jormungandr.service import JormungandrRuntime


@pytest.fixture(scope="module")
def local_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-language-backbone")
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[UNK]",
                "[PAD]",
                "[EOS]",
                "left",
                "right",
                "wait",
                "good",
                "bad",
                "state",
                "action",
                "reward",
                "history",
                "observation",
                "outcome",
                "0",
                "1",
                "-",
                "2",
            ]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    ).save_pretrained(path)
    torch.manual_seed(42)
    LlamaModel(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            pad_token_id=1,
        )
    ).save_pretrained(path)
    return str(path)


def config(model):
    return {
        "llm_model": model,
        "llm_max_tokens": 512,
        "llm_lora_rank": 2,
        "llm_forward_batch_size": 2,
        "epochs": 1,
        "minibatch_size": 2,
        "lr": 0.01,
        "value_backbone_gradient_scale": 0.0,
    }


def observation(context=None, legal=(True, True)):
    context = context or AgentContext("actor", "trial", "train")
    return EntityCandidateObservation(
        global_features=np.array([0.0]),
        entity_features=np.zeros((1, 1)),
        entity_type_ids=np.array([0]),
        entity_ids=("state",),
        candidate_features=np.array([[0.0], [1.0]]),
        candidate_ids=("left", "right"),
        legal_action_mask=np.array(legal),
        metadata=context.metadata("state", {"left": "left", "right": "right"}),
    )


def build(model):
    return algorithm_registry.get("llm_ppo").build_structured(
        StructuredPolicySpec(1, 1, 1, 1), config(model), "cpu"
    )


def test_adapter_update_frozen_base_masks_and_roundtrip(local_model, tmp_path):
    agent = build(local_model)
    obs = observation()
    before = {name: p.detach().clone() for name, p in agent.policy.named_parameters()}
    scores = agent.score_results_structured((obs,))[0]
    old_logp = torch.log_softmax(torch.tensor(scores.candidate_logits), dim=-1)
    # Both actions are collected under the same behavior distribution; opposite
    # rewards exercise the policy-gradient path, not merely the fresh critic.
    trajectories = [
        (
            [
                StructuredTransition(
                    str(i),
                    0,
                    obs,
                    action,
                    i,
                    float(old_logp[i]),
                    scores.value,
                    1.0 if i == 0 else -1.0,
                    True,
                )
            ]
        )
        for i, action in enumerate(obs.candidate_ids)
    ]
    result = agent.update_structured(trajectories)
    assert np.isfinite(result.loss)
    assert any(
        not torch.equal(before[n], p)
        for n, p in agent.policy.named_parameters()
        if "lora_" in n
    )
    assert all(
        torch.equal(before[n], p)
        for n, p in agent.policy.named_parameters()
        if not p.requires_grad
    )
    assert (
        agent.action_result_structured(
            observation(legal=(False, True)), deterministic=False
        ).candidate_id
        == "right"
    )
    agent.policy.train()
    training_logits = agent.policy(agent._collate_observations([obs])).logits.detach()
    agent.policy.eval()
    assert torch.allclose(
        training_logits, agent.policy(agent._collate_observations([obs])).logits
    )
    bundle = export_structured_policy_bundle(
        tmp_path / "bundle",
        agent=agent,
        spec=agent.spec,
        algorithm="llm_ppo",
        agent_config=config(local_model),
    )
    restored, _ = load_structured_policy_bundle(bundle["output_dir"])
    assert np.allclose(
        agent.score_results_structured([obs])[0].candidate_logits,
        restored.score_results_structured([obs])[0].candidate_logits,
    )
    state = agent.state_dict()
    assert "policy" not in state and all("lora_" in name for name in state["adapter"])
    invalid = deepcopy(state)
    invalid["llm_identity"]["rank"] = 99
    with pytest.raises(ValueError, match="does not match"):
        restored.load_state_dict(invalid)


def test_context_changes_scores_without_changing_weights_and_rejects_leakage(
    local_model,
):
    agent = build(local_model)
    before = {n: p.detach().clone() for n, p in agent.policy.named_parameters()}
    context = AgentContext("actor", "trial", "validation")
    first = observation(context)
    context = context.append(Experience("previous", 0, "state", "left", "good", 1.0, 0))
    second = observation(context)
    a, b = agent.score_results_structured([first, second])
    assert a.candidate_logits != b.candidate_logits
    assert all(torch.equal(before[n], p) for n, p in agent.policy.named_parameters())
    step = StructuredTransition("episode", 0, second, "left", 0, -0.7, 0.0, 1.0, True)
    with pytest.raises(ValueError, match="training context"):
        agent.update_structured([[step]])
    bad = AgentContext("actor", "trial", "train").append(
        Experience("episode", 0, "state", "left", "good", 1.0, 0)
    )
    with pytest.raises(ValueError, match="future"):
        agent.update_structured([[replace(step, observation=observation(bad))]])


def test_missing_text_oversize_and_checkpoint_base_validation(local_model):
    agent = build(local_model)
    with pytest.raises(ValueError, match="metadata"):
        agent.score_results_structured([replace(observation(), metadata={})])
    obs = observation()
    obs.metadata["llm"]["observation"] = "state " * 2000
    with pytest.raises(ValueError, match="exceed"):
        agent.score_results_structured([obs])
    with pytest.raises(ValueError, match="immutable"):
        algorithm_registry.get("llm_ppo").build_structured(
            agent.spec, {"llm_model": "unversioned/model"}, "cpu"
        )


def test_existing_runtime_checkpoint_and_frozen_inference(local_model, tmp_path):
    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path))
    learner = {
        "enabled": False,
        "algo": "llm_ppo",
        "device": "cpu",
        "plugin_config": config(local_model),
    }
    representation = {
        "global_dim": 1,
        "entity_dim": 1,
        "candidate_dim": 1,
        "entity_type_count": 1,
    }
    try:
        runtime.create_structured_model(
            model_id="source", representation=representation, learner=learner
        )
        payload = entity_candidate_observation_to_payload(
            observation(AgentContext("a", "validation-trial", "validation"))
        )
        initial = runtime.structured_policy_score("source", observations=[payload])
        checkpoint = runtime.force_structured_checkpoint("source")["checkpoint"]
        runtime.create_structured_model(
            model_id="restored",
            representation=representation,
            learner=learner,
            checkpoint_path=checkpoint,
        )
        restored = runtime.structured_policy_score("restored", observations=[payload])
        assert (
            initial["items"][0]["candidate_logits"]
            == restored["items"][0]["candidate_logits"]
        )
        with pytest.raises(ValueError, match="frozen"):
            runtime.structured_experience_add("restored", [{}])
    finally:
        runtime.close_all()


def test_multi_actor_runtime_training_and_validation_smoke(local_model, tmp_path):
    from argparse import Namespace
    from examples.llm_agent_prototype import run

    result = run(
        Namespace(
            model=local_model,
            revision=None,
            device="cpu",
            dtype="float32",
            output=str(tmp_path / "smoke"),
            seed=19,
            actors=2,
            rounds=1,
            episodes=2,
            max_tokens=512,
            learning_rate=0.001,
            update_timeout=30,
        )
    )
    assert result["updates"] == 1
    assert result["checkpoint_reload_verified"]
    assert result["runtime_status"]["experience"]["train_items"] == 4
    assert result["runtime_status"]["experience"]["validation_items"] == 4
    assert result["runtime_status"]["trajectories"]["trained_steps"] == 4


def test_active_learner_isolates_validation_and_rejects_misattributed_history(
    local_model, tmp_path
):
    import time
    from examples.llm_agent_prototype import collect

    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path))
    try:
        runtime.create_structured_model(
            model_id="active",
            representation=dict(
                global_dim=1, entity_dim=1, candidate_dim=1, entity_type_count=1
            ),
            learner={
                "enabled": True,
                "algo": "llm_ppo",
                "device": "cpu",
                "min_trajectory_steps": 2,
                "tick_interval_s": 0.01,
                "checkpoint_every": 0,
                "plugin_config": config(local_model),
            },
            tensorboard_enabled=False,
        )
        trajectories, _ = collect(
            runtime,
            "active",
            [AgentContext("v", "held-out", "validation")],
            episodes=2,
            episode_offset=0,
            rng=np.random.default_rng(11),
        )
        broken = deepcopy(trajectories)
        broken[0][0]["observation"]["metadata"]["llm"]["actor_id"] = "another-agent"
        with pytest.raises(ValueError, match="provenance"):
            runtime.structured_trajectory_add("active", broken)
        before = runtime.structured_policy_score(
            "active",
            observations=[entity_candidate_observation_to_payload(observation())],
        )
        result = runtime.structured_trajectory_add("active", trajectories)
        assert result["steps_by_split"] == {"train": 0, "validation": 2}
        assert result["trajectory_store"]["train_steps"] == 0
        time.sleep(0.05)
        after = runtime.structured_policy_score(
            "active",
            observations=[entity_candidate_observation_to_payload(observation())],
        )
        assert after["updates"] == before["updates"] == 0
        assert before["items"] == after["items"]
    finally:
        runtime.close_all()


def test_adaptation_benchmark_with_optional_language_policy(local_model, tmp_path):
    from argparse import Namespace
    from examples.adaptation_benchmark import run

    result = run(
        Namespace(
            output=str(tmp_path / "adaptation"),
            seeds="13",
            rounds=1,
            train_trials=1,
            eval_trials=2,
            steps=16,
            history=2,
            cpu_threads=1,
            no_llm=False,
            model=local_model,
            revision=None,
            device="cpu",
            dtype="float32",
            max_tokens=1024,
            learning_rate=0.0001,
        )
    )
    paired = result["results"]["paired_llm_reward_differences"]
    assert set(paired) == {
        "rolling",
        "numeric_trained",
        "llm_untrained_history",
        "llm_trained_no_history",
    }
    assert len(paired["rolling"]["seed_differences"]) == 1


def test_language_context_diagnostics_report_truncation(local_model):
    agent = build(local_model)
    context = AgentContext("actor", "trial", "validation", max_events=16)
    for i in range(16):
        context = context.append(
            Experience(str(i), 0, "state " * 20, "left", "good", 1.0, 0)
        )
    diagnostic = agent.policy.context_diagnostics(observation(context))
    assert diagnostic["requested_events"] == 16
    assert diagnostic["retained_events"] < 16
    assert diagnostic["max_sequence_tokens"] <= 512
    assert len(context.events) == 16
