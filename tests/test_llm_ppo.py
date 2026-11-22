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


@pytest.mark.parametrize("hybrid", [False, True])
def test_adapter_update_frozen_base_masks_and_roundtrip(local_model, tmp_path, hybrid):
    agent_config = dict(config(local_model), llm_numeric_branch=hybrid)
    agent = algorithm_registry.get("llm_ppo").build_structured(
        StructuredPolicySpec(1, 1, 1, 1), agent_config, "cpu"
    )
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
        agent_config=agent_config,
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


def test_representation_benchmark_pairs_initialization_and_freezes_evaluation(
    local_model, tmp_path
):
    from examples.representation_benchmark import parser, run

    args = parser().parse_args(
        [
            "--output",
            str(tmp_path / "representations"),
            "--model",
            local_model,
            "--seeds",
            "13",
            "--rounds",
            "1",
            "--train-trials",
            "1",
            "--eval-trials",
            "2",
            "--steps",
            "16",
            "--history",
            "2",
        ]
    )
    args.revision = None
    args.max_tokens = 1024
    results = run(args)
    assert len({r["initial_digest"] for r in results}) == 1
    assert {r["representation"] for r in results} == {"raw", "summary", "both"}
    for result in results:
        assert result["final_digest"] != result["initial_digest"]
        assert len(result["learning"]) == 1
        evaluation = result["evaluation"]
        assert [t["trial_id"] for t in evaluation["heldout"]] == [
            t["trial_id"] for t in evaluation["heldout_no_history"]
        ]
        assert not {t["trial_id"] for t in evaluation["fit_after"]} & {
            t["trial_id"] for t in evaluation["heldout"]
        }
    assert (tmp_path / "representations" / "summary.json").exists()


def test_language_scorer_can_fit_explicit_candidate_cues(local_model):
    from examples.learning_diagnostic import teach, imitation_metrics

    torch.manual_seed(7)
    agent = build(local_model)
    data = []
    for action, word in enumerate(("left", "right")):
        obs = observation()
        meta = dict(obs.metadata)
        meta["llm"] = dict(
            meta["llm"],
            candidate_descriptions={
                key: "good" if key == word else "bad" for key in ("left", "right")
            },
        )
        data.append((replace(obs, metadata=meta), np.eye(2)[action]))
    before = imitation_metrics(agent, data)
    teach(agent, data, 64, 7)
    after = imitation_metrics(agent, data)
    assert after["teacher_agreement"] == 1.0
    assert after["cross_entropy"] < before["cross_entropy"] * 0.5
    invalid = [
        (replace(data[0][0], metadata={"llm": {"split": "validation"}}), data[0][1])
    ]
    with pytest.raises(ValueError, match="training context"):
        teach(agent, invalid, 1, 7)


def test_hybrid_actor_gradients_components_masks_and_checkpoint(local_model, tmp_path):
    from examples.learning_diagnostic import teach

    spec = StructuredPolicySpec(1, 1, 1, 1)
    hybrid_config = dict(
        config(local_model), llm_numeric_branch=True, llm_numeric_lr=0.003
    )
    agent = algorithm_registry.get("llm_ppo").build_structured(
        spec, hybrid_config, "cpu"
    )
    observations = [
        observation(),
        replace(observation(), candidate_features=np.array([[1.0], [0.0]])),
    ]
    data = [
        (observations[0], np.array([0.0, 1.0])),
        (observations[1], np.array([1.0, 0.0])),
    ]
    branch = agent.policy.numeric_branch
    before = {k: p.detach().clone() for k, p in branch.named_parameters()}
    teach(agent, data, 8, 7)
    assert any(
        not torch.equal(before[k], p)
        for k, p in branch.named_parameters()
        if p.requires_grad
    )
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad)
        for p in branch.parameters()
        if p.requires_grad
    )
    assert all(
        torch.equal(before[k], p)
        for k, p in branch.named_parameters()
        if k.startswith("value_head")
    )
    batch = agent._collate_observations(observations)
    agent.policy.eval()
    with torch.no_grad():
        full = agent.policy(batch).logits
        language = agent.policy(batch, numeric_scale=0).logits
        numeric = agent.policy(batch, language_scale=0).logits
    torch.testing.assert_close(full, language + numeric)
    masked = agent._collate_observations([observation(legal=(True, False))])
    for language_scale, numeric_scale in ((1, 1), (0, 1), (1, 0), (0, 0)):
        assert (
            agent.policy(
                masked, language_scale=language_scale, numeric_scale=numeric_scale
            ).logits[0, 1]
            < -1e8
        )
    export_structured_policy_bundle(
        tmp_path / "hybrid",
        agent=agent,
        spec=spec,
        algorithm="llm_ppo",
        agent_config=hybrid_config,
    )
    restored, _ = load_structured_policy_bundle(tmp_path / "hybrid", device="cpu")
    assert restored.score_results_structured(
        observations
    ) == agent.score_results_structured(observations)
    assert len(restored.optimizer.param_groups) == 2
    with pytest.raises(ValueError, match="does not match"):
        build(local_model).load_state_dict(agent.state_dict())


def test_numeric_warmup_leaves_language_parameters_unchanged(local_model):
    from examples.hybrid_adaptation import warmup_numeric

    cfg = dict(config(local_model), llm_numeric_branch=True)
    agent = algorithm_registry.get("llm_ppo").build_structured(
        StructuredPolicySpec(1, 1, 1, 1), cfg, "cpu"
    )
    before = {name: p.detach().clone() for name, p in agent.policy.named_parameters()}
    warmup_numeric(agent, [(observation(), np.array([0.0, 1.0]))], 3, 7)
    assert any(
        not torch.equal(before[name], p)
        for name, p in agent.policy.named_parameters()
        if name.startswith("numeric_branch")
    )
    assert all(
        torch.equal(before[name], p)
        for name, p in agent.policy.named_parameters()
        if not name.startswith("numeric_branch")
    )
    restored = algorithm_registry.get("llm_ppo").build_structured(
        agent.spec, cfg, "cpu"
    )
    restored.load_state_dict(deepcopy(agent.state_dict()))
    for learner in (agent, restored):
        warmup_numeric(learner, [(observation(), np.array([1.0, 0.0]))], 1, 11)
    assert agent.score_results_structured(
        [observation()]
    ) == restored.score_results_structured([observation()])


def test_legacy_language_checkpoint_without_numeric_field_loads(local_model):
    agent = build(local_model)
    state = agent.state_dict()
    state.pop("numeric_branch")
    restored = build(local_model)
    restored.load_state_dict(state)
    assert agent.score_results_structured(
        [observation()]
    ) == restored.score_results_structured([observation()])


def test_hybrid_imports_only_compatible_fresh_numeric_actor(local_model):
    spec = StructuredPolicySpec(1, 1, 1, 1)
    cfg = dict(
        config(local_model), llm_numeric_branch=True, llm_numeric_logit_scale=10.0
    )
    agent = algorithm_registry.get("llm_ppo").build_structured(spec, cfg, "cpu")
    numeric = algorithm_registry.get("structured_dqn").build_structured(
        spec,
        dict(structured_model_dim=32, structured_heads=4, structured_layers=1),
        "cpu",
    )
    language_before = {
        n: p.detach().clone()
        for n, p in agent.policy.named_parameters()
        if not n.startswith("numeric_branch")
    }
    agent.initialize_numeric_actor_from_state(numeric.state_dict())
    assert all(
        torch.equal(language_before[n], p)
        for n, p in agent.policy.named_parameters()
        if n in language_before
    )
    batch = agent._collate_observations([observation()])
    with torch.no_grad():
        hybrid_scores = agent.policy(batch, language_scale=0).logits
        numeric_scores = numeric.q(batch.numeric).logits
    torch.testing.assert_close(hybrid_scores, 10.0 * numeric_scores)
    bad = deepcopy(numeric.state_dict())
    bad["spec"]["entity_dim"] = 2
    with pytest.raises(ValueError, match="schema"):
        agent.initialize_numeric_actor_from_state(bad)
    bad = deepcopy(numeric.state_dict())
    bad["q"].pop(next(iter(bad["q"])))
    with pytest.raises(ValueError, match="architecture"):
        agent.initialize_numeric_actor_from_state(bad)
    agent.update_steps = 1
    with pytest.raises(ValueError, match="fresh"):
        agent.initialize_numeric_actor_from_state(numeric.state_dict())


def test_hybrid_dqn_transfer_experiment_runs_end_to_end(local_model, tmp_path):
    import json
    from examples.hybrid_adaptation import parser, run

    args = parser().parse_args(
        [
            "--output",
            str(tmp_path / "hybrid-run"),
            "--model",
            local_model,
            "--teacher",
            "dqn",
            "--seeds",
            "13",
            "--rounds",
            "1",
            "--train-trials",
            "1",
            "--eval-trials",
            "2",
            "--steps",
            "16",
            "--max-tokens",
            "1024",
        ]
    )
    args.revision = None
    records = run(args)
    assert len(records) == 1
    assert len(records[0]["teacher_losses"]) == 16
    assert "dqn_teacher" in records[0]["evaluation"]
    assert records[0]["initial_digest"] != records[0]["final_digest"]
    summary = json.loads((tmp_path / "hybrid-run" / "summary.json").read_text())
    assert isinstance(summary["aggregate_target_met"], bool)
    assert set(summary["hybrid_minus_rolling"]) == {
        "mean_reward",
        "mean_expected_regret",
        "optimal_action_fraction",
    }


def test_language_tool_composition_runs_with_offline_backbone(local_model, tmp_path):
    from examples.language_tool_benchmark import parser, run

    args = parser().parse_args(
        [
            "--output",
            str(tmp_path / "language-tools"),
            "--model",
            local_model,
            "--seeds",
            "41",
            "--rounds",
            "1",
            "--train-trials",
            "1",
            "--eval-trials",
            "2",
            "--steps",
            "16",
            "--language-examples",
            "4",
            "--language-epochs",
            "1",
            "--max-tokens",
            "1024",
        ]
    )
    args.revision = None
    result = run(args)[0]
    assert len(result["numeric_losses"]) == 16
    assert len(result["language_losses"]) == 1
    assert set(result["evaluation"]) == {
        "random",
        "numeric_only",
        "language_only",
        "combined",
        "language_rolling",
        "oracle_numeric",
        "oracle_rolling",
        "combined_no_history",
    }
    assert all(
        t["compatibility"] == 1.0 for t in result["evaluation"]["oracle_numeric"]
    )
    assert all(
        t["compatibility"] == 1.0 for t in result["evaluation"]["oracle_rolling"]
    )


def test_independent_suitability_training_export_and_frozen_scoring(
    local_model, tmp_path
):
    from examples.adaptation_benchmark import SPEC, trainable_digest
    from examples.tool_suitability_benchmark import (
        pair_observation,
        score_pairs,
        train_suitability,
    )

    config = dict(
        llm_model=local_model,
        llm_max_tokens=768,
        llm_forward_batch_size=3,
        llm_lora_rank=2,
        llm_lora_alpha=4,
        lr=0.0001,
    )
    agent = algorithm_registry.get("llm_ppo").build_structured(SPEC, config, "cpu")
    pairs = [("translate", "translates"), ("summarize", "translates")]
    data = [
        (pair_observation(*p, split="train"), np.array(y))
        for p, y in zip(pairs, ([0.0, 1.0], [1.0, 0.0]))
    ]
    initial_digest = trainable_digest(agent)
    losses = train_suitability(agent, data, 1, 11, batch_size=2)
    assert np.isfinite(losses).all()
    digest, updates = trainable_digest(agent), agent.update_steps
    assert digest != initial_digest and updates == 1
    scores, probabilities = score_pairs(agent, pairs)
    export_structured_policy_bundle(
        tmp_path / "suitability",
        agent=agent,
        spec=SPEC,
        algorithm="llm_ppo",
        agent_config=config,
    )
    restored, _ = load_structured_policy_bundle(tmp_path / "suitability")
    restored_scores, restored_probabilities = score_pairs(restored, pairs)
    np.testing.assert_allclose(scores, restored_scores, atol=1e-6)
    np.testing.assert_allclose(probabilities, restored_probabilities, atol=1e-6)
    assert trainable_digest(agent) == digest and agent.update_steps == updates


def test_suitability_experiment_runs_offline(local_model, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from examples import tool_suitability_benchmark as benchmark
    from examples.adaptation_benchmark import SPEC

    numeric_config = dict(
        structured_model_dim=16, structured_heads=2, structured_layers=1
    )
    numeric = algorithm_registry.get("structured_dqn").build_structured(
        SPEC, numeric_config, "cpu"
    )
    export_structured_policy_bundle(
        tmp_path / "numeric-41",
        agent=numeric,
        spec=SPEC,
        algorithm="structured_dqn",
        agent_config=numeric_config,
    )
    data = benchmark.training_data()[:2]
    monkeypatch.setattr(benchmark, "training_data", lambda: data)
    result = benchmark.run(
        SimpleNamespace(
            output=str(tmp_path / "run"),
            numeric_root=str(tmp_path),
            seeds="41",
            epochs=1,
            learning_rate=0.0001,
            model=local_model,
            revision=None,
            device="cpu",
            eval_trials=1,
        )
    )[0]
    assert result["semantics"]["cases"] == 36
    assert len(result["semantics"]["catalogs"]) == 5
    assert (
        result["evaluation"]["parser"][0]["mean_reward"]
        == result["evaluation"]["oracle_numeric"][0]["mean_reward"]
    )
    assert (tmp_path / "run" / "41" / "language" / "manifest.json").is_file()
    assert (tmp_path / "run" / "41" / "semantic-traces.json").is_file()


def test_matched_gate_training_uses_same_initial_weights_and_update_budget(local_model):
    from examples.adaptation_benchmark import SPEC, trainable_digest
    from examples.tool_gate_comparison import train, score, training_cases, metrics

    cases = training_cases(11, "narrow", 2)
    initial = []
    for representation in ("ranking", "binary"):
        torch.manual_seed(11)
        agent = algorithm_registry.get("llm_ppo").build_structured(
            SPEC,
            dict(
                llm_model=local_model,
                llm_max_tokens=1024,
                llm_forward_batch_size=24,
                llm_lora_rank=2,
                lr=0.0001,
            ),
            "cpu",
        )
        initial.append(trainable_digest(agent))
        losses = train(agent, cases, representation)
        assert len(losses) == agent.update_steps == 1
        assert np.isfinite(losses).all()
        assert trainable_digest(agent) != initial[-1]
        digest = trainable_digest(agent)
        scores = score(agent, cases, representation)
        assert scores.shape == (2, 4)
        assert metrics(cases, scores, representation)["primary_cases"] == 2
        assert trainable_digest(agent) == digest
    assert initial[0] == initial[1]


def test_language_dqn_bootstrap_target_and_adapter_checkpoint(local_model, tmp_path):
    from examples.adaptation_benchmark import SPEC
    from examples.context_adaptation_benchmark import collect

    cfg = dict(
        llm_model=local_model,
        llm_max_tokens=1024,
        llm_forward_batch_size=6,
        llm_lora_rank=2,
        lr=0.0001,
        gamma=0.9,
        llm_target_tau=0.25,
        demonstration_weight=0.0,
    )
    agent = algorithm_registry.get("llm_dqn").build_structured(SPEC, cfg, "cpu")
    replay = collect(31, 4, 8, 1)
    step = replace(replay[0], reward=1.0)
    with torch.no_grad():
        agent.q.action_head.weight.zero_()
        agent.q.action_head.bias.zero_()
        agent.q_target.action_head.weight.zero_()
        agent.q_target.action_head.bias.fill_(2.0)
    result = agent.update_structured([step])
    assert result.metrics["target_mean"] == pytest.approx(2.8)
    torch.testing.assert_close(
        agent.q_target.action_head.bias,
        0.75 * torch.full_like(agent.q.action_head.bias, 2.0)
        + 0.25 * agent.q.action_head.bias,
    )
    assert all(not p.requires_grad for p in agent.q_target.parameters())
    with torch.no_grad():
        agent.q_target.action_head.weight.zero_()
        agent.q_target.action_head.bias.fill_(100.0)
    terminal = agent.update_structured([replace(step, done=True)])
    assert terminal.metrics["target_mean"] == pytest.approx(1.0)
    state = agent.state_dict()
    assert all("lora_" in k for k in state["q"]["adapter"])
    assert "backbone" not in state["q"]
    export_structured_policy_bundle(
        tmp_path / "dqn", agent=agent, spec=SPEC, algorithm="llm_dqn", agent_config=cfg
    )
    restored, _ = load_structured_policy_bundle(tmp_path / "dqn")
    a = agent.action_result_structured(step.observation, deterministic=True)
    b = restored.action_result_structured(step.observation, deterministic=True)
    assert a.candidate_values == b.candidate_values
    torch.testing.assert_close(
        agent.q_target.action_head.bias, restored.q_target.action_head.bias
    )
    assert restored.update_steps == 2


def test_language_dqn_public_text_cannot_reveal_hidden_winner(local_model):
    from examples.adaptation_benchmark import SPEC
    from examples.context_adaptation_benchmark import (
        make_trial,
        context_for,
        observation,
    )

    agent = algorithm_registry.get("llm_dqn").build_structured(
        SPEC, dict(llm_model=local_model, llm_max_tokens=1024, llm_lora_rank=2), "cpu"
    )
    worlds = [make_trial(31, 2, i, 8) for i in (0, 1)]
    texts = [agent.q._texts(observation(w, 0, context_for(w, 1))) for w in worlds]
    assert texts[0] == texts[1]


def test_language_dqn_uses_existing_frozen_runtime(local_model, tmp_path):
    from examples.context_adaptation_benchmark import (
        make_trial,
        context_for,
        observation as context_observation,
    )

    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path))
    learner = dict(
        enabled=False,
        algo="llm_dqn",
        device="cpu",
        plugin_config=dict(llm_model=local_model, llm_max_tokens=1024, llm_lora_rank=2),
    )
    representation = dict(
        global_dim=1, entity_dim=3, candidate_dim=3, entity_type_count=1
    )
    world = make_trial(31, 2, 0, 8)
    payload = entity_candidate_observation_to_payload(
        context_observation(world, 0, context_for(world, 1))
    )
    try:
        runtime.create_structured_model(
            model_id="source", representation=representation, learner=learner
        )
        before = runtime.structured_policy_infer(
            "source", observations=[payload], deterministic=True, epsilon=0.0
        )
        checkpoint = runtime.force_structured_checkpoint("source")["checkpoint"]
        runtime.create_structured_model(
            model_id="restored",
            representation=representation,
            learner=learner,
            checkpoint_path=checkpoint,
        )
        after = runtime.structured_policy_infer(
            "restored", observations=[payload], deterministic=True, epsilon=0.0
        )
        assert (
            before["items"][0]["candidate_values"]
            == after["items"][0]["candidate_values"]
        )
        with pytest.raises(ValueError, match="frozen"):
            runtime.structured_experience_add("restored", [{}])
    finally:
        runtime.close_all()


def test_language_dqn_runtime_learns_replay(local_model, tmp_path):
    import time
    from examples.context_adaptation_benchmark import collect

    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path))
    learner = dict(
        enabled=True,
        algo="llm_dqn",
        device="cpu",
        batch_size=2,
        min_replay=2,
        max_updates=1,
        tick_interval_s=0.001,
        checkpoint_every=0,
        plugin_config=dict(
            llm_model=local_model, llm_max_tokens=1024, llm_lora_rank=2, gamma=0.9
        ),
    )
    try:
        runtime.create_structured_model(
            model_id="context-dqn",
            representation=dict(
                global_dim=1, entity_dim=3, candidate_dim=3, entity_type_count=1
            ),
            learner=learner,
        )
        rows = []
        for t in collect(31, 4, 8, 1)[:2]:
            rows.append(
                dict(
                    actor_id=t.actor_id,
                    episode_id=t.episode_id,
                    timestep=t.timestep,
                    policy_version=0,
                    split="train",
                    candidate_id=t.candidate_id,
                    reward=t.reward,
                    done=t.done,
                    observation=entity_candidate_observation_to_payload(t.observation),
                    next_observation=entity_candidate_observation_to_payload(
                        t.next_observation
                    ),
                )
            )
        runtime.structured_experience_add("context-dqn", rows)
        deadline = time.monotonic() + 10
        model = runtime.get_structured_model("context-dqn")
        while model["updates"] < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
            model = runtime.get_structured_model("context-dqn")
        assert model["updates"] == 1
        assert model["last_error"] == ""
        bad = {**rows[0], "observation": rows[0]["next_observation"]}
        with pytest.raises(ValueError, match="future"):
            runtime.structured_experience_add("context-dqn", [rows[0], bad])
        assert runtime.get_structured_model("context-dqn")["replay"]["size"] == 2
    finally:
        runtime.close_all()


@pytest.mark.parametrize("candidate_context", ["local", "evidence"])
def test_local_language_scores_ignore_other_candidates_and_survive_export(
    local_model, tmp_path, candidate_context
):
    from examples.multiservice_adaptation import Task, append, context, observe, world
    from examples.adaptation_benchmark import SPEC

    cfg = {
        **config(local_model),
        "llm_candidate_context": candidate_context,
        "llm_max_tokens": 768,
    }
    agent = algorithm_registry.get("llm_dqn").build_structured(SPEC, cfg, "cpu")
    w = world(31, Task(4, 1.0, 0.0, 0), 0)
    c = append(context(w), 0, w.ids[0], 1.0)
    obs = observe(w, 1, c)
    rev = observe(w, 1, c, reverse=True)
    result, reordered = agent.action_results_structured([obs, rev], deterministic=True)
    scores = dict(zip(obs.candidate_ids, result.candidate_values))
    other = dict(zip(rev.candidate_ids, reordered.candidate_values))
    assert scores == pytest.approx(other, abs=1e-6)
    texts = dict(zip(obs.candidate_ids, agent.q._texts(obs)[1:]))
    altered = replace(
        obs,
        metadata={
            "llm": {
                **obs.metadata["llm"],
                "candidate_descriptions": {
                    **obs.metadata["llm"]["candidate_descriptions"],
                    w.ids[1]: "unrelated changed evidence",
                },
            }
        },
    )
    altered_texts = dict(zip(altered.candidate_ids, agent.q._texts(altered)[1:]))
    assert texts[w.ids[0]] == altered_texts[w.ids[0]]
    assert texts[w.ids[1]] != altered_texts[w.ids[1]]
    assert agent.q.identity["serialization"].endswith(f"{candidate_context}.v1")
    export_structured_policy_bundle(
        tmp_path / "local",
        agent=agent,
        spec=SPEC,
        algorithm="llm_dqn",
        agent_config=cfg,
    )
    restored, _ = load_structured_policy_bundle(tmp_path / "local", device="cpu")
    actual = restored.action_results_structured([obs], deterministic=True)[0]
    assert actual.candidate_values == pytest.approx(result.candidate_values, abs=1e-6)
    joint = algorithm_registry.get("llm_dqn").build_structured(
        SPEC, {**cfg, "llm_candidate_context": "joint"}, "cpu"
    )
    with pytest.raises(ValueError, match="identity"):
        joint.load_state_dict(agent.state_dict())


def test_local_sdpa_and_cached_history_free_evaluation(local_model):
    from examples.multiservice_adaptation import Task, context, evaluate, observe, world
    from examples.adaptation_benchmark import SPEC

    cfg = {
        **config(local_model),
        "llm_candidate_context": "local",
        "llm_attention_implementation": "sdpa",
        "llm_max_tokens": 512,
    }
    agent = algorithm_registry.get("llm_dqn").build_structured(SPEC, cfg, "cpu")
    worlds = [world(31, Task(4, 1.0, 0.0, 0, 8), i) for i in range(4)]
    original = agent.action_results_structured
    calls = []

    def tracked(observations, **kwargs):
        calls.append(len(observations))
        return original(observations, **kwargs)

    agent.action_results_structured = tracked
    result = evaluate(agent, worlds, "no_history")
    assert calls == [4]
    assert result["weights_frozen"]
    first = observe(worlds[0], 0, context(worlds[0]), "no_history")
    later = observe(worlds[0], 7, context(worlds[0]), "no_history")
    by_id = lambda obs: dict(zip(obs.candidate_ids, agent.q._texts(obs)[1:]))
    assert by_id(first) == by_id(later)


@pytest.mark.parametrize("candidate_context", ["local", "evidence"])
def test_local_bandit_chosen_only_loss_matches_full_candidate_loss(
    local_model, candidate_context
):
    from examples.multiservice_adaptation import collect
    from examples.adaptation_benchmark import SPEC
    from jormungandr.algorithms.structured_dqn import StructuredDQNAgent

    cfg = {
        **config(local_model),
        "llm_candidate_context": candidate_context,
        "llm_max_tokens": 768,
        "gamma": 0.0,
        "demonstration_weight": 0.0,
        "structured_microbatch_size": 2,
    }
    full = algorithm_registry.get("llm_dqn").build_structured(SPEC, cfg, "cpu")
    reduced = algorithm_registry.get("llm_dqn").build_structured(SPEC, cfg, "cpu")
    reduced.load_state_dict(deepcopy(full.state_dict()))
    full.optimizer = torch.optim.SGD(full.q.parameters(), lr=0.01)
    reduced.optimizer = torch.optim.SGD(reduced.q.parameters(), lr=0.01)
    rows = collect(31, 1)[5:10]
    weights = np.array([0.3, 0.7, 1.0, 0.2, 0.9])
    a = StructuredDQNAgent.update_structured(full, rows, weights)
    b = reduced.update_structured(rows, weights)
    assert a.loss == pytest.approx(b.loss, abs=1e-6)
    np.testing.assert_allclose(a.priorities, b.priorities, atol=1e-6)
    assert reduced.update_steps == 1
    for x, y in zip(full.q.parameters(), reduced.q.parameters()):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
    for x, y in zip(full.q_target.parameters(), reduced.q_target.parameters()):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)


def test_language_reward_learner_estimates_stochastic_mean(local_model):
    from examples.multiservice_adaptation import collect
    from examples.adaptation_benchmark import SPEC

    cfg = {
        **config(local_model),
        "llm_candidate_context": "local",
        "gamma": 0.0,
        "demonstration_weight": 0.0,
        "structured_microbatch_size": 3,
        "lr": 0.003,
    }
    torch.manual_seed(73)
    agent = algorithm_registry.get("llm_dqn").build_structured(SPEC, cfg, "cpu")
    row = collect(31, 1)[0]
    # Identical inputs with stochastic outcomes: the target is a mean, not
    # the last outcome or a majority-class label.
    batch = [
        replace(
            row, reward=float(i % 3 == 0), next_observation=row.observation, done=True
        )
        for i in range(12)
    ]
    for _ in range(30):
        agent.update_structured(batch)
    result = agent.action_results_structured([row.observation], deterministic=True)[0]
    prediction = result.candidate_values[row.candidate_index]
    assert 0.23 < prediction < 0.43
    assert agent.update_steps == 30


def test_actor_evidence_excludes_routing_ids_but_keeps_causal_validation(local_model):
    from examples.multiservice_adaptation import collect
    from examples.multiservice_evidence import evidence_view
    from examples.adaptation_benchmark import SPEC

    cfg = {
        **config(local_model),
        "llm_candidate_context": "evidence",
        "llm_max_tokens": 1024,
        "gamma": 0.0,
    }
    agent = algorithm_registry.get("llm_dqn").build_structured(SPEC, cfg, "cpu")
    row = collect(31, 1)[5]
    obs = evidence_view(row.observation)
    texts = agent.q._texts(obs)
    assert all(key not in text for key in obs.candidate_ids for text in texts)
    assert all("timestep" not in text for text in texts)
    assert agent.q.context_diagnostics(obs)["actor_evidence"] is True
    raw = obs.metadata["llm"]
    bad = replace(
        obs,
        metadata={
            "llm": {**raw, "history": [{**raw["history"][0], "timestep": row.timestep}]}
        },
    )
    with pytest.raises(ValueError, match="future feedback"):
        agent.update_structured([replace(row, observation=bad)])
    agent.q.max_tokens = 8
    with pytest.raises(ValueError, match="actor evidence exceeds"):
        agent.q._texts(obs)


def test_evidence_inference_encodes_identical_public_inputs_once(local_model):
    from examples.multiservice_adaptation import Task, world, context, observe
    from examples.multiservice_evidence import evidence_view
    from examples.adaptation_benchmark import SPEC

    cfg = {
        **config(local_model),
        "llm_candidate_context": "evidence",
        "llm_max_tokens": 1024,
    }
    agent = algorithm_registry.get("llm_dqn").build_structured(SPEC, cfg, "cpu")
    w = world(19, Task(8, 1.0, 0.0, 0), 0)
    obs = evidence_view(observe(w, 0, context(w)))
    sizes = []
    hook = agent.q.backbone.register_forward_pre_hook(
        lambda module, args, kwargs: sizes.append(len(kwargs["input_ids"])),
        with_kwargs=True,
    )
    try:
        results = agent.action_results_structured([obs, obs], deterministic=True)
    finally:
        hook.remove()
    assert (
        sum(sizes) == len(set(agent.q._texts(obs))) == 2
    )  # State plus one common candidate view.
    assert len(results[0].candidate_values) == 8
    assert len(set(results[0].candidate_values)) == 1
    assert results[0].candidate_values == results[1].candidate_values
