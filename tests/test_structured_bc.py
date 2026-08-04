from __future__ import annotations

from dataclasses import fields, replace
import time

import numpy as np
import pytest
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.structured import EntityCandidateObservation, StructuredPolicySpec
from jormungandr.structured_supervision import StructuredSupervisionExample
from jormungandr.structured_supervision import structured_supervision_to_payload
from jormungandr.service import JormungandrRuntime


SPEC = StructuredPolicySpec(2, 3, 4, 2)
CONFIG = {
    "structured_model_dim": 16,
    "structured_heads": 4,
    "structured_layers": 1,
    "structured_feedforward_dim": 32,
    "structured_dropout": 0.0,
    "lr": 1e-2,
    "max_grad": 5.0,
}


def _observation(step: int, order=(0, 1, 2, 3)):
    candidate_ids = ("move:pass", "move:go", "market:pass", "market:buy")
    # Targets have the first candidate feature; the order changes independently
    # of semantic IDs so slot memorization cannot solve the corpus.
    features = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return EntityCandidateObservation(
        global_features=np.asarray([step / 10.0, 1.0], dtype=np.float32),
        entity_features=np.asarray(
            [[step, 0.0, 1.0], [0.0, 1.0, step]], dtype=np.float32
        ),
        entity_type_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=("worker", "inventory"),
        candidate_features=features[list(order)],
        candidate_ids=tuple(candidate_ids[index] for index in order),
        legal_action_mask=np.ones(4, dtype=np.bool_),
    )


def _corpus(split: str):
    examples = []
    orders = ((0, 1, 2, 3), (1, 0, 3, 2), (2, 0, 3, 1), (3, 2, 1, 0))
    for step, order in enumerate(orders):
        observation = _observation(step, order)
        examples.extend(
            (
                StructuredSupervisionExample(
                    actor_id="expert",
                    episode_id=f"episode:{step}",
                    timestep=step,
                    observation=observation,
                    factor_id="move",
                    candidate_ids=("move:pass", "move:go"),
                    target_candidate_id="move:go",
                    split=split,
                    source_group="arena_expert",
                    factor_group="movement",
                    target_group="advance",
                    sample_weight=2.0,
                ),
                StructuredSupervisionExample(
                    actor_id="expert",
                    episode_id=f"episode:{step}",
                    timestep=step,
                    observation=observation,
                    factor_id="market",
                    candidate_ids=("market:pass", "market:buy"),
                    target_candidate_id="market:buy",
                    split=split,
                    source_group="scripted_expert",
                    factor_group="market",
                    target_group="purchase",
                    sample_weight=1.0,
                ),
            )
        )
    return tuple(examples)


def test_structured_bc_memorizes_tiny_corpus_without_reward_and_reports_groups() -> None:
    torch.manual_seed(113)
    agent = algorithm_registry.get("structured_bc").build_structured(
        SPEC, CONFIG, "cpu"
    )
    train = _corpus("train")
    validation = _corpus("validation")

    result = None
    for _ in range(120):
        result = agent.update_structured_supervision(train)
        evaluated = agent.evaluate_structured_supervision(validation)
        if evaluated.accuracy == 1.0 and evaluated.nll < 0.05:
            break

    assert result is not None
    assert evaluated.accuracy == 1.0
    assert evaluated.nll < 0.05
    assert "group/source/arena_expert/accuracy" in evaluated.metrics
    assert "group/factor/market/nll" in evaluated.metrics
    assert "group/target/purchase/accuracy" in evaluated.metrics
    assert "weighted_accuracy" in evaluated.metrics
    assert "reward" not in {field.name for field in fields(StructuredSupervisionExample)}


def test_structured_bc_reports_raw_and_weighted_accuracy_separately() -> None:
    torch.manual_seed(119)
    agent = algorithm_registry.get("structured_bc").build_structured(
        SPEC, CONFIG, "cpu"
    )
    base = _corpus("validation")
    # The first semantic class carries much more optimization weight.  The
    # reported raw score must still be the ordinary count-based accuracy.
    weighted = tuple(
        replace(item, sample_weight=(17.0 if item.factor_group == "movement" else 1.0))
        for item in base
    )
    result = agent.evaluate_structured_supervision(weighted)
    scores = agent.score_results_structured(
        tuple(item.observation for item in weighted)
    )
    correct = np.asarray(
        [
            max(
                item.candidate_ids,
                key=dict(zip(score.candidate_ids, score.candidate_logits)).__getitem__,
            )
            == item.target_candidate_id
            for score, item in zip(scores, weighted, strict=True)
        ],
        dtype=np.float64,
    )
    weights = np.asarray([item.sample_weight for item in weighted])

    assert result.metrics["accuracy"] == pytest.approx(float(correct.mean()))
    assert result.metrics["weighted_accuracy"] == pytest.approx(
        float(np.average(correct, weights=weights))
    )


def test_candidate_shuffle_and_validation_evaluation_do_not_change_weights() -> None:
    torch.manual_seed(127)
    agent = algorithm_registry.get("structured_bc").build_structured(
        SPEC, CONFIG, "cpu"
    )
    for _ in range(80):
        agent.update_structured_supervision(_corpus("train"))
    before = [parameter.detach().clone() for parameter in agent.policy.parameters()]

    evaluation = agent.evaluate_structured_supervision(_corpus("validation"))
    canonical = _observation(7, (0, 1, 2, 3))
    shuffled = _observation(7, (3, 1, 0, 2))
    canonical_scores = agent.score_results_structured((canonical,))[0]
    shuffled_scores = agent.score_results_structured((shuffled,))[0]
    left = dict(zip(canonical_scores.candidate_ids, canonical_scores.candidate_logits))
    right = dict(zip(shuffled_scores.candidate_ids, shuffled_scores.candidate_logits))

    assert evaluation.accuracy == 1.0
    assert all(torch.equal(old, new) for old, new in zip(before, agent.policy.parameters()))
    assert all(abs(left[key] - right[key]) <= 1e-5 for key in left)


def test_structured_bc_checkpoint_initializes_ppo_without_schema_conversion(
    tmp_path,
) -> None:
    torch.manual_seed(131)
    bc = algorithm_registry.get("structured_bc").build_structured(
        SPEC, CONFIG, "cpu"
    )
    for _ in range(60):
        bc.update_structured_supervision(_corpus("train"))
    checkpoint = tmp_path / "bc.pt"
    torch.save({"agent": bc.state_dict()}, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ppo = algorithm_registry.get("structured_ppo").build_structured(
        SPEC, CONFIG, "cpu"
    )

    ppo.initialize_policy_from_state(payload["agent"])
    observation = _observation(3, (2, 1, 3, 0))
    bc_scores = bc.score_results_structured((observation,))[0]
    ppo_scores = ppo.score_results_structured((observation,))[0]

    assert ppo_scores.candidate_ids == bc_scores.candidate_ids
    assert np.allclose(ppo_scores.candidate_logits, bc_scores.candidate_logits)


def test_central_structured_bc_keeps_validation_isolated_and_initializes_ppo(
    tmp_path,
) -> None:
    runtime = JormungandrRuntime(
        checkpoint_root=str(tmp_path), tensorboard_root=str(tmp_path)
    )
    representation = {
        "global_dim": SPEC.global_dim,
        "entity_dim": SPEC.entity_dim,
        "candidate_dim": SPEC.candidate_dim,
        "entity_type_count": SPEC.entity_type_count,
    }
    learner = {
        "enabled": True,
        "algo": "structured_bc",
        "device": "cpu",
        "batch_size": 8,
        "min_replay": 8,
        "replay_ratio": 40.0,
        "tick_interval_s": 0.005,
        "checkpoint_every": 0,
        **CONFIG,
    }
    try:
        runtime.create_structured_model(
            representation=representation,
            learner=learner,
            model_id="bc-service",
            capacity=32,
            validation_capacity=16,
        )
        examples = tuple(
            structured_supervision_to_payload(item)
            for item in (*_corpus("train"), *_corpus("validation"))
        )
        added = runtime.structured_supervision_add("bc-service", examples)
        assert added["added_by_split"] == {"train": 8, "validation": 8}

        deadline = time.time() + 10.0
        model = runtime.get_structured_model("bc-service")
        while (
            model["last_metrics"].get("validation/accuracy", 0.0) < 1.0
            and time.time() < deadline
        ):
            if model["last_error"]:
                raise AssertionError(model["last_error"])
            time.sleep(0.01)
            model = runtime.get_structured_model("bc-service")
        assert model["last_metrics"]["validation/accuracy"] == 1.0
        assert model["supervision"]["train_size"] == 8
        assert model["supervision"]["validation_size"] == 8
        assert model["supervision"]["trained_items"] > 0
        assert "validation/group/factor/market/accuracy" in model["last_metrics"]

        probe = _observation(9, (3, 1, 0, 2))
        payload = probe
        from jormungandr.structured import entity_candidate_observation_to_payload

        wire = entity_candidate_observation_to_payload(payload)
        checkpoint = runtime.force_structured_checkpoint("bc-service")[
            "checkpoint"
        ]
        frozen_bc_learner = {
            **learner,
            "enabled": False,
        }
        runtime.create_structured_model(
            representation=representation,
            learner=frozen_bc_learner,
            model_id="bc-frozen",
            capacity=8,
            validation_capacity=8,
            checkpoint_path=checkpoint,
        )
        source = runtime.structured_policy_score(
            "bc-frozen", observations=(wire,)
        )["items"][0]
        ppo_learner = {
            "enabled": False,
            "algo": "structured_ppo",
            "device": "cpu",
            **CONFIG,
        }
        initialized = runtime.create_structured_model(
            representation=representation,
            learner=ppo_learner,
            model_id="ppo-from-bc",
            capacity=8,
            validation_capacity=8,
            policy_initialization_path=checkpoint,
        )
        restored = runtime.structured_policy_score(
            "ppo-from-bc", observations=(wire,)
        )["items"][0]

        assert initialized["policy_initialization_source"] == checkpoint
        assert initialized["updates"] == 0
        assert np.allclose(
            restored["candidate_logits"], source["candidate_logits"]
        )
    finally:
        runtime.close_all()
