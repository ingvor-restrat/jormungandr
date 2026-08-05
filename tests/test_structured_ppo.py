from dataclasses import replace

import numpy as np
import pytest
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.algorithms.structured_ppo import StructuredTransition
from jormungandr.service import JormungandrRuntime, JormungandrServiceError
from jormungandr.structured import (
    EntityCandidateObservation,
    EntityCandidatePolicyOutput,
    StructuredPolicySpec,
    collate_entity_candidate_observations,
)
from jormungandr.structured_trajectory import (
    StructuredActionFactor,
    StructuredFactorChoice,
    StructuredJointTrajectoryStep,
    sample_structured_joint_action,
)


def _observation(step: int, candidates: int) -> EntityCandidateObservation:
    return EntityCandidateObservation(
        global_features=np.asarray([step / 10.0, 1.0], dtype=np.float32),
        entity_features=np.asarray(
            [[step, 0.0, 1.0], [0.0, 1.0, step]], dtype=np.float32
        ),
        entity_type_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=(f"entity-a:{step}", f"entity-b:{step}"),
        candidate_features=np.arange(
            candidates * 4, dtype=np.float32
        ).reshape(candidates, 4),
        candidate_ids=tuple(
            f"episode-action:{step}:{index}" for index in range(candidates)
        ),
        legal_action_mask=np.ones(candidates, dtype=np.bool_),
    )


def test_structured_ppo_plugin_updates_variable_candidate_trajectories() -> None:
    torch.manual_seed(31)
    plugin = algorithm_registry.get("structured_ppo")
    assert plugin.build is None
    assert plugin.build_structured is not None
    agent = plugin.build_structured(
        StructuredPolicySpec(2, 3, 4, 2),
        {
            "structured_model_dim": 16,
            "structured_heads": 4,
            "structured_layers": 1,
            "structured_feedforward_dim": 32,
            "epochs": 2,
            "minibatch_size": 3,
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "lr": 1e-3,
        },
        "cpu",
    )
    trajectories = []
    for episode_index, terminal_reward in enumerate((1.0, -0.5)):
        trajectory = []
        for timestep, candidate_count in enumerate((2, 4, 3)):
            observation = _observation(timestep, candidate_count)
            selected = agent.action_result_structured(
                observation, deterministic=False
            )
            trajectory.append(
                StructuredTransition(
                    episode_id=f"episode-{episode_index}",
                    timestep=timestep,
                    observation=observation,
                    candidate_id=selected.candidate_id,
                    candidate_index=selected.candidate_index,
                    behavior_log_probability=selected.log_probability,
                    behavior_value=selected.value,
                    reward=terminal_reward if timestep == 2 else 0.0,
                    done=timestep == 2,
                )
            )
        trajectories.append(trajectory)
    before = [parameter.detach().clone() for parameter in agent.policy.parameters()]

    result = agent.update_structured(trajectories)

    assert result.transitions == 6
    assert result.episodes == 2
    assert result.minibatches == 4
    assert result.episode_return_mean == pytest.approx(0.25)
    assert result.episode_return_std == pytest.approx(0.75)
    assert result.episode_return_min == pytest.approx(-0.5)
    assert result.episode_return_max == pytest.approx(1.0)
    assert result.episode_return_unique_count == 2
    assert result.episode_length_mean == pytest.approx(3.0)
    assert result.episode_length_min == 3
    assert result.episode_length_max == 3
    assert result.reward_nonzero_fraction == pytest.approx(1.0 / 3.0)
    assert result.gae_decay == pytest.approx(1.0)
    assert result.gae_oldest_delta_weight_mean == pytest.approx(1.0)
    assert np.isfinite(result.loss)
    assert np.isfinite(result.approximate_kl)
    assert np.isfinite(result.importance_ratio_mean)
    assert np.isfinite(result.importance_ratio_std)
    assert 0.0 < result.importance_ratio_min <= result.importance_ratio_max
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.policy.parameters())
    )


def test_vector_service_rejects_structured_only_algorithm_clearly() -> None:
    with pytest.raises(JormungandrServiceError, match="in-process structured"):
        JormungandrRuntime._parse_learner_config(
            {"enabled": True, "algo": "structured_ppo"}
        )


def test_structured_ppo_updates_exact_joint_factor_log_probabilities() -> None:
    torch.manual_seed(37)
    agent = algorithm_registry.get("structured_ppo").build_structured(
        StructuredPolicySpec(2, 3, 4, 2),
        {
            "structured_model_dim": 16,
            "structured_heads": 4,
            "structured_layers": 1,
            "structured_feedforward_dim": 32,
            "epochs": 2,
            "minibatch_size": 2,
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "lr": 1e-3,
        },
        "cpu",
    )
    observations = (_observation(0, 4), _observation(1, 4), _observation(2, 4))
    factors = (
        StructuredActionFactor(
            "worker:0", observations[0].candidate_ids[:2]
        ),
        StructuredActionFactor(
            "worker:1", observations[0].candidate_ids[2:]
        ),
    )
    trajectory = []
    rng = np.random.default_rng(103)

    def restrict_second_factor(factor_index, selected, mask):
        del selected
        if factor_index == 1:
            mask[3] = False
        return mask

    for timestep in range(2):
        observation = observations[timestep]
        local_factors = tuple(
            StructuredActionFactor(
                factor.factor_id,
                observation.candidate_ids[index * 2 : index * 2 + 2],
            )
            for index, factor in enumerate(factors)
        )
        scores = agent.score_results_structured((observation,))[0]
        sampled = sample_structured_joint_action(
            observation,
            local_factors,
            scores.candidate_logits,
            behavior_value=scores.value,
            rng=rng,
            legal_mask_update=restrict_second_factor,
        )
        trajectory.append(
            StructuredJointTrajectoryStep(
                actor_id="actor:0",
                episode_id="episode:0",
                timestep=timestep,
                policy_version=0,
                observation=sampled.observation,
                factors=sampled.factors,
                joint_behavior_log_probability=sampled.joint_log_probability,
                behavior_value=sampled.behavior_value,
                reward=1.0 if timestep == 1 else 0.0,
                next_observation=observations[timestep + 1],
                terminated=timestep == 1,
            )
        )
    trajectory[0] = replace(
        trajectory[0], next_observation=trajectory[1].observation
    )
    batch = collate_entity_candidate_observations(
        [step.observation for step in trajectory]
    ).to_torch("cpu")
    logits = agent.policy(batch).logits.detach().requires_grad_(True)
    policy_output = EntityCandidatePolicyOutput(
        logits=logits,
        values=torch.zeros(len(trajectory)),
    )
    vector_log_probability, vector_entropy = agent._joint_statistics(
        policy_output, trajectory
    )
    assert vector_log_probability.detach().numpy() == pytest.approx(
        [step.joint_behavior_log_probability for step in trajectory],
        abs=1e-5,
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_log_probabilities = []
    reference_entropies = []
    for row, step in enumerate(trajectory):
        candidate_index = {
            candidate_id: index
            for index, candidate_id in enumerate(step.observation.candidate_ids)
        }
        joint_log_probability = reference_logits.new_zeros(())
        joint_entropy = reference_logits.new_zeros(())
        for factor in step.factors:
            conditional_candidate_ids = factor.conditional_candidate_ids
            indices = torch.as_tensor(
                [candidate_index[value] for value in conditional_candidate_ids],
                dtype=torch.long,
            )
            distribution = torch.distributions.Categorical(
                logits=reference_logits[row].index_select(0, indices)
            )
            selected = conditional_candidate_ids.index(
                factor.selected_candidate_id
            )
            joint_log_probability = (
                joint_log_probability
                + distribution.log_prob(torch.as_tensor(selected))
            )
            joint_entropy = joint_entropy + distribution.entropy()
        reference_log_probabilities.append(joint_log_probability)
        reference_entropies.append(joint_entropy)
    reference_log_probability = torch.stack(reference_log_probabilities)
    reference_entropy = torch.stack(reference_entropies)

    assert torch.allclose(
        vector_log_probability, reference_log_probability, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        vector_entropy, reference_entropy, atol=1e-6, rtol=1e-6
    )
    weights = torch.tensor([0.7, -1.3])
    (vector_log_probability * weights + vector_entropy).sum().backward()
    (reference_log_probability * weights + reference_entropy).sum().backward()
    assert torch.allclose(
        logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-6
    )

    before = [parameter.detach().clone() for parameter in agent.policy.parameters()]

    result = agent.update_joint_structured((trajectory,))

    assert result.transitions == 2
    assert result.episodes == 1
    assert result.minibatches == 2
    assert result.episode_return_mean == pytest.approx(1.0)
    assert result.episode_return_std == pytest.approx(0.0)
    assert result.episode_return_unique_count == 1
    assert result.reward_nonzero_fraction == pytest.approx(0.5)
    assert result.gae_oldest_delta_weight_mean == pytest.approx(1.0)
    assert np.isfinite(result.approximate_kl)
    assert np.isfinite(result.importance_ratio_mean)
    assert np.isfinite(result.importance_ratio_std)
    assert 0.0 < result.importance_ratio_min <= result.importance_ratio_max
    assert np.isfinite(result.gradient_norm)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.policy.parameters())
    )


def _run_value_only_joint_update(
    value_backbone_gradient_scale: float,
):
    torch.manual_seed(113)
    agent = algorithm_registry.get("structured_ppo").build_structured(
        StructuredPolicySpec(2, 3, 4, 2),
        {
            "structured_model_dim": 8,
            "structured_heads": 2,
            "structured_layers": 1,
            "structured_feedforward_dim": 16,
            "epochs": 1,
            "minibatch_size": 1,
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "lr": 1e-2,
            "entropy_coef": 0.0,
            "value_coef": 1.0,
            "value_backbone_gradient_scale": value_backbone_gradient_scale,
        },
        "cpu",
    )
    observation = _observation(0, 1)
    score = agent.score_results_structured((observation,))[0]
    sampled = sample_structured_joint_action(
        observation,
        (StructuredActionFactor("only", observation.candidate_ids),),
        score.candidate_logits,
        behavior_value=score.value,
        deterministic=True,
    )
    step = StructuredJointTrajectoryStep(
        actor_id="actor:value-only",
        episode_id="episode:value-only",
        timestep=0,
        policy_version=0,
        observation=sampled.observation,
        factors=sampled.factors,
        joint_behavior_log_probability=sampled.joint_log_probability,
        behavior_value=sampled.behavior_value,
        reward=1.0,
        next_observation=_observation(1, 1),
        terminated=True,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.policy.named_parameters()
    }
    result = agent.update_joint_structured(((step,),))
    after = {
        name: parameter.detach().clone()
        for name, parameter in agent.policy.named_parameters()
    }
    return before, after, result


def test_value_head_can_train_without_changing_shared_policy_backbone() -> None:
    isolated_before, isolated_after, isolated = _run_value_only_joint_update(0.0)
    shared_before, shared_after, shared = _run_value_only_joint_update(1.0)

    assert isolated.value_backbone_gradient_scale == 0.0
    assert shared.value_backbone_gradient_scale == 1.0
    assert any(
        not torch.equal(isolated_before[name], isolated_after[name])
        for name in isolated_before
        if name.startswith("value_head.")
    )
    assert all(
        torch.equal(isolated_before[name], isolated_after[name])
        for name in isolated_before
        if not name.startswith("value_head.")
    )
    assert any(
        not torch.equal(shared_before[name], shared_after[name])
        for name in shared_before
        if not name.startswith("value_head.")
    )


def test_value_backbone_gradient_scale_is_bounded() -> None:
    plugin = algorithm_registry.get("structured_ppo")
    with pytest.raises(ValueError, match="must be in"):
        plugin.build_structured(
            StructuredPolicySpec(2, 3, 4, 2),
            {"value_backbone_gradient_scale": 1.01},
            "cpu",
        )


def _build_ratio_guard_case(
    *,
    ratio_min: float,
    ratio_max: float,
    max_backtracks: int,
):
    torch.manual_seed(127)
    agent = algorithm_registry.get("structured_ppo").build_structured(
        StructuredPolicySpec(2, 3, 4, 2),
        {
            "structured_model_dim": 8,
            "structured_heads": 2,
            "structured_layers": 1,
            "structured_feedforward_dim": 16,
            "epochs": 2,
            "minibatch_size": 2,
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "lr": 1.0,
            "entropy_coef": 0.0,
            "value_coef": 0.0,
            "policy_ratio_guard_min": ratio_min,
            "policy_ratio_guard_max": ratio_max,
            "policy_ratio_guard_backoff_factor": 0.1,
            "policy_ratio_guard_max_backtracks": max_backtracks,
        },
        "cpu",
    )
    observation = _observation(0, 2)
    score = agent.score_results_structured((observation,))[0]
    log_probabilities = torch.log_softmax(
        torch.as_tensor(score.candidate_logits), dim=0
    )
    trajectories = tuple(
        (
            StructuredTransition(
                episode_id=f"guard:{index}",
                timestep=0,
                observation=observation,
                candidate_id=observation.candidate_ids[index],
                candidate_index=index,
                behavior_log_probability=float(log_probabilities[index]),
                behavior_value=score.value,
                reward=reward,
                done=True,
            ),
        )
        for index, reward in enumerate((1.0, -1.0))
    )
    return agent, trajectories


def test_ratio_guard_rolls_back_a_rejected_update_transactionally() -> None:
    agent, trajectories = _build_ratio_guard_case(
        ratio_min=0.999,
        ratio_max=1.001,
        max_backtracks=0,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.policy.named_parameters()
    }

    result = agent.update_structured(trajectories)

    assert result.trust_region_enabled
    assert not result.trust_region_update_accepted
    assert result.trust_region_proposal_attempts == 1
    assert result.trust_region_backtracks == 1
    assert result.post_update_importance_ratio_min == pytest.approx(1.0)
    assert result.post_update_importance_ratio_max == pytest.approx(1.0)
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in agent.policy.named_parameters()
    )
    assert not agent.optimizer.state
    assert agent.optimizer.param_groups[0]["lr"] == 1.0


def test_ratio_guard_backtracks_until_full_batch_is_contained() -> None:
    agent, trajectories = _build_ratio_guard_case(
        ratio_min=0.5,
        ratio_max=2.0,
        max_backtracks=6,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.policy.named_parameters()
    }

    result = agent.update_structured(trajectories)

    assert result.trust_region_update_accepted
    assert result.trust_region_proposal_attempts > 1
    assert result.trust_region_backtracks == (
        result.trust_region_proposal_attempts - 1
    )
    assert result.effective_learning_rate < 1.0
    assert result.post_update_importance_ratio_min >= 0.5
    assert result.post_update_importance_ratio_max <= 2.0
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in agent.policy.named_parameters()
    )


def test_ratio_guard_requires_bounds_around_one() -> None:
    plugin = algorithm_registry.get("structured_ppo")
    with pytest.raises(ValueError, match="requires"):
        plugin.build_structured(
            StructuredPolicySpec(2, 3, 4, 2),
            {
                "policy_ratio_guard_min": 0.5,
                "policy_ratio_guard_max": 0.9,
            },
            "cpu",
        )


def test_joint_ratio_guard_audits_and_rolls_back_exact_joint_probabilities() -> None:
    agent, _ = _build_ratio_guard_case(
        ratio_min=0.999,
        ratio_max=1.001,
        max_backtracks=0,
    )
    observation = _observation(0, 2)
    score = agent.score_results_structured((observation,))[0]
    log_probabilities = torch.log_softmax(
        torch.as_tensor(score.candidate_logits), dim=0
    )
    trajectories = tuple(
        (
            StructuredJointTrajectoryStep(
                actor_id="actor:guard",
                episode_id=f"joint-guard:{index}",
                timestep=0,
                policy_version=0,
                observation=observation,
                factors=(
                    StructuredFactorChoice(
                        factor_id="choice",
                        candidate_ids=observation.candidate_ids,
                        selected_candidate_id=(
                            observation.candidate_ids[index]
                        ),
                        behavior_log_probability=float(
                            log_probabilities[index]
                        ),
                    ),
                ),
                joint_behavior_log_probability=float(
                    log_probabilities[index]
                ),
                behavior_value=score.value,
                reward=reward,
                next_observation=_observation(1, 2),
                terminated=True,
            ),
        )
        for index, reward in enumerate((1.0, -1.0))
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.policy.named_parameters()
    }

    result = agent.update_joint_structured(trajectories)

    assert not result.trust_region_update_accepted
    assert result.post_update_importance_ratio_min == pytest.approx(1.0)
    assert result.post_update_importance_ratio_max == pytest.approx(1.0)
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in agent.policy.named_parameters()
    )


def test_prefix_conditioned_behavior_probability_matches_joint_ppo() -> None:
    torch.manual_seed(41)
    agent = algorithm_registry.get("structured_ppo").build_structured(
        StructuredPolicySpec(2, 3, 4, 2),
        {
            "structured_model_dim": 16,
            "structured_heads": 4,
            "structured_layers": 1,
            "structured_feedforward_dim": 32,
            "structured_prefix_dim": 4,
            "epochs": 1,
            "minibatch_size": 1,
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "lr": 1e-3,
        },
        "cpu",
    )
    observation = _observation(0, 4)
    factors = (
        StructuredActionFactor("worker:0", observation.candidate_ids[:2]),
        StructuredActionFactor("worker:1", observation.candidate_ids[2:]),
    )
    score = agent.score_results_structured((observation,))[0]
    sampled = sample_structured_joint_action(
        observation,
        factors,
        score.candidate_logits,
        behavior_value=score.value,
        deterministic=False,
        rng=np.random.default_rng(211),
        candidate_prefix_keys=score.candidate_prefix_keys,
        candidate_prefix_values=score.candidate_prefix_values,
    )
    step = StructuredJointTrajectoryStep(
        actor_id="actor:prefix",
        episode_id="episode:prefix",
        timestep=0,
        policy_version=0,
        observation=sampled.observation,
        factors=sampled.factors,
        joint_behavior_log_probability=sampled.joint_log_probability,
        behavior_value=sampled.behavior_value,
        reward=1.0,
        next_observation=_observation(1, 4),
        terminated=True,
    )
    batch = collate_entity_candidate_observations(
        (sampled.observation,)
    ).to_torch("cpu")
    output = agent.policy(batch)
    log_probability, _ = agent._joint_statistics(output, (step,))

    assert score.candidate_prefix_keys
    assert log_probability.detach().cpu().item() == pytest.approx(
        sampled.joint_log_probability, abs=1e-5
    )
    updated = agent.update_joint_structured(((step,),))
    assert updated.importance_ratio_mean == pytest.approx(1.0, abs=1e-5)


def test_gae_diagnostic_exposes_719_step_terminal_credit_attenuation() -> None:
    agent = algorithm_registry.get("structured_ppo").build_structured(
        StructuredPolicySpec(2, 3, 4, 2),
        {
            "structured_model_dim": 8,
            "structured_heads": 2,
            "structured_layers": 1,
            "structured_feedforward_dim": 16,
            "gamma": 1.0,
            "gae_lambda": 0.98,
            "epochs": 1,
            "minibatch_size": 1024,
        },
        "cpu",
    )
    observation = _observation(0, 2)
    trajectory = [
        StructuredTransition(
            episode_id="delayed-terminal",
            timestep=timestep,
            observation=observation,
            candidate_id=observation.candidate_ids[0],
            candidate_index=0,
            behavior_log_probability=0.0,
            behavior_value=0.0,
            reward=1.0 if timestep == 718 else 0.0,
            done=timestep == 718,
        )
        for timestep in range(719)
    ]

    _, advantages, returns = agent._targets((trajectory,))

    expected = 0.98**718
    assert advantages[0] == pytest.approx(expected, rel=1e-5)
    assert returns[0] == pytest.approx(expected, rel=1e-5)
    assert advantages[-1] == pytest.approx(1.0)

    result = agent.update_structured((trajectory,))
    assert result.episode_length_min == 719
    assert result.episode_length_max == 719
    assert result.gae_decay == pytest.approx(0.98)
    assert result.gae_oldest_delta_weight_mean == pytest.approx(expected)
