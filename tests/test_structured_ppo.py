import numpy as np
import pytest
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.algorithms.structured_ppo import StructuredTransition
from jormungandr.service import JormungandrRuntime, JormungandrServiceError
from jormungandr.structured import (
    EntityCandidateObservation,
    StructuredPolicySpec,
)
from jormungandr.structured_trajectory import (
    StructuredActionFactor,
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
    assert result.reward_nonzero_fraction == pytest.approx(1.0 / 3.0)
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
    before = [parameter.detach().clone() for parameter in agent.policy.parameters()]

    result = agent.update_joint_structured((trajectory,))

    assert result.transitions == 2
    assert result.episodes == 1
    assert result.minibatches == 2
    assert result.episode_return_mean == pytest.approx(1.0)
    assert result.episode_return_std == pytest.approx(0.0)
    assert result.episode_return_unique_count == 1
    assert result.reward_nonzero_fraction == pytest.approx(0.5)
    assert np.isfinite(result.approximate_kl)
    assert np.isfinite(result.importance_ratio_mean)
    assert np.isfinite(result.importance_ratio_std)
    assert 0.0 < result.importance_ratio_min <= result.importance_ratio_max
    assert np.isfinite(result.gradient_norm)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.policy.parameters())
    )
