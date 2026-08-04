from __future__ import annotations

import numpy as np

from jormungandr.benchmarks import ConstrainedWorkbench
from jormungandr.structured_trajectory import sample_structured_joint_action


def test_workbench_gym_observation_oracle_and_delayed_terminal_reward() -> None:
    environment = ConstrainedWorkbench(
        horizon=3, worker_counts=(3,), job_counts=(2, 3, 4)
    )
    observation, info = environment.reset(seed=20260804)
    assert environment.observation_space.contains(observation)
    assert info["oracle_terminal_return"] == 1.0

    rewards = []
    utilities = []
    for _ in range(environment.horizon):
        decision = environment.oracle_joint_action()
        assert environment.is_feasible(decision.selected_candidate_ids)
        observation, reward, terminated, truncated, step_info = (
            environment.step_semantic(decision.selected_candidate_ids)
        )
        assert environment.observation_space.contains(observation)
        assert not truncated
        assert step_info["feasible"]
        rewards.append(reward)
        utilities.append(decision.utility)
    assert terminated
    assert rewards[:-1] == [0.0, 0.0]
    assert rewards[-1] == 1.0
    assert sum(utilities) == environment.oracle_terminal_utility
    assert environment.invalid_action_attempts == 0


def test_sequential_masked_samples_are_feasible_without_projection() -> None:
    environment = ConstrainedWorkbench(horizon=1)
    for seed in range(40):
        environment.reset(seed=seed)
        observation = environment.structured_observation()
        sampled = sample_structured_joint_action(
            observation,
            environment.action_factors(),
            np.zeros(len(observation.candidate_ids), dtype=np.float64),
            behavior_value=0.0,
            rng=np.random.default_rng(seed + 10_000),
            legal_mask_update=environment.legal_mask_update,
        )
        assert environment.is_feasible(sampled.selected_candidate_ids)
        _, _, terminated, _, info = environment.step_semantic(
            sampled.selected_candidate_ids
        )
        assert terminated and info["invalid_action_attempts"] == 0


def test_worker_count_changes_factor_count_not_reward_or_step_cardinality() -> None:
    episode_shapes = []
    for worker_count in (2, 4):
        environment = ConstrainedWorkbench(
            horizon=4,
            worker_counts=(worker_count,),
            job_counts=(3,),
        )
        environment.reset(seed=91)
        factors_per_step = []
        rewards = []
        for turn in range(environment.horizon):
            factors_per_step.append(len(environment.action_factors()))
            selected = environment.random_legal_joint_action(
                np.random.default_rng(100 + turn)
            )
            _, reward, terminated, _, _ = environment.step_semantic(selected)
            rewards.append(reward)
        assert terminated
        episode_shapes.append(
            (len(rewards), sum(value != 0.0 for value in rewards), factors_per_step)
        )

    assert episode_shapes[0][:2] == episode_shapes[1][:2] == (4, 1)
    assert episode_shapes[0][2] == [2, 2, 2, 2]
    assert episode_shapes[1][2] == [4, 4, 4, 4]
