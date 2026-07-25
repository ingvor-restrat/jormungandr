import math

from examples.train_ou_spread import OrnsteinUhlenbeckSpread, interleaved_jobs


def test_ou_spread_is_seeded_and_reports_finite_research_metrics() -> None:
    left = OrnsteinUhlenbeckSpread(horizon=12)
    right = OrnsteinUhlenbeckSpread(horizon=12)

    assert left.reset(42) == right.reset(42)
    for action in [-1.0, 0.0, 1.0, -1.0]:
        left_step = left.step(action)
        right_step = right.step(action)
        assert left_step == right_step
        assert len(left_step.observation) == left.observation_dim
        assert 0 <= left_step.aux_label <= 2
        assert math.isfinite(left_step.reward)
        assert math.isfinite(left_step.reference_reward)
        assert left_step.turnover >= 0.0


def test_ou_reference_policy_has_positive_expected_reward() -> None:
    rewards: list[float] = []
    for seed in range(24):
        environment = OrnsteinUhlenbeckSpread(horizon=48)
        observation = environment.reset(seed)
        episode_reward = 0.0
        for _ in range(48):
            z = observation[0]
            action = -1.0 if z > 0.65 else 1.0 if z < -0.65 else 0.0
            step = environment.step(action)
            episode_reward += step.reward
            assert math.isclose(step.reward, step.reference_reward)
            observation = step.observation
        rewards.append(episode_reward)

    assert sum(rewards) / len(rewards) > 1.0


def test_ou_schedule_interleaves_held_out_paths() -> None:
    jobs = interleaved_jobs(8, 2, seed=11)

    assert [job.split for job in jobs] == [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "train",
        "train",
        "train",
        "train",
        "validation",
    ]
    assert len({job.seed for job in jobs}) == len(jobs)
