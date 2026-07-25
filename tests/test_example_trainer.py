from examples.train_synthetic_control import (
    SyntheticTrackingEnvironment,
    interleaved_jobs,
)


def test_example_schedule_interleaves_validation_episodes() -> None:
    jobs = interleaved_jobs(train_episodes=8, validation_episodes=2, seed=17)

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


def test_synthetic_environment_is_seeded_and_domain_neutral() -> None:
    left = SyntheticTrackingEnvironment(horizon=8)
    right = SyntheticTrackingEnvironment(horizon=8)

    assert left.reset(42) == right.reset(42)
    for action in [-1.0, 0.0, 1.0, 0.0]:
        left_step = left.step(action)
        right_step = right.step(action)
        assert left_step == right_step
        assert len(left_step.observation) == left.observation_dim
        assert 0 <= left_step.aux_label <= 2
        assert -2.0 <= left_step.reward <= 1.0
