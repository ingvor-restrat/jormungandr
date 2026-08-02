import pytest

from examples.compare_ou_algorithms import BenchmarkConfig, run_benchmark


def _small_config(algorithm: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        algorithms=(algorithm,),
        runs=1,
        train_episodes=2,
        eval_interval=1,
        eval_episodes=2,
        horizon=4,
        batch_size=2,
        warmup_transitions=2,
        replay_capacity=16,
        hidden=8,
        sensor_noise_std=0.1,
        base_seed=101,
    )


def test_online_algorithm_benchmark_is_seeded_and_checkpointable() -> None:
    report = run_benchmark(_small_config("dqn"), quiet=True)

    assert report["format"] == "jormungandr.ou_algorithm_benchmark.v1"
    assert report["config"]["algorithms"] == ("dqn",)
    assert [row["episode"] for row in report["runs"][0]["checkpoints"]] == [
        0,
        1,
        2,
    ]
    assert report["runs"][0]["checkpoint_roundtrip"] is True
    assert len(report["summaries"]) == 3
    assert len(report["playback"]["dqn"]["action_indices"]) == 4
    assert len(report["environment"]["held_out_path_seeds"]) == 2


@pytest.mark.parametrize("algorithm", ["ppo", "cql"])
def test_online_transition_benchmark_rejects_incompatible_cohorts(
    algorithm: str,
) -> None:
    with pytest.raises(ValueError, match="not an online transition-control"):
        run_benchmark(_small_config(algorithm), quiet=True)
