from examples.benchmark_structured_ppo_safety import run_benchmark


def test_structured_ppo_safety_benchmark_passes_all_controls() -> None:
    result = run_benchmark((211, 223, 227))

    assert result["decision"]["passed"]
