from __future__ import annotations

import pytest

from examples.benchmark_conditional_supervision_sampling import (
    ConditionalSupervisionSamplingConfig,
    run_benchmark,
)


def test_conditional_supervision_importance_sampling_gate_passes() -> None:
    result = run_benchmark()

    uniform = result["arms"]["uniform_weighted_loss"]
    importance = result["arms"]["sample_weight_importance"]
    assert result["passed"]
    assert all(result["conditions"].values())
    assert uniform["rare_zero_batch_rate"] > 0.20
    assert importance["rare_zero_batch_rate"] == 0.0
    assert importance["objective_absolute_bias"] < 0.001
    assert importance["objective_mean_absolute_error"] < (
        0.5 * uniform["objective_mean_absolute_error"]
    )


def test_conditional_supervision_sampling_config_rejects_bad_exponent() -> None:
    with pytest.raises(ValueError, match="balance exponent"):
        ConditionalSupervisionSamplingConfig(balance_exponent=1.1)
