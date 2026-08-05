from __future__ import annotations

import pytest

from examples.benchmark_delayed_terminal_credit import (
    DelayedCreditConfig,
    run_benchmark,
)


def test_delayed_credit_core_matches_closed_form_and_exposes_attenuation() -> None:
    result = run_benchmark(
        DelayedCreditConfig(horizon=719), include_sb3=False
    )

    current = result["implementations"]["jormungandr_structured_ppo"]["0.98"]
    episodic = result["implementations"]["jormungandr_structured_ppo"]["1.0"]
    assert result["conditions"]["jormungandr_matches_closed_form"]
    assert result["conditions"]["current_opening_weight_below_bound"]
    assert result["conditions"]["episodic_opening_weight_preserved"]
    assert not result["passed"]
    assert current["analytic_opening_weight"] == pytest.approx(0.98**718)
    assert current["normalized_positive_opening_advantage"] < 3e-6
    assert episodic["analytic_opening_weight"] == pytest.approx(1.0)
    assert episodic["normalized_positive_opening_advantage"] == pytest.approx(1.0)


def test_delayed_credit_config_rejects_undeclared_decision_arms() -> None:
    with pytest.raises(ValueError, match="current lambda"):
        DelayedCreditConfig(gae_lambdas=(1.0,))
