from jormungandr.cli_monitor import _view


def test_monitor_view_extracts_structured_learner_progress() -> None:
    view = _view(
        {
            "model_id": "farm-policy",
            "algorithm": {"name": "structured_dqn"},
            "device": "cuda",
            "replay": {"size": 123, "capacity": 1000},
            "experience": {"items": 456},
            "updates": 7,
            "policy_version": 7,
            "inference_calls": 789,
            "last_loss": 0.25,
            "last_td_abs": 0.5,
            "last_metrics": {"demonstration_fraction": 0.25},
            "last_error": "",
        }
    )

    assert view["model_id"] == "farm-policy"
    assert view["replay"] == "123/1000"
    assert view["updates"] == 7
    assert view["last_metrics"]["demonstration_fraction"] == 0.25
