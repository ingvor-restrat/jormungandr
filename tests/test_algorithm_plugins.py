import json

import numpy as np
import pytest
import torch

from jormungandr.algorithms import algorithm_registry, available_algorithms
from jormungandr.export import export_inference_bundle, inspect_checkpoint
from jormungandr.selectors import (
    QUBORolloutSelector,
    RolloutCandidate,
    select_trajectory_replay,
)
from jormungandr.core import PrioritizedReplayBuffer
from jormungandr.service import JormungandrRuntime


BUILTIN_ALGORITHMS = {
    "appo",
    "bc",
    "c51",
    "cql",
    "dqn",
    "dreamerv3",
    "impala",
    "marwil",
    "maxent",
    "ppo",
    "qrdqn",
    "sac",
}


def _batch(seed: int = 11):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(8, 4)).astype(np.float32),
        rng.integers(0, 3, size=(8, 1)).astype(np.float32),
        rng.normal(size=(8, 1)).astype(np.float32),
        rng.normal(size=(8, 4)).astype(np.float32),
        np.asarray([[0], [0], [0], [0], [0], [0], [0], [1]], dtype=np.float32),
    )


def _metadata():
    return [
        {
            "actor_id": "actor-a",
            "episode_id": "episode-a",
            "timestep": index,
            "policy_version": 0,
            "behavior_logp": -np.log(3.0),
            "behavior_value": 0.0,
            "action_mask": [True, True, True],
            "next_action_mask": [True, True, True],
        }
        for index in range(8)
    ]


def test_all_documented_algorithms_are_registered() -> None:
    assert BUILTIN_ALGORITHMS.issubset(set(available_algorithms()))


@pytest.mark.parametrize("algorithm", sorted(BUILTIN_ALGORITHMS))
def test_builtin_plugin_updates_infers_and_round_trips(algorithm: str) -> None:
    config = {
        "action_values": [-1.0, 0.0, 1.0],
        "hidden": 16,
        "dreamer_latent": 12,
        "imagination_horizon": 3,
        "quantiles": 17,
        "epochs": 1,
        "minibatch_size": 8,
        "target_update": 2,
        "lr": 1e-3,
    }
    plugin = algorithm_registry.get(algorithm)
    agent = plugin.build(4, config, "cpu")
    result = agent.update(
        _batch(),
        np.ones(8, dtype=np.float32),
        metadata=_metadata(),
    )

    assert np.isfinite(result.loss)
    assert result.priorities.shape == (8,)
    assert np.isfinite(result.priorities).all()
    action = agent.action_result(
        np.zeros(4, dtype=np.float32),
        deterministic=True,
        action_mask=np.asarray([False, True, True]),
    )
    assert action.action_idx in {1, 2}
    json.dumps(dict(action.extras), allow_nan=False)

    restored = plugin.build(4, config, "cpu")
    restored.load_state_dict(agent.state_dict())
    restored_action = restored.action_result(
        np.zeros(4, dtype=np.float32), deterministic=True
    )
    assert restored_action.action_idx in {0, 1, 2}


def test_qrdqn_exposes_return_quantiles_and_downside_score() -> None:
    agent = algorithm_registry.get("qrdqn").build(
        4,
        {
            "action_values": [-1.0, 0.0, 1.0],
            "hidden": 8,
            "quantiles": 13,
            "quantile_risk_measure": "cvar",
            "quantile_risk_level": 0.2,
        },
        "cpu",
    )
    result = agent.action_result(np.zeros(4, dtype=np.float32), deterministic=True)

    assert result.extras["risk_measure"] == "cvar"
    assert len(result.extras["quantiles"]) == 3
    assert all(len(row) == 13 for row in result.extras["quantiles"])
    assert len(result.extras["risk_values"]) == 3


def test_categorical_sac_masked_entropy_stays_finite() -> None:
    agent = algorithm_registry.get("sac").build(
        4,
        {"action_values": [-1.0, 0.0, 1.0], "hidden": 8, "lr": 1e-3},
        "cpu",
    )
    batch = list(_batch())
    # Keep every demonstrated action legal while excluding another slot.
    actions = batch[1].astype(np.int64).reshape(-1)
    metadata = []
    for index, action in enumerate(actions.tolist()):
        mask = [True, True, True]
        mask[(action + 1) % 3] = False
        metadata.append(
            {
                "actor_id": "actor-a",
                "episode_id": "episode-a",
                "timestep": index,
                "action_mask": mask,
                "next_action_mask": [True, False, True],
            }
        )

    result = agent.update(
        tuple(batch), np.ones(8, dtype=np.float32), metadata=metadata
    )

    assert np.isfinite(result.loss)
    assert all(np.isfinite(value) for value in result.metrics.values())
    assert all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters())


def test_marwil_uses_contiguous_returns_without_online_policy_aging() -> None:
    plugin = algorithm_registry.get("marwil")
    assert plugin.replay_mode == "trajectory"
    assert plugin.enforce_policy_lag is False
    assert algorithm_registry.get("ppo").enforce_policy_lag is True
    assert JormungandrRuntime._parse_learner_config(
        {"enabled": True, "algo": "ppo"}
    ).max_policy_lag == 0

    agent = plugin.build(
        2,
        {
            "action_values": [-1.0, 0.0, 1.0],
            "hidden": 8,
            "gamma": 0.5,
        },
        "cpu",
    )
    with torch.no_grad():
        for parameter in agent.value.parameters():
            parameter.zero_()
    batch = (
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        np.ones((3, 2), dtype=np.float32),
        np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32),
    )
    metadata = [
        {
            "actor_id": "actor-a",
            "episode_id": "episode-a",
            "timestep": index,
            "action_mask": [True, True, True],
        }
        for index in range(3)
    ]

    with torch.no_grad():
        *_, advantage, _weights = agent._losses(
            batch, training=False, metadata=metadata
        )

    np.testing.assert_allclose(
        advantage.cpu().numpy(), [2.75, 3.5, 3.0], rtol=1e-6
    )


def test_qubo_selects_exactly_k_diverse_candidates() -> None:
    candidates = [
        RolloutCandidate(
            key=str(index),
            utility=float(6 - index),
            embedding=np.asarray([float(index // 2), float(index % 2)]),
        )
        for index in range(6)
    ]
    result = QUBORolloutSelector(
        utility_weight=1.0,
        diversity_weight=0.5,
        cardinality_penalty=4.0,
    ).select(candidates, 3)

    assert result.decisions.tolist().count(1) == 3
    assert result.selected_indices.shape == (3,)
    assert result.qubo.shape == (6, 6)
    assert np.isfinite(result.energy)


def test_qubo_trajectory_selection_uses_fragments_and_excludes_stale_data() -> None:
    replay = PrioritizedReplayBuffer(capacity=16, obs_dim=2)
    metadata = {}
    for timestep in range(12):
        obs = np.asarray([float(timestep), 0.0], dtype=np.float32)
        replay.add(obs, timestep % 3, float(timestep % 2), obs + 1.0, timestep == 11)
        metadata[timestep] = {
            "actor_id": "actor-a",
            "episode_id": "episode-a",
            "timestep": timestep,
            "policy_version": 0 if timestep < 3 else 9,
        }

    selected = select_trajectory_replay(
        replay,
        metadata,
        batch_size=6,
        beta=0.4,
        rollout_length=3,
        current_policy_version=10,
        max_policy_lag=2,
        selector_name="qubo",
        config={"qubo_pool_factor": 4.0},
    )

    assert selected is not None
    assert selected.metrics["selector_stale_transition_count"] == 3.0
    assert selected.metrics["selector_selected_rollouts"] == 2.0
    assert selected.audit["unit"] == "rollout_fragment"
    assert sum(selected.audit["decisions"]) == 2
    assert len(selected.indices) == 6
    assert (selected.indices >= 3).all()


def test_qrdqn_checkpoint_restores_and_exports_quantile_shape(tmp_path) -> None:
    runtime = JormungandrRuntime(checkpoint_root=str(tmp_path / "checkpoints"))
    runtime.create_model(
        obs_dim=4,
        model_id="quantile-test",
        tensorboard_enabled=False,
        learner={
            "enabled": True,
            "algo": "qrdqn",
            "device": "cpu",
            "hidden": 8,
            "plugin_config": {
                "quantiles": 17,
                "quantile_risk_measure": "lower_quantile",
                "quantile_risk_level": 0.1,
            },
            "checkpoint_every": 0,
            "checkpoint_dir": "models",
        },
    )
    try:
        checkpoint = runtime.force_policy_checkpoint("quantile-test")["checkpoint"]
        inspected = inspect_checkpoint(checkpoint)
        assert inspected["spec"]["algo"] == "qrdqn"
        assert inspected["spec"]["quantiles"] == 17

        export_inference_bundle(checkpoint, str(tmp_path / "bundle"))
        manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
        assert manifest["io"]["output"]["shape"] == ["N", 3, 17]

        restored = JormungandrRuntime(checkpoint_root=str(tmp_path / "restored"))
        try:
            restored.create_model(
                obs_dim=4,
                model_id="quantile-restored",
                checkpoint_path=checkpoint,
                tensorboard_enabled=False,
            )
            inference = restored.policy_infer(
                "quantile-restored", obs=[0.0, 0.0, 0.0, 0.0]
            )
            assert inference["algo"] == "qrdqn"
            assert len(inference["quantiles"]) == 3
        finally:
            restored.close_all()
    finally:
        runtime.close_all()
