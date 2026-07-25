import numpy as np
import torch

from jormungandr.core import C51Agent, PrioritizedReplayBuffer


def make_agent() -> C51Agent:
    return C51Agent(
        obs_dim=2,
        num_actions=3,
        action_values=[-1.0, 0.0, 1.0],
        hidden=8,
        lr=1e-3,
        gamma=0.99,
        v_min=-10.0,
        v_max=10.0,
        atoms=51,
        target_update=10,
        max_grad_norm=1.0,
        device="cpu",
    )


def test_prioritized_replay_keeps_zero_updates_sampleable() -> None:
    replay = PrioritizedReplayBuffer(capacity=4, obs_dim=2)
    for value in range(4):
        obs = np.asarray([value, 0], dtype=np.float32)
        replay.add(obs, 0, 0.0, obs, False)

    replay.update_priorities(np.arange(4), np.zeros(4, dtype=np.float32))
    batch, idxs, weights = replay.sample(batch_size=4, beta=0.4)

    assert batch[0].shape == (4, 2)
    assert idxs.shape == (4,)
    assert np.isfinite(weights).all()
    assert (replay.priorities[:4] > 0.0).all()


def test_c51_projection_preserves_probability_at_exact_atoms() -> None:
    agent = make_agent()
    with torch.no_grad():
        projected = agent._project_distribution(
            torch.zeros((3, 2)),
            torch.zeros(3),
            torch.ones(3),
        )

    assert torch.allclose(projected.sum(dim=1), torch.ones(3), atol=1e-6)


def test_c51_held_out_evaluation_does_not_change_weights() -> None:
    agent = make_agent()
    batch = (
        np.zeros((2, 2), dtype=np.float32),
        np.asarray([[0], [2]], dtype=np.float32),
        np.asarray([[0], [1]], dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.asarray([[0], [1]], dtype=np.float32),
    )
    before = [param.detach().clone() for param in agent.q.parameters()]
    metrics = agent.evaluate_batch(batch)

    assert metrics["count"] == 2
    assert metrics["loss"] > 0
    assert all(
        torch.equal(old, new)
        for old, new in zip(before, agent.q.parameters())
    )
