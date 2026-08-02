import numpy as np
import torch

from jormungandr.core import C51Agent, PrioritizedReplayBuffer
from jormungandr.policy import (
    GraphTrajectoryBuffer,
    GraphTrajectoryStep,
    apply_legal_action_mask,
    masked_actor_critic_loss,
    select_masked_actions,
)


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


def test_c51_action_selection_respects_point_in_time_legal_mask() -> None:
    agent = make_agent()
    observation = np.zeros(2, dtype=np.float32)
    mask = np.asarray([False, True, False])

    _, deterministic_index = agent.act(
        observation,
        deterministic=True,
        action_mask=mask,
    )
    sampled_indices = {
        agent.act(observation, epsilon=1.0, action_mask=mask)[1]
        for _ in range(20)
    }

    assert deterministic_index == 1
    assert sampled_indices == {1}


def test_c51_bootstrap_requires_one_legal_next_action_per_row() -> None:
    agent = make_agent()
    next_obs = torch.zeros((2, 2))
    reward = torch.zeros(2)
    done = torch.zeros(2)

    projected = agent._project_distribution(
        next_obs,
        reward,
        done,
        next_action_mask=torch.tensor(
            [[False, True, False], [True, False, True]]
        ),
    )

    assert torch.allclose(projected.sum(dim=1), torch.ones(2), atol=1e-6)
    with np.testing.assert_raises(ValueError):
        agent._project_distribution(
            next_obs,
            reward,
            done,
            next_action_mask=torch.zeros((2, 3), dtype=torch.bool),
        )


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


def test_common_policy_loss_accepts_any_encoder_output_and_trains() -> None:
    torch.manual_seed(19)
    logits = torch.randn(3, 4, requires_grad=True)
    values = torch.randn(3, requires_grad=True)
    legal = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, False],
            [True, True, False, True],
        ]
    )
    actions = torch.tensor([2, 1, 3])
    loss = masked_actor_critic_loss(
        policy_logits=logits,
        state_values=values,
        action_indices=actions,
        advantages=torch.tensor([1.0, -0.5, 0.25]),
        returns=torch.tensor([0.5, -1.0, 1.5]),
        legal_action_mask=legal,
    )

    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.policy)
    assert torch.isfinite(loss.value)
    assert torch.isfinite(loss.entropy)
    assert logits.grad is not None
    assert values.grad is not None


def test_common_policy_selection_never_chooses_an_illegal_slot() -> None:
    logits = torch.tensor([[100.0, 1.0, 99.0], [2.0, 3.0, 4.0]])
    legal = torch.tensor([[False, True, False], [True, False, False]])

    masked = apply_legal_action_mask(logits, legal)
    actions, _ = select_masked_actions(
        logits, legal, deterministic=True
    )

    assert torch.isneginf(masked[~legal]).all()
    assert actions.tolist() == [1, 0]


def test_graph_trajectory_buffer_computes_returns_from_ordered_steps() -> None:
    buffer = GraphTrajectoryBuffer()
    buffer.add(
        GraphTrajectoryStep(
            episode_id="episode-1",
            timestep=0,
            state_reference="state-0",
            action_index=1,
            reward=2.0,
            done=False,
            legal_action_mask=(True, True),
            log_probability=-0.2,
            value=0.5,
        )
    )
    buffer.add(
        GraphTrajectoryStep(
            episode_id="episode-1",
            timestep=1,
            state_reference="state-1",
            action_index=0,
            reward=3.0,
            done=True,
            legal_action_mask=(True, False),
            log_probability=0.0,
            value=1.0,
        )
    )

    batch = buffer.finish(gamma=1.0, gae_lambda=1.0)

    assert batch.state_references == ("state-0", "state-1")
    np.testing.assert_allclose(batch.returns, [5.0, 3.0])
    np.testing.assert_allclose(batch.advantages, [4.5, 2.0])
    assert len(buffer.records()) == 2


def test_graph_trajectory_buffer_rejects_illegal_or_late_actions() -> None:
    with np.testing.assert_raises(ValueError):
        GraphTrajectoryStep(
            episode_id="episode-1",
            timestep=0,
            state_reference="state-0",
            action_index=1,
            reward=0.0,
            done=True,
            legal_action_mask=(True, False),
            log_probability=0.0,
            value=0.0,
        )
    buffer = GraphTrajectoryBuffer()
    buffer.add(
        GraphTrajectoryStep(
            episode_id="episode-2",
            timestep=0,
            state_reference="state-0",
            action_index=0,
            reward=0.0,
            done=True,
            legal_action_mask=(True,),
            log_probability=0.0,
            value=0.0,
        )
    )
    with np.testing.assert_raises(ValueError):
        buffer.add(
            GraphTrajectoryStep(
                episode_id="episode-2",
                timestep=1,
                state_reference="state-1",
                action_index=0,
                reward=0.0,
                done=True,
                legal_action_mask=(True,),
                log_probability=0.0,
                value=0.0,
            )
        )
