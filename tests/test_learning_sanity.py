"""Behavioral learning contracts, not just finite-loss/update smoke tests.

A balanced two-action contextual bandit exposes its correct action explicitly.
A separate PyTorch learner establishes that the targets and input are learnable.
"""

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry
from jormungandr.algorithms.structured_ppo import StructuredTransition
from jormungandr.structured import EntityCandidateObservation, StructuredPolicySpec
from jormungandr.structured_replay import StructuredReplayTransition


SPEC = StructuredPolicySpec(1, 1, 1, 1)
CONFIG = dict(
    structured_model_dim=16,
    structured_heads=2,
    structured_layers=1,
    lr=0.003,
    gamma=0.0,
    epochs=2,
    minibatch_size=32,
)


def cue_observation(cue):
    return EntityCandidateObservation(
        global_features=np.zeros(1),
        entity_features=np.array([[float(cue == 0)], [float(cue == 1)]]),
        entity_type_ids=np.zeros(2, dtype=np.int64),
        entity_ids=("left", "right"),
        candidate_features=np.zeros((2, 1)),
        candidate_ids=("left", "right"),
        candidate_entity_indices=np.arange(2),
        legal_action_mask=np.ones(2, dtype=bool),
    )


def test_independent_pytorch_learns_balanced_contextual_bandit():
    """No Jormungandr policy, collation, optimizer or update code is used."""
    torch.manual_seed(7)
    inputs = torch.eye(2).repeat(16, 1)
    labels = torch.arange(2).repeat(16)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(model(inputs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert (model(inputs).argmax(-1) == labels).float().mean().item() == 1.0
    # Removing the informative cue makes both classes indistinguishable.
    assert (
        model(torch.zeros_like(inputs)).argmax(-1) == labels
    ).float().mean().item() == 0.5


def test_structured_dqn_learns_explicit_cue_from_rewards():
    torch.manual_seed(7)
    agent = algorithm_registry.get("structured_dqn").build_structured(
        SPEC, CONFIG, "cpu"
    )
    samples = []
    for cue in (0, 1):
        obs = cue_observation(cue)
        for action in (0, 1):
            samples.append(
                StructuredReplayTransition(
                    obs,
                    obs.candidate_ids[action],
                    float(action == cue),
                    obs,
                    True,
                    "actor",
                    f"{cue}-{action}",
                    0,
                    0,
                )
            )
    for _ in range(60):
        agent.update_structured(samples)
    for cue in (0, 1):
        result = agent.action_result_structured(
            cue_observation(cue), deterministic=True
        )
        assert result.candidate_index == cue
        assert result.candidate_values[cue] - result.candidate_values[1 - cue] > 0.5


def test_structured_ppo_learns_explicit_cue_from_sampled_rewards():
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    agent = algorithm_registry.get("structured_ppo").build_structured(
        SPEC, CONFIG, "cpu"
    )
    observations = [cue_observation(i % 2) for i in range(64)]
    for update in range(12):
        scores = agent.score_results_structured(observations)
        trajectories = []
        for i, (obs, score) in enumerate(zip(observations, scores)):
            logits = np.asarray(score.candidate_logits)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            action = int(rng.choice(2, p=probabilities))
            trajectories.append(
                [
                    StructuredTransition(
                        f"{update}-{i}",
                        0,
                        obs,
                        obs.candidate_ids[action],
                        action,
                        float(np.log(probabilities[action])),
                        score.value,
                        float(action == i % 2),
                        True,
                    )
                ]
            )
        agent.update_structured(trajectories)
    for cue, score in enumerate(
        agent.score_results_structured([cue_observation(0), cue_observation(1)])
    ):
        probabilities = torch.softmax(torch.tensor(score.candidate_logits), -1)
        assert probabilities[cue].item() > 0.9
