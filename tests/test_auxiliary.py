import math
import numpy as np
import pytest
import torch
from torch import nn
from copy import deepcopy
from dataclasses import replace

from jormungandr.auxiliary import AuxiliaryLoss, compose_losses, context_selection_loss
from jormungandr.algorithms.structured_dqn import StructuredDQNAgent
from jormungandr.structured import (
    EntityCandidateObservation,
    StructuredPolicySpec,
    EntityCandidatePolicyOutput,
)
from jormungandr.structured_replay import StructuredReplayTransition
from examples.context_auxiliary_research import context_pairs
from examples.multiservice_learning_control import collect_arrays


def test_clipped_context_loss_matches_equation_and_swapped_options():
    logits = torch.tensor([[1.0, 0.0], [-2.0, 1.0], [20.0, -20.0]], requires_grad=True)
    labels = torch.tensor([0, 1, 1])
    actual = context_selection_loss(logits, labels, margin_clip=5.0)
    expected = torch.nn.functional.softplus(-torch.tensor([1.0, 3.0, -5.0])).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual, context_selection_loss(logits.flip(1), 1 - labels, margin_clip=5.0)
    )
    actual.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    assert torch.equal(logits.grad[2], torch.zeros(2))  # Paper's hard clipping.


def test_masked_weighted_loss_and_composition_have_exact_microbatch_scaling():
    source = torch.randn(7, 2)
    full = source.clone().requires_grad_()
    split = source.clone().requires_grad_()
    labels = torch.tensor([0, 1, 1, 0, 1, 0, 0])
    valid = torch.tensor([True, False, True, True, False, False, True])
    weights = torch.arange(1, 8, dtype=torch.float32) / 7
    base = (full**2).mean()
    loss, metrics = compose_losses(
        base,
        [
            AuxiliaryLoss(
                "grounding",
                context_selection_loss(full, labels, valid=valid, weights=weights),
                0.3,
            )
        ],
    )
    loss.backward()
    for start, end in ((0, 3), (3, 6), (6, 7)):
        chunk = split[start:end]
        total, _ = compose_losses(
            (chunk**2).mean(),
            [
                AuxiliaryLoss(
                    "grounding",
                    context_selection_loss(
                        chunk,
                        labels[start:end],
                        valid=valid[start:end],
                        weights=weights[start:end],
                    ),
                    0.3,
                )
            ],
        )
        (total * ((end - start) / 7)).backward()
    torch.testing.assert_close(full.grad, split.grad)
    assert metrics["auxiliary_total"] == pytest.approx(
        metrics["auxiliary/grounding"] * 0.3
    )
    empty = context_selection_loss(full, labels, valid=torch.zeros(7, dtype=torch.bool))
    assert empty.item() == 0 and empty.requires_grad


@pytest.mark.parametrize("coefficient", [-1.0, float("nan"), float("inf")])
def test_invalid_auxiliary_coefficients_rejected(coefficient):
    with pytest.raises(ValueError, match="coefficient"):
        compose_losses(
            torch.tensor(1.0), [AuxiliaryLoss("x", torch.tensor(2.0), coefficient)]
        )


def test_invalid_values_names_and_masked_nan_rejected():
    with pytest.raises(ValueError, match="unique"):
        compose_losses(torch.tensor(1.0), [AuxiliaryLoss("x", torch.tensor(1.0))] * 2)
    with pytest.raises(ValueError, match="finite"):
        context_selection_loss(
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([0]),
            valid=torch.tensor([False]),
        )
    with pytest.raises(ValueError, match="indices"):
        context_selection_loss(torch.zeros(1, 2), torch.tensor([2]))
    with pytest.raises(ValueError, match="weights"):
        context_selection_loss(
            torch.zeros(1, 2), torch.tensor([0]), weights=torch.tensor([-1.0])
        )
    x = torch.tensor(1.0, requires_grad=True)
    total, _ = compose_losses(x * x, [AuxiliaryLoss("unused", x * 100, 0.0)])
    total.backward()
    assert x.grad.item() == 2.0


def test_contrast_pairs_preserve_counts_and_change_only_the_anchor_fact():
    x = torch.from_numpy(collect_arrays(87, trials=8)[0])
    original = x.clone()
    left, right, positive, anchor, valid = context_pairs(
        x, torch.Generator().manual_seed(13)
    )
    assert valid.sum() > 100
    torch.testing.assert_close(x, original)
    pos = torch.where((positive == 0)[:, None], left, right)
    neg = torch.where((positive == 0)[:, None], right, left)
    torch.testing.assert_close(pos, x)
    a = pos[:, :-1].reshape(len(x), 16, 4)
    b = neg[:, :-1].reshape_as(a)
    torch.testing.assert_close(a[:, :, [0, 1, 3]], b[:, :, [0, 1, 3]])
    torch.testing.assert_close(
        (a[:, :, 0] * a[:, :, 2]).sum(1), (b[:, :, 0] * b[:, :, 2]).sum(1)
    )
    own = a[:, :, 0] > 0.5
    first = own.long().argmax(1)
    rows = torch.arange(len(x))[valid]
    torch.testing.assert_close(a[rows, first[valid], 2], anchor[valid])
    torch.testing.assert_close(b[rows, first[valid], 2], 1 - anchor[valid])
    assert set(positive[valid].tolist()) == {0, 1}
    assert torch.equal(neg[~valid], pos[~valid])


class SharedQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 4)
        self.q_head = nn.Linear(4, 1)
        self.extra_head = nn.Linear(4, 1)

    def forward(self, batch):
        h = self.encoder(batch.global_features).tanh()
        return EntityCandidatePolicyOutput(logits=self.q_head(h), values=h[:, 0] * 0)


class AuxiliaryAgent(StructuredDQNAgent):
    def _build_q(self, spec, config):
        return SharedQ().to(self.device)

    def _auxiliary_losses(self, transitions, weights=None):
        assert all(t.split == "train" for t in transitions)
        batch = self._collate_observations([t.observation for t in transitions])
        h = self.q.encoder(batch.global_features).tanh()
        per_item = (
            self.q.extra_head(h).squeeze(1) - batch.global_features[:, 0]
        ).square()
        if weights is not None:
            per_item = per_item * torch.as_tensor(weights, device=self.device)
        return [AuxiliaryLoss("prediction", per_item.mean(), 0.2)]


def replay():
    rows = []
    for t in range(5):
        obs = EntityCandidateObservation(
            global_features=np.array([t / 5]),
            entity_features=np.zeros((1, 1)),
            entity_type_ids=np.array([0]),
            entity_ids=("e",),
            candidate_features=np.zeros((1, 1)),
            candidate_ids=("a",),
            legal_action_mask=np.array([True]),
        )
        rows.append(
            StructuredReplayTransition(
                obs,
                "a",
                float(t % 2),
                obs,
                False,
                "actor",
                "episode",
                t,
                0,
                split="train",
            )
        )
    return rows


def test_native_auxiliary_update_microbatch_checkpoint_and_validation_boundary():
    cfg = {"gamma": 0.0, "target_update": 1, "max_grad": 1.0, "lr": 0.01}
    full = AuxiliaryAgent(StructuredPolicySpec(1, 1, 1, 1), cfg, "cpu")
    split = AuxiliaryAgent(full.spec, {**cfg, "structured_microbatch_size": 2}, "cpu")
    split.load_state_dict(deepcopy(full.state_dict()))
    before = deepcopy(full.q.state_dict())
    rows = replay()
    weights = np.array([0.2, 0.5, 1.0, 0.4, 0.3])
    a = full.update_structured(rows, weights)
    b = split.update_structured(rows, weights)
    assert a.loss == pytest.approx(b.loss, abs=1e-6)
    assert a.metrics["loss"] == pytest.approx(
        a.metrics["td_loss"] + a.metrics["auxiliary_total"]
    )
    assert not torch.equal(before["extra_head.weight"], full.q.extra_head.weight)
    for name, value in full.q.state_dict().items():
        torch.testing.assert_close(value, split.q.state_dict()[name])
        torch.testing.assert_close(value, full.q_target.state_dict()[name])
    restored = AuxiliaryAgent(full.spec, cfg, "cpu")
    restored.load_state_dict(deepcopy(full.state_dict()))
    c = full.update_structured(rows)
    d = restored.update_structured(rows)
    assert c.loss == pytest.approx(d.loss)
    before = deepcopy(restored.q.state_dict())
    with pytest.raises(ValueError, match="training transitions"):
        restored.update_structured([replace(rows[0], split="validation")])
    for name, value in before.items():
        torch.testing.assert_close(value, restored.q.state_dict()[name])
