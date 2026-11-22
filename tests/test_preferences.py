from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from jormungandr.preferences import (
    PreferenceFeedback,
    PairwisePreference,
    PreferenceLedger,
    exposure_to_payload,
    exposure_from_payload,
    feedback_to_payload,
    feedback_from_payload,
    preference_examples,
    preference_batch_loss,
    pairwise_preference_loss,
    ranked_shortlist,
)
from examples.shortlist_preference_benchmark import generate, features


def sample():
    row = generate(42, 1, "train")[0]
    return row["exposures"]["biased"], row["feedback"]["biased"]


def test_roundtrip_preserves_actual_presentation_order_and_signals():
    exposure, feedback = sample()
    a, b = exposure.displayed_ids[:2]
    feedback = replace(
        feedback, ratings={a: 1.0, b: 1.0}, comparisons=(PairwisePreference(a, b, 0.5),)
    )
    restored = exposure_from_payload(
        json.loads(json.dumps(exposure_to_payload(exposure)))
    )
    restored_feedback = feedback_from_payload(
        json.loads(json.dumps(feedback_to_payload(feedback)))
    )
    assert exposure_to_payload(restored) == exposure_to_payload(exposure)
    assert feedback_to_payload(restored_feedback) == feedback_to_payload(feedback)
    pairs = preference_examples(restored, restored_feedback)
    assert len(pairs) == 1 and pairs[0].comparison.target == 0.5


def test_selection_and_abstention_never_imply_correctness_labels():
    exposure, feedback = sample()
    assert preference_examples(exposure, feedback) == ()
    assert len(preference_examples(exposure, feedback, infer_selection=True)) == 5
    abstention = replace(feedback, selected_id=None, abstained=True)
    assert preference_examples(exposure, abstention, infer_selection=True) == ()
    ratings = replace(
        feedback, selected_id=None, ratings={k: 1.0 for k in exposure.displayed_ids}
    )
    assert preference_examples(exposure, ratings) == ()


def test_late_feedback_join_is_idempotent_snapshot_based_and_split_isolated():
    exposure, feedback = sample()
    ledger = PreferenceLedger(capacity=2)
    assert ledger.add_feedback(feedback)
    assert ledger.pending_count == 1
    assert list(ledger.resolved(split="train")) == []
    assert ledger.add_exposure(exposure)
    assert ledger.pending_count == 0
    assert not ledger.add_feedback(feedback)
    assert not ledger.add_exposure(exposure)
    before = exposure_to_payload(next(ledger.resolved(split="train"))[0])
    exposure.observation.candidate_features[:] = 99
    assert exposure_to_payload(next(ledger.resolved(split="train"))[0]) == before
    with pytest.raises(ValueError, match="conflicting"):
        ledger.add_feedback(replace(feedback, received_at_ns=100))
    with pytest.raises(ValueError, match="conflicting"):
        ledger.add_exposure(replace(exposure, split="validation"))
    assert list(ledger.resolved(split="validation")) == []


def test_invalid_pending_feedback_does_not_partially_admit_exposure():
    exposure, feedback = sample()
    ledger = PreferenceLedger()
    ledger.add_feedback(replace(feedback, selected_id="not-shown"))
    with pytest.raises(ValueError, match="not displayed"):
        ledger.add_exposure(exposure)
    assert ledger.pending_count == 1
    assert list(ledger.resolved(split="train")) == []


@pytest.mark.parametrize(
    "bad", ["unknown", "early", "duplicate_pair", "rating", "selection_abstention"]
)
def test_bad_feedback_rejected(bad):
    exposure, feedback = sample()
    a, b = exposure.displayed_ids[:2]
    with pytest.raises(ValueError):
        if bad == "unknown":
            preference_examples(exposure, replace(feedback, selected_id="absent"))
        elif bad == "early":
            preference_examples(replace(exposure, shown_at_ns=100), feedback)
        elif bad == "duplicate_pair":
            replace(
                feedback,
                comparisons=(PairwisePreference(a, b, 1), PairwisePreference(b, a, 1)),
            )
        elif bad == "rating":
            replace(feedback, ratings={a: float("nan")})
        else:
            replace(feedback, abstained=True)


def test_capacity_and_legality_checks():
    exposure, feedback = sample()
    ledger = PreferenceLedger(capacity=1)
    ledger.add_exposure(exposure)
    with pytest.raises(ValueError, match="capacity"):
        ledger.add_exposure(replace(exposure, exposure_id="another"))
    with pytest.raises(ValueError):
        replace(exposure, displayed_ids=("absent",), displayed_scores=(0.0,))
    with pytest.raises(ValueError):
        replace(exposure, display_probability=0.0)
    with pytest.raises(ValueError):
        replace(exposure, displayed_scores=(float("nan"),) * 6)
    masked = replace(
        exposure.observation,
        legal_action_mask=np.array([False, True, True, True, True, True]),
    )
    assert all(
        k != "candidate-0"
        for k, _ in ranked_shortlist(masked, [99, 2, 2, 0, 0, 0], k=2)
    )
    assert ranked_shortlist(masked, [99, 2, 2, 0, 0, 0], k=2) == (
        ("candidate-1", 2.0),
        ("candidate-2", 2.0),
    )


def test_pair_loss_invariance_ties_and_microbatch_gradients():
    a = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
    b = torch.tensor([0.0, 1.0, -0.5], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 0.5])
    weight = torch.tensor([1.0, 0.2, 0.0])
    full = pairwise_preference_loss(a, b, target, weights=weight)
    torch.testing.assert_close(
        full, pairwise_preference_loss(b, a, 1 - target, weights=weight)
    )
    torch.testing.assert_close(
        full, pairwise_preference_loss(a + 4, b + 4, target, weights=weight)
    )
    gradient = torch.autograd.grad(full, (a, b), retain_graph=True)
    chunks = sum(
        pairwise_preference_loss(a[s], b[s], target[s], weights=weight[s])
        * len(a[s])
        / len(a)
        for s in [slice(0, 1), slice(1, 3)]
    )
    for x, y in zip(gradient, torch.autograd.grad(chunks, (a, b))):
        torch.testing.assert_close(x, y)
    tied = torch.zeros(1, requires_grad=True)
    tie_loss = pairwise_preference_loss(tied, torch.zeros(1), torch.tensor([0.5]))
    assert torch.autograd.grad(tie_loss, tied)[0].item() == 0.0
    with pytest.raises(ValueError):
        pairwise_preference_loss(a, b, torch.tensor([float("nan"), 0.0, 1.0]))


def test_batch_loss_rejects_validation_and_preserves_catalog_alignment():
    exposure, feedback = sample()
    a, b = exposure.displayed_ids[:2]
    feedback = replace(
        feedback, selected_id=None, comparisons=(PairwisePreference(a, b, 1.0),)
    )
    examples = preference_examples(exposure, feedback)
    logits = torch.arange(6.0).reshape(1, 6).requires_grad_()
    expected = torch.nn.functional.softplus(
        -(
            logits[0, exposure.observation.candidate_ids.index(a)]
            - logits[0, exposure.observation.candidate_ids.index(b)]
        )
    )
    torch.testing.assert_close(preference_batch_loss(logits, examples), expected)
    invalid = preference_examples(replace(exposure, split="validation"), feedback)
    with pytest.raises(ValueError, match="training"):
        preference_batch_loss(logits, invalid)


def test_user_simulation_keeps_preferences_out_of_policy_inputs():
    rows = generate(17, 16, "train")
    original = features(rows)
    for row in rows:
        row["utility"].fill(float("nan"))
        row["feedback"].clear()
    torch.testing.assert_close(features(rows), original)


def test_empty_entity_shape_survives_json_and_legacy_spec_decoding():
    from jormungandr.structured import (
        entity_candidate_observation_from_payload,
        entity_candidate_observation_to_payload,
        StructuredPolicySpec,
    )

    exposure, _ = sample()
    payload = entity_candidate_observation_to_payload(exposure.observation)
    restored = entity_candidate_observation_from_payload(
        json.loads(json.dumps(payload))
    )
    assert restored.entity_features.shape == (0, 1)
    del payload["entities"]["feature_dim"]
    restored = entity_candidate_observation_from_payload(
        payload, spec=StructuredPolicySpec(4, 1, 4, 1)
    )
    assert restored.entity_features.shape == (0, 1)
    with pytest.raises(ValueError, match="feature_dim"):
        entity_candidate_observation_from_payload(payload)
    payload["entities"]["feature_dim"] = -1
    with pytest.raises(ValueError):
        entity_candidate_observation_from_payload(payload)


def test_shortlist_accepts_native_negative_infinity_for_illegal_candidates():
    exposure, _ = sample()
    masked = replace(
        exposure.observation,
        legal_action_mask=np.array([False, True, True, True, True, True]),
    )
    assert ranked_shortlist(masked, [float("-inf"), 2, 1, 0, 0, 0], k=2) == (
        ("candidate-1", 2.0),
        ("candidate-2", 1.0),
    )
    with pytest.raises(ValueError, match="finite"):
        ranked_shortlist(masked, [0, float("nan"), 1, 0, 0, 0], k=2)
