from dataclasses import replace

import numpy as np
import pytest

from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_metrics import (
    StructuredSupervisionFrameMetricsAccumulator,
    StructuredSupervisionMetricsAccumulator,
    structured_supervision_frame_policy_metrics,
    structured_supervision_policy_metrics,
)
from jormungandr.structured_supervision import (
    StructuredSupervisionExample,
    StructuredSupervisionFrame,
    StructuredSupervisionLabel,
)


def _example(target: str, *, weight: float = 1.0, prefix=()):
    observation = EntityCandidateObservation(
        global_features=np.asarray([0.0], dtype=np.float32),
        entity_features=np.asarray([[0.0]], dtype=np.float32),
        entity_type_ids=np.asarray([0], dtype=np.int64),
        entity_ids=("entity",),
        candidate_features=np.eye(3, dtype=np.float32),
        candidate_ids=("prefix", "left", "right"),
        legal_action_mask=np.ones(3, dtype=np.bool_),
    )
    return StructuredSupervisionExample(
        observation=observation,
        factor_id="choice",
        candidate_ids=("left", "right"),
        target_candidate_id=target,
        selected_prefix_candidate_ids=prefix,
        actor_id="actor",
        episode_id=f"episode:{target}:{prefix}",
        timestep=0,
        split="validation",
        source_group="source",
        factor_group="choice",
        target_group=target,
        sample_weight=weight,
    )


def test_policy_metrics_are_grouped_weighted_and_streamable() -> None:
    examples = (_example("left", weight=1.0), _example("right", weight=3.0))
    scores = (
        {"candidate_ids": ("prefix", "left", "right"), "candidate_logits": (0, 2, 0)},
        {"candidate_ids": ("prefix", "left", "right"), "candidate_logits": (0, 2, 0)},
    )

    report = structured_supervision_policy_metrics(examples, scores)
    accumulator = StructuredSupervisionMetricsAccumulator()
    for example, score in zip(examples, scores, strict=True):
        accumulator.add(example, score)

    assert report == accumulator.summary()
    assert report["overall"]["accuracy"] == 0.5
    assert report["overall"]["weighted_accuracy"] == 0.25
    assert report["target_macro_accuracy"] == 0.5
    assert report["groups"]["target"]["left"]["accuracy"] == 1.0
    assert report["groups"]["target"]["right"]["accuracy"] == 0.0


def test_policy_metrics_apply_selected_prefix_exactly() -> None:
    example = _example("right", prefix=("prefix",))
    score = {
        "candidate_ids": ("prefix", "left", "right"),
        "candidate_logits": (0.0, 0.0, 0.0),
        "candidate_prefix_keys": ((0.0,), (-1.0,), (1.0,)),
        "candidate_prefix_values": ((2.0,), (0.0,), (0.0,)),
    }

    report = structured_supervision_policy_metrics((example,), (score,))

    assert report["overall"]["accuracy"] == 1.0


def test_policy_metrics_reject_misaligned_scores() -> None:
    example = _example("left")
    with pytest.raises(ValueError, match="align"):
        structured_supervision_policy_metrics((example,), ())
    with pytest.raises(ValueError, match="candidate IDs"):
        structured_supervision_policy_metrics(
            (example,),
            ({"candidate_ids": ("wrong",), "candidate_logits": (0.0,)},),
        )


def _frame(left_target: str, right_target: str) -> StructuredSupervisionFrame:
    example = _example(left_target)
    return StructuredSupervisionFrame(
        actor_id="actor",
        episode_id=f"frame:{left_target}:{right_target}",
        timestep=0,
        observation=example.observation,
        labels=(
            StructuredSupervisionLabel(
                factor_id="first",
                candidate_ids=("left", "right"),
                target_candidate_id=left_target,
                factor_group="choice",
                target_group=left_target,
            ),
            StructuredSupervisionLabel(
                factor_id="second",
                candidate_ids=("left", "right"),
                target_candidate_id=right_target,
                factor_group="choice",
                target_group=right_target,
            ),
        ),
        split="validation",
        source_group="source",
    )


def test_frame_policy_metrics_reuse_one_score_and_report_exact_states() -> None:
    frames = (_frame("left", "left"), _frame("left", "right"))
    scores = (
        {"candidate_ids": ("prefix", "left", "right"), "candidate_logits": (0, 2, 0)},
        {"candidate_ids": ("prefix", "left", "right"), "candidate_logits": (0, 2, 0)},
    )

    report = structured_supervision_frame_policy_metrics(frames, scores)
    accumulator = StructuredSupervisionFrameMetricsAccumulator()
    for frame, score in zip(frames, scores, strict=True):
        accumulator.add(frame, score)

    assert report == accumulator.summary()
    assert report["frames"] == 2
    assert report["examples"] == 4
    assert report["mean_within_frame_accuracy"] == 0.75
    assert report["exact_frame_accuracy"] == 0.5
    assert report["labels"]["overall"]["accuracy"] == 0.75
    assert report["labels"]["target_macro_accuracy"] == 0.5
