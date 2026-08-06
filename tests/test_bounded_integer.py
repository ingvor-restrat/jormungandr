from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from jormungandr.bounded_integer import BoundedIntegerRadixCodec
from jormungandr.structured import EntityCandidateObservation
from jormungandr.structured_trajectory import (
    StructuredActionFactor,
    sample_structured_joint_action,
)


def _legal_sequences(
    codec: BoundedIntegerRadixCodec,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
):
    prefixes = [()]
    for _ in range(codec.width):
        prefixes = [
            (*prefix, digit)
            for prefix in prefixes
            for digit in codec.legal_next_digits(
                prefix, minimum=minimum, maximum=maximum
            )
        ]
    return tuple(prefixes)


def test_binary_codec_round_trips_complete_interval_without_flat_vocabulary() -> None:
    codec = BoundedIntegerRadixCodec(1, 1_000_000)

    assert codec.width == 20
    assert codec.categorical_candidate_count == 40
    for value in (1, 2, 17, 999_999, 1_000_000):
        assert codec.decode(codec.encode(value)) == value


def test_prefix_masks_exactly_cover_interval_and_have_no_dead_ends() -> None:
    codec = BoundedIntegerRadixCodec(1, 100)
    sequences = _legal_sequences(codec)
    decoded = tuple(codec.decode(sequence) for sequence in sequences)

    assert decoded == tuple(range(1, 101))
    assert len(set(sequences)) == 100


def test_conditional_subinterval_reuses_same_fixed_factor_topology() -> None:
    codec = BoundedIntegerRadixCodec(0, 255, radix=4)
    sequences = _legal_sequences(codec, minimum=73, maximum=91)

    assert codec.width == 4
    assert tuple(
        codec.decode(sequence, minimum=73, maximum=91)
        for sequence in sequences
    ) == tuple(range(73, 92))


def test_completion_interval_matches_brute_force_digit_suffixes() -> None:
    codec = BoundedIntegerRadixCodec(0, 80, radix=3)
    prefix = (1, 2)
    lower, upper = codec.completion_interval(prefix)
    suffix_width = codec.width - len(prefix)
    values = []
    for suffix in itertools.product(range(codec.radix), repeat=suffix_width):
        digits = (*prefix, *suffix)
        value = 0
        for digit in digits:
            value = value * codec.radix + digit
        values.append(value)

    assert (lower, upper) == (min(values), max(values))


@pytest.mark.parametrize(
    "args,exception",
    [
        ((-1, 3), ValueError),
        ((4, 3), ValueError),
        ((0, 3, 1), ValueError),
        ((0, 3, 37), ValueError),
        ((True, 3), TypeError),
    ],
)
def test_codec_rejects_invalid_domains(args, exception) -> None:
    with pytest.raises(exception):
        BoundedIntegerRadixCodec(*args)


def test_codec_rejects_invalid_values_prefixes_and_subintervals() -> None:
    codec = BoundedIntegerRadixCodec(1, 10)

    with pytest.raises(ValueError, match="outside"):
        codec.encode(0)
    with pytest.raises(ValueError, match="wrong number"):
        codec.decode((1,))
    with pytest.raises(ValueError, match="radix"):
        codec.legal_next_digits((2,))
    with pytest.raises(ValueError, match="subinterval"):
        codec.legal_next_digits((), minimum=0, maximum=5)
    with pytest.raises(ValueError, match="wrong number"):
        codec.legal_next_digits((0,) * codec.width)


def test_codec_drives_exact_autoregressive_sampling_and_joint_log_probability() -> None:
    codec = BoundedIntegerRadixCodec(1, 13)
    candidate_ids = tuple(
        f"digit:{index}:{digit}"
        for index in range(codec.width)
        for digit in range(codec.radix)
    )
    factors = tuple(
        StructuredActionFactor(
            f"digit:{index}",
            tuple(
                f"digit:{index}:{digit}" for digit in range(codec.radix)
            ),
        )
        for index in range(codec.width)
    )
    observation = EntityCandidateObservation(
        global_features=np.asarray([1.0], dtype=np.float32),
        entity_features=np.asarray([[1.0]], dtype=np.float32),
        entity_type_ids=np.asarray([0], dtype=np.int64),
        entity_ids=("bound",),
        candidate_features=np.zeros((len(candidate_ids), 1), dtype=np.float32),
        candidate_ids=candidate_ids,
        legal_action_mask=np.ones(len(candidate_ids), dtype=np.bool_),
    )
    union_index = {value: index for index, value in enumerate(candidate_ids)}

    def restrict_to_interval(index, selected, mask):
        prefix = tuple(int(value.rsplit(":", 1)[1]) for value in selected)
        legal = set(
            codec.legal_next_digits(prefix, minimum=3, maximum=10)
        )
        for digit in range(codec.radix):
            mask[union_index[f"digit:{index}:{digit}"]] = digit in legal
        return mask

    target = codec.encode(7)
    logits = np.asarray(
        [
            4.0 if digit == target[index] else -2.0
            for index in range(codec.width)
            for digit in range(codec.radix)
        ],
        dtype=np.float64,
    )
    selected = sample_structured_joint_action(
        observation,
        factors,
        logits,
        behavior_value=0.0,
        deterministic=True,
        legal_mask_update=restrict_to_interval,
    )
    selected_digits = tuple(
        int(value.rsplit(":", 1)[1])
        for value in selected.selected_candidate_ids
    )

    assert codec.decode(selected_digits, minimum=3, maximum=10) == 7
    assert selected.joint_log_probability == pytest.approx(
        sum(item.behavior_log_probability for item in selected.factors)
    )
    assert math.isfinite(selected.joint_log_probability)

    rng = np.random.default_rng(20260804)
    for _ in range(64):
        sampled = sample_structured_joint_action(
            observation,
            factors,
            np.zeros(len(candidate_ids), dtype=np.float64),
            behavior_value=0.0,
            deterministic=False,
            rng=rng,
            legal_mask_update=restrict_to_interval,
        )
        digits = tuple(
            int(value.rsplit(":", 1)[1])
            for value in sampled.selected_candidate_ids
        )
        assert 3 <= codec.decode(digits) <= 10
        assert math.isfinite(sampled.joint_log_probability)
