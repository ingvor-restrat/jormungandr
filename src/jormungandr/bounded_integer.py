"""Compact exact encodings for bounded integer action parameters.

Structured policies already know how to sample a sequence of conditionally
masked categorical factors and retain the exact joint log probability.  This
module turns a bounded integer into a fixed-width radix sequence so callers do
not need one categorical candidate per numeric value.  A binary codec, for
example, represents every value through one million with twenty two-way
decisions.

The codec owns arithmetic only.  Environments still own the bounds and may
select a narrower interval for each conditional action prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class BoundedIntegerRadixCodec:
    """Encode an inclusive non-negative interval as most-significant digits.

    ``legal_next_digits`` is the important actor-facing operation.  It returns
    only digits whose numeric subtree intersects the requested conditional
    interval.  Repeating that operation before every sampled digit therefore
    has no dead ends and exactly covers the interval.
    """

    minimum: int
    maximum: int
    radix: int = 2

    def __post_init__(self) -> None:
        minimum = _integer(self.minimum, name="minimum")
        maximum = _integer(self.maximum, name="maximum")
        radix = _integer(self.radix, name="radix")
        if minimum < 0:
            raise ValueError("minimum must be non-negative")
        if maximum < minimum:
            raise ValueError("maximum must be at least minimum")
        if not 2 <= radix <= 36:
            raise ValueError("radix must be between 2 and 36")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "radix", radix)

    @property
    def width(self) -> int:
        """Fixed digit count needed to include the configured maximum."""

        width = 1
        capacity = self.radix
        while self.maximum >= capacity:
            width += 1
            capacity *= self.radix
        return width

    @property
    def categorical_candidate_count(self) -> int:
        """Total candidates exposed across all fixed-width digit factors."""

        return self.width * self.radix

    def _conditional_bounds(
        self,
        minimum: int | None,
        maximum: int | None,
    ) -> tuple[int, int]:
        lower = self.minimum if minimum is None else _integer(
            minimum, name="conditional minimum"
        )
        upper = self.maximum if maximum is None else _integer(
            maximum, name="conditional maximum"
        )
        if lower < self.minimum or upper > self.maximum or upper < lower:
            raise ValueError(
                "conditional bounds must be a non-empty subinterval of the codec"
            )
        return lower, upper

    def _digits(self, digits: Sequence[int], *, allow_prefix: bool) -> tuple[int, ...]:
        normalized = tuple(
            _integer(value, name="digit") for value in digits
        )
        limit = self.width if allow_prefix else self.width - 1
        if len(normalized) > limit:
            qualifier = "prefix" if allow_prefix else "encoded value"
            raise ValueError(f"{qualifier} has the wrong number of digits")
        if any(value < 0 or value >= self.radix for value in normalized):
            raise ValueError("digit is outside the codec radix")
        return normalized

    def encode(self, value: int) -> tuple[int, ...]:
        """Return the unique fixed-width digit sequence for one legal value."""

        number = _integer(value, name="value")
        if not self.minimum <= number <= self.maximum:
            raise ValueError("value is outside the codec interval")
        digits = [0] * self.width
        remaining = number
        for index in range(self.width - 1, -1, -1):
            digits[index] = remaining % self.radix
            remaining //= self.radix
        if remaining:
            raise RuntimeError("codec width cannot represent its configured maximum")
        return tuple(digits)

    def decode(
        self,
        digits: Sequence[int],
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        """Decode a complete sequence and validate optional conditional bounds."""

        normalized = tuple(
            _integer(value, name="digit") for value in digits
        )
        if len(normalized) != self.width:
            raise ValueError("encoded value has the wrong number of digits")
        if any(value < 0 or value >= self.radix for value in normalized):
            raise ValueError("digit is outside the codec radix")
        value = 0
        for digit in normalized:
            value = value * self.radix + digit
        lower, upper = self._conditional_bounds(minimum, maximum)
        if not lower <= value <= upper:
            raise ValueError("decoded value is outside the conditional interval")
        return value

    def completion_interval(self, prefix: Sequence[int]) -> tuple[int, int]:
        """Return the smallest and largest values below one digit prefix."""

        normalized = self._digits(prefix, allow_prefix=True)
        prefix_value = 0
        for digit in normalized:
            prefix_value = prefix_value * self.radix + digit
        remaining = self.width - len(normalized)
        scale = self.radix**remaining
        return prefix_value * scale, (prefix_value + 1) * scale - 1

    def legal_next_digits(
        self,
        prefix: Sequence[int],
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> tuple[int, ...]:
        """Return digits with at least one legal completion after ``prefix``."""

        normalized = self._digits(prefix, allow_prefix=False)
        lower, upper = self._conditional_bounds(minimum, maximum)
        legal: list[int] = []
        for digit in range(self.radix):
            completion_lower, completion_upper = self.completion_interval(
                (*normalized, digit)
            )
            if completion_upper >= lower and completion_lower <= upper:
                legal.append(digit)
        return tuple(legal)

