"""Actor-side n-step returns for finalized, complete structured episodes.

This does not join late reward receipts, reorder events, or impute missing rewards.
"""

from dataclasses import replace
import math
from numbers import Integral
from typing import Sequence

import numpy as np

from jormungandr.structured_replay import StructuredReplayTransition


def _same_observation(a, b):
    return (
        a.candidate_ids == b.candidate_ids
        and a.entity_ids == b.entity_ids
        and all(
            np.array_equal(getattr(a, name), getattr(b, name))
            for name in (
                "global_features",
                "entity_features",
                "entity_type_ids",
                "candidate_features",
                "candidate_entity_indices",
                "legal_action_mask",
            )
        )
    )


def n_step_transitions(
    episode: Sequence[StructuredReplayTransition], *, n: int, gamma: float
) -> tuple[StructuredReplayTransition, ...]:
    """One return per original decision, shortened at termination/truncation.

    Input is one ordered episode starting at timestep zero. All rewards must be
    resolved. `done` means true termination; `truncated` closes the sequence but
    preserves bootstrap from the final observation (never the reset observation).
    Mid-episode policy-version changes are allowed for off-policy learning.
    """
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not episode:
        raise ValueError("a complete episode is required")
    first = episode[0]
    identity = (first.actor_id, first.episode_id, first.split)
    for i, step in enumerate(episode):
        if (
            step.actor_id,
            step.episode_id,
            step.split,
        ) != identity or step.timestep != i:
            raise ValueError("episode identity/split and timesteps must be contiguous")
        if step.bootstrap_steps != 1 or step.return_gamma is not None:
            raise ValueError("input must contain original one-step transitions")
        if i < len(episode) - 1:
            if step.done or step.truncated:
                raise ValueError("episode boundary before final transition")
            if not _same_observation(step.next_observation, episode[i + 1].observation):
                raise ValueError("adjacent transition observations do not match")
    if not (episode[-1].done or episode[-1].truncated):
        raise ValueError("episode must end with termination or explicit truncation")
    output = []
    for start, step in enumerate(episode):
        chunk = episode[start : start + n]
        last = chunk[-1]
        output.append(
            replace(
                step,
                reward=sum(gamma**k * t.reward for k, t in enumerate(chunk)),
                next_observation=last.next_observation,
                done=last.done,
                truncated=last.truncated,
                bootstrap_steps=len(chunk),
                return_gamma=float(gamma),
            )
        )
    return tuple(output)
