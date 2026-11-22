"""Composable scalar auxiliary objectives for shared-encoder learners.

Applications own supervision, causal boundaries and heads. Losses must be means
across the complete learner batch (including zero-weight/masked examples), so
ordinary microbatch gradient accumulation preserves their scale.
"""

from dataclasses import dataclass
import math
import re
from typing import Iterable

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class AuxiliaryLoss:
    name: str
    loss: torch.Tensor
    coefficient: float = 1.0


def compose_losses(task_loss: torch.Tensor, terms: Iterable[AuxiliaryLoss] = ()):
    """Return a differentiable total and detached, namespaced auxiliary metrics."""
    if task_loss.ndim != 0 or not torch.isfinite(task_loss).item():
        raise ValueError("task loss must be a finite scalar")
    total = task_loss
    metrics = {}
    weighted = 0.0
    seen = set()
    for term in terms:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", term.name) or term.name in seen:
            raise ValueError("auxiliary names must be unique lowercase identifiers")
        seen.add(term.name)
        coefficient = float(term.coefficient)
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("auxiliary coefficient must be finite and nonnegative")
        if term.loss.ndim != 0 or not torch.isfinite(term.loss).item():
            raise ValueError("auxiliary loss must be a finite scalar")
        if term.loss.device != task_loss.device:
            raise ValueError("task and auxiliary losses must share a device")
        value = float(term.loss.detach())
        metrics[f"auxiliary/{term.name}"] = value
        metrics[f"auxiliary_weighted/{term.name}"] = coefficient * value
        if coefficient:
            total = total + coefficient * term.loss
        weighted += coefficient * value
    if seen:
        metrics["auxiliary_total"] = weighted
    return total, metrics


def context_selection_loss(
    logits, positive, *, valid=None, weights=None, margin_clip=5.0
):
    """Two-context logit-margin loss inspired by ContextRL equation (1).

    `positive` selects the supporting context (0 or 1), not an action/reward label.
    Masked items contribute zero; the denominator is the entire batch size, which
    makes uneven microbatches equivalent. Labels and masks are application supplied.
    This primitive alone is not a replication of the paper's GRPO training recipe.
    """
    if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] == 0:
        raise ValueError("context logits must have shape (nonempty batch, 2)")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("context logits must be finite floating-point values")
    if positive.shape != logits.shape[:1] or positive.dtype != torch.long:
        raise ValueError("positive indices must be a matching int64 vector")
    if positive.device != logits.device or ((positive < 0) | (positive > 1)).any():
        raise ValueError("positive context indices must be 0 or 1 on the logits device")
    if not math.isfinite(margin_clip) or margin_clip <= 0:
        raise ValueError("margin_clip must be finite and positive")
    if valid is None:
        valid = torch.ones_like(positive, dtype=torch.bool)
    if (
        valid.shape != positive.shape
        or valid.dtype != torch.bool
        or valid.device != logits.device
    ):
        raise ValueError("valid must be a matching boolean mask on the logits device")
    if weights is not None:
        if (
            weights.shape != positive.shape
            or weights.device != logits.device
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError(
                "weights must be a matching finite nonnegative vector on the logits device"
            )
    chosen = logits.gather(1, positive[:, None]).squeeze(1)
    other = logits.gather(1, (1 - positive)[:, None]).squeeze(1)
    margin = (chosen - other).clamp(-margin_clip, margin_clip)
    losses = F.softplus(-margin)
    losses = torch.where(valid, losses, torch.zeros_like(losses))
    if weights is not None:
        losses = losses * weights
    return losses.mean()
