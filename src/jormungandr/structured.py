"""Variable-size entity observations and state-dependent action candidates.

This module is deliberately environment agnostic.  An actor remains the
authority for domain entities, candidate semantics, and hard feasibility.  The
learner receives numeric descriptors plus stable, point-in-time identifiers;
it never assumes that candidate slot ``i`` has the same meaning in two states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class StructuredPolicySpec:
    """Feature schema required to build a structured policy plugin."""

    global_dim: int
    entity_dim: int
    candidate_dim: int
    entity_type_count: int

    def __post_init__(self) -> None:
        if min(
            self.global_dim,
            self.entity_dim,
            self.candidate_dim,
            self.entity_type_count,
        ) <= 0:
            raise ValueError("structured policy dimensions must be positive")


def _float_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [features]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _float_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [items, features]")
    if array.shape[1] <= 0:
        raise ValueError(f"{name} must contain at least one feature column")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _identifiers(value: Sequence[str], *, count: int, name: str) -> tuple[str, ...]:
    identifiers = tuple(str(item).strip() for item in value)
    if len(identifiers) != count:
        raise ValueError(f"{name} must contain one identifier per item")
    if any(not item for item in identifiers):
        raise ValueError(f"{name} must not contain empty identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must be unique within one observation")
    return identifiers


@dataclass(frozen=True)
class EntityCandidateObservation:
    """One state represented by variable entities and action candidates.

    Candidate identifiers are local to this observation.  They carry semantic
    identity for audit and execution, while ``candidate_features`` carry the
    numeric description scored by a policy.  A legal mask is still useful for
    padded or speculative candidates, although actors should normally omit
    known no-ops before constructing this object.
    """

    global_features: np.ndarray
    entity_features: np.ndarray
    entity_type_ids: np.ndarray
    entity_ids: tuple[str, ...]
    candidate_features: np.ndarray
    candidate_ids: tuple[str, ...]
    legal_action_mask: np.ndarray
    candidate_entity_indices: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        global_features = _float_vector(
            self.global_features, name="global_features"
        )
        entity_features = _float_matrix(
            self.entity_features, name="entity_features"
        )
        candidate_features = _float_matrix(
            self.candidate_features, name="candidate_features"
        )
        entity_count = entity_features.shape[0]
        candidate_count = candidate_features.shape[0]
        if candidate_count == 0:
            raise ValueError("an observation must contain at least one candidate")

        entity_type_ids = np.asarray(self.entity_type_ids, dtype=np.int64)
        if entity_type_ids.shape != (entity_count,):
            raise ValueError("entity_type_ids must contain one type per entity")
        if np.any(entity_type_ids < 0):
            raise ValueError("entity_type_ids must be non-negative")
        legal_action_mask = np.asarray(self.legal_action_mask, dtype=np.bool_)
        if legal_action_mask.shape != (candidate_count,):
            raise ValueError(
                "legal_action_mask must contain one value per candidate"
            )
        if not legal_action_mask.any():
            raise ValueError("an observation must admit at least one candidate")
        candidate_entity_indices = (
            np.full(candidate_count, -1, dtype=np.int64)
            if self.candidate_entity_indices is None
            else np.asarray(self.candidate_entity_indices, dtype=np.int64)
        )
        if candidate_entity_indices.ndim == 1:
            expected_prefix = candidate_entity_indices.shape == (candidate_count,)
        elif candidate_entity_indices.ndim == 2:
            expected_prefix = (
                candidate_entity_indices.shape[0] == candidate_count
                and candidate_entity_indices.shape[1] > 0
            )
        else:
            expected_prefix = False
        if not expected_prefix:
            raise ValueError(
                "candidate_entity_indices must have shape [candidates] or "
                "[candidates, references]"
            )
        if np.any(candidate_entity_indices < -1) or np.any(
            candidate_entity_indices >= entity_count
        ):
            raise ValueError(
                "candidate entity pointers must be -1 or index an observed entity"
            )

        object.__setattr__(self, "global_features", global_features)
        object.__setattr__(self, "entity_features", entity_features)
        object.__setattr__(self, "entity_type_ids", entity_type_ids.copy())
        object.__setattr__(
            self,
            "entity_ids",
            _identifiers(self.entity_ids, count=entity_count, name="entity_ids"),
        )
        object.__setattr__(self, "candidate_features", candidate_features)
        object.__setattr__(
            self,
            "candidate_ids",
            _identifiers(
                self.candidate_ids,
                count=candidate_count,
                name="candidate_ids",
            ),
        )
        object.__setattr__(self, "legal_action_mask", legal_action_mask.copy())
        object.__setattr__(
            self,
            "candidate_entity_indices",
            candidate_entity_indices.copy(),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def global_dim(self) -> int:
        return int(self.global_features.shape[0])

    @property
    def entity_dim(self) -> int:
        return int(self.entity_features.shape[1])

    @property
    def candidate_dim(self) -> int:
        return int(self.candidate_features.shape[1])


ENTITY_CANDIDATE_SCHEMA = "jormungandr.entity_candidates.v1"


def entity_candidate_observation_to_payload(
    observation: EntityCandidateObservation,
) -> dict[str, Any]:
    """Encode one structured observation as a JSON-compatible wire object."""

    return {
        "schema": ENTITY_CANDIDATE_SCHEMA,
        "global_features": observation.global_features.tolist(),
        "entities": {
            "features": observation.entity_features.tolist(),
            "type_ids": observation.entity_type_ids.tolist(),
            "ids": list(observation.entity_ids),
        },
        "candidates": {
            "features": observation.candidate_features.tolist(),
            "ids": list(observation.candidate_ids),
            "legal_mask": observation.legal_action_mask.tolist(),
            "entity_indices": observation.candidate_entity_indices.tolist(),
        },
        "metadata": dict(observation.metadata),
    }


def entity_candidate_observation_from_payload(
    payload: Mapping[str, Any],
    *,
    spec: StructuredPolicySpec | None = None,
) -> EntityCandidateObservation:
    """Validate and decode one entity/candidate wire object."""

    if not isinstance(payload, Mapping):
        raise ValueError("structured observation must be an object")
    if str(payload.get("schema", "")) != ENTITY_CANDIDATE_SCHEMA:
        raise ValueError(
            f"structured observation schema must be {ENTITY_CANDIDATE_SCHEMA!r}"
        )
    entities = payload.get("entities")
    candidates = payload.get("candidates")
    metadata = payload.get("metadata", {})
    if not isinstance(entities, Mapping):
        raise ValueError("structured observation entities must be an object")
    if not isinstance(candidates, Mapping):
        raise ValueError("structured observation candidates must be an object")
    if not isinstance(metadata, Mapping):
        raise ValueError("structured observation metadata must be an object")
    observation = EntityCandidateObservation(
        global_features=np.asarray(
            payload.get("global_features", []), dtype=np.float32
        ),
        entity_features=np.asarray(
            entities.get("features", []), dtype=np.float32
        ),
        entity_type_ids=np.asarray(
            entities.get("type_ids", []), dtype=np.int64
        ),
        entity_ids=tuple(str(value) for value in entities.get("ids", [])),
        candidate_features=np.asarray(
            candidates.get("features", []), dtype=np.float32
        ),
        candidate_ids=tuple(
            str(value) for value in candidates.get("ids", [])
        ),
        legal_action_mask=np.asarray(
            candidates.get("legal_mask", []), dtype=np.bool_
        ),
        candidate_entity_indices=np.asarray(
            candidates.get(
                "entity_indices",
                [-1] * len(candidates.get("ids", [])),
            ),
            dtype=np.int64,
        ),
        metadata=dict(metadata),
    )
    if spec is not None:
        observed = (
            observation.global_dim,
            observation.entity_dim,
            observation.candidate_dim,
        )
        expected = (spec.global_dim, spec.entity_dim, spec.candidate_dim)
        if observed != expected:
            raise ValueError(
                "structured observation feature dimensions do not match model "
                f"specification: observed={observed}, expected={expected}"
            )
        if np.any(observation.entity_type_ids >= spec.entity_type_count):
            raise ValueError(
                "structured observation entity type exceeds model vocabulary"
            )
    return observation


@dataclass(frozen=True)
class EntityCandidateBatch:
    """NumPy batch padded only at the collation boundary."""

    global_features: np.ndarray
    entity_features: np.ndarray
    entity_type_ids: np.ndarray
    entity_mask: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    legal_action_mask: np.ndarray
    candidate_entity_indices: np.ndarray
    entity_ids: tuple[tuple[str, ...], ...]
    candidate_ids: tuple[tuple[str, ...], ...]

    def to_torch(self, device: str | torch.device = "cpu") -> "TorchEntityCandidateBatch":
        return TorchEntityCandidateBatch(
            global_features=torch.as_tensor(
                self.global_features, dtype=torch.float32, device=device
            ),
            entity_features=torch.as_tensor(
                self.entity_features, dtype=torch.float32, device=device
            ),
            entity_type_ids=torch.as_tensor(
                self.entity_type_ids, dtype=torch.long, device=device
            ),
            entity_mask=torch.as_tensor(
                self.entity_mask, dtype=torch.bool, device=device
            ),
            candidate_features=torch.as_tensor(
                self.candidate_features, dtype=torch.float32, device=device
            ),
            candidate_mask=torch.as_tensor(
                self.candidate_mask, dtype=torch.bool, device=device
            ),
            legal_action_mask=torch.as_tensor(
                self.legal_action_mask, dtype=torch.bool, device=device
            ),
            candidate_entity_indices=torch.as_tensor(
                self.candidate_entity_indices, dtype=torch.long, device=device
            ),
            entity_ids=self.entity_ids,
            candidate_ids=self.candidate_ids,
        )


@dataclass(frozen=True)
class TorchEntityCandidateBatch:
    """Torch counterpart consumed by structured policy modules."""

    global_features: torch.Tensor
    entity_features: torch.Tensor
    entity_type_ids: torch.Tensor
    entity_mask: torch.Tensor
    candidate_features: torch.Tensor
    candidate_mask: torch.Tensor
    legal_action_mask: torch.Tensor
    candidate_entity_indices: torch.Tensor
    entity_ids: tuple[tuple[str, ...], ...]
    candidate_ids: tuple[tuple[str, ...], ...]


def collate_entity_candidate_observations(
    observations: Sequence[EntityCandidateObservation],
) -> EntityCandidateBatch:
    """Pad a heterogeneous list without assigning global action meanings."""

    items = tuple(observations)
    if not items:
        raise ValueError("cannot collate an empty observation sequence")
    dimensions = {
        (item.global_dim, item.entity_dim, item.candidate_dim) for item in items
    }
    if len(dimensions) != 1:
        raise ValueError(
            "all observations in a batch must share feature dimensions"
        )
    global_dim, entity_dim, candidate_dim = dimensions.pop()
    batch_size = len(items)
    max_entities = max(item.entity_features.shape[0] for item in items)
    max_candidates = max(item.candidate_features.shape[0] for item in items)
    max_candidate_references = max(
        1
        if item.candidate_entity_indices.ndim == 1
        else item.candidate_entity_indices.shape[1]
        for item in items
    )

    global_features = np.empty((batch_size, global_dim), dtype=np.float32)
    entity_features = np.zeros(
        (batch_size, max_entities, entity_dim), dtype=np.float32
    )
    entity_type_ids = np.zeros((batch_size, max_entities), dtype=np.int64)
    entity_mask = np.zeros((batch_size, max_entities), dtype=np.bool_)
    candidate_features = np.zeros(
        (batch_size, max_candidates, candidate_dim), dtype=np.float32
    )
    candidate_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    legal_action_mask = np.zeros(
        (batch_size, max_candidates), dtype=np.bool_
    )
    candidate_entity_indices = np.full(
        (batch_size, max_candidates, max_candidate_references),
        -1,
        dtype=np.int64,
    )

    for row, item in enumerate(items):
        entity_count = item.entity_features.shape[0]
        candidate_count = item.candidate_features.shape[0]
        global_features[row] = item.global_features
        entity_features[row, :entity_count] = item.entity_features
        entity_type_ids[row, :entity_count] = item.entity_type_ids
        entity_mask[row, :entity_count] = True
        candidate_features[row, :candidate_count] = item.candidate_features
        candidate_mask[row, :candidate_count] = True
        legal_action_mask[row, :candidate_count] = item.legal_action_mask
        item_entity_indices = item.candidate_entity_indices
        if item_entity_indices.ndim == 1:
            item_entity_indices = item_entity_indices[:, None]
        candidate_entity_indices[
            row,
            :candidate_count,
            : item_entity_indices.shape[1],
        ] = item_entity_indices

    return EntityCandidateBatch(
        global_features=global_features,
        entity_features=entity_features,
        entity_type_ids=entity_type_ids,
        entity_mask=entity_mask,
        candidate_features=candidate_features,
        candidate_mask=candidate_mask,
        legal_action_mask=legal_action_mask,
        candidate_entity_indices=candidate_entity_indices,
        entity_ids=tuple(item.entity_ids for item in items),
        candidate_ids=tuple(item.candidate_ids for item in items),
    )


@dataclass(frozen=True)
class EntityCandidatePolicyOutput:
    """Policy logits over local candidates plus one value per state."""

    logits: torch.Tensor
    values: torch.Tensor


class EntityCandidateTransformer(nn.Module):
    """Small generic transformer for entity sets and dynamic candidates.

    The module defines a useful baseline representation, not a domain policy.
    Candidate descriptors are scored in the context of a transformer-encoded
    entity set.  Padding and actor-declared illegality are excluded exactly.
    """

    def __init__(
        self,
        *,
        global_dim: int,
        entity_dim: int,
        candidate_dim: int,
        entity_type_count: int,
        model_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        spec = StructuredPolicySpec(
            global_dim=global_dim,
            entity_dim=entity_dim,
            candidate_dim=candidate_dim,
            entity_type_count=entity_type_count,
        )
        if model_dim <= 0 or heads <= 0 or model_dim % heads:
            raise ValueError("model_dim must be positive and divisible by heads")
        if layers <= 0 or feedforward_dim <= 0:
            raise ValueError("layers and feedforward_dim must be positive")

        self.spec = spec
        self.global_projection = nn.Linear(spec.global_dim, model_dim)
        self.entity_projection = nn.Linear(spec.entity_dim, model_dim)
        self.entity_type_embedding = nn.Embedding(spec.entity_type_count, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.entity_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.candidate_projection = nn.Linear(spec.candidate_dim, model_dim)
        self.candidate_scorer = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self, batch: TorchEntityCandidateBatch
    ) -> EntityCandidatePolicyOutput:
        if torch.any(batch.entity_type_ids < 0):
            raise ValueError("entity type identifiers must be non-negative")
        if torch.any(batch.entity_type_ids >= self.entity_type_embedding.num_embeddings):
            raise ValueError("entity type identifier exceeds configured vocabulary")
        global_token = self.global_projection(batch.global_features).unsqueeze(1)
        entity_tokens = (
            self.entity_projection(batch.entity_features)
            + self.entity_type_embedding(batch.entity_type_ids)
        )
        tokens = torch.cat((global_token, entity_tokens), dim=1)
        global_mask = torch.ones(
            (batch.entity_mask.shape[0], 1),
            dtype=torch.bool,
            device=batch.entity_mask.device,
        )
        token_mask = torch.cat((global_mask, batch.entity_mask), dim=1)
        encoded = self.entity_encoder(
            tokens, src_key_padding_mask=~token_mask
        )
        context = encoded[:, 0]

        candidate_tokens = self.candidate_projection(batch.candidate_features)
        pointer_indices = batch.candidate_entity_indices
        safe_indices = torch.clamp(pointer_indices, min=0)
        batch_indices = torch.arange(
            encoded.shape[0], device=encoded.device
        )[:, None, None]
        referenced_entities = encoded[:, 1:][batch_indices, safe_indices]
        reference_mask = (
            (pointer_indices >= 0) & batch.candidate_mask.unsqueeze(-1)
        ).unsqueeze(-1)
        referenced_entities = (referenced_entities * reference_mask).sum(dim=2)
        candidate_tokens = candidate_tokens + referenced_entities
        expanded_context = context.unsqueeze(1).expand_as(candidate_tokens)
        logits = self.candidate_scorer(
            torch.cat((expanded_context, candidate_tokens), dim=-1)
        ).squeeze(-1)
        effective_mask = batch.candidate_mask & batch.legal_action_mask
        if not torch.all(effective_mask.any(dim=1)):
            raise ValueError("every batch row must admit at least one candidate")
        logits = logits.masked_fill(~effective_mask, -torch.inf)
        values = self.value_head(context).squeeze(-1)
        return EntityCandidatePolicyOutput(logits=logits, values=values)


@dataclass(frozen=True)
class DynamicActionResult:
    """Selection of a state-local semantic candidate."""

    candidate_id: str
    candidate_index: int
    log_probability: float
    value: float


@dataclass(frozen=True)
class DynamicQActionResult(DynamicActionResult):
    """Dynamic action accompanied by Q values in local candidate order."""

    candidate_values: tuple[float, ...] = ()


def select_dynamic_actions(
    output: EntityCandidatePolicyOutput,
    batch: TorchEntityCandidateBatch,
    *,
    deterministic: bool = False,
) -> tuple[DynamicActionResult, ...]:
    """Select candidates and translate padded slots back to semantic IDs."""

    distribution = torch.distributions.Categorical(logits=output.logits)
    indices = (
        torch.argmax(output.logits, dim=-1)
        if deterministic
        else distribution.sample()
    )
    log_probabilities = distribution.log_prob(indices)
    results: list[DynamicActionResult] = []
    for row, index_tensor in enumerate(indices):
        index = int(index_tensor.item())
        if index >= len(batch.candidate_ids[row]):
            raise RuntimeError("policy selected a padded candidate")
        results.append(
            DynamicActionResult(
                candidate_id=batch.candidate_ids[row][index],
                candidate_index=index,
                log_probability=float(log_probabilities[row].detach().cpu()),
                value=float(output.values[row].detach().cpu()),
            )
        )
    return tuple(results)
