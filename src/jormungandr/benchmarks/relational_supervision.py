"""Synthetic structured-BC benchmark for candidate-conditioned relations.

Each world contains several workers and one destination assigned to each
worker.  A supervision factor asks for the first Manhattan move of one worker
toward its own destination.  Worker/destination pairs share an identity
feature, entity order is shuffled, and validation worlds contain unseen
coordinates.  Solving the task therefore requires a candidate that references
one worker to recover the matching destination from a variable entity set.

The benchmark is intentionally independent of any game or business domain.  It
isolates the architectural distinction between a single pooled entity summary
and candidate-to-entity cross-attention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jormungandr.algorithms.structured_bc import StructuredBCAgent
from jormungandr.structured import (
    EntityCandidateObservation,
    StructuredPolicySpec,
)
from jormungandr.structured_supervision import StructuredSupervisionExample


_DIRECTIONS = (
    ("east", 1.0, 0.0),
    ("west", -1.0, 0.0),
    ("south", 0.0, 1.0),
    ("north", 0.0, -1.0),
)


@dataclass(frozen=True)
class RelationalSupervisionConfig:
    seed: int = 31
    workers: int = 6
    grid_size: int = 11
    train_worlds: int = 192
    validation_worlds: int = 64
    updates: int = 500
    batch_size: int = 128
    model_dim: int = 32
    heads: int = 4
    layers: int = 1
    feedforward_dim: int = 64
    learning_rate: float = 3e-4

    def __post_init__(self) -> None:
        if self.workers < 2:
            raise ValueError("relational benchmark requires at least two workers")
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least three")
        if self.train_worlds <= 0 or self.validation_worlds <= 0:
            raise ValueError("world counts must be positive")
        if self.updates <= 0 or self.batch_size <= 0:
            raise ValueError("updates and batch_size must be positive")


def _target_direction(worker: np.ndarray, destination: np.ndarray) -> str:
    delta = destination - worker
    if int(delta[0]) > 0:
        return "east"
    if int(delta[0]) < 0:
        return "west"
    if int(delta[1]) > 0:
        return "south"
    return "north"


def _world_examples(
    *,
    rng: np.random.Generator,
    config: RelationalSupervisionConfig,
    world_index: int,
    split: str,
) -> tuple[StructuredSupervisionExample, ...]:
    worker_positions = rng.integers(
        0, config.grid_size, size=(config.workers, 2), dtype=np.int64
    )
    destination_positions = rng.integers(
        0, config.grid_size, size=(config.workers, 2), dtype=np.int64
    )
    for index in range(config.workers):
        if np.array_equal(worker_positions[index], destination_positions[index]):
            destination_positions[index, 0] = (
                int(destination_positions[index, 0]) + 1
            ) % config.grid_size

    scale = float(config.grid_size - 1)
    feature_dim = 4 + config.workers
    entities: list[tuple[str, int, np.ndarray]] = []
    for pair_index in range(config.workers):
        identity = np.zeros(config.workers, dtype=np.float32)
        identity[pair_index] = 1.0
        worker_feature = np.concatenate(
            (
                worker_positions[pair_index].astype(np.float32) / scale,
                np.asarray([1.0, 0.0], dtype=np.float32),
                identity,
            )
        )
        destination_feature = np.concatenate(
            (
                destination_positions[pair_index].astype(np.float32) / scale,
                np.asarray([0.0, 1.0], dtype=np.float32),
                identity,
            )
        )
        if worker_feature.shape != (feature_dim,) or destination_feature.shape != (
            feature_dim,
        ):
            raise RuntimeError("relational entity feature width changed")
        entities.extend(
            (
                (f"worker:{pair_index}", 0, worker_feature),
                (f"destination:{pair_index}", 1, destination_feature),
            )
        )
    order = rng.permutation(len(entities))
    ordered = [entities[int(index)] for index in order]
    entity_ids = tuple(item[0] for item in ordered)
    entity_index = {
        identifier: index for index, identifier in enumerate(entity_ids)
    }
    entity_type_ids = np.asarray([item[1] for item in ordered], dtype=np.int64)
    entity_features = np.stack([item[2] for item in ordered]).astype(np.float32)

    candidate_features = np.asarray(
        [
            [float(index == selected) for index in range(len(_DIRECTIONS))]
            + [dx, dy]
            for selected, (_, dx, dy) in enumerate(_DIRECTIONS)
        ],
        dtype=np.float32,
    )
    candidate_ids = tuple(f"move:{name}" for name, _, _ in _DIRECTIONS)
    examples: list[StructuredSupervisionExample] = []
    for pair_index in range(config.workers):
        worker_pointer = entity_index[f"worker:{pair_index}"]
        observation = EntityCandidateObservation(
            global_features=np.asarray([1.0], dtype=np.float32),
            entity_features=entity_features,
            entity_type_ids=entity_type_ids,
            entity_ids=entity_ids,
            candidate_features=candidate_features,
            candidate_ids=candidate_ids,
            legal_action_mask=np.ones(len(candidate_ids), dtype=np.bool_),
            candidate_entity_indices=np.asarray(
                [[worker_pointer] for _ in candidate_ids], dtype=np.int64
            ),
            metadata={
                "benchmark": "relational_assignment_v1",
                "world": world_index,
                "worker": pair_index,
            },
        )
        target = _target_direction(
            worker_positions[pair_index], destination_positions[pair_index]
        )
        examples.append(
            StructuredSupervisionExample(
                observation=observation,
                factor_id=f"worker:{pair_index}",
                candidate_ids=candidate_ids,
                target_candidate_id=f"move:{target}",
                actor_id="relational-benchmark",
                episode_id=f"{split}:world:{world_index}",
                timestep=pair_index,
                split=split,
                source_group="synthetic-relational-oracle",
                factor_group="worker-routing",
                target_group=f"move:{target}",
                balance_group=f"move:{target}",
            )
        )
    return tuple(examples)


def relational_supervision_corpus(
    config: RelationalSupervisionConfig,
) -> tuple[
    StructuredPolicySpec,
    tuple[StructuredSupervisionExample, ...],
    tuple[StructuredSupervisionExample, ...],
]:
    """Build a deterministic train/validation relational corpus."""

    rng = np.random.default_rng(config.seed)
    train = tuple(
        example
        for world in range(config.train_worlds)
        for example in _world_examples(
            rng=rng,
            config=config,
            world_index=world,
            split="train",
        )
    )
    validation = tuple(
        example
        for world in range(config.validation_worlds)
        for example in _world_examples(
            rng=rng,
            config=config,
            world_index=world,
            split="validation",
        )
    )
    spec = StructuredPolicySpec(
        global_dim=1,
        entity_dim=4 + config.workers,
        candidate_dim=6,
        entity_type_count=2,
    )
    return spec, train, validation


def _accuracy(
    agent: StructuredBCAgent,
    examples: Sequence[StructuredSupervisionExample],
    *,
    batch_size: int,
) -> float:
    correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        scored = agent.score_results_structured(
            tuple(example.observation for example in batch)
        )
        for example, result in zip(batch, scored, strict=True):
            by_id = dict(
                zip(result.candidate_ids, result.candidate_logits, strict=True)
            )
            predicted = max(example.candidate_ids, key=by_id.__getitem__)
            correct += int(predicted == example.target_candidate_id)
    return float(correct / len(examples))


def _fit_arm(
    config: RelationalSupervisionConfig,
    *,
    candidate_attention_layers: int,
    spec: StructuredPolicySpec,
    train: Sequence[StructuredSupervisionExample],
    validation: Sequence[StructuredSupervisionExample],
) -> Mapping[str, Any]:
    torch.manual_seed(config.seed)
    # Both arms receive the exact same minibatch indices. Architecture is the
    # only intended arm difference.
    rng = np.random.default_rng(config.seed + 1_000)
    agent = StructuredBCAgent(
        spec,
        {
            "structured_model_dim": config.model_dim,
            "structured_heads": config.heads,
            "structured_layers": config.layers,
            "structured_feedforward_dim": config.feedforward_dim,
            "structured_candidate_attention_layers": candidate_attention_layers,
            "structured_dropout": 0.0,
            "lr": config.learning_rate,
            "max_grad": 1.0,
        },
        "cpu",
    )
    initial_validation_accuracy = _accuracy(
        agent, validation, batch_size=config.batch_size
    )
    last_loss = float("nan")
    for _ in range(config.updates):
        indices = rng.choice(len(train), size=config.batch_size, replace=True)
        result = agent.update_structured_supervision(
            tuple(train[int(index)] for index in indices)
        )
        last_loss = result.loss
    return {
        "candidate_attention_layers": candidate_attention_layers,
        "trainable_parameters": sum(
            parameter.numel() for parameter in agent.policy.parameters()
        ),
        "updates": config.updates,
        "last_batch_loss": last_loss,
        "initial_validation_accuracy": initial_validation_accuracy,
        "train_accuracy": _accuracy(
            agent, train, batch_size=config.batch_size
        ),
        "validation_accuracy": _accuracy(
            agent, validation, batch_size=config.batch_size
        ),
    }


def run_relational_supervision_benchmark(
    config: RelationalSupervisionConfig = RelationalSupervisionConfig(),
) -> Mapping[str, Any]:
    """Compare pooled-context and candidate-attention BC under one budget."""

    spec, train, validation = relational_supervision_corpus(config)
    pooled = _fit_arm(
        config,
        candidate_attention_layers=0,
        spec=spec,
        train=train,
        validation=validation,
    )
    relational = _fit_arm(
        config,
        candidate_attention_layers=1,
        spec=spec,
        train=train,
        validation=validation,
    )
    return {
        "schema": "jormungandr.relational_supervision_benchmark.v1",
        "question": (
            "Can a candidate referring to one entity recover its matching "
            "entity from a shuffled variable set?"
        ),
        "config": asdict(config),
        "corpus": {
            "train_examples": len(train),
            "validation_examples": len(validation),
            "unseen_validation_worlds": True,
            "shuffled_entity_order": True,
        },
        "arms": {
            "pooled_context": pooled,
            "candidate_entity_attention": relational,
        },
        "validation_accuracy_delta": float(
            relational["validation_accuracy"] - pooled["validation_accuracy"]
        ),
    }


__all__ = [
    "RelationalSupervisionConfig",
    "relational_supervision_corpus",
    "run_relational_supervision_benchmark",
]
