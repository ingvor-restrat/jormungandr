"""Weighted behavior cloning for variable state-local action factors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jormungandr.structured import (
    DynamicActionResult,
    EntityCandidateObservation,
    EntityCandidatePolicyOutput,
    EntityCandidateTransformer,
    StructuredPolicySpec,
    apply_candidate_prefix,
    collate_entity_candidate_observations,
    select_dynamic_actions,
)
from jormungandr.structured_supervision import (
    StructuredSupervisionExample,
    StructuredSupervisionFrame,
)

from .base import AlgorithmPlugin
from .common import cfg, optimizer_to
from .registry import algorithm_registry
from .structured_ppo import StructuredPolicyScore, _resolve_device


@dataclass(frozen=True)
class StructuredBCUpdate:
    examples: int
    loss: float
    nll: float
    accuracy: float
    entropy: float
    calibration_error: float
    gradient_norm: float
    metrics: Mapping[str, float]


def _group_key(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip())
    return normalized or "unnamed"


class StructuredBCAgent:
    """Categorical NLL over semantic candidates; rewards are not an input."""

    def __init__(
        self,
        spec: StructuredPolicySpec,
        config: Mapping[str, Any],
        device: str = "auto",
    ) -> None:
        self.spec = spec
        self.device = _resolve_device(device)
        model_dim = max(8, int(cfg(config, "structured_model_dim", 64)))
        heads = max(1, int(cfg(config, "structured_heads", 4)))
        layers = max(1, int(cfg(config, "structured_layers", 2)))
        feedforward = max(
            model_dim,
            int(cfg(config, "structured_feedforward_dim", model_dim * 2)),
        )
        self.policy = EntityCandidateTransformer(
            global_dim=spec.global_dim,
            entity_dim=spec.entity_dim,
            candidate_dim=spec.candidate_dim,
            entity_type_count=spec.entity_type_count,
            model_dim=model_dim,
            heads=heads,
            layers=layers,
            feedforward_dim=feedforward,
            dropout=max(0.0, float(cfg(config, "structured_dropout", 0.0))),
            prefix_dim=max(0, int(cfg(config, "structured_prefix_dim", 0))),
            candidate_attention_layers=max(
                0,
                int(cfg(config, "structured_candidate_attention_layers", 0)),
            ),
        ).to(self.device)
        self.max_grad_norm = max(0.0, float(cfg(config, "max_grad", 1.0)))
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=float(cfg(config, "lr", 3e-4))
        )
        self.update_steps = 0
        self.last_metrics: Mapping[str, float] = {}

    def action_results_structured(
        self,
        observations: Sequence[EntityCandidateObservation],
        *,
        deterministic: bool,
    ) -> tuple[DynamicActionResult, ...]:
        if not observations:
            return ()
        was_training = self.policy.training
        self.policy.eval()
        batch = collate_entity_candidate_observations(observations).to_torch(
            self.device
        )
        with torch.no_grad():
            output = self.policy(batch)
            results = select_dynamic_actions(
                output, batch, deterministic=deterministic
            )
        self.policy.train(was_training)
        return results

    def action_result_structured(
        self,
        observation: EntityCandidateObservation,
        *,
        deterministic: bool,
    ) -> DynamicActionResult:
        return self.action_results_structured(
            (observation,), deterministic=deterministic
        )[0]

    def score_results_structured(
        self,
        observations: Sequence[EntityCandidateObservation],
    ) -> tuple[StructuredPolicyScore, ...]:
        if not observations:
            return ()
        was_training = self.policy.training
        self.policy.eval()
        batch = collate_entity_candidate_observations(observations).to_torch(
            self.device
        )
        with torch.no_grad():
            output = self.policy(batch)
        results = tuple(
            StructuredPolicyScore(
                candidate_ids=observation.candidate_ids,
                candidate_logits=tuple(
                    float(value)
                    for value in output.logits[row, : len(observation.candidate_ids)]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                value=float(output.values[row].detach().cpu()),
                candidate_prefix_keys=(
                    tuple(
                        tuple(float(component) for component in vector)
                        for vector in output.candidate_prefix_keys[
                            row, : len(observation.candidate_ids)
                        ].detach().cpu().tolist()
                    )
                    if output.candidate_prefix_keys is not None
                    else ()
                ),
                candidate_prefix_values=(
                    tuple(
                        tuple(float(component) for component in vector)
                        for vector in output.candidate_prefix_values[
                            row, : len(observation.candidate_ids)
                        ].detach().cpu().tolist()
                    )
                    if output.candidate_prefix_values is not None
                    else ()
                ),
            )
            for row, observation in enumerate(observations)
        )
        self.policy.train(was_training)
        return results

    @staticmethod
    def _calibration_error(
        confidence: np.ndarray, correct: np.ndarray, bins: int = 10
    ) -> float:
        result = 0.0
        edges = np.linspace(0.0, 1.0, bins + 1)
        for index in range(bins):
            selected = (confidence > edges[index]) & (
                confidence <= edges[index + 1]
            )
            if index == 0:
                selected |= confidence == 0.0
            if selected.any():
                result += float(selected.mean()) * abs(
                    float(confidence[selected].mean())
                    - float(correct[selected].mean())
                )
        return result

    def _forward_examples(
        self,
        examples: Sequence[StructuredSupervisionExample],
    ) -> tuple[torch.Tensor, Mapping[str, float]]:
        items = tuple(examples)
        if not items:
            raise ValueError("at least one supervision example is required")
        if any(item.split != items[0].split for item in items):
            raise ValueError("one supervision batch cannot mix train and validation")
        batch = collate_entity_candidate_observations(
            [item.observation for item in items]
        ).to_torch(self.device)
        output = self.policy(batch)
        maximum_prefix = max(
            1, max(len(item.selected_prefix_candidate_ids) for item in items)
        )
        prefix_indices = np.full(
            (len(items), maximum_prefix), -1, dtype=np.int64
        )
        for row, item in enumerate(items):
            by_id = {
                candidate_id: index
                for index, candidate_id in enumerate(
                    item.observation.candidate_ids
                )
            }
            prefix = [
                by_id[candidate_id]
                for candidate_id in item.selected_prefix_candidate_ids
            ]
            prefix_indices[row, : len(prefix)] = prefix
        conditioned_logits = apply_candidate_prefix(
            output,
            torch.as_tensor(
                prefix_indices, dtype=torch.long, device=self.device
            ),
        )
        nlls = []
        entropies = []
        confidences = []
        correct = []
        for row, item in enumerate(items):
            by_id = {
                candidate_id: index
                for index, candidate_id in enumerate(item.observation.candidate_ids)
            }
            indices = torch.as_tensor(
                [by_id[candidate_id] for candidate_id in item.candidate_ids],
                dtype=torch.long,
                device=self.device,
            )
            distribution = torch.distributions.Categorical(
                logits=conditioned_logits[row].index_select(0, indices)
            )
            target = item.candidate_ids.index(item.target_candidate_id)
            target_tensor = torch.as_tensor(target, device=self.device)
            nlls.append(-distribution.log_prob(target_tensor))
            entropies.append(distribution.entropy())
            probabilities = distribution.probs
            prediction = int(torch.argmax(probabilities).item())
            confidences.append(float(probabilities[prediction].detach().cpu()))
            correct.append(float(prediction == target))
        nll = torch.stack(nlls)
        entropy = torch.stack(entropies)
        weights = torch.as_tensor(
            [item.sample_weight for item in items],
            dtype=torch.float32,
            device=self.device,
        )
        loss = (weights * nll).sum() / weights.sum()
        nll_values = nll.detach().cpu().numpy()
        entropy_values = entropy.detach().cpu().numpy()
        confidence_values = np.asarray(confidences, dtype=np.float64)
        correct_values = np.asarray(correct, dtype=np.float64)
        sample_weights = weights.detach().cpu().numpy()

        def raw_mean(values: np.ndarray, mask: np.ndarray) -> float:
            return float(np.mean(values[mask]))

        def weighted_mean(values: np.ndarray, mask: np.ndarray) -> float:
            selected_weights = sample_weights[mask]
            return float(
                np.sum(values[mask] * selected_weights)
                / np.sum(selected_weights)
            )

        all_items = np.ones(len(items), dtype=bool)
        metrics: dict[str, float] = {
            # The ordinary names describe the empirical corpus.  Optimization
            # metrics are explicit so balancing cannot make a rare-class
            # weighted score look like raw imitation fidelity.
            "nll": raw_mean(nll_values, all_items),
            "accuracy": raw_mean(correct_values, all_items),
            "entropy": raw_mean(entropy_values, all_items),
            "weighted_nll": weighted_mean(nll_values, all_items),
            "weighted_accuracy": weighted_mean(correct_values, all_items),
            "weighted_entropy": weighted_mean(entropy_values, all_items),
            "calibration_error": self._calibration_error(
                confidence_values, correct_values
            ),
            "sample_weight_mean": float(np.mean(sample_weights)),
            "sample_weight_min": float(np.min(sample_weights)),
            "sample_weight_max": float(np.max(sample_weights)),
        }
        for group_name, values in (
            ("source", [item.source_group for item in items]),
            ("factor", [item.factor_group for item in items]),
            ("target", [item.target_group for item in items]),
        ):
            for group in sorted(set(values)):
                mask = np.asarray([value == group for value in values], dtype=bool)
                prefix = f"group/{group_name}/{_group_key(group)}"
                metrics[f"{prefix}/nll"] = raw_mean(nll_values, mask)
                metrics[f"{prefix}/accuracy"] = raw_mean(correct_values, mask)
                metrics[f"{prefix}/entropy"] = raw_mean(entropy_values, mask)
                metrics[f"{prefix}/weighted_nll"] = weighted_mean(
                    nll_values, mask
                )
                metrics[f"{prefix}/weighted_accuracy"] = weighted_mean(
                    correct_values, mask
                )
                metrics[f"{prefix}/weighted_entropy"] = weighted_mean(
                    entropy_values, mask
                )
        return loss, metrics

    def _forward_frames(
        self,
        frames: Sequence[StructuredSupervisionFrame],
    ) -> tuple[torch.Tensor, Mapping[str, float], int]:
        """Score each shared observation once and supervise all of its labels."""

        items = tuple(frames)
        if not items:
            raise ValueError("at least one supervision frame is required")
        if any(item.split != items[0].split for item in items):
            raise ValueError("one supervision batch cannot mix train and validation")
        batch = collate_entity_candidate_observations(
            [item.observation for item in items]
        ).to_torch(self.device)
        output = self.policy(batch)
        flattened = tuple(
            (row, frame, label)
            for row, frame in enumerate(items)
            for label in frame.labels
        )
        row_indices = torch.as_tensor(
            [row for row, _, _ in flattened],
            dtype=torch.long,
            device=self.device,
        )
        expanded_output = EntityCandidatePolicyOutput(
            logits=output.logits.index_select(0, row_indices),
            values=output.values.index_select(0, row_indices),
            candidate_prefix_keys=(
                None
                if output.candidate_prefix_keys is None
                else output.candidate_prefix_keys.index_select(0, row_indices)
            ),
            candidate_prefix_values=(
                None
                if output.candidate_prefix_values is None
                else output.candidate_prefix_values.index_select(0, row_indices)
            ),
        )
        maximum_prefix = max(
            1,
            max(
                len(label.selected_prefix_candidate_ids)
                for _, _, label in flattened
            ),
        )
        prefix_indices = np.full(
            (len(flattened), maximum_prefix), -1, dtype=np.int64
        )
        for label_row, (_, frame, label) in enumerate(flattened):
            by_id = {
                candidate_id: index
                for index, candidate_id in enumerate(
                    frame.observation.candidate_ids
                )
            }
            prefix = [
                by_id[value] for value in label.selected_prefix_candidate_ids
            ]
            prefix_indices[label_row, : len(prefix)] = prefix
        conditioned_logits = apply_candidate_prefix(
            expanded_output,
            torch.as_tensor(
                prefix_indices, dtype=torch.long, device=self.device
            ),
        )

        nlls = []
        entropies = []
        confidences = []
        correct = []
        for label_row, (_, frame, label) in enumerate(flattened):
            by_id = {
                candidate_id: index
                for index, candidate_id in enumerate(
                    frame.observation.candidate_ids
                )
            }
            indices = torch.as_tensor(
                [by_id[value] for value in label.candidate_ids],
                dtype=torch.long,
                device=self.device,
            )
            distribution = torch.distributions.Categorical(
                logits=conditioned_logits[label_row].index_select(0, indices)
            )
            target = label.candidate_ids.index(label.target_candidate_id)
            target_tensor = torch.as_tensor(target, device=self.device)
            nlls.append(-distribution.log_prob(target_tensor))
            entropies.append(distribution.entropy())
            probabilities = distribution.probs
            prediction = int(torch.argmax(probabilities).item())
            confidences.append(float(probabilities[prediction].detach().cpu()))
            correct.append(float(prediction == target))

        nll = torch.stack(nlls)
        entropy = torch.stack(entropies)
        weights = torch.as_tensor(
            [label.sample_weight for _, _, label in flattened],
            dtype=torch.float32,
            device=self.device,
        )
        loss = (weights * nll).sum() / weights.sum()
        nll_values = nll.detach().cpu().numpy()
        entropy_values = entropy.detach().cpu().numpy()
        confidence_values = np.asarray(confidences, dtype=np.float64)
        correct_values = np.asarray(correct, dtype=np.float64)
        sample_weights = weights.detach().cpu().numpy()

        def raw_mean(values: np.ndarray, mask: np.ndarray) -> float:
            return float(np.mean(values[mask]))

        def weighted_mean(values: np.ndarray, mask: np.ndarray) -> float:
            selected_weights = sample_weights[mask]
            return float(
                np.sum(values[mask] * selected_weights)
                / np.sum(selected_weights)
            )

        all_labels = np.ones(len(flattened), dtype=bool)
        metrics: dict[str, float] = {
            "nll": raw_mean(nll_values, all_labels),
            "accuracy": raw_mean(correct_values, all_labels),
            "entropy": raw_mean(entropy_values, all_labels),
            "weighted_nll": weighted_mean(nll_values, all_labels),
            "weighted_accuracy": weighted_mean(correct_values, all_labels),
            "weighted_entropy": weighted_mean(entropy_values, all_labels),
            "calibration_error": self._calibration_error(
                confidence_values, correct_values
            ),
            "sample_weight_mean": float(np.mean(sample_weights)),
            "sample_weight_min": float(np.min(sample_weights)),
            "sample_weight_max": float(np.max(sample_weights)),
            "frames": float(len(items)),
            "labels_per_frame": float(len(flattened) / len(items)),
        }
        for group_name, values in (
            ("source", [frame.source_group for _, frame, _ in flattened]),
            ("factor", [label.factor_group for _, _, label in flattened]),
            ("target", [label.target_group for _, _, label in flattened]),
        ):
            for group in sorted(set(values)):
                mask = np.asarray([value == group for value in values], dtype=bool)
                prefix = f"group/{group_name}/{_group_key(group)}"
                metrics[f"{prefix}/nll"] = raw_mean(nll_values, mask)
                metrics[f"{prefix}/accuracy"] = raw_mean(correct_values, mask)
                metrics[f"{prefix}/entropy"] = raw_mean(entropy_values, mask)
                metrics[f"{prefix}/weighted_nll"] = weighted_mean(
                    nll_values, mask
                )
                metrics[f"{prefix}/weighted_accuracy"] = weighted_mean(
                    correct_values, mask
                )
                metrics[f"{prefix}/weighted_entropy"] = weighted_mean(
                    entropy_values, mask
                )
        return loss, metrics, len(flattened)

    def update_structured_supervision(
        self,
        examples: Sequence[StructuredSupervisionExample],
    ) -> StructuredBCUpdate:
        items = tuple(examples)
        if any(item.split != "train" for item in items):
            raise ValueError("validation supervision cannot update weights")
        self.policy.train()
        loss, metrics = self._forward_examples(items)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.max_grad_norm if self.max_grad_norm > 0.0 else float("inf"),
        )
        self.optimizer.step()
        self.update_steps += 1
        combined = {
            **metrics,
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        result = StructuredBCUpdate(
            examples=len(items),
            loss=float(loss.detach().cpu()),
            nll=metrics["nll"],
            accuracy=metrics["accuracy"],
            entropy=metrics["entropy"],
            calibration_error=metrics["calibration_error"],
            gradient_norm=combined["gradient_norm"],
            metrics=combined,
        )
        self.last_metrics = {
            key: float(value)
            for key, value in combined.items()
            if np.isfinite(float(value))
        }
        return result

    def evaluate_structured_supervision(
        self,
        examples: Sequence[StructuredSupervisionExample],
    ) -> StructuredBCUpdate:
        items = tuple(examples)
        if any(item.split != "validation" for item in items):
            raise ValueError("evaluation requires validation supervision")
        was_training = self.policy.training
        self.policy.eval()
        with torch.no_grad():
            loss, metrics = self._forward_examples(items)
        self.policy.train(was_training)
        return StructuredBCUpdate(
            examples=len(items),
            loss=float(loss.detach().cpu()),
            nll=metrics["nll"],
            accuracy=metrics["accuracy"],
            entropy=metrics["entropy"],
            calibration_error=metrics["calibration_error"],
            gradient_norm=0.0,
            metrics=metrics,
        )

    def update_structured_supervision_frames(
        self,
        frames: Sequence[StructuredSupervisionFrame],
    ) -> StructuredBCUpdate:
        items = tuple(frames)
        if any(item.split != "train" for item in items):
            raise ValueError("validation supervision cannot update weights")
        self.policy.train()
        loss, metrics, labels = self._forward_frames(items)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.max_grad_norm if self.max_grad_norm > 0.0 else float("inf"),
        )
        self.optimizer.step()
        self.update_steps += 1
        combined = {
            **metrics,
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        self.last_metrics = {
            key: float(value)
            for key, value in combined.items()
            if np.isfinite(float(value))
        }
        return StructuredBCUpdate(
            examples=labels,
            loss=float(loss.detach().cpu()),
            nll=metrics["nll"],
            accuracy=metrics["accuracy"],
            entropy=metrics["entropy"],
            calibration_error=metrics["calibration_error"],
            gradient_norm=combined["gradient_norm"],
            metrics=combined,
        )

    def evaluate_structured_supervision_frames(
        self,
        frames: Sequence[StructuredSupervisionFrame],
    ) -> StructuredBCUpdate:
        items = tuple(frames)
        if any(item.split != "validation" for item in items):
            raise ValueError("evaluation requires validation supervision")
        was_training = self.policy.training
        self.policy.eval()
        with torch.no_grad():
            loss, metrics, labels = self._forward_frames(items)
        self.policy.train(was_training)
        return StructuredBCUpdate(
            examples=labels,
            loss=float(loss.detach().cpu()),
            nll=metrics["nll"],
            accuracy=metrics["accuracy"],
            entropy=metrics["entropy"],
            calibration_error=metrics["calibration_error"],
            gradient_norm=0.0,
            metrics=metrics,
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "spec": asdict(self.spec),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
            "last_metrics": dict(self.last_metrics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_spec = state.get("spec")
        if isinstance(saved_spec, Mapping) and dict(saved_spec) != asdict(self.spec):
            raise ValueError("structured checkpoint feature schema does not match")
        self.policy.load_state_dict(state["policy"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            optimizer_to(self.optimizer, self.device)
        self.update_steps = int(state.get("update_steps", 0))
        self.last_metrics = {
            str(key): float(value)
            for key, value in dict(state.get("last_metrics", {})).items()
        }


def _build_structured(spec, config, device):
    return StructuredBCAgent(spec, config, device)


PLUGIN = AlgorithmPlugin(
    name="structured_bc",
    version="1.3.0",
    family="reward-free structured supervision",
    build=None,
    build_structured=_build_structured,
    representation_modes=("entity_candidates",),
    default_export_module="policy",
    replay_mode="supervision",
    enforce_policy_lag=False,
    description=(
        "Weighted behavior cloning over factor-local semantic candidates with "
        "source/factor/target metrics, optional selected-prefix conditioning, "
        "shared-observation multi-label frames, explicit raw and weighted "
        "scores, and no reward objective."
    ),
)
algorithm_registry.register(PLUGIN)
