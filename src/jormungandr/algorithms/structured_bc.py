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
    EntityCandidateTransformer,
    StructuredPolicySpec,
    collate_entity_candidate_observations,
    select_dynamic_actions,
)
from jormungandr.structured_supervision import StructuredSupervisionExample

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
                logits=output.logits[row].index_select(0, indices)
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
    version="1.1.0",
    family="reward-free structured supervision",
    build=None,
    build_structured=_build_structured,
    representation_modes=("entity_candidates",),
    default_export_module="policy",
    replay_mode="supervision",
    enforce_policy_lag=False,
    description=(
        "Weighted behavior cloning over factor-local semantic candidates with "
        "source/factor/target metrics, explicit raw and weighted scores, and "
        "no reward objective."
    ),
)
algorithm_registry.register(PLUGIN)
