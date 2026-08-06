"""Off-policy Double-DQN over variable entities and local candidates."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jormungandr.structured import (
    DynamicQActionResult,
    EntityCandidateObservation,
    EntityCandidateTransformer,
    StructuredPolicySpec,
    collate_entity_candidate_observations,
)
from jormungandr.structured_replay import StructuredReplayTransition

from .base import AlgorithmPlugin, UpdateResult
from .common import cfg, optimizer_to
from .registry import algorithm_registry


def _resolve_device(value: str) -> torch.device:
    raw = str(value).strip().lower()
    if raw in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


class StructuredDQNAgent:
    """Double-DQN whose Q slots are state-local semantic candidates."""

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
        self.q = EntityCandidateTransformer(
            global_dim=spec.global_dim,
            entity_dim=spec.entity_dim,
            candidate_dim=spec.candidate_dim,
            entity_type_count=spec.entity_type_count,
            model_dim=model_dim,
            heads=heads,
            layers=layers,
            feedforward_dim=feedforward,
            dropout=max(0.0, float(cfg(config, "structured_dropout", 0.0))),
            candidate_attention_layers=max(
                0,
                int(cfg(config, "structured_candidate_attention_layers", 0)),
            ),
        ).to(self.device)
        self.q_target = deepcopy(self.q).to(self.device)
        self.q_target.eval()
        for parameter in self.q_target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.q.parameters(), lr=float(cfg(config, "lr", 3e-4))
        )
        self.gamma = min(1.0, max(0.0, float(cfg(config, "gamma", 0.99))))
        self.target_update = max(1, int(cfg(config, "target_update", 500)))
        self.max_grad_norm = max(0.0, float(cfg(config, "max_grad", 1.0)))
        self.huber_delta = max(1e-6, float(cfg(config, "huber_delta", 1.0)))
        self.default_epsilon = min(
            1.0, max(0.0, float(cfg(config, "epsilon", 0.1)))
        )
        self.demonstration_margin = max(
            0.0, float(cfg(config, "demonstration_margin", 0.2))
        )
        self.demonstration_weight = max(
            0.0, float(cfg(config, "demonstration_weight", 1.0))
        )
        self.update_steps = 0
        self.last_metrics: Mapping[str, float] = {}

    def action_results_structured(
        self,
        observations: Sequence[EntityCandidateObservation],
        *,
        deterministic: bool,
        epsilon: float | None = None,
    ) -> tuple[DynamicQActionResult, ...]:
        if not observations:
            return ()
        exploration = (
            0.0
            if deterministic
            else min(
                1.0,
                max(
                    0.0,
                    self.default_epsilon if epsilon is None else float(epsilon),
                ),
            )
        )
        was_training = self.q.training
        self.q.eval()
        batch = collate_entity_candidate_observations(observations).to_torch(
            self.device
        )
        with torch.no_grad():
            values = self.q(batch).logits
            greedy = torch.argmax(values, dim=-1)
            selected = greedy.clone()
            if exploration > 0.0:
                explore_rows = torch.rand(
                    len(observations), device=self.device
                ) < exploration
                effective_mask = batch.candidate_mask & batch.legal_action_mask
                for row in torch.nonzero(explore_rows, as_tuple=False).flatten():
                    legal = torch.nonzero(
                        effective_mask[int(row)], as_tuple=False
                    ).flatten()
                    choice = torch.randint(
                        0, len(legal), (1,), device=self.device
                    )
                    selected[int(row)] = legal[choice]
        self.q.train(was_training)

        results = []
        for row, index_tensor in enumerate(selected):
            index = int(index_tensor.item())
            candidate_count = len(batch.candidate_ids[row])
            if index >= candidate_count:
                raise RuntimeError("Q policy selected a padded candidate")
            legal_count = int(
                batch.legal_action_mask[row, :candidate_count].sum().item()
            )
            probability = exploration / max(1, legal_count)
            if index == int(greedy[row].item()):
                probability += 1.0 - exploration
            row_values = values[row, :candidate_count].detach().cpu().tolist()
            results.append(
                DynamicQActionResult(
                    candidate_id=batch.candidate_ids[row][index],
                    candidate_index=index,
                    log_probability=math.log(max(probability, 1e-12)),
                    value=float(values[row, index].detach().cpu()),
                    candidate_values=tuple(float(value) for value in row_values),
                )
            )
        return tuple(results)

    def action_result_structured(
        self,
        observation: EntityCandidateObservation,
        *,
        deterministic: bool,
        epsilon: float | None = None,
    ) -> DynamicQActionResult:
        return self.action_results_structured(
            (observation,), deterministic=deterministic, epsilon=epsilon
        )[0]

    def update_structured(
        self,
        transitions: Sequence[StructuredReplayTransition],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> UpdateResult:
        if not transitions:
            raise ValueError("at least one structured transition is required")
        observations = [item.observation for item in transitions]
        next_observations = [item.next_observation for item in transitions]
        batch = collate_entity_candidate_observations(observations).to_torch(
            self.device
        )
        next_batch = collate_entity_candidate_observations(
            next_observations
        ).to_torch(self.device)
        actions = torch.as_tensor(
            [item.candidate_index for item in transitions],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        importance = torch.as_tensor(
            np.ones(len(transitions), dtype=np.float32)
            if weights is None
            else np.asarray(weights, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        if importance.shape != (len(transitions),):
            raise ValueError("importance weights must align with transitions")

        self.q.train()
        action_values = self.q(batch).logits
        predicted = action_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            online_next = self.q(next_batch).logits
            next_actions = torch.argmax(online_next, dim=1)
            target_next = self.q_target(next_batch).logits.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * target_next
        td_error = targets - predicted
        per_item = torch.nn.functional.huber_loss(
            predicted,
            targets,
            reduction="none",
            delta=self.huber_delta,
        )
        td_loss = (importance * per_item).mean()
        demonstration_mask = torch.as_tensor(
            [
                bool(item.metadata.get("demonstration", False))
                for item in transitions
            ],
            dtype=torch.bool,
            device=self.device,
        )
        demonstration_loss = torch.zeros((), device=self.device)
        if self.demonstration_weight > 0.0 and demonstration_mask.any():
            margins = torch.full_like(action_values, self.demonstration_margin)
            margins.scatter_(1, actions.unsqueeze(1), 0.0)
            best_margin_value = torch.max(action_values + margins, dim=1).values
            per_demo = torch.clamp(best_margin_value - predicted, min=0.0)
            demonstration_loss = (
                importance[demonstration_mask]
                * per_demo[demonstration_mask]
            ).mean()
        loss = td_loss + self.demonstration_weight * demonstration_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = 0.0
        if self.max_grad_norm > 0.0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.q.parameters(), self.max_grad_norm
                ).detach().cpu()
            )
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        priorities = td_error.detach().abs().cpu().numpy().astype(np.float32)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "td_loss": float(td_loss.detach().cpu()),
            "demonstration_loss": float(demonstration_loss.detach().cpu()),
            "demonstration_fraction": float(
                demonstration_mask.float().mean().detach().cpu()
            ),
            "td_abs_mean": float(np.mean(priorities)),
            "q_mean": float(predicted.detach().mean().cpu()),
            "target_mean": float(targets.detach().mean().cpu()),
            "grad_norm": grad_norm,
        }
        self.last_metrics = metrics
        return UpdateResult(
            loss=metrics["loss"], priorities=priorities, metrics=metrics
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "spec": {
                "global_dim": self.spec.global_dim,
                "entity_dim": self.spec.entity_dim,
                "candidate_dim": self.spec.candidate_dim,
                "entity_type_count": self.spec.entity_type_count,
            },
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
            "last_metrics": dict(self.last_metrics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_spec = state.get("spec")
        expected = {
            "global_dim": self.spec.global_dim,
            "entity_dim": self.spec.entity_dim,
            "candidate_dim": self.spec.candidate_dim,
            "entity_type_count": self.spec.entity_type_count,
        }
        if isinstance(saved_spec, Mapping) and dict(saved_spec) != expected:
            raise ValueError("structured checkpoint feature schema does not match")
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state.get("q_target", state["q"]))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            optimizer_to(self.optimizer, self.device)
        self.update_steps = int(state.get("update_steps", 0))
        self.last_metrics = {
            str(key): float(value)
            for key, value in dict(state.get("last_metrics", {})).items()
        }


def _build_structured(
    spec: StructuredPolicySpec,
    config: Mapping[str, Any],
    device: str,
) -> StructuredDQNAgent:
    return StructuredDQNAgent(spec, config, device)


PLUGIN = AlgorithmPlugin(
    name="structured_dqn",
    version="1.0.0",
    family="off-policy structured value",
    build=None,
    build_structured=_build_structured,
    representation_modes=("entity_candidates",),
    default_export_module="q",
    replay_mode="transition",
    description=(
        "Double-DQN over variable typed entities and state-local semantic "
        "action candidates."
    ),
    noise_profile=(
        "Huber TD loss, prioritized replay, and Double-DQN targets reduce "
        "reward-outlier and value-overestimation sensitivity."
    ),
)
algorithm_registry.register(PLUGIN)
