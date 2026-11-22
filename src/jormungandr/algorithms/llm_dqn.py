"""Optional history-conditioned language Double-DQN with adapter-only checkpoints.

Uses native structured replay/TD updates. This is not a reproduction of the
Llama 8B/IA3 experiments in arXiv:2501.14176.
"""

from dataclasses import asdict, replace
from collections.abc import Mapping
import math

import torch
from jormungandr.llm_context import Experience

from .base import AlgorithmPlugin
from .common import optimizer_to
from .llm_ppo import LanguageCandidatePolicy, _TextBatch
from .registry import algorithm_registry
from .structured_dqn import StructuredDQNAgent


class LLMDQNAgent(StructuredDQNAgent):
    def __init__(self, spec, config, device="auto"):
        self.target_tau = float(config.get("llm_target_tau", 0.1))
        if not math.isfinite(self.target_tau) or not 0 < self.target_tau <= 1:
            raise ValueError("llm_target_tau must be finite in (0,1]")
        super().__init__(spec, config, device)

    @property
    def policy(self):
        return self.q

    def _build_q(self, spec, config):
        if config.get("llm_numeric_branch", False):
            raise ValueError("llm_dqn currently requires a language-only policy")
        model = LanguageCandidatePolicy(config, self.device, spec).to(self.device)
        for parameter in model.value_head.parameters():
            parameter.requires_grad_(False)
        return model

    def _collate_observations(self, observations):
        return _TextBatch(super()._collate_observations(observations), observations)

    def _update_target(self):
        target = dict(self.q_target.named_parameters())
        with torch.no_grad():
            for name, parameter in self.q.named_parameters():
                if parameter.requires_grad:
                    target[name].lerp_(parameter, self.target_tau)

    @staticmethod
    def _validate_transition(step, *, require_training=True):
        if require_training and step.split != "train":
            raise ValueError("only training replay may update the language model")
        contexts = []
        for observation, next_state in (
            (step.observation, False),
            (step.next_observation, True),
        ):
            context = observation.metadata.get("llm", {})
            if (
                not isinstance(context, Mapping)
                or context.get("schema") != "jormungandr.llm_context.v1"
                or context.get("split") != step.split
                or context.get("actor_id") != step.actor_id
                or not context.get("trial_id")
            ):
                raise ValueError(
                    "LLM replay requires matching training actor/trial context"
                )
            contexts.append(context)
            history = context.get("history", [])
            if not isinstance(history, list):
                raise ValueError("LLM history must be a list")
            for event in history:
                try:
                    Experience(**event)
                except (TypeError, ValueError) as exc:
                    raise ValueError("invalid LLM history event") from exc
                if event.get("episode_id") == step.episode_id:
                    limit = step.timestep + (step.bootstrap_steps if next_state else 0)
                    if int(event.get("timestep", -1)) >= limit:
                        raise ValueError("LLM replay context contains future feedback")
        if contexts[0]["trial_id"] != contexts[1]["trial_id"]:
            raise ValueError("LLM replay cannot cross trial boundaries")

    def validate_transition(self, step):
        """Ingress accepts valid held-out data, but update_structured never trains it."""
        self._validate_transition(step, require_training=False)

    def update_structured(self, transitions, weights=None):
        for step in transitions:
            self._validate_transition(step)
        if (
            self.gamma == 0
            and self.q.candidate_context in {"local", "evidence"}
            and self.demonstration_weight == 0
            # Auxiliary objectives may need alternatives omitted by TD's
            # chosen-candidate shortcut. Preserve their full input contract.
            and not self._has_auxiliary_objectives()
        ):
            # The contextual reward loss uses only the chosen candidate. Local
            # serialization makes this exactly equivalent to scoring every arm.
            # Preserve original entity indices and the causal source history.
            selected = []
            for step in transitions:
                obs, i = step.observation, step.candidate_index
                metadata = dict(obs.metadata)
                metadata["llm"] = {
                    **metadata["llm"],
                    "candidate_descriptions": {
                        step.candidate_id: metadata["llm"]["candidate_descriptions"][
                            step.candidate_id
                        ]
                    },
                }
                current = replace(
                    obs,
                    candidate_features=obs.candidate_features[[i]],
                    candidate_ids=(step.candidate_id,),
                    candidate_entity_indices=obs.candidate_entity_indices[[i]],
                    legal_action_mask=obs.legal_action_mask[[i]],
                    metadata=metadata,
                )
                selected.append(replace(step, observation=current))
            transitions = selected
        return super().update_structured(transitions, weights)

    @staticmethod
    def _pack(model):
        from peft import get_peft_model_state_dict

        return dict(
            adapter=get_peft_model_state_dict(model.backbone),
            action_head=model.action_head.state_dict(),
            value_head=model.value_head.state_dict(),
        )

    @staticmethod
    def _validate_pack(model, state):
        expected = LLMDQNAgent._pack(model)
        for part, weights in expected.items():
            actual = state.get(part, {})
            if set(weights) != set(actual) or any(
                weights[k].shape != actual[k].shape for k in weights
            ):
                raise ValueError("LLM DQN checkpoint tensors do not match")

    @staticmethod
    def _restore(model, state):
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model.backbone, state["adapter"])
        model.action_head.load_state_dict(state["action_head"])
        model.value_head.load_state_dict(state["value_head"])

    def state_dict(self):
        return dict(
            spec=asdict(self.spec),
            llm_identity=self.q.identity,
            q=self._pack(self.q),
            q_target=self._pack(self.q_target),
            optimizer=self.optimizer.state_dict(),
            update_steps=self.update_steps,
            last_metrics=dict(self.last_metrics),
        )

    def load_state_dict(self, state):
        if (
            state.get("spec") != asdict(self.spec)
            or state.get("llm_identity") != self.q.identity
        ):
            raise ValueError("LLM DQN checkpoint identity or schema does not match")
        target = state.get("q_target", state["q"])
        self._validate_pack(self.q, state["q"])
        self._validate_pack(self.q_target, target)
        self._restore(self.q, state["q"])
        self._restore(self.q_target, target)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            optimizer_to(self.optimizer, self.device)
        self.update_steps = int(state.get("update_steps", 0))
        self.last_metrics = dict(state.get("last_metrics", {}))


PLUGIN = AlgorithmPlugin(
    name="llm_dqn",
    version="1.0.0",
    family="off-policy language value",
    build=None,
    build_structured=LLMDQNAgent,
    representation_modes=("entity_candidates",),
    default_export_module="q",
    replay_mode="transition",
    runtime_defaults={"batch_size": 4, "min_replay": 32},
    description="History-conditioned language Double-DQN with LoRA and Polyak targets.",
)
algorithm_registry.register(PLUGIN)
