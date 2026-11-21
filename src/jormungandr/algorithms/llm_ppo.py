"""Optional LoRA language backbone under the existing structured PPO runtime.

This scores actor-approved candidates. It does not generate arbitrary tool
arguments or implement the DQN/IA3 algorithm from arXiv:2501.14176.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from jormungandr.structured import EntityCandidatePolicyOutput
from jormungandr.llm_context import Experience
from .base import AlgorithmPlugin
from .common import optimizer_to
from .registry import algorithm_registry
from .structured_ppo import StructuredPPOAgent


class _TextBatch:
    def __init__(self, numeric, observations):
        self.numeric = numeric
        self.observations = tuple(observations)

    def __getattr__(self, name):
        return getattr(self.numeric, name)


class LanguageCandidatePolicy(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("Install jormungandr-rl[llm] for llm_ppo") from exc

        model_id = str(config.get("llm_model", "")).strip()
        if not model_id:
            raise ValueError("llm_model is required")
        revision = config.get("llm_revision")
        if not Path(model_id).is_dir() and not revision:
            raise ValueError(
                "remote llm_model requires an immutable llm_revision commit"
            )
        if revision and (
            len(str(revision)) != 40
            or any(c not in "0123456789abcdef" for c in str(revision))
        ):
            raise ValueError("llm_revision must be a 40-character commit hash")
        dtype_name = str(config.get("llm_dtype", "float32"))
        if dtype_name not in {"float32", "bfloat16"}:
            raise ValueError("llm_dtype must be float32 or bfloat16")
        self.max_tokens = int(config.get("llm_max_tokens", 512))
        self.forward_batch_size = int(config.get("llm_forward_batch_size", 4))
        if self.max_tokens < 16 or self.forward_batch_size < 1:
            raise ValueError("invalid token or forward batch limit")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs a padding or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        backbone = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            torch_dtype=getattr(torch, dtype_name),
            attn_implementation="eager",
        )
        if self.max_tokens > getattr(
            backbone.config, "max_position_embeddings", self.max_tokens
        ):
            raise ValueError("llm_max_tokens exceeds model context limit")
        backbone.config.use_cache = False
        targets = config.get("llm_target_modules", "all-linear")
        rank = int(config.get("llm_lora_rank", 8))
        alpha = int(config.get("llm_lora_alpha", 16))
        if rank < 1 or alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        self.backbone = get_peft_model(
            backbone,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.0,
                target_modules=targets,
            ),
        )
        # PPO must reproduce behavior probabilities: disable backbone dropout
        # during both collection and gradient-enabled updates.
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        hidden = backbone.config.hidden_size
        self.action_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)
        self.identity = {
            "model": model_id,
            "revision": revision,
            "dtype": dtype_name,
            "rank": rank,
            "alpha": alpha,
            "target_modules": targets,
            "max_tokens": self.max_tokens,
            "serialization": "jormungandr.llm_candidate.v1",
        }
        if Path(model_id).is_dir():
            # A mutable local pathname is not a sufficient base-model identity.
            digest = hashlib.sha256()
            for path in sorted(Path(model_id).iterdir()):
                if path.is_file() and path.suffix in {
                    ".json",
                    ".safetensors",
                    ".bin",
                    ".model",
                    ".txt",
                }:
                    digest.update(path.name.encode())
                    with path.open("rb") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
            self.identity["local_sha256"] = digest.hexdigest()
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        # Eval mode still permits autograd. Also disables attention's functional
        # dropout, which need not be represented by an nn.Dropout module.
        self.backbone.eval()
        return self

    def _prepare_texts(self, observation):
        raw = observation.metadata.get("llm")
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema") != "jormungandr.llm_context.v1"
        ):
            raise ValueError("observation requires llm_context.v1 metadata")
        if (
            raw.get("split") not in {"train", "validation"}
            or not raw.get("actor_id")
            or not raw.get("trial_id")
        ):
            raise ValueError("LLM context requires actor, trial and split")
        descriptions = raw.get("candidate_descriptions", {})
        if set(descriptions) != set(observation.candidate_ids):
            raise ValueError("candidate descriptions must match candidate IDs exactly")
        current = raw.get("observation")
        if not isinstance(current, str) or not current.strip():
            raise ValueError("LLM observation text is required")
        history = raw.get("history", [])
        if not isinstance(history, list):
            raise ValueError("LLM history must be a list")
        try:
            for event in history:
                Experience(**event)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid LLM history event") from exc
        # Include numeric inputs so text does not silently discard actor state.
        state = {
            "observation": current,
            "global": observation.global_features.tolist(),
            "entities": [
                {"id": key, "type": int(kind), "features": features.tolist()}
                for key, kind, features in zip(
                    observation.entity_ids,
                    observation.entity_type_ids,
                    observation.entity_features,
                )
            ],
        }
        choices = [
            {
                "id": key,
                "description": descriptions[key],
                "features": observation.candidate_features[i].tolist(),
                "entity_indices": observation.candidate_entity_indices[i].tolist(),
                "legal": bool(observation.legal_action_mask[i]),
            }
            for i, key in enumerate(observation.candidate_ids)
        ]
        suffixes = ["\nState value:"] + [
            "\nCandidate to score: "
            + json.dumps(choice, sort_keys=True, allow_nan=False)
            for choice in choices
        ]
        # Drop whole oldest events; never silently truncate the current state
        # or candidate identity. The same policy input yields the same window.
        history = list(history)
        while True:
            prompt = (
                "Choose an allowed action using observed experience.\n"
                + json.dumps(
                    {"history": history, "state": state, "candidates": choices},
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            texts = [prompt + suffix for suffix in suffixes]
            max_tokens = max(len(self.tokenizer(text)["input_ids"]) for text in texts)
            if max_tokens <= self.max_tokens:
                return texts, {
                    "requested_events": len(raw.get("history", [])),
                    "retained_events": len(history),
                    "max_sequence_tokens": max_tokens,
                }
            if not history:
                raise ValueError(
                    "current state and candidates exceed llm_max_tokens; shorten actor text or increase limit"
                )
            history.pop(0)

    def _texts(self, observation):
        return self._prepare_texts(observation)[0]

    def context_diagnostics(self, observation):
        """Report effective context without model inference or memory mutation."""
        return self._prepare_texts(observation)[1]

    def forward(self, batch):
        groups = [self._texts(observation) for observation in batch.observations]
        texts = [text for group in groups for text in group]
        features = []
        device = self.action_head.weight.device
        for start in range(0, len(texts), self.forward_batch_size):
            tokens = self.tokenizer(
                texts[start : start + self.forward_batch_size],
                padding=True,
                return_tensors="pt",
            )
            tokens = {
                key: value.to(device)
                for key, value in tokens.items()
                if key in {"input_ids", "attention_mask"}
            }
            hidden = self.backbone(**tokens).last_hidden_state
            last = tokens["attention_mask"].sum(-1) - 1
            features.append(
                hidden[torch.arange(len(last), device=device), last].float()
            )
        features = torch.cat(features)
        rows, values, offset = [], [], 0
        width = batch.candidate_mask.shape[1]
        for group in groups:
            count = len(group) - 1
            values.append(self.value_head(features[offset]).squeeze(-1))
            logits = self.action_head(
                features[offset + 1 : offset + 1 + count]
            ).squeeze(-1)
            rows.append(torch.nn.functional.pad(logits, (0, width - count), value=-1e9))
            offset += len(group)
        logits = torch.stack(rows).masked_fill(
            ~(batch.candidate_mask & batch.legal_action_mask), -1e9
        )
        return EntityCandidatePolicyOutput(logits=logits, values=torch.stack(values))


class LLMPPOAgent(StructuredPPOAgent):
    def _build_policy(self, spec, config):
        if int(config.get("structured_prefix_dim", 0)):
            raise ValueError(
                "llm_ppo does not support learned joint-prefix interactions yet"
            )
        if config.get("policy_ratio_guard_min", 0) or config.get(
            "policy_ratio_guard_max", 0
        ):
            raise ValueError(
                "llm_ppo does not support full-model transactional snapshots"
            )
        return LanguageCandidatePolicy(config, self.device)

    def _collate_observations(self, observations):
        return _TextBatch(super()._collate_observations(observations), observations)

    @staticmethod
    def _validate_training_context(step):
        context = step.observation.metadata.get("llm", {})
        if context.get("split") != "train":
            raise ValueError("only training context may update the LLM")
        if hasattr(step, "split") and (
            step.split != "train" or context.get("actor_id") != step.actor_id
        ):
            raise ValueError("context actor/split does not match trajectory provenance")
        for event in context.get("history", []):
            if (
                event.get("episode_id") == step.episode_id
                and int(event.get("timestep", -1)) >= step.timestep
            ):
                raise ValueError("context contains current or future episode feedback")

    def update_structured(self, trajectories):
        for trajectory in trajectories:
            for step in trajectory:
                self._validate_training_context(step)
        return super().update_structured(trajectories)

    def update_joint_structured(self, trajectories):
        for trajectory in trajectories:
            self.validate_trajectory(trajectory)
            for step in trajectory:
                self._validate_training_context(step)
        return super().update_joint_structured(trajectories)

    def validate_trajectory(self, trajectory):
        trial_id = None
        for step in trajectory:
            if len(step.factors) != 1:
                raise ValueError(
                    "llm_ppo prototype supports one decision factor per agent turn"
                )
            for observation in (step.observation, step.next_observation):
                context = observation.metadata.get("llm", {})
                if context.get("schema") != "jormungandr.llm_context.v1":
                    raise ValueError("trajectory requires llm_context.v1 metadata")
                if (
                    context.get("actor_id") != step.actor_id
                    or context.get("split") != step.split
                ):
                    raise ValueError(
                        "context actor/split does not match trajectory provenance"
                    )
                if not context.get("trial_id"):
                    raise ValueError("context requires a trial_id")
                if trial_id is not None and trial_id != context["trial_id"]:
                    raise ValueError("trial cannot change within a trajectory")
                trial_id = context["trial_id"]
            for event in step.observation.metadata["llm"].get("history", []):
                if (
                    event.get("episode_id") == step.episode_id
                    and int(event.get("timestep", -1)) >= step.timestep
                ):
                    raise ValueError(
                        "context contains current or future episode feedback"
                    )

    def state_dict(self):
        from peft import get_peft_model_state_dict

        return {
            "spec": asdict(self.spec),
            "llm_identity": self.policy.identity,
            "adapter": get_peft_model_state_dict(self.policy.backbone),
            "action_head": self.policy.action_head.state_dict(),
            "value_head": self.policy.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
            "last_metrics": dict(self.last_metrics),
        }

    def load_state_dict(self, state):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        if (
            state.get("spec") != asdict(self.spec)
            or state.get("llm_identity") != self.policy.identity
        ):
            raise ValueError(
                "LLM checkpoint base, adapter, tokenizer contract or feature schema does not match"
            )
        expected = get_peft_model_state_dict(self.policy.backbone)
        actual = state["adapter"]
        if set(expected) != set(actual) or any(
            expected[k].shape != actual[k].shape for k in expected
        ):
            raise ValueError("LLM checkpoint adapter tensors do not match")
        set_peft_model_state_dict(self.policy.backbone, actual)
        self.policy.action_head.load_state_dict(state["action_head"])
        self.policy.value_head.load_state_dict(state["value_head"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            optimizer_to(self.optimizer, self.device)
        self.update_steps = int(state.get("update_steps", 0))
        self.last_metrics = dict(state.get("last_metrics", {}))

    def initialize_policy_from_state(self, state):
        raise ValueError(
            "llm_ppo requires its own adapter checkpoint; numeric BC initialization is incompatible"
        )


PLUGIN = AlgorithmPlugin(
    name="llm_ppo",
    version="0.1.0",
    family="on-policy language candidate actor critic",
    build=None,
    build_structured=LLMPPOAgent,
    representation_modes=("entity_candidates",),
    default_export_module="policy",
    replay_mode="trajectory",
    enforce_policy_lag=True,
    description="Experimental pretrained language backbone with LoRA, actor-owned history and bounded candidate PPO.",
)
algorithm_registry.register(PLUGIN)
