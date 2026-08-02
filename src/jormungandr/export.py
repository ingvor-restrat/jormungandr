from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from jormungandr.algorithms import algorithm_registry, canonical_algorithm_name


JsonDict = Dict[str, Any]


def _first_not_none(*values: Any, default: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


@dataclass
class CheckpointSpec:
    checkpoint_path: str
    style: str
    algo: str
    model_id: str
    obs_dim: int
    hidden: int
    aux_hidden: int
    lr: float
    max_grad: float
    gamma: float
    v_min: float
    v_max: float
    atoms: int
    quantiles: int
    quantile_risk_measure: str
    quantile_risk_level: float
    target_update: int
    aux_classes: int
    action_values: List[float]
    feature_keys: List[str]
    updates: int
    policy_version: int
    timestamp: float

    def to_dict(self) -> JsonDict:
        return {
            "checkpoint_path": self.checkpoint_path,
            "style": self.style,
            "algo": self.algo,
            "model_id": self.model_id,
            "obs_dim": int(self.obs_dim),
            "hidden": int(self.hidden),
            "aux_hidden": int(self.aux_hidden),
            "lr": float(self.lr),
            "max_grad": float(self.max_grad),
            "gamma": float(self.gamma),
            "v_min": float(self.v_min),
            "v_max": float(self.v_max),
            "atoms": int(self.atoms),
            "quantiles": int(self.quantiles),
            "quantile_risk_measure": self.quantile_risk_measure,
            "quantile_risk_level": float(self.quantile_risk_level),
            "target_update": int(self.target_update),
            "aux_classes": int(self.aux_classes),
            "action_values": list(self.action_values),
            "feature_keys": list(self.feature_keys),
            "updates": int(self.updates),
            "policy_version": int(self.policy_version),
            "timestamp": float(self.timestamp),
        }


def _load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    # Torch checkpoints can execute pickle payloads. Trust local artifacts only.
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility with older torch versions that do not expose weights_only.
        return torch.load(checkpoint_path, map_location="cpu")


def _first_linear_in_dim(agent_state: Mapping[str, Any]) -> int:
    keys = (
        "q.net.0.weight",
        "policy.net.0.weight",
        "net.0.weight",
    )
    for key in keys:
        weight = agent_state.get(key)
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[1])
    return 0


def _first_linear_hidden(agent_state: Mapping[str, Any]) -> int:
    keys = (
        "q.net.0.weight",
        "policy.net.0.weight",
        "net.0.weight",
    )
    for key in keys:
        weight = agent_state.get(key)
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[0])
    return 0


def _aux_linear_hidden(agent_state: Mapping[str, Any]) -> int:
    keys = (
        "aux_head.net.0.weight",
        "aux_head.0.weight",
    )
    for key in keys:
        weight = agent_state.get(key)
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[0])
    return 0


def _detect_style(ckpt: Mapping[str, Any]) -> str:
    if "learner_config" in ckpt:
        return "service"
    if "train_config" in ckpt or "model_spec" in ckpt:
        return "train"
    return "unknown"


def _checkpoint_spec_from_payload(checkpoint_path: str, ckpt: Mapping[str, Any]) -> CheckpointSpec:
    style = _detect_style(ckpt)
    model_spec = ckpt.get("model_spec") or {}
    learner_cfg = ckpt.get("learner_config") or {}
    train_cfg = ckpt.get("train_config") or {}
    agent_state = ckpt.get("agent") or {}
    plugin_cfg_raw = learner_cfg.get("plugin_config")
    plugin_cfg = plugin_cfg_raw if isinstance(plugin_cfg_raw, Mapping) else {}

    algo = str(ckpt.get("algo") or learner_cfg.get("algo") or train_cfg.get("algo") or "").strip().lower()
    if not algo:
        # Service checkpoints are c51 today; keep fallback conservative.
        algo = "c51"

    feature_keys = [str(x) for x in (ckpt.get("feature_keys") or [])]

    obs_dim = int(
        ckpt.get("obs_dim")
        or model_spec.get("obs_dim")
        or learner_cfg.get("obs_dim")
        or train_cfg.get("obs_dim")
        or (len(feature_keys) if feature_keys else 0)
        or _first_linear_in_dim(agent_state)
    )

    hidden = int(
        plugin_cfg.get("hidden")
        or model_spec.get("hidden")
        or learner_cfg.get("hidden")
        or train_cfg.get("hidden")
        or _first_linear_hidden(agent_state)
        or 256
    )

    aux_hidden = int(
        plugin_cfg.get("aux_hidden")
        or model_spec.get("aux_hidden")
        or learner_cfg.get("aux_hidden")
        or train_cfg.get("aux_hidden")
        or _aux_linear_hidden(agent_state)
        or 0
    )

    lr = float(
        plugin_cfg.get("lr")
        or model_spec.get("lr")
        or learner_cfg.get("lr")
        or train_cfg.get("lr")
        or 1e-4
    )
    max_grad = float(
        _first_not_none(
            plugin_cfg.get("max_grad"),
            model_spec.get("max_grad"),
            learner_cfg.get("max_grad"),
            train_cfg.get("max_grad"),
            default=1.0,
        )
    )
    gamma = float(
        _first_not_none(
            plugin_cfg.get("gamma"),
            model_spec.get("gamma"),
            learner_cfg.get("gamma"),
            train_cfg.get("gamma"),
            default=0.99,
        )
    )
    v_min = float(
        _first_not_none(
            plugin_cfg.get("v_min"),
            ckpt.get("v_min"),
            model_spec.get("v_min"),
            learner_cfg.get("v_min"),
            train_cfg.get("v_min"),
            default=-10.0,
        )
    )
    v_max = float(
        _first_not_none(
            plugin_cfg.get("v_max"),
            ckpt.get("v_max"),
            model_spec.get("v_max"),
            learner_cfg.get("v_max"),
            train_cfg.get("v_max"),
            default=10.0,
        )
    )
    atoms = int(
        plugin_cfg.get("atoms")
        or ckpt.get("atoms")
        or model_spec.get("atoms")
        or learner_cfg.get("atoms")
        or train_cfg.get("atoms")
        or 51
    )
    quantiles = int(
        plugin_cfg.get("quantiles")
        or model_spec.get("quantiles")
        or learner_cfg.get("quantiles")
        or train_cfg.get("quantiles")
        or 51
    )
    quantile_risk_measure = str(
        plugin_cfg.get("quantile_risk_measure")
        or model_spec.get("quantile_risk_measure")
        or learner_cfg.get("quantile_risk_measure")
        or train_cfg.get("quantile_risk_measure")
        or "mean"
    )
    quantile_risk_level = float(
        plugin_cfg.get("quantile_risk_level")
        or model_spec.get("quantile_risk_level")
        or learner_cfg.get("quantile_risk_level")
        or train_cfg.get("quantile_risk_level")
        or 0.1
    )
    target_update = int(
        plugin_cfg.get("target_update")
        or model_spec.get("target_update")
        or learner_cfg.get("target_update")
        or train_cfg.get("target_update")
        or 1000
    )

    aux_classes = int(
        plugin_cfg.get("aux_classes")
        or model_spec.get("aux_classes")
        or learner_cfg.get("aux_classes")
        or train_cfg.get("aux_classes")
        or 0
    )
    aux_enabled = plugin_cfg.get("aux_enabled", learner_cfg.get("aux_enabled"))
    if style == "service" and aux_enabled is False:
        aux_classes = 0

    action_values = (
        plugin_cfg.get("action_values")
        or ckpt.get("action_values")
        or learner_cfg.get("action_values")
        or train_cfg.get("action_values")
    )
    if action_values is None:
        action_values = [-1.0, 0.0, 1.0]
    action_values = [float(x) for x in action_values]

    model_id = str(ckpt.get("model_id") or "")
    updates = int(ckpt.get("updates") or ckpt.get("global_step") or 0)
    policy_version = int(ckpt.get("policy_version") or updates)
    timestamp = float(ckpt.get("ts") or 0.0)

    return CheckpointSpec(
        checkpoint_path=str(Path(checkpoint_path).expanduser().resolve()),
        style=style,
        algo=algo,
        model_id=model_id,
        obs_dim=obs_dim,
        hidden=hidden,
        aux_hidden=aux_hidden,
        lr=lr,
        max_grad=max_grad,
        gamma=gamma,
        v_min=v_min,
        v_max=v_max,
        atoms=atoms,
        quantiles=quantiles,
        quantile_risk_measure=quantile_risk_measure,
        quantile_risk_level=quantile_risk_level,
        target_update=target_update,
        aux_classes=aux_classes,
        action_values=action_values,
        feature_keys=feature_keys,
        updates=updates,
        policy_version=policy_version,
        timestamp=timestamp,
    )


def inspect_checkpoint(checkpoint_path: str) -> JsonDict:
    ckpt = _load_checkpoint(checkpoint_path)
    spec = _checkpoint_spec_from_payload(checkpoint_path, ckpt)
    obs_normalizer = _obs_normalizer_from_checkpoint(ckpt, spec.obs_dim)
    top_keys = sorted([str(k) for k in ckpt.keys()])
    return {
        "ok": True,
        "spec": spec.to_dict(),
        "preprocessing": _obs_normalizer_summary(obs_normalizer),
        "algorithm_plugin": dict(ckpt.get("algorithm_plugin") or {}),
        "replay_selector": dict(ckpt.get("replay_selector") or {}),
        "top_level_keys": top_keys,
    }


def _build_agent_from_checkpoint(
    ckpt: Mapping[str, Any],
    spec: CheckpointSpec,
    device: str = "cpu",
    *,
    load_optimizer: bool = False,
):
    if "agent" not in ckpt:
        raise ValueError("checkpoint payload is missing 'agent' state")
    learner_config = dict(ckpt.get("learner_config") or {})
    model_spec = dict(ckpt.get("model_spec") or {})
    learner_config.update(
        {
            "algo": spec.algo,
            "action_values": list(spec.action_values),
            "hidden": int(spec.hidden),
            "aux_hidden": int(spec.aux_hidden),
            "lr": float(spec.lr),
            "gamma": float(spec.gamma),
            "v_min": float(spec.v_min),
            "v_max": float(spec.v_max),
            "atoms": int(spec.atoms),
            "quantiles": int(spec.quantiles),
            "quantile_risk_measure": spec.quantile_risk_measure,
            "quantile_risk_level": float(spec.quantile_risk_level),
            "target_update": int(spec.target_update),
            "max_grad": float(spec.max_grad),
            "aux_enabled": bool(spec.aux_classes > 0),
            "aux_classes": int(spec.aux_classes),
        }
    )
    # Translate the original standalone PPO checkpoint vocabulary.
    if "ppo_clip" in model_spec:
        learner_config.setdefault("clip_ratio", model_spec["ppo_clip"])
    if "ppo_entropy" in model_spec:
        learner_config.setdefault("entropy_coef", model_spec["ppo_entropy"])
    if "ppo_value" in model_spec:
        learner_config.setdefault("value_coef", model_spec["ppo_value"])
    try:
        plugin = algorithm_registry.get(spec.algo)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    saved_plugin = ckpt.get("algorithm_plugin")
    if isinstance(saved_plugin, Mapping):
        saved_name = canonical_algorithm_name(str(saved_plugin.get("name", "")))
        if saved_name and saved_name != canonical_algorithm_name(spec.algo):
            raise ValueError(
                f"checkpoint algorithm plugin {saved_name!r} does not match "
                f"payload algorithm {spec.algo!r}"
            )
        saved_version = str(saved_plugin.get("version", "")).strip()
        if saved_version and saved_version != plugin.version:
            raise ValueError(
                f"checkpoint requires {plugin.name}@{saved_version}, but the "
                f"installed plugin is {plugin.checkpoint_id}; install a compatible "
                "plugin or migrate the checkpoint"
            )
    agent = plugin.build(int(spec.obs_dim), learner_config, device)
    agent_state = dict(ckpt["agent"])
    if not load_optimizer:
        for key in (
            "opt",
            "actor_opt",
            "critic_opt",
            "alpha_opt",
            "world_opt",
            "value_opt",
        ):
            agent_state.pop(key, None)
    agent.load_state_dict(agent_state)
    return agent


def _obs_normalizer_from_checkpoint(ckpt: Mapping[str, Any], obs_dim: int) -> Optional[JsonDict]:
    raw = ckpt.get("obs_normalizer") or ckpt.get("normalizer")
    if not isinstance(raw, Mapping):
        return None
    try:
        count = int(raw.get("count", 0))
        eps = float(raw.get("eps", 1e-6))
        mean = np.asarray(raw.get("mean"), dtype=np.float32).reshape(-1)
        m2 = np.asarray(raw.get("m2"), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if count < 2 or mean.size != int(obs_dim) or m2.size != int(obs_dim):
        return None
    var = m2 / max(count - 1, 1)
    std = np.sqrt(var + eps).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32, copy=False)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0).astype(np.float32, copy=False)
    return {
        "kind": "running_zscore",
        "count": int(count),
        "mean": mean,
        "std": std,
    }


def _obs_normalizer_summary(normalizer: Optional[Mapping[str, Any]]) -> JsonDict:
    if not isinstance(normalizer, Mapping):
        return {"kind": "none", "present": False}
    return {
        "kind": str(normalizer.get("kind", "running_zscore")),
        "present": True,
        "count": int(normalizer.get("count", 0)),
    }


def _resolve_module(algo: str, module: str) -> str:
    if module == "auto":
        try:
            return algorithm_registry.get(algo).default_export_module
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
    if module not in {"policy", "q", "heads"}:
        raise ValueError("module must be one of: auto, policy, q, heads")
    return module


class _C51Heads(torch.nn.Module):
    def __init__(self, q: torch.nn.Module, aux_head: Optional[torch.nn.Module]):
        super().__init__()
        self.q = q
        self.aux_head = aux_head

    def forward(self, obs: torch.Tensor):
        q_logits = self.q(obs)
        if self.aux_head is None:
            return q_logits
        aux_logits = self.aux_head(obs)
        return q_logits, aux_logits


class _ObsNormalizeModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.module = module
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        return self.module((obs - self.mean) / self.std)


def _script_module(agent: Any, module: str, *, obs_normalizer: Optional[Mapping[str, Any]] = None):
    if hasattr(agent, "export_module"):
        torch_module = agent.export_module(module)
    elif module == "policy":
        torch_module = agent.policy
    elif module == "q":
        torch_module = getattr(agent, "q", None)
        if torch_module is None:
            torch_module = getattr(agent, "q1", None)
        if torch_module is None:
            raise ValueError("module=q requires an agent with a q or q1 network")
    elif module == "heads":
        if not hasattr(agent, "q"):
            raise ValueError("module=heads requires a C51-style agent with q network")
        aux_head = getattr(agent, "aux_head", None)
        torch_module = _C51Heads(agent.q, aux_head)
    else:
        raise ValueError("module must be one of: policy, q, heads")
    if isinstance(obs_normalizer, Mapping):
        torch_module = _ObsNormalizeModule(
            torch_module,
            mean=obs_normalizer.get("mean", []),
            std=obs_normalizer.get("std", []),
        )
    torch_module.eval()
    return torch.jit.script(torch_module)


def _cxx_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _format_cxx_float(x: float) -> str:
    v = float(x)
    if np.isfinite(v):
        return f"{v:.10g}f"
    return "0.0f"


def _generate_cpp_header(spec: CheckpointSpec, module: str, torchscript_relpath: str) -> str:
    n_actions = len(spec.action_values)
    feature_keys = spec.feature_keys
    ts_rel = _cxx_escape(torchscript_relpath)

    action_values_body = ", ".join(_format_cxx_float(v) for v in spec.action_values) if n_actions else ""
    feature_body = ", ".join(f'"{_cxx_escape(k)}"' for k in feature_keys) if feature_keys else ""
    if n_actions > 0:
        action_values_lines = [
            f"inline constexpr std::array<float, {int(n_actions)}> kActionValues = {{",
            f"    {action_values_body}",
            "};",
            "",
        ]
    else:
        action_values_lines = [
            "inline constexpr std::array<float, 0> kActionValues = {};",
            "",
        ]

    if len(feature_keys) > 0:
        feature_key_lines = [
            f"inline constexpr std::array<const char*, {int(len(feature_keys))}> kFeatureKeys = {{",
            f"    {feature_body}",
            "};",
            "",
        ]
    else:
        feature_key_lines = [
            "inline constexpr std::array<const char*, 0> kFeatureKeys = {};",
            "",
        ]

    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "",
        "namespace jormungandr_bundle {",
        "",
        "struct ModelSpec {",
        f"  static constexpr const char* kAlgo = \"{_cxx_escape(spec.algo)}\";",
        f"  static constexpr const char* kModule = \"{_cxx_escape(module)}\";",
        f"  static constexpr const char* kTorchScriptRelativePath = \"{ts_rel}\";",
        f"  static constexpr std::size_t kObsDim = {int(spec.obs_dim)};",
        f"  static constexpr std::size_t kNumActions = {int(n_actions)};",
        f"  static constexpr std::size_t kNumFeatureKeys = {int(len(feature_keys))};",
        f"  static constexpr std::size_t kC51Atoms = {int(spec.atoms)};",
        f"  static constexpr std::size_t kQuantiles = {int(spec.quantiles)};",
        f"  static constexpr const char* kQuantileRiskMeasure = \"{_cxx_escape(spec.quantile_risk_measure)}\";",
        f"  static constexpr float kQuantileRiskLevel = {_format_cxx_float(spec.quantile_risk_level)};",
        f"  static constexpr float kC51VMin = {_format_cxx_float(spec.v_min)};",
        f"  static constexpr float kC51VMax = {_format_cxx_float(spec.v_max)};",
        f"  static constexpr std::size_t kAuxClasses = {int(spec.aux_classes)};",
        f"  static constexpr bool kHasAuxHead = {str(int(spec.aux_classes) > 0).lower()};",
        "};",
        "",
        *action_values_lines,
        *feature_key_lines,
        "inline constexpr float c51_support_value(std::size_t atom_idx) {",
        "  if (ModelSpec::kC51Atoms <= 1) return ModelSpec::kC51VMin;",
        "  const float step = (ModelSpec::kC51VMax - ModelSpec::kC51VMin) / static_cast<float>(ModelSpec::kC51Atoms - 1);",
        "  return ModelSpec::kC51VMin + step * static_cast<float>(atom_idx);",
        "}",
        "",
        "}  // namespace jormungandr_bundle",
        "",
    ]
    return "\n".join(lines)


def export_torchscript_from_checkpoint(
    checkpoint_path: str,
    out_path: str,
    module: str = "auto",
    metadata_path: Optional[str] = None,
) -> Dict[str, Any]:
    ckpt = _load_checkpoint(checkpoint_path)
    spec = _checkpoint_spec_from_payload(checkpoint_path, ckpt)
    agent = _build_agent_from_checkpoint(ckpt, spec=spec, device="cpu")
    obs_normalizer = _obs_normalizer_from_checkpoint(ckpt, spec.obs_dim)

    resolved_module = _resolve_module(spec.algo, module)
    scripted = _script_module(agent, resolved_module, obs_normalizer=obs_normalizer)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_file))

    info = {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "algo": spec.algo,
        "module": resolved_module,
        "obs_dim": int(spec.obs_dim),
        "feature_keys": list(spec.feature_keys),
        "action_values": list(spec.action_values),
        "torchscript": str(out_file.resolve()),
        "preprocessing": {
            **_obs_normalizer_summary(obs_normalizer),
            "embedded_in_artifact": bool(obs_normalizer),
        },
    }

    meta_file = Path(metadata_path) if metadata_path else out_file.with_suffix(out_file.suffix + ".meta.json")
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def export_inference_bundle(
    checkpoint_path: str,
    bundle_dir: str,
    *,
    module: str = "auto",
    torchscript_name: str = "policy.ts.pt",
    manifest_name: str = "manifest.json",
    header_name: str = "jormungandr_model_spec.hpp",
) -> JsonDict:
    ckpt = _load_checkpoint(checkpoint_path)
    spec = _checkpoint_spec_from_payload(checkpoint_path, ckpt)
    agent = _build_agent_from_checkpoint(ckpt, spec=spec, device="cpu")
    obs_normalizer = _obs_normalizer_from_checkpoint(ckpt, spec.obs_dim)
    resolved_module = _resolve_module(spec.algo, module)

    out_dir = Path(bundle_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_path = out_dir / torchscript_name
    manifest_path = out_dir / manifest_name
    header_path = out_dir / header_name

    scripted = _script_module(agent, resolved_module, obs_normalizer=obs_normalizer)
    scripted.save(str(ts_path))

    manifest: JsonDict = {
        "format": "jormungandr.inference_bundle.v1",
        "exported_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source": {
            "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
            "style": spec.style,
            "model_id": spec.model_id,
            "updates": int(spec.updates),
            "policy_version": int(spec.policy_version),
            "timestamp": float(spec.timestamp),
        },
        "model": {
            "algo": spec.algo,
            "module": resolved_module,
            "obs_dim": int(spec.obs_dim),
            "feature_keys": list(spec.feature_keys),
            "action_values": list(spec.action_values),
            "aux_classes": int(spec.aux_classes),
            "c51": {
                "atoms": int(spec.atoms),
                "v_min": float(spec.v_min),
                "v_max": float(spec.v_max),
            },
            "quantile": {
                "count": int(spec.quantiles),
                "risk_measure": spec.quantile_risk_measure,
                "risk_level": float(spec.quantile_risk_level),
            },
        },
        "io": {
            "input": {
                "name": "obs",
                "dtype": "float32",
                "shape": ["N", int(spec.obs_dim)],
            },
            "output": {
                **(
                    {
                        "type": "tuple",
                        "items": [
                            {
                                "name": "q_logits",
                                "dtype": "float32",
                                "shape": ["N", len(spec.action_values), int(spec.atoms)],
                            },
                            {
                                "name": "aux_logits",
                                "dtype": "float32",
                                "shape": ["N", int(max(0, spec.aux_classes))],
                            },
                        ],
                    }
                    if resolved_module == "heads" and spec.algo == "c51"
                    else {
                        "dtype": "float32",
                        "shape": (
                            ["N", len(spec.action_values), int(spec.atoms)]
                            if resolved_module == "q" and spec.algo == "c51"
                            else (
                                ["N", len(spec.action_values), int(spec.quantiles)]
                                if resolved_module == "q" and spec.algo == "qrdqn"
                                else ["N", len(spec.action_values)]
                            )
                        ),
                    }
                ),
            },
        },
        "preprocessing": {
            **_obs_normalizer_summary(obs_normalizer),
            "embedded_in_artifact": bool(obs_normalizer),
        },
        "artifacts": {
            "torchscript": ts_path.name,
            "header": header_path.name,
        },
    }

    header = _generate_cpp_header(
        spec=spec,
        module=resolved_module,
        torchscript_relpath=ts_path.name,
    )

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    header_path.write_text(header, encoding="utf-8")

    return {
        "ok": True,
        "bundle_dir": str(out_dir),
        "manifest": str(manifest_path),
        "torchscript": str(ts_path),
        "header": str(header_path),
        "model": manifest["model"],
    }


def _inference_from_checkpoint(ckpt: Mapping[str, Any], spec: CheckpointSpec, obs: np.ndarray) -> JsonDict:
    agent = _build_agent_from_checkpoint(ckpt, spec=spec, device="cpu")
    obs_np = np.asarray(obs, dtype=np.float32)
    if obs_np.ndim != 2:
        raise ValueError("obs must be a 2D array [N, obs_dim]")
    if obs_np.shape[1] != int(spec.obs_dim):
        raise ValueError(f"obs dim mismatch: got {obs_np.shape[1]}, expected {spec.obs_dim}")
    obs_normalizer = _obs_normalizer_from_checkpoint(ckpt, spec.obs_dim)
    if isinstance(obs_normalizer, Mapping):
        mean = np.asarray(obs_normalizer.get("mean", []), dtype=np.float32).reshape(1, -1)
        std = np.asarray(obs_normalizer.get("std", []), dtype=np.float32).reshape(1, -1)
        obs_np = (obs_np - mean) / std

    out: JsonDict = {
        "algo": spec.algo,
        "obs_rows": int(obs_np.shape[0]),
        "preprocessing": _obs_normalizer_summary(obs_normalizer),
    }

    if not hasattr(agent, "inference_batch"):
        raise ValueError(
            f"algorithm plugin {spec.algo} does not expose checkpoint comparison inference"
        )
    results = agent.inference_batch(
        obs_np,
        deterministic=True,
        epsilon=0.0,
        action_masks=None,
    )
    out["action"] = [float(item.action) for item in results]
    out["action_idx"] = [int(item.action_idx) for item in results]
    for key in (
        "q_values",
        "risk_values",
        "quantiles",
        "policy_logits",
        "policy_probs",
    ):
        if results and all(key in item.extras for item in results):
            out[key] = [item.extras[key] for item in results]
    if results and all(item.value is not None for item in results):
        out["value"] = [float(item.value) for item in results]
    return out


def compare_checkpoints(
    left_checkpoint: str,
    right_checkpoint: str,
    *,
    obs: Optional[np.ndarray] = None,
    num_obs: int = 64,
    seed: int = 7,
) -> JsonDict:
    left_payload = _load_checkpoint(left_checkpoint)
    right_payload = _load_checkpoint(right_checkpoint)

    left_spec = _checkpoint_spec_from_payload(left_checkpoint, left_payload)
    right_spec = _checkpoint_spec_from_payload(right_checkpoint, right_payload)

    if int(left_spec.obs_dim) <= 0 or int(right_spec.obs_dim) <= 0:
        raise ValueError("could not infer obs_dim from one or both checkpoints")
    if int(left_spec.obs_dim) != int(right_spec.obs_dim):
        raise ValueError(
            f"obs_dim mismatch: left={left_spec.obs_dim} right={right_spec.obs_dim}; provide matching checkpoints"
        )

    if obs is None:
        rng = np.random.default_rng(int(seed))
        obs_np = rng.standard_normal(size=(int(num_obs), int(left_spec.obs_dim))).astype(np.float32)
    else:
        obs_np = np.asarray(obs, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)

    left_inf = _inference_from_checkpoint(left_payload, left_spec, obs_np)
    right_inf = _inference_from_checkpoint(right_payload, right_spec, obs_np)

    summary: JsonDict = {
        "obs_rows": int(obs_np.shape[0]),
        "obs_dim": int(obs_np.shape[1]),
        "left": {
            "checkpoint": str(Path(left_checkpoint).expanduser().resolve()),
            "algo": left_spec.algo,
            "updates": int(left_spec.updates),
            "policy_version": int(left_spec.policy_version),
        },
        "right": {
            "checkpoint": str(Path(right_checkpoint).expanduser().resolve()),
            "algo": right_spec.algo,
            "updates": int(right_spec.updates),
            "policy_version": int(right_spec.policy_version),
        },
        "comparisons": {},
    }

    left_actions = np.asarray(left_inf.get("action", []), dtype=np.float32).reshape(-1)
    right_actions = np.asarray(right_inf.get("action", []), dtype=np.float32).reshape(-1)
    if left_actions.size == right_actions.size and left_actions.size > 0:
        summary["comparisons"]["action_mae"] = float(np.mean(np.abs(left_actions - right_actions)))

    left_idx = left_inf.get("action_idx")
    right_idx = right_inf.get("action_idx")
    if left_idx is not None and right_idx is not None:
        li = np.asarray(left_idx, dtype=np.int64).reshape(-1)
        ri = np.asarray(right_idx, dtype=np.int64).reshape(-1)
        if li.size == ri.size and li.size > 0:
            summary["comparisons"]["action_match_rate"] = float(np.mean(li == ri))

    left_q = left_inf.get("q_values")
    right_q = right_inf.get("q_values")
    if left_q is not None and right_q is not None:
        lq = np.asarray(left_q, dtype=np.float32)
        rq = np.asarray(right_q, dtype=np.float32)
        if lq.shape == rq.shape and lq.size > 0:
            diff = np.abs(lq - rq)
            summary["comparisons"]["q_mae"] = float(np.mean(diff))
            summary["comparisons"]["q_max_abs"] = float(np.max(diff))

    left_logits = left_inf.get("policy_logits")
    right_logits = right_inf.get("policy_logits")
    if left_logits is not None and right_logits is not None:
        ll = np.asarray(left_logits, dtype=np.float32)
        rl = np.asarray(right_logits, dtype=np.float32)
        if ll.shape == rl.shape and ll.size > 0:
            diff = np.abs(ll - rl)
            summary["comparisons"]["policy_logits_mae"] = float(np.mean(diff))
            summary["comparisons"]["policy_logits_max_abs"] = float(np.max(diff))

    left_risk = left_inf.get("risk_values")
    right_risk = right_inf.get("risk_values")
    if left_risk is not None and right_risk is not None:
        lrisk = np.asarray(left_risk, dtype=np.float32)
        rrisk = np.asarray(right_risk, dtype=np.float32)
        if lrisk.shape == rrisk.shape and lrisk.size > 0:
            summary["comparisons"]["risk_value_mae"] = float(
                np.mean(np.abs(lrisk - rrisk))
            )

    left_quantiles = left_inf.get("quantiles")
    right_quantiles = right_inf.get("quantiles")
    if left_quantiles is not None and right_quantiles is not None:
        lquantiles = np.sort(
            np.asarray(left_quantiles, dtype=np.float32), axis=-1
        )
        rquantiles = np.sort(
            np.asarray(right_quantiles, dtype=np.float32), axis=-1
        )
        if lquantiles.shape == rquantiles.shape and lquantiles.size > 0:
            summary["comparisons"]["quantile_mae"] = float(
                np.mean(np.abs(lquantiles - rquantiles))
            )

    return {
        "ok": True,
        "summary": summary,
    }


def load_obs_json(path: str) -> np.ndarray:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "obs" in raw:
        raw = raw["obs"]
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("obs json must be a 1D or 2D numeric array")
    return arr
