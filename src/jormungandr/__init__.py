"""Reusable Jörmungandr service runtime (generic trainer/checkpoint/inference over HTTP)."""

from importlib import import_module
from typing import Any

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "JormungandrRuntime",
    "JormungandrHttpServer",
    "GraphTrajectoryBatch",
    "GraphTrajectoryBuffer",
    "GraphTrajectoryStep",
    "MaskedActorCriticLoss",
    "BeamSearchResult",
    "FrontierSelection",
    "QUBOFrontierPruner",
    "SearchNode",
    "UtilityFrontierPruner",
    "AlgorithmPlugin",
    "algorithm_registry",
    "available_algorithms",
    "apply_legal_action_mask",
    "export_torchscript_from_checkpoint",
    "export_inference_bundle",
    "inspect_checkpoint",
    "compare_checkpoints",
    "bounded_beam_search",
    "build_frontier_pruner",
    "masked_actor_critic_loss",
    "select_masked_actions",
]


def __getattr__(name: str) -> Any:
    if name in {"JormungandrRuntime", "JormungandrHttpServer"}:
        module = import_module("jormungandr.service")
        return getattr(module, name)
    if name in {
        "compare_checkpoints",
        "export_inference_bundle",
        "export_torchscript_from_checkpoint",
        "inspect_checkpoint",
    }:
        module = import_module("jormungandr.export")
        return getattr(module, name)
    if name in {
        "BeamSearchResult",
        "FrontierSelection",
        "QUBOFrontierPruner",
        "SearchNode",
        "UtilityFrontierPruner",
        "bounded_beam_search",
        "build_frontier_pruner",
    }:
        module = import_module("jormungandr.search")
        return getattr(module, name)
    if name in {
        "AlgorithmPlugin",
        "algorithm_registry",
        "available_algorithms",
    }:
        module = import_module("jormungandr.algorithms")
        return getattr(module, name)
    if name in {
        "GraphTrajectoryBatch",
        "GraphTrajectoryBuffer",
        "GraphTrajectoryStep",
        "MaskedActorCriticLoss",
        "apply_legal_action_mask",
        "masked_actor_critic_loss",
        "select_masked_actions",
    }:
        module = import_module("jormungandr.policy")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
