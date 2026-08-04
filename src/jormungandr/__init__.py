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
    "DynamicActionResult",
    "EntityCandidateBatch",
    "EntityCandidateObservation",
    "EntityCandidatePolicyOutput",
    "EntityCandidateTransformer",
    "TorchEntityCandidateBatch",
    "StructuredPolicySpec",
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
    "collate_entity_candidate_observations",
    "select_dynamic_actions",
    "ProcessActorPool",
    "JormungandrClient",
    "JormungandrClientError",
    "JointActionAudit",
    "JointActionChoice",
    "JointActionFactor",
    "JointActionSolution",
    "compose_joint_action",
    "StructuredFactorChoice",
    "StructuredJointTrajectoryStep",
    "StructuredActionFactor",
    "StructuredJointActionResult",
    "sample_structured_joint_action",
    "structured_joint_step_from_payload",
    "structured_joint_step_to_payload",
    "structured_joint_trajectory_from_sequence_payload",
    "structured_joint_trajectory_to_sequence_payload",
    "validate_structured_joint_trajectory",
    "STRUCTURED_SUPERVISION_SCHEMA",
    "StructuredSupervisionExample",
    "structured_supervision_from_payload",
    "structured_supervision_to_payload",
    "STRUCTURED_BUNDLE_SCHEMA",
    "export_structured_policy_bundle",
    "load_structured_policy_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {"JormungandrClient", "JormungandrClientError"}:
        module = import_module("jormungandr.client")
        return getattr(module, name)
    if name == "ProcessActorPool":
        module = import_module("jormungandr.actors")
        return getattr(module, name)
    if name in {
        "StructuredFactorChoice",
        "StructuredJointTrajectoryStep",
        "StructuredActionFactor",
        "StructuredJointActionResult",
        "sample_structured_joint_action",
        "structured_joint_step_from_payload",
        "structured_joint_step_to_payload",
        "structured_joint_trajectory_from_sequence_payload",
        "structured_joint_trajectory_to_sequence_payload",
        "validate_structured_joint_trajectory",
    }:
        module = import_module("jormungandr.structured_trajectory")
        return getattr(module, name)
    if name in {
        "STRUCTURED_SUPERVISION_SCHEMA",
        "StructuredSupervisionExample",
        "structured_supervision_from_payload",
        "structured_supervision_to_payload",
    }:
        module = import_module("jormungandr.structured_supervision")
        return getattr(module, name)
    if name in {
        "STRUCTURED_BUNDLE_SCHEMA",
        "export_structured_policy_bundle",
        "load_structured_policy_bundle",
    }:
        module = import_module("jormungandr.structured_export")
        return getattr(module, name)
    if name in {
        "JointActionAudit",
        "JointActionChoice",
        "JointActionFactor",
        "JointActionSolution",
        "compose_joint_action",
    }:
        module = import_module("jormungandr.joint_actions")
        return getattr(module, name)
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
        "DynamicActionResult",
        "EntityCandidateBatch",
        "EntityCandidateObservation",
        "EntityCandidatePolicyOutput",
        "EntityCandidateTransformer",
        "TorchEntityCandidateBatch",
        "StructuredPolicySpec",
        "collate_entity_candidate_observations",
        "select_dynamic_actions",
    }:
        module = import_module("jormungandr.structured")
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
