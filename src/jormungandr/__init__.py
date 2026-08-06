"""Reusable Jörmungandr service runtime (generic trainer/checkpoint/inference over HTTP)."""

from importlib import import_module
from typing import Any

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "JormungandrRuntime",
    "JormungandrHttpServer",
    "BoundedIntegerRadixCodec",
    "PAIRWISE_SUMMARY_SCHEMA",
    "PairwiseOutcome",
    "TASK_ASSIGNMENT_SCHEMA",
    "TASK_ASSIGNMENT_OBJECTIVES",
    "TaskAssignmentCandidate",
    "TaskAssignmentChoice",
    "TaskAssignmentResult",
    "CONSTRAINED_TASK_ASSIGNMENT_SCHEMA",
    "ConstrainedTaskAssignmentCandidate",
    "ConstrainedTaskAssignmentChoice",
    "ConstrainedTaskAssignmentResult",
    "solve_resource_constrained_task_assignment",
    "RANKING_METRICS_SCHEMA",
    "SELECTION_SET_METRICS_SCHEMA",
    "SELECTION_MULTISET_METRICS_SCHEMA",
    "RankingMetricsAccumulator",
    "SelectionMultisetMetricsAccumulator",
    "SelectionSetMetricsAccumulator",
    "SHORTEST_PATH_SCHEMA",
    "DirectedRouteEdge",
    "ShortestPathResult",
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
    "STRUCTURED_CHECKPOINT_SCHEMA",
    "save_structured_checkpoint",
    "structured_checkpoint_payload",
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
    "apply_candidate_prefix",
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
    "apply_candidate_prefix_numpy",
    "structured_joint_step_from_payload",
    "structured_joint_step_to_payload",
    "structured_joint_trajectory_from_sequence_payload",
    "structured_joint_trajectory_to_sequence_payload",
    "validate_structured_joint_trajectory",
    "STRUCTURED_SUPERVISION_SCHEMA",
    "STRUCTURED_SUPERVISION_FRAME_SCHEMA",
    "STRUCTURED_SUPERVISION_CEILING_SCHEMA",
    "STRUCTURED_SUPERVISION_TIME_DEPENDENCE_SCHEMA",
    "STRUCTURED_SUPERVISION_STRATIFIED_SUBSET_SCHEMA",
    "StructuredSupervisionExample",
    "StructuredSupervisionFrame",
    "StructuredSupervisionLabel",
    "structured_supervision_balance_weights",
    "structured_supervision_balance_weights_from_counts",
    "structured_supervision_frame_balance_weights",
    "structured_supervision_examples_from_frame",
    "apply_structured_supervision_balance_weights",
    "apply_structured_supervision_frame_balance_weights",
    "structured_supervision_from_payload",
    "structured_supervision_frame_from_payload",
    "structured_supervision_frame_to_payload",
    "structured_supervision_to_payload",
    "structured_supervision_deterministic_ceiling",
    "structured_supervision_model_input_fingerprint",
    "structured_supervision_time_dependence",
    "structured_supervision_stratified_subset",
    "STRUCTURED_SUPERVISION_POLICY_METRICS_SCHEMA",
    "STRUCTURED_SUPERVISION_FRAME_POLICY_METRICS_SCHEMA",
    "StructuredSupervisionFrameMetricsAccumulator",
    "StructuredSupervisionMetricsAccumulator",
    "structured_supervision_frame_policy_metrics",
    "structured_supervision_policy_metrics",
    "PYTHON_POLICY_SOURCE_AUDIT_SCHEMA",
    "POLICY_COUNTERFACTUAL_RESPONSE_SCHEMA",
    "PolicyCounterfactualOutcome",
    "audit_python_policy_source",
    "summarize_policy_counterfactual_responses",
    "STRUCTURED_BUNDLE_SCHEMA",
    "export_structured_policy_bundle",
    "load_structured_policy_bundle",
    "fit_bradley_terry",
    "outcome_from_values",
    "pairwise_outcome_from_payload",
    "summarize_pairwise_outcomes",
    "solve_shortest_path",
    "solve_task_assignment",
]


def __getattr__(name: str) -> Any:
    if name == "BoundedIntegerRadixCodec":
        module = import_module("jormungandr.bounded_integer")
        return getattr(module, name)
    if name in {
        "PAIRWISE_SUMMARY_SCHEMA",
        "PairwiseOutcome",
        "fit_bradley_terry",
        "outcome_from_values",
        "pairwise_outcome_from_payload",
        "summarize_pairwise_outcomes",
    }:
        module = import_module("jormungandr.pairwise")
        return getattr(module, name)
    if name in {
        "STRUCTURED_CHECKPOINT_SCHEMA",
        "save_structured_checkpoint",
        "structured_checkpoint_payload",
    }:
        module = import_module("jormungandr.structured_checkpoint")
        return getattr(module, name)
    if name in {
        "STRUCTURED_SUPERVISION_FRAME_POLICY_METRICS_SCHEMA",
        "STRUCTURED_SUPERVISION_POLICY_METRICS_SCHEMA",
        "StructuredSupervisionFrameMetricsAccumulator",
        "StructuredSupervisionMetricsAccumulator",
        "structured_supervision_frame_policy_metrics",
        "structured_supervision_policy_metrics",
    }:
        module = import_module("jormungandr.structured_metrics")
        return getattr(module, name)
    if name in {
        "RANKING_METRICS_SCHEMA",
        "SELECTION_SET_METRICS_SCHEMA",
        "SELECTION_MULTISET_METRICS_SCHEMA",
        "RankingMetricsAccumulator",
        "SelectionMultisetMetricsAccumulator",
        "SelectionSetMetricsAccumulator",
    }:
        module = import_module("jormungandr.ranking_metrics")
        return getattr(module, name)
    if name in {
        "CONSTRAINED_TASK_ASSIGNMENT_SCHEMA",
        "ConstrainedTaskAssignmentCandidate",
        "ConstrainedTaskAssignmentChoice",
        "ConstrainedTaskAssignmentResult",
        "solve_resource_constrained_task_assignment",
    }:
        module = import_module("jormungandr.constrained_assignment")
        return getattr(module, name)
    if name in {
        "SHORTEST_PATH_SCHEMA",
        "DirectedRouteEdge",
        "ShortestPathResult",
        "solve_shortest_path",
    }:
        module = import_module("jormungandr.path_planning")
        return getattr(module, name)
    if name in {
        "TASK_ASSIGNMENT_SCHEMA",
        "TASK_ASSIGNMENT_OBJECTIVES",
        "TaskAssignmentCandidate",
        "TaskAssignmentChoice",
        "TaskAssignmentResult",
        "solve_task_assignment",
    }:
        module = import_module("jormungandr.task_assignment")
        return getattr(module, name)
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
        "apply_candidate_prefix_numpy",
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
        "STRUCTURED_SUPERVISION_FRAME_SCHEMA",
        "StructuredSupervisionExample",
        "StructuredSupervisionFrame",
        "StructuredSupervisionLabel",
        "structured_supervision_balance_weights",
        "structured_supervision_balance_weights_from_counts",
        "structured_supervision_frame_balance_weights",
        "structured_supervision_examples_from_frame",
        "apply_structured_supervision_balance_weights",
        "apply_structured_supervision_frame_balance_weights",
        "structured_supervision_from_payload",
        "structured_supervision_frame_from_payload",
        "structured_supervision_frame_to_payload",
        "structured_supervision_to_payload",
    }:
        module = import_module("jormungandr.structured_supervision")
        return getattr(module, name)
    if name in {
        "STRUCTURED_SUPERVISION_CEILING_SCHEMA",
        "STRUCTURED_SUPERVISION_TIME_DEPENDENCE_SCHEMA",
        "STRUCTURED_SUPERVISION_STRATIFIED_SUBSET_SCHEMA",
        "structured_supervision_deterministic_ceiling",
        "structured_supervision_model_input_fingerprint",
        "structured_supervision_time_dependence",
        "structured_supervision_stratified_subset",
    }:
        module = import_module("jormungandr.supervision_diagnostics")
        return getattr(module, name)
    if name in {
        "PYTHON_POLICY_SOURCE_AUDIT_SCHEMA",
        "POLICY_COUNTERFACTUAL_RESPONSE_SCHEMA",
        "PolicyCounterfactualOutcome",
        "audit_python_policy_source",
        "summarize_policy_counterfactual_responses",
    }:
        module = import_module("jormungandr.policy_provenance")
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
        "apply_candidate_prefix",
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
