"""Evaluation framework for RB-KAIROS parameter studies."""

from evaluation.campaign import (
    CampaignSpec,
    build_candidate_manifest,
    build_initial_campaign_state,
    load_campaign_spec,
    validate_campaign_state,
)
from evaluation.config import EvaluationConfig, load_config
from evaluation.campaign_executor import execute_managed_campaign
from evaluation.costs import (
    CostBreakdown,
    compute_multispeed_total_cost,
    compute_traction_cost,
)
from evaluation.evaluators import evaluate_frf_offline, evaluate_traction_offline
from evaluation.managed import evaluate_managed
from evaluation.metadata_validation import (
    RuntimeMetadataValidation,
    validate_runtime_metadata,
    validate_runtime_metadata_multi,
)
from evaluation.parameters import EvaluationParameters
from evaluation.result import EvaluationResult, evaluate_offline

__all__ = [
    "CampaignSpec",
    "CostBreakdown",
    "EvaluationConfig",
    "EvaluationParameters",
    "EvaluationResult",
    "RuntimeMetadataValidation",
    "build_candidate_manifest",
    "build_initial_campaign_state",
    "compute_multispeed_total_cost",
    "compute_traction_cost",
    "evaluate_frf_offline",
    "evaluate_managed",
    "execute_managed_campaign",
    "evaluate_offline",
    "evaluate_traction_offline",
    "load_config",
    "load_campaign_spec",
    "validate_runtime_metadata",
    "validate_runtime_metadata_multi",
    "validate_campaign_state",
]
