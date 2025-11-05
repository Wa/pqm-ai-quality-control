"""Utilities for文件要素检查."""

from .config import (
    DELIVERABLE_INDEX,
    DELIVERABLE_PROFILES,
    PHASE_TO_DELIVERABLES,
    SEVERITY_LABELS,
    SEVERITY_ORDER,
    DeliverableProfile,
    ElementRequirement,
)
from .evaluator import (
    ElementEvaluation,
    EvaluationOrchestrator,
    EvaluationResult,
    auto_convert_sources,
    parse_deliverable_stub,
    save_result_payload,
)

__all__ = [
    "DeliverableProfile",
    "ElementRequirement",
    "DELIVERABLE_INDEX",
    "DELIVERABLE_PROFILES",
    "PHASE_TO_DELIVERABLES",
    "SEVERITY_LABELS",
    "SEVERITY_ORDER",
    "ElementEvaluation",
    "EvaluationOrchestrator",
    "EvaluationResult",
    "auto_convert_sources",
    "parse_deliverable_stub",
    "save_result_payload",
]
