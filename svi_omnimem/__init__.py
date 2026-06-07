"""SVI-OmniMem extension package.

This package keeps the SVI implementation outside the OmniSimpleMem source tree.
Pass an existing OmniMemoryOrchestrator into SVIOmniMemAdapter to use structured
visual indexing without changing the baseline code.
"""

from .adapter import SVIOmniMemAdapter
from .config import SVIConfig
from .executor import VisualMemoryPlanExecutor
from .extractor import StructuredVisualExtractor
from .models import (
    IndexedAttribute,
    OCRObservation,
    PlanExecutionResult,
    PlanStep,
    RetrievalAnchor,
    RetrievalCandidate,
    StructuredVisualCard,
    StructuredVisualQueryResult,
    VerifiedVisualFact,
    VerificationResult,
    VisualMemoryPlan,
    VisualQueryRequirement,
)
from .planner import VisualMemoryPlanner
from .promoter import VerifiedFactPromoter
from .query_parser import VisualQueryParser
from .retriever import StructuredVisualRetriever
from .stores import StructuredVisualStore, VerifiedFactStore
from .verifier import RawEvidenceVerifier

__all__ = [
    "SVIOmniMemAdapter",
    "SVIConfig",
    "StructuredVisualExtractor",
    "VisualMemoryPlanner",
    "VisualMemoryPlanExecutor",
    "VisualQueryParser",
    "StructuredVisualRetriever",
    "RawEvidenceVerifier",
    "VerifiedFactPromoter",
    "StructuredVisualStore",
    "VerifiedFactStore",
    "IndexedAttribute",
    "OCRObservation",
    "PlanExecutionResult",
    "PlanStep",
    "RetrievalAnchor",
    "RetrievalCandidate",
    "StructuredVisualCard",
    "StructuredVisualQueryResult",
    "VerifiedVisualFact",
    "VerificationResult",
    "VisualMemoryPlan",
    "VisualQueryRequirement",
]
