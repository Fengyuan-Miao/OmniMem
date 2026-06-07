"""Configuration for the SVI-OmniMem extension."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ExtractionBudget:
    max_card_tokens: int = 220
    max_retrieval_anchors: int = 5
    max_attributes_per_anchor: int = 3
    max_ocr_observations: int = 4


@dataclass
class SVIConfig:
    schema_version: str = "svi_v1"
    extraction_budget: ExtractionBudget = field(default_factory=ExtractionBudget)
    retrieval_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "caption_dense": 0.22,
            "card_text": 1.00,
            "embedding_dense": 0.18,
            "entity_alias": 0.24,
            "attribute": 0.10,
            "ocr": 0.14,
            "global_visual_fallback": 0.04,
            "verified_fact": 0.06,
            "recent_fallback": 0.03,
        }
    )
    verification_enabled: bool = True
    verification_budget: int = 5
    batch_verification: bool = True
    allow_verifier_abstain: bool = True
    require_provenance: bool = True
    promote_verified_facts: bool = True
    min_verification_confidence: float = 0.80
    fallback_recent_images: int = 3
    fallback_same_session_images: int = 3
    enable_fallback_when_structured_score_below: float = 0.35
    index_text_mirror: bool = True
    store_only_requested_facts: bool = True
    deduplicate_same_fact: bool = True
    planner_mode: str = "off"
    planner_model: Optional[str] = None
    planner_temperature: float = 0.0
    planner_max_tokens: int = 700
    planner_json_repair_attempts: int = 1
    planner_fallback_mode: str = "generic_retrieve_verify"


DEFAULT_CONFIG = SVIConfig()
