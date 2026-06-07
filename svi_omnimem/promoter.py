"""Verified visual fact promotion for SVI-OmniMem."""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import SVIConfig
from .models import VerifiedVisualFact, VerificationResult, VisualQueryRequirement
from .stores import VerifiedFactStore


class VerifiedFactPromoter:
    """Persist conservative query-verified facts for reuse."""

    def __init__(
        self,
        fact_store: VerifiedFactStore,
        config: Optional[SVIConfig] = None,
    ):
        self.fact_store = fact_store
        self.config = config or SVIConfig()

    def promote(
        self,
        verification_results: List[VerificationResult],
        requirement: VisualQueryRequirement,
    ) -> List[VerifiedVisualFact]:
        if not self.config.promote_verified_facts:
            return []

        promoted: List[VerifiedVisualFact] = []
        for result in verification_results:
            if not self._eligible_result(result):
                continue
            raw_facts = result.verified_facts or [
                {
                    "subject": "visual_memory",
                    "predicate": requirement.query_type,
                    "value": result.answer_fragment,
                    "evidence_description": result.visible_evidence,
                    "evidence_scope": "full_image",
                }
            ]
            for raw_fact in raw_facts:
                fact = self._build_fact(raw_fact, result, requirement)
                if not fact:
                    continue
                if self.fact_store.conflicting_after(
                    fact.subject,
                    fact.predicate,
                    fact.value,
                    fact.observation_time,
                ):
                    continue
                self.fact_store.append(
                    fact,
                    deduplicate=self.config.deduplicate_same_fact,
                )
                promoted.append(fact)
        return promoted

    def _eligible_result(self, result: VerificationResult) -> bool:
        if not result.supports or result.abstained:
            return False
        if result.confidence < self.config.min_verification_confidence:
            return False
        if not result.answer_fragment.strip() and not result.verified_facts:
            return False
        if not result.raw_pointer:
            return False
        if not result.source_card_id or not result.source_image_mau_id:
            return False
        return True

    def _build_fact(
        self,
        raw_fact: Dict,
        result: VerificationResult,
        requirement: VisualQueryRequirement,
    ) -> Optional[VerifiedVisualFact]:
        subject = str(raw_fact.get("subject") or "visual_memory").strip()
        predicate = str(raw_fact.get("predicate") or requirement.query_type).strip()
        value = str(raw_fact.get("value") or result.answer_fragment).strip()
        if not subject or not predicate or not value:
            return None
        return VerifiedVisualFact.new(
            source_image_mau_id=result.source_image_mau_id or "",
            source_card_id=result.source_card_id or "",
            subject=subject,
            predicate=predicate,
            value=value,
            evidence_description=str(
                raw_fact.get("evidence_description")
                or raw_fact.get("visible_evidence")
                or result.visible_evidence
            ),
            observation_time=str(
                raw_fact.get("observation_time") or result.observation_time or ""
            ),
            confidence=result.confidence,
            query_type=requirement.query_type,
            evidence_scope=str(raw_fact.get("evidence_scope") or "full_image"),
            raw_pointer=result.raw_pointer or "",
        )
