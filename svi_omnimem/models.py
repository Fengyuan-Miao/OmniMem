"""Core data models for Structured Visual Indexing."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils import ensure_iso_time, safe_float, unique_list, utcnow_iso

UNVERIFIED = "unverified_extraction"
QUERY_VERIFIED = "query_verified"
CONTRADICTED = "contradicted"
SUPERSEDED = "superseded"
UNCERTAIN = "uncertain"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


@dataclass
class IndexedAttribute:
    value: str

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: Any) -> "IndexedAttribute":
        if isinstance(data, str):
            return cls(value=data)
        data = data or {}
        return cls(value=str(data.get("value", "")))


@dataclass
class RetrievalAnchor:
    anchor_id: str
    category: str
    salient_attributes: Dict[str, IndexedAttribute] = field(default_factory=dict)

    def all_names(self) -> List[str]:
        return unique_list([self.category])

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "anchor_id": self.anchor_id,
            "category": self.category,
        }
        if self.salient_attributes:
            data["salient_attributes"] = {
                key: value.to_dict()
                for key, value in self.salient_attributes.items()
            }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalAnchor":
        attrs = data.get("salient_attributes") or {}
        return cls(
            anchor_id=str(data.get("anchor_id") or _new_id("anchor")),
            category=str(data.get("category") or data.get("name") or "object"),
            salient_attributes={
                str(k): IndexedAttribute.from_dict(v) for k, v in attrs.items()
            },
        )


@dataclass
class VisualRelation:
    subject_anchor_id: str
    predicate: str
    object_anchor_id: Optional[str] = None
    object_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_anchor_id": self.subject_anchor_id,
            "predicate": self.predicate,
            "object_anchor_id": self.object_anchor_id,
            "object_text": self.object_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualRelation":
        return cls(
            subject_anchor_id=str(data.get("subject_anchor_id", "")),
            predicate=str(data.get("predicate", "")),
            object_anchor_id=data.get("object_anchor_id"),
            object_text=data.get("object_text"),
        )


@dataclass
class OCRObservation:
    text: str
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {"text": self.text}
        if self.context:
            data["context"] = self.context
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRObservation":
        return cls(
            text=str(data.get("text", "")),
            context=data.get("context"),
        )


@dataclass
class StateObservation:
    slot: List[str]
    value: str
    observed_at: str
    source_anchor_id: Optional[str] = None

    def slot_key(self) -> str:
        return ".".join(part.strip().lower() for part in self.slot if part)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "value": self.value,
            "observed_at": self.observed_at,
            "source_anchor_id": self.source_anchor_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateObservation":
        return cls(
            slot=[str(v) for v in (data.get("slot") or [])],
            value=str(data.get("value", "")),
            observed_at=ensure_iso_time(data.get("observed_at")),
            source_anchor_id=data.get("source_anchor_id"),
        )


@dataclass
class StructuredVisualCard:
    card_id: str
    image_mau_id: str
    session_id: Optional[str]
    turn_id: Optional[str]
    observed_at: str
    raw_pointer: str
    global_caption: str
    retrieval_anchors: List[RetrievalAnchor] = field(default_factory=list)
    relations: List[VisualRelation] = field(default_factory=list)
    ocr_observations: List[OCRObservation] = field(default_factory=list)
    state_observations: List[StateObservation] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    source_text_context: str = ""
    extraction_scope: str = "salient_entities_only"
    schema_version: str = "svi_v1"

    @classmethod
    def new(
        cls,
        image_mau_id: str,
        raw_pointer: str,
        global_caption: str,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        observed_at: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        source_text_context: str = "",
        schema_version: str = "svi_v1",
    ) -> "StructuredVisualCard":
        return cls(
            card_id=_new_id("visual_card"),
            image_mau_id=image_mau_id,
            session_id=session_id,
            turn_id=turn_id,
            observed_at=ensure_iso_time(observed_at),
            raw_pointer=raw_pointer,
            global_caption=global_caption or "Image captured",
            tags=tags or [],
            source_text_context=source_text_context,
            schema_version=schema_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "card_id": self.card_id,
            "image_mau_id": self.image_mau_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "observed_at": self.observed_at,
            "raw_pointer": self.raw_pointer,
            "global_caption": self.global_caption,
            "tags": self.tags,
            "source_text_context": self.source_text_context,
        }
        if self.retrieval_anchors:
            data["retrieval_anchors"] = [
                item.to_dict() for item in self.retrieval_anchors
            ]
        if self.ocr_observations:
            data["ocr_observations"] = [
                item.to_dict() for item in self.ocr_observations
            ]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredVisualCard":
        return cls(
            card_id=str(data.get("card_id") or _new_id("visual_card")),
            image_mau_id=str(data.get("image_mau_id", "")),
            session_id=data.get("session_id"),
            turn_id=data.get("turn_id"),
            observed_at=ensure_iso_time(data.get("observed_at")),
            raw_pointer=str(data.get("raw_pointer", "")),
            global_caption=str(data.get("global_caption") or "Image captured"),
            retrieval_anchors=[
                RetrievalAnchor.from_dict(item)
                for item in data.get("retrieval_anchors", []) or []
            ],
            relations=[
                VisualRelation.from_dict(item)
                for item in data.get("relations", []) or []
            ],
            ocr_observations=[
                OCRObservation.from_dict(item)
                for item in data.get("ocr_observations", []) or []
            ],
            state_observations=[
                StateObservation.from_dict(item)
                for item in data.get("state_observations", []) or []
            ],
            tags=unique_list(data.get("tags") or []),
            source_text_context=str(data.get("source_text_context") or ""),
            extraction_scope=data.get("extraction_scope", "salient_entities_only"),
            schema_version=data.get("schema_version", "svi_v1"),
        )

    def to_mirror_text(self) -> str:
        entity_parts = []
        for anchor in self.retrieval_anchors:
            attrs = []
            for name, attr in anchor.salient_attributes.items():
                attrs.append(f"{name}={attr.value}")
            alias_text = "/".join(anchor.all_names())
            if attrs:
                entity_parts.append(f"{alias_text} ({', '.join(attrs)})")
            else:
                entity_parts.append(alias_text)

        ocr_parts = [
            f'"{obs.text}"' + (f" in {obs.context}" if obs.context else "")
            for obs in self.ocr_observations
        ]
        return "\n".join(
            line
            for line in [
                "[Structured visual index; raw image verification required for fine-grained claims]",
                f"Scene: {self.global_caption}",
                "Entities: " + "; ".join(entity_parts) if entity_parts else "",
                "OCR: " + "; ".join(ocr_parts) if ocr_parts else "",
                f"Source dialogue turn: {self.source_text_context[:800]}"
                if self.source_text_context
                else "",
                f"Source image: {self.image_mau_id}",
                "Verification status: extracted, not yet query-verified.",
            ]
            if line
        )


@dataclass
class VerifiedVisualFact:
    fact_id: str
    source_image_mau_id: str
    source_card_id: str
    subject: str
    predicate: str
    value: str
    evidence_description: str
    observation_time: str
    verified_at: str
    confidence: float
    query_type: str
    evidence_scope: str
    raw_pointer: str
    status: str = QUERY_VERIFIED

    @classmethod
    def new(
        cls,
        source_image_mau_id: str,
        source_card_id: str,
        subject: str,
        predicate: str,
        value: str,
        evidence_description: str,
        observation_time: str,
        confidence: float,
        query_type: str,
        evidence_scope: str,
        raw_pointer: str,
    ) -> "VerifiedVisualFact":
        return cls(
            fact_id=_new_id("verified_fact"),
            source_image_mau_id=source_image_mau_id,
            source_card_id=source_card_id,
            subject=subject,
            predicate=predicate,
            value=value,
            evidence_description=evidence_description,
            observation_time=observation_time,
            verified_at=utcnow_iso(),
            confidence=confidence,
            query_type=query_type,
            evidence_scope=evidence_scope,
            raw_pointer=raw_pointer,
        )

    def key(self) -> str:
        return "|".join(
            [
                self.source_image_mau_id,
                self.subject.lower(),
                self.predicate.lower(),
                self.value.lower(),
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "source_image_mau_id": self.source_image_mau_id,
            "source_card_id": self.source_card_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "evidence_description": self.evidence_description,
            "observation_time": self.observation_time,
            "verified_at": self.verified_at,
            "confidence": self.confidence,
            "query_type": self.query_type,
            "evidence_scope": self.evidence_scope,
            "raw_pointer": self.raw_pointer,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifiedVisualFact":
        return cls(
            fact_id=str(data.get("fact_id") or _new_id("verified_fact")),
            source_image_mau_id=str(data.get("source_image_mau_id", "")),
            source_card_id=str(data.get("source_card_id", "")),
            subject=str(data.get("subject", "")),
            predicate=str(data.get("predicate", "")),
            value=str(data.get("value", "")),
            evidence_description=str(data.get("evidence_description", "")),
            observation_time=ensure_iso_time(data.get("observation_time")),
            verified_at=ensure_iso_time(data.get("verified_at")),
            confidence=safe_float(data.get("confidence")),
            query_type=str(data.get("query_type", "visual")),
            evidence_scope=str(data.get("evidence_scope", "full_image")),
            raw_pointer=str(data.get("raw_pointer", "")),
            status=data.get("status", QUERY_VERIFIED),
        )


@dataclass
class VisualQueryRequirement:
    requires_visual_evidence: bool
    entities: List[str] = field(default_factory=list)
    requested_attributes: List[str] = field(default_factory=list)
    relation_constraints: List[str] = field(default_factory=list)
    ocr_terms: List[str] = field(default_factory=list)
    state_slots: List[List[str]] = field(default_factory=list)
    temporal_scope: str = "any"
    target_time: Optional[str] = None
    requires_raw_verification: bool = False
    query_type: str = "visual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requires_visual_evidence": self.requires_visual_evidence,
            "entities": self.entities,
            "requested_attributes": self.requested_attributes,
            "relation_constraints": self.relation_constraints,
            "ocr_terms": self.ocr_terms,
            "state_slots": self.state_slots,
            "temporal_scope": self.temporal_scope,
            "target_time": self.target_time,
            "requires_raw_verification": self.requires_raw_verification,
            "query_type": self.query_type,
        }


@dataclass
class RetrievalCandidate:
    image_mau_id: str
    card_id: str
    score: float
    routes: List[str] = field(default_factory=list)
    matched_fields: List[str] = field(default_factory=list)
    observation_time: Optional[str] = None
    raw_pointer: Optional[str] = None

    def merge(self, route: str, score: float, matched_field: str) -> None:
        self.score += score
        if route not in self.routes:
            self.routes.append(route)
        if matched_field and matched_field not in self.matched_fields:
            self.matched_fields.append(matched_field)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_mau_id": self.image_mau_id,
            "card_id": self.card_id,
            "score": self.score,
            "routes": self.routes,
            "matched_fields": self.matched_fields,
            "observation_time": self.observation_time,
            "raw_pointer": self.raw_pointer,
        }


@dataclass
class PlanStep:
    op: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "args": self.args,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        return cls(
            op=str(data.get("op", "")),
            args=dict(data.get("args") or {}),
        )


@dataclass
class VisualMemoryPlan:
    plan_id: str
    query: str
    scope: Dict[str, Any] = field(default_factory=dict)
    steps: List[PlanStep] = field(default_factory=list)
    answer: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    planner: str = "deterministic"
    requires_raw_verification: bool = True

    @classmethod
    def new(
        cls,
        query: str,
        scope: Optional[Dict[str, Any]] = None,
        steps: Optional[List[PlanStep]] = None,
        answer: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        planner: str = "deterministic",
        requires_raw_verification: bool = True,
    ) -> "VisualMemoryPlan":
        return cls(
            plan_id=_new_id("visual_plan"),
            query=query,
            scope=scope or {"modality": "image", "session": "all"},
            steps=steps or [],
            answer=answer or {"type": "short_text", "target": "visual_answer"},
            confidence=confidence,
            planner=planner,
            requires_raw_verification=requires_raw_verification,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "scope": self.scope,
            "steps": [step.to_dict() for step in self.steps],
            "answer": self.answer,
            "confidence": self.confidence,
            "planner": self.planner,
            "requires_raw_verification": self.requires_raw_verification,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualMemoryPlan":
        return cls(
            plan_id=str(data.get("plan_id") or _new_id("visual_plan")),
            query=str(data.get("query", "")),
            scope=dict(data.get("scope") or {}),
            steps=[PlanStep.from_dict(item) for item in data.get("steps", []) or []],
            answer=dict(data.get("answer") or {}),
            confidence=safe_float(data.get("confidence")),
            planner=str(data.get("planner", "deterministic")),
            requires_raw_verification=bool(data.get("requires_raw_verification", True)),
        )


@dataclass
class PlanExecutionResult:
    plan: VisualMemoryPlan
    requirement: VisualQueryRequirement
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        retrieval_policy = {
            "mode": "planned",
            "requires_raw_verification": self.requirement.requires_raw_verification,
        }
        return {
            "plan": self.plan.to_dict(),
            "retrieval_policy": retrieval_policy,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "claims": self.claims,
            "execution_trace": self.execution_trace,
        }


@dataclass
class VerificationResult:
    supports: bool
    answer_fragment: str = ""
    visible_evidence: str = ""
    verified_facts: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    source_card_id: Optional[str] = None
    source_image_mau_id: Optional[str] = None
    raw_pointer: Optional[str] = None
    observation_time: Optional[str] = None
    error: Optional[str] = None
    abstained: bool = False

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        source_card_id: Optional[str] = None,
        source_image_mau_id: Optional[str] = None,
        raw_pointer: Optional[str] = None,
        observation_time: Optional[str] = None,
    ) -> "VerificationResult":
        return cls(
            supports=bool(data.get("supports", False)),
            answer_fragment=str(data.get("answer_fragment") or ""),
            visible_evidence=str(
                data.get("visible_evidence") or data.get("evidence_description") or ""
            ),
            verified_facts=list(data.get("verified_facts") or []),
            confidence=safe_float(data.get("confidence")),
            source_card_id=source_card_id,
            source_image_mau_id=source_image_mau_id,
            raw_pointer=raw_pointer,
            observation_time=observation_time or data.get("observation_time"),
            abstained=bool(data.get("abstained", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supports": self.supports,
            "answer_fragment": self.answer_fragment,
            "visible_evidence": self.visible_evidence,
            "verified_facts": self.verified_facts,
            "confidence": self.confidence,
            "source_card_id": self.source_card_id,
            "source_image_mau_id": self.source_image_mau_id,
            "raw_pointer": self.raw_pointer,
            "observation_time": self.observation_time,
            "error": self.error,
            "abstained": self.abstained,
        }


@dataclass
class StructuredVisualQueryResult:
    query: str
    requirement: VisualQueryRequirement
    candidates: List[RetrievalCandidate]
    verified_evidence: List[VerificationResult] = field(default_factory=list)
    promoted_facts: List[VerifiedVisualFact] = field(default_factory=list)
    retrieval_claims: List[Dict[str, Any]] = field(default_factory=list)
    answer_context: str = ""
    plan: Optional[VisualMemoryPlan] = None
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        retrieval_policy = {
            "mode": "generic_card_text",
            "requires_raw_verification": self.requirement.requires_raw_verification,
        }
        if self.plan is not None:
            retrieval_policy["mode"] = "planned"
        return {
            "query": self.query,
            "retrieval_policy": retrieval_policy,
            "candidates": [item.to_dict() for item in self.candidates],
            "verified_evidence": [item.to_dict() for item in self.verified_evidence],
            "promoted_facts": [item.to_dict() for item in self.promoted_facts],
            "retrieval_claims": self.retrieval_claims,
            "answer_context": self.answer_context,
            "plan": self.plan.to_dict() if self.plan else None,
            "execution_trace": self.execution_trace,
        }
