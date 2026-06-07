"""VLM-first query planner for Plan-Grounded SVI."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import SVIConfig
from .models import PlanStep, VisualMemoryPlan, VisualQueryRequirement
from .stores import StructuredVisualStore, VerifiedFactStore
from .utils import extract_first_json_object, safe_float, unique_list

logger = logging.getLogger(__name__)


ALLOWED_OPERATORS = {
    "scope",
    "retrieve",
    "filter",
    "order_by",
    "group_by",
    "aggregate",
    "prioritize",
    "diversify",
    "limit",
    "verify_raw_image",
    "answer",
}

ALLOWED_RETRIEVAL_ROUTES = {
    "caption_dense",
    "entity_alias",
    "attribute",
    "relation",
    "ocr",
    "temporal_state",
    "verified_fact",
    "recent_fallback",
}


class VisualMemoryPlanner:
    """Compile a query into VisualMemoryPlan using a text-only VLM call.

    The planner does not use keyword lists for query semantics. It asks the VLM
    to choose generic memory operators, then validates the returned plan against
    a small schema. If planning fails, it falls back to a conservative generic
    retrieve-and-verify plan.
    """

    def __init__(
        self,
        orchestrator: Any = None,
        config: Optional[SVIConfig] = None,
        visual_store: Optional[StructuredVisualStore] = None,
        verified_fact_store: Optional[VerifiedFactStore] = None,
    ):
        self.orchestrator = orchestrator
        self.config = config or SVIConfig()
        self.visual_store = visual_store
        self.verified_fact_store = verified_fact_store

    def plan(self, query: str) -> Tuple[VisualMemoryPlan, VisualQueryRequirement]:
        if self.config.planner_mode == "deterministic_legacy":
            plan = self._fallback_plan(query, reason="legacy_mode_disabled")
            return plan, self.requirement_from_plan(plan)

        try:
            payload = self._call_vlm_planner(query)
            plan = self._parse_plan_json(query, payload)
            plan = self._repair_plan_if_needed(plan)
            self._validate_plan(plan)
            return plan, self.requirement_from_plan(plan)
        except Exception as exc:
            logger.warning("SVI VLM planner failed, using generic fallback: %s", exc)
            plan = self._fallback_plan(query, reason=f"{type(exc).__name__}: {exc}")
            return plan, self.requirement_from_plan(plan)

    def requirement_from_plan(self, plan: VisualMemoryPlan) -> VisualQueryRequirement:
        if plan.planner == "generic_retrieve_verify_fallback":
            return VisualQueryRequirement(
                requires_visual_evidence=True,
                entities=[],
                requested_attributes=[],
                relation_constraints=[],
                ocr_terms=[],
                state_slots=[],
                temporal_scope="any",
                target_time=None,
                requires_raw_verification=plan.requires_raw_verification,
                query_type="visual",
            )

        routes = set()
        entities: List[str] = []
        attrs: List[str] = []
        ocr_terms: List[str] = []
        relation_constraints: List[str] = []
        state_slots: List[List[str]] = []
        query_type = str(plan.answer.get("target") or "visual_answer")
        temporal_scope = "any"

        for step in plan.steps:
            args = step.args or {}
            if step.op == "retrieve":
                routes.update(str(route) for route in args.get("routes", []) or [])
                entities.extend(str(item) for item in args.get("entities", []) or [])
                attrs.extend(str(item) for item in args.get("attributes", []) or [])
                ocr_terms.extend(str(item) for item in args.get("ocr_terms", []) or [])
                relation_constraints.extend(
                    str(item) for item in args.get("relations", []) or []
                )
                for slot in args.get("state_slots", []) or []:
                    if isinstance(slot, list):
                        state_slots.append([str(item) for item in slot])
            elif step.op == "order_by" and args.get("field") == "observed_at":
                direction = str(args.get("direction", "")).lower()
                if direction == "asc":
                    temporal_scope = "earliest_observed"
                elif direction == "desc":
                    temporal_scope = "latest_observed"

        if "ocr" in routes:
            query_type = "ocr"
        elif "relation" in routes:
            query_type = "relation"
        elif "attribute" in routes:
            query_type = "attribute"
        elif any(step.op == "aggregate" for step in plan.steps):
            query_type = "count"
        elif temporal_scope != "any":
            query_type = "latest_state"

        return VisualQueryRequirement(
            requires_visual_evidence=True,
            entities=unique_list(entities, 12),
            requested_attributes=unique_list(attrs, 12),
            relation_constraints=unique_list(relation_constraints, 12),
            ocr_terms=unique_list(ocr_terms, 12),
            state_slots=state_slots[:4],
            temporal_scope=temporal_scope,
            target_time=None,
            requires_raw_verification=plan.requires_raw_verification,
            query_type=query_type,
        )

    def _call_vlm_planner(
        self,
        query: str,
        repair_feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.orchestrator:
            raise RuntimeError("planner_orchestrator_missing")
        client = self.orchestrator._get_llm_client()
        model = (
            self.config.planner_model
            or getattr(self.orchestrator.config.llm, "caption_model", None)
        )
        messages = [
            {
                "role": "user",
                "content": self._build_prompt(query, repair_feedback=repair_feedback),
            }
        ]
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": self.config.planner_temperature,
            "max_tokens": self.config.planner_max_tokens,
        }
        try:
            response = client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.debug("SVI planner JSON mode unavailable, retrying plain: %s", exc)
            response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        data = extract_first_json_object(text)
        if not data:
            raise ValueError("planner_non_json")
        return data

    def _build_prompt(
        self,
        query: str,
        repair_feedback: Optional[str] = None,
    ) -> str:
        stats = self._index_stats()
        repair_section = ""
        if repair_feedback:
            repair_section = f"""
Previous plan issue:
{repair_feedback}

Return a corrected JSON plan only.
"""
        return f"""You are the autonomous query planner for a multimodal memory system.

Convert the user question into a JSON VisualMemoryPlan. Do not answer the question.
Choose generic memory operators instead of relying on fixed keywords.

Available memory fields:
- StructuredVisualCard: card_id, image_mau_id, session_id, turn_id, observed_at, raw_pointer, global_caption, retrieval_anchors, relations, ocr_observations, state_observations, tags
- VerifiedVisualFact: subject, predicate, value, observation_time, confidence, query_type, source_card_id, source_image_mau_id

Allowed operators:
- scope: choose modality/session/time constraints
- retrieve: select retrieval routes and optional semantic hints
- filter: filter candidates by field/value, use soft=true if uncertain
- order_by: order candidates by observed_at or score
- group_by: group candidates by a field
- aggregate: aggregate grouped candidates, e.g. count
- prioritize: prefer candidates with routes or fields
- diversify: keep diverse candidates
- limit: keep k candidates, k can be an integer or "verification_budget"
- verify_raw_image: request raw image verification
- answer: describe target answer type

Allowed retrieval routes:
caption_dense, entity_alias, attribute, relation, ocr, temporal_state, verified_fact, recent_fallback

Planning policy:
- First decide the reasoning need: semantic lookup, visual attribute verification, text/OCR verification, counting/aggregation, or memory-order selection.
- If the question asks for an item by its position in the conversation/memory timeline, do NOT use semantic retrieval as the selector. Use all in-scope visual cards, then order_by observed_at, then limit 1.
- Use order_by observed_at asc for earliest/first memory-order selection; use order_by observed_at desc for latest/last memory-order selection.
- If a plan uses order_by observed_at to select one image, place limit 1 before verify_raw_image.
- Use retrieve only when the question names a specific entity, visual attribute, relation, OCR text, or semantic content to locate.
- StructuredVisualCard fields are retrieval hints only. Raw visual facts must be verified with verify_raw_image.

Examples of operator selection:
- Question needs the earliest image in memory: scope -> order_by observed_at asc -> limit 1 -> verify_raw_image -> answer.
- Question needs the latest image in memory: scope -> order_by observed_at desc -> limit 1 -> verify_raw_image -> answer.
- Question asks a visual attribute of a named object: scope -> retrieve(entity_alias, attribute, caption_dense) -> limit verification_budget -> verify_raw_image -> answer.
- Question asks how many images satisfy a condition: scope -> retrieve or filter -> aggregate count -> verify_raw_image -> answer.

Index stats:
{json.dumps(stats, ensure_ascii=False)}

Return JSON only with this shape:
{{
  "scope": {{"modality": "image", "session": "all|current_or_all", "time_range": null}},
  "steps": [
    {{"op": "scope", "args": {{"modality": "image", "session": "all"}}}},
    {{"op": "retrieve", "args": {{"routes": ["caption_dense", "entity_alias"], "entities": [], "attributes": [], "ocr_terms": [], "relations": [], "state_slots": []}}}},
    {{"op": "limit", "args": {{"k": "verification_budget"}}}},
    {{"op": "verify_raw_image", "args": {{"ask": "what to verify in the raw image"}}}},
    {{"op": "answer", "args": {{"type": "short_text", "target": "visual_answer"}}}}
  ],
  "answer": {{"type": "short_text", "target": "visual_answer"}},
  "requires_raw_verification": true,
  "confidence": 0.0
}}

{repair_section}
User question: {query}
"""

    def _index_stats(self) -> Dict[str, Any]:
        cards = self.visual_store.all_cards() if self.visual_store else []
        facts = self.verified_fact_store.all_facts() if self.verified_fact_store else []
        times = sorted(card.observed_at for card in cards if card.observed_at)
        sessions = sorted({card.session_id for card in cards if card.session_id})
        return {
            "num_visual_cards": len(cards),
            "num_verified_facts": len(facts),
            "num_sessions": len(sessions),
            "sample_sessions": sessions[:8],
            "time_min": times[0] if times else None,
            "time_max": times[-1] if times else None,
        }

    def _parse_plan_json(self, query: str, data: Dict[str, Any]) -> VisualMemoryPlan:
        steps_data = data.get("steps") or []
        steps = [
            PlanStep(op=str(item.get("op", "")), args=dict(item.get("args") or {}))
            for item in steps_data
            if isinstance(item, dict)
        ]
        return VisualMemoryPlan.new(
            query=query,
            scope=dict(data.get("scope") or {"modality": "image", "session": "all"}),
            steps=steps,
            answer=dict(data.get("answer") or {"type": "short_text", "target": "visual_answer"}),
            confidence=safe_float(data.get("confidence")),
            planner="vlm_plan_grounded_svi",
            requires_raw_verification=bool(data.get("requires_raw_verification", True)),
        )

    def _validate_plan(self, plan: VisualMemoryPlan) -> None:
        if not plan.steps:
            raise ValueError("planner_empty_steps")
        for step in plan.steps:
            if step.op not in ALLOWED_OPERATORS:
                raise ValueError(f"planner_unknown_operator:{step.op}")
            if step.op == "retrieve":
                routes = step.args.get("routes", []) or []
                invalid = [route for route in routes if route not in ALLOWED_RETRIEVAL_ROUTES]
                if invalid:
                    raise ValueError(f"planner_unknown_routes:{invalid}")
            if step.op == "order_by":
                field = step.args.get("field")
                if field not in {"observed_at", "score"}:
                    raise ValueError(f"planner_bad_order_field:{field}")
            if step.op == "limit":
                k = step.args.get("k")
                if k != "verification_budget":
                    try:
                        if int(k) < 0:
                            raise ValueError
                    except (TypeError, ValueError):
                        raise ValueError(f"planner_bad_limit:{k}")

    def _repair_plan_if_needed(self, plan: VisualMemoryPlan) -> VisualMemoryPlan:
        issue = self._plan_issue(plan)
        if not issue or self.config.planner_json_repair_attempts <= 0:
            return plan
        try:
            payload = self._call_vlm_planner(plan.query, repair_feedback=issue)
            repaired = self._parse_plan_json(plan.query, payload)
            self._validate_plan(repaired)
            return repaired
        except Exception as exc:
            logger.debug("SVI planner repair failed, keeping original plan: %s", exc)
            return plan

    def _plan_issue(self, plan: VisualMemoryPlan) -> Optional[str]:
        has_order_by_time = any(
            step.op == "order_by" and step.args.get("field") == "observed_at"
            for step in plan.steps
        )
        has_limit_one = any(
            step.op == "limit" and str(step.args.get("k")) == "1"
            for step in plan.steps
        )
        answer_text = " ".join(
            [
                str(plan.answer.get("target", "")),
                str(plan.answer.get("type", "")),
                *[
                    str(step.args.get("ask", ""))
                    for step in plan.steps
                    if step.op == "verify_raw_image"
                ],
            ]
        ).lower()
        temporal_intent_terms = [
            "first",
            "earliest",
            "initial",
            "last",
            "latest",
            "final",
            "most recent",
            "timeline",
            "conversation",
            "memory order",
            "observed",
            "mentioned",
        ]
        looks_temporal = any(term in answer_text for term in temporal_intent_terms)
        if looks_temporal and not (has_order_by_time and has_limit_one):
            return (
                "The plan appears to answer a memory-order selection question, but it "
                "does not select by observed_at with limit 1. Use scope over the "
                "relevant visual cards, order_by observed_at asc/desc as appropriate, "
                "then limit 1 before verify_raw_image."
            )
        return None

    def _fallback_plan(self, query: str, reason: str) -> VisualMemoryPlan:
        del reason
        return VisualMemoryPlan.new(
            query=query,
            scope={"modality": "image", "session": "all"},
            steps=[
                PlanStep("scope", {"modality": "image", "session": "all"}),
                PlanStep(
                    "retrieve",
                    {
                        "routes": [
                            "caption_dense",
                            "entity_alias",
                            "attribute",
                            "relation",
                            "ocr",
                            "temporal_state",
                            "verified_fact",
                        ],
                        "entities": [],
                        "attributes": [],
                        "ocr_terms": [],
                        "relations": [],
                        "state_slots": [],
                        "fallback": "omission_aware",
                    },
                ),
                PlanStep("limit", {"k": "verification_budget"}),
                PlanStep(
                    "verify_raw_image",
                    {"ask": f"answer the question using the raw image: {query}"},
                ),
                PlanStep("answer", {"type": "short_text", "target": "visual_answer"}),
            ],
            answer={"type": "short_text", "target": "visual_answer"},
            confidence=0.0,
            planner="generic_retrieve_verify_fallback",
            requires_raw_verification=True,
        )
