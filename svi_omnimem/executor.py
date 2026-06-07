"""Executor for VisualMemoryPlan operators."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import SVIConfig
from .models import (
    PlanExecutionResult,
    PlanStep,
    RetrievalCandidate,
    StructuredVisualCard,
    VisualMemoryPlan,
    VisualQueryRequirement,
)
from .retriever import StructuredVisualRetriever
from .stores import StructuredVisualStore, VerifiedFactStore
from .utils import normalize_text


class VisualMemoryPlanExecutor:
    """Execute generic visual-memory plan operators over SVI stores."""

    def __init__(
        self,
        visual_store: StructuredVisualStore,
        verified_fact_store: VerifiedFactStore,
        retriever: StructuredVisualRetriever,
        config: Optional[SVIConfig] = None,
    ):
        self.visual_store = visual_store
        self.verified_fact_store = verified_fact_store
        self.retriever = retriever
        self.config = config or SVIConfig()

    def execute(
        self,
        plan: VisualMemoryPlan,
        requirement: VisualQueryRequirement,
        top_k: int = 10,
        verification_budget: int = 3,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
        current_session_id: Optional[str] = None,
    ) -> PlanExecutionResult:
        cards = self.visual_store.filtered_cards(tags_filter, time_range)
        candidates: List[RetrievalCandidate] = []
        claims: List[Dict[str, Any]] = []
        trace: List[Dict[str, Any]] = [
            {
                "op": "load_scope",
                "count": len(cards),
                "scope": plan.scope,
            }
        ]

        for step in plan.steps:
            op = step.op
            args = step.args
            if op == "scope":
                trace.append({"op": op, "args": args, "count": len(cards)})
            elif op == "filter":
                before = len(cards)
                cards = self._filter_cards(cards, args)
                trace.append(
                    {
                        "op": op,
                        "args": args,
                        "before": before,
                        "after": len(cards),
                    }
                )
            elif op == "retrieve":
                candidates, claims = self._retrieve_candidates(
                    plan=plan,
                    requirement=requirement,
                    top_k=top_k,
                    tags_filter=tags_filter,
                    time_range=time_range,
                    current_session_id=current_session_id,
                )
                candidates = self._intersect_with_cards(candidates, cards)
                trace.append(
                    {
                        "op": op,
                        "args": args,
                        "count": len(candidates),
                    }
                )
            elif op == "order_by":
                if not candidates:
                    candidates = self._cards_to_equal_candidates(cards, "temporal_sequence")
                candidates = self._order_candidates(candidates, args)
                trace.append(
                    {
                        "op": op,
                        "args": args,
                        "count": len(candidates),
                    }
                )
            elif op == "prioritize":
                candidates = self._prioritize_candidates(candidates, args)
                trace.append(
                    {
                        "op": op,
                        "args": args,
                        "count": len(candidates),
                    }
                )
            elif op == "diversify":
                candidates = self._diverse(candidates)
                trace.append(
                    {
                        "op": op,
                        "args": args,
                        "count": len(candidates),
                    }
                )
            elif op == "group_by":
                trace.append(
                    {
                        "op": op,
                        "args": args,
                        "count": len(candidates),
                        "note": "grouping is recorded for answer-time context",
                    }
                )
            elif op == "aggregate":
                candidates = self._aggregate_candidates(candidates, args)
                trace.append(
                    {
                        "op": op,
                        "args": args,
                        "count": len(candidates),
                    }
                )
            elif op == "limit":
                k = self._limit_value(args.get("k"), verification_budget, top_k)
                candidates = candidates[:k]
                trace.append({"op": op, "args": args, "k": k, "count": len(candidates)})
            elif op == "verify_raw_image":
                trace.append(
                    {
                        "op": op,
                        "args": args,
                        "count": len(candidates),
                        "note": "verification is executed by RawEvidenceVerifier",
                    }
                )
            elif op == "answer":
                trace.append({"op": op, "args": args, "count": len(candidates)})
            else:
                trace.append(
                    {
                        "op": op,
                        "args": args,
                        "count": len(candidates),
                        "warning": "unknown operator skipped",
                    }
                )

        if not candidates:
            candidates, claims = self._retrieve_candidates(
                plan=plan,
                requirement=requirement,
                top_k=top_k,
                tags_filter=tags_filter,
                time_range=time_range,
                current_session_id=current_session_id,
            )
            trace.append({"op": "fallback_retrieve", "count": len(candidates)})

        if not claims and candidates:
            claims = [
                {
                    "source": "visual_memory_plan",
                    "route": ",".join(candidate.routes) or "plan",
                    "card_id": candidate.card_id,
                    "image_mau_id": candidate.image_mau_id,
                    "score": candidate.score,
                    "matched_field": ",".join(candidate.matched_fields),
                    "note": "candidate selected by plan execution",
                }
                for candidate in candidates
            ]

        return PlanExecutionResult(
            plan=plan,
            requirement=requirement,
            candidates=candidates[:top_k],
            claims=claims,
            execution_trace=trace,
        )

    def _retrieve_candidates(
        self,
        plan: VisualMemoryPlan,
        requirement: VisualQueryRequirement,
        top_k: int,
        tags_filter: Optional[List[str]],
        time_range: Optional[Tuple[Any, Any]],
        current_session_id: Optional[str],
    ) -> Tuple[List[RetrievalCandidate], List[Dict[str, Any]]]:
        return self.retriever.retrieve(
            query=plan.query,
            requirement=requirement,
            top_k=top_k,
            tags_filter=tags_filter,
            time_range=time_range,
            current_session_id=current_session_id,
        )

    def _filter_cards(
        self,
        cards: List[StructuredVisualCard],
        args: Dict[str, Any],
    ) -> List[StructuredVisualCard]:
        field = args.get("field")
        value = normalize_text(args.get("value"))
        soft = bool(args.get("soft", False))
        if field != "visual_type" or not value:
            return cards

        matched = [
            card
            for card in cards
            if self._card_contains(card, value)
        ]
        if soft and not matched:
            return cards
        return matched

    def _card_contains(self, card: StructuredVisualCard, value: str) -> bool:
        haystack_parts = [card.global_caption, " ".join(card.tags)]
        for anchor in card.retrieval_anchors:
            haystack_parts.append(anchor.category)
        for obs in card.ocr_observations:
            haystack_parts.append(obs.text)
            haystack_parts.append(obs.context or "")
        haystack = normalize_text(" ".join(haystack_parts))
        return value in haystack

    def _cards_to_equal_candidates(
        self,
        cards: List[StructuredVisualCard],
        route: str,
    ) -> List[RetrievalCandidate]:
        return [
            RetrievalCandidate(
                image_mau_id=card.image_mau_id,
                card_id=card.card_id,
                score=1.0,
                routes=[route],
                matched_fields=["plan_scope"],
                observation_time=card.observed_at,
                raw_pointer=card.raw_pointer,
            )
            for card in cards
        ]

    def _intersect_with_cards(
        self,
        candidates: List[RetrievalCandidate],
        cards: List[StructuredVisualCard],
    ) -> List[RetrievalCandidate]:
        if not cards:
            return []
        allowed = {card.card_id for card in cards}
        return [candidate for candidate in candidates if candidate.card_id in allowed]

    def _order_candidates(
        self,
        candidates: List[RetrievalCandidate],
        args: Dict[str, Any],
    ) -> List[RetrievalCandidate]:
        field = args.get("field")
        direction = str(args.get("direction", "desc")).lower()
        reverse = direction == "desc"
        if field == "observed_at":
            return sorted(
                candidates,
                key=lambda item: (item.observation_time or "", item.score),
                reverse=reverse,
            )
        if field == "score":
            return sorted(candidates, key=lambda item: item.score, reverse=reverse)
        return candidates

    def _prioritize_candidates(
        self,
        candidates: List[RetrievalCandidate],
        args: Dict[str, Any],
    ) -> List[RetrievalCandidate]:
        routes = args.get("routes") or [args.get("route")]
        wanted = {str(route) for route in routes if route}
        if not wanted:
            return candidates
        return sorted(
            candidates,
            key=lambda item: (
                not bool(wanted.intersection(item.routes)),
                -item.score,
            ),
        )

    def _diverse(
        self,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        selected: List[RetrievalCandidate] = []
        seen = set()
        for candidate in candidates:
            if candidate.card_id in seen:
                continue
            selected.append(candidate)
            seen.add(candidate.card_id)
        return selected

    def _aggregate_candidates(
        self,
        candidates: List[RetrievalCandidate],
        args: Dict[str, Any],
    ) -> List[RetrievalCandidate]:
        method = str(args.get("method") or args.get("type") or "").lower()
        if method == "count":
            return self._diverse(candidates)
        return candidates

    def _limit_value(self, raw: Any, verification_budget: int, top_k: int) -> int:
        if raw == "verification_budget":
            return max(0, verification_budget)
        if raw == "top_k":
            return max(0, top_k)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return max(0, verification_budget)
