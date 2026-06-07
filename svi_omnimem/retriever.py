"""Structured visual retrieval and candidate packing."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import SVIConfig
from .models import RetrievalCandidate, StructuredVisualCard, VisualQueryRequirement
from .stores import StructuredVisualStore, VerifiedFactStore
from .utils import tokenize


class StructuredVisualRetriever:
    """Run low-prior SVI card retrieval and merge candidates by source image."""

    def __init__(
        self,
        visual_store: StructuredVisualStore,
        verified_fact_store: VerifiedFactStore,
        config: Optional[SVIConfig] = None,
        embedding_service: Optional[Any] = None,
    ):
        self.visual_store = visual_store
        self.verified_fact_store = verified_fact_store
        self.config = config or SVIConfig()
        self.embedding_service = embedding_service
        self._embedding_cache: Dict[str, List[float]] = {}

    def retrieve(
        self,
        query: str,
        requirement: VisualQueryRequirement,
        top_k: int = 10,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
        current_session_id: Optional[str] = None,
    ) -> Tuple[List[RetrievalCandidate], List[Dict[str, Any]]]:
        del current_session_id
        candidates: Dict[str, RetrievalCandidate] = {}
        retrieval_claims: List[Dict[str, Any]] = []

        def add_card_route(
            route: str,
            card: StructuredVisualCard,
            score: float,
            matched_field: str,
        ) -> None:
            weight = self.config.retrieval_weights.get(route, 0.05)
            weighted_score = score * weight
            candidate = candidates.get(card.image_mau_id)
            if candidate is None:
                candidate = RetrievalCandidate(
                    image_mau_id=card.image_mau_id,
                    card_id=card.card_id,
                    score=0.0,
                    observation_time=card.observed_at,
                    raw_pointer=card.raw_pointer,
                )
                candidates[card.image_mau_id] = candidate
            candidate.merge(route, weighted_score, matched_field)
            retrieval_claims.append(
                {
                    "source": "structured_visual_card",
                    "route": route,
                    "card_id": card.card_id,
                    "image_mau_id": card.image_mau_id,
                    "score": weighted_score,
                    "matched_field": matched_field,
                    "note": "retrieval hint only; raw image verification required for final evidence",
                }
            )

        for card, score, field in self.visual_store.search_all_text(
            query,
            tags_filter=tags_filter,
            time_range=time_range,
        ):
            add_card_route("card_text", card, score, field)

        for fact, score, field in self.verified_fact_store.search(
            tokenize(query),
            query_type=None,
            tags_filter=tags_filter,
            time_range=time_range,
        ):
            card = self.visual_store.get_by_card_id(fact.source_card_id)
            if not card:
                continue
            add_card_route("verified_fact", card, score, field)
            retrieval_claims.append(
                {
                    "source": "verified_fact",
                    "fact_id": fact.fact_id,
                    "card_id": fact.source_card_id,
                    "image_mau_id": fact.source_image_mau_id,
                    "score": score,
                    "fact": fact.to_dict(),
                }
            )

        self._merge_dense_scores(
            query=query,
            candidates=candidates,
            tags_filter=tags_filter,
            time_range=time_range,
            add_card_route=add_card_route,
        )

        merged = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        return self.pack_for_verification(merged, requirement, top_k), retrieval_claims

    def _merge_dense_scores(
        self,
        query: str,
        candidates: Dict[str, RetrievalCandidate],
        tags_filter: Optional[List[str]],
        time_range: Optional[Tuple[Any, Any]],
        add_card_route,
    ) -> None:
        if self.embedding_service is None:
            return
        visual_query = query.startswith("VISUAL_COMPARE::")
        if query.startswith("VISUAL_COMPARE::") or query.startswith("IMAGE_RECALL::"):
            query = query.split("\n", 1)[1] if "\n" in query else ""
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return
        for card in self.visual_store.filtered_cards(tags_filter, time_range):
            search_text = self._dense_card_text(card, visual_query=visual_query)
            if not search_text:
                continue
            card_embedding = self._cached_embedding(card.card_id, search_text)
            if not card_embedding:
                continue
            score = self._cosine_similarity(query_embedding, card_embedding)
            if score <= 0:
                continue
            add_card_route("embedding_dense", card, score, f"dense:{score:.3f}")

    def _dense_card_text(
        self,
        card: StructuredVisualCard,
        visual_query: bool = False,
    ) -> str:
        parts: List[str] = [card.global_caption]
        for anchor in card.retrieval_anchors:
            parts.append(anchor.category)
            for attr in anchor.salient_attributes.values():
                parts.append(attr.value)
        for obs in card.ocr_observations:
            parts.append(obs.text)
        if card.observed_at:
            try:
                parsed = datetime.fromisoformat(str(card.observed_at).replace("Z", "+00:00"))
            except ValueError:
                try:
                    parsed = datetime.strptime(str(card.observed_at)[:10], "%Y-%m-%d")
                except ValueError:
                    parsed = None
            if parsed is not None:
                parts.append(
                    " ".join(
                        part
                        for part in [
                            parsed.strftime("%B"),
                            parsed.strftime("%b"),
                            str(parsed.year),
                            str(parsed.month),
                            str(parsed.day),
                        ]
                        if part
                    )
                )
        if card.source_text_context and not visual_query:
            parts.append(card.source_text_context[:240])
        return " ".join(part for part in parts if part).strip()

    def _cached_embedding(self, cache_key: str, text: str) -> List[float]:
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        embedding = self._embed_text(text)
        if embedding:
            self._embedding_cache[cache_key] = embedding
        return embedding

    def _embed_text(self, text: str) -> List[float]:
        if not self.embedding_service:
            return []
        try:
            return list(self.embedding_service.embed_text(text))
        except Exception:
            return []

    def _cosine_similarity(self, left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(l * l for l in left))
        right_norm = math.sqrt(sum(r * r for r in right))
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def pack_for_verification(
        self,
        candidates: List[RetrievalCandidate],
        requirement: VisualQueryRequirement,
        budget: int,
    ) -> List[RetrievalCandidate]:
        del requirement
        if budget <= 0:
            return []
        return candidates[:budget]
