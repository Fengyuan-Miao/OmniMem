"""Strict topic-scoped dual-route retrieval."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from dual_encoder_memory.models import RankedMemory, RetrievalResult, RouteHit
from dual_encoder_memory.retriever import date_match_score, lexical_overlap

from .store import TopicMemoryStore


class TopicScopedRetriever:
    def __init__(
        self,
        store: TopicMemoryStore,
        text_encoder: Any,
        vision_encoder: Any,
    ):
        self.store = store
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder

    def retrieve(
        self,
        query: str,
        topic_ids: Sequence[str],
        modalities: Sequence[str],
        question_image: Optional[Any] = None,
        top_k_text: int = 20,
        top_k_image: int = 20,
        rerank_top_k: int = 10,
    ) -> RetrievalResult:
        normalized_modalities = {str(item).lower() for item in modalities}
        if not normalized_modalities:
            normalized_modalities = {"text"}
        text_hits = (
            self._text_route(query, topic_ids, top_k_text)
            if "text" in normalized_modalities
            else []
        )
        image_hits = (
            self._image_route(query, topic_ids, question_image, top_k_image)
            if "image" in normalized_modalities
            else []
        )
        ranked = self._rerank(query, text_hits, image_hits, rerank_top_k)
        return RetrievalResult(
            query=query,
            text_hits=text_hits,
            image_hits=image_hits,
            bm25_hits=[],
            ranked_memories=ranked,
        )

    def _text_route(
        self,
        query: str,
        topic_ids: Sequence[str],
        top_k: int,
    ) -> List[RouteHit]:
        embedding = self.text_encoder.encode(query)
        hits = []
        for rank, (memory_id, score) in enumerate(
            self.store.search_text_in_topics(embedding, topic_ids, top_k),
            start=1,
        ):
            hits.append(RouteHit(route="text", memory_id=memory_id, score=score, rank=rank))
        return hits

    def _image_route(
        self,
        query: str,
        topic_ids: Sequence[str],
        question_image: Optional[Any],
        top_k: int,
    ) -> List[RouteHit]:
        if top_k <= 0:
            return []
        if question_image is not None:
            embedding = self.vision_encoder.encode_image(question_image)
            route = "image_by_image"
        else:
            embedding = self.vision_encoder.encode_text(query)
            route = "image_by_text"
        hits = []
        for rank, (image_row_id, memory_id, image_id, score) in enumerate(
            self.store.search_image_in_topics(embedding, topic_ids, top_k),
            start=1,
        ):
            hits.append(
                RouteHit(
                    route=route,
                    memory_id=memory_id,
                    score=score,
                    rank=rank,
                    image_row_id=image_row_id,
                    image_id=image_id,
                )
            )
        return hits

    def _rerank(
        self,
        query: str,
        text_hits: List[RouteHit],
        image_hits: List[RouteHit],
        top_k: int,
    ) -> List[RankedMemory]:
        hits_by_memory: Dict[str, List[RouteHit]] = defaultdict(list)
        for hit in [*text_hits, *image_hits]:
            hits_by_memory[hit.memory_id].append(hit)

        ranked: List[RankedMemory] = []
        for memory_id, hits in hits_by_memory.items():
            memory = self.store.get_memory(memory_id)
            if not memory:
                continue
            text_score = max((hit.score for hit in hits if hit.route == "text"), default=0.0)
            image_score = max((hit.score for hit in hits if hit.route.startswith("image")), default=0.0)
            best_rank = min((hit.rank for hit in hits), default=10**6)
            route_bonus = 1.0 / math.sqrt(max(best_rank, 1))
            doc_text = " ".join(
                [
                    memory.text,
                    " ".join(image.caption for image in memory.images),
                    memory.date,
                    memory.session_id,
                    memory.turn_id,
                ]
            )
            lexical_score = lexical_overlap(query, doc_text)
            date_score = date_match_score(query, memory.date)
            score = (
                0.42 * text_score
                + 0.34 * image_score
                + 0.12 * route_bonus
                + 0.08 * lexical_score
                + 0.04 * date_score
            )
            ranked.append(
                RankedMemory(
                    memory=memory,
                    score=score,
                    text_score=text_score,
                    image_score=image_score,
                    lexical_score=lexical_score,
                    date_score=date_score,
                    route_bonus=route_bonus,
                    route_hits=sorted(hits, key=lambda item: (item.rank, item.route)),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: max(top_k, 0)]
