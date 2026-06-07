"""Three-route retrieval with RRF fusion for unified memories."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .models import RankedMemory, RetrievalResult, RouteHit
from .store import DualEncoderMemoryStore


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
}


MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def normalize_text(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def retrieval_tokens(text: Any) -> List[str]:
    tokens = []
    for token in normalize_text(text).split():
        if token in STOPWORDS:
            continue
        if len(token) <= 2 and not token.isdigit():
            continue
        tokens.append(token)
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            tokens.append(token[:-1])
        if len(token) > 5 and token.endswith("ing"):
            tokens.append(token[:-3])
    return tokens


def lexical_overlap(query: str, document: str) -> float:
    query_terms = set(retrieval_tokens(query))
    if not query_terms:
        return 0.0
    doc_terms = set(retrieval_tokens(document))
    return len(query_terms & doc_terms) / max(len(query_terms), 1)


def date_match_score(query: str, date_value: str) -> float:
    if not date_value:
        return 0.0
    q = normalize_text(query)
    try:
        parsed = datetime.strptime(str(date_value)[:10], "%Y-%m-%d")
    except ValueError:
        parsed = None
    if parsed is not None:
        exact = parsed.strftime("%Y-%m-%d")
        if exact in str(query):
            return 1.0
        if parsed.strftime("%B").lower() in q or parsed.strftime("%b").lower() in q:
            if "early" in q and parsed.day <= 10:
                return 0.8
            if "mid" in q and 11 <= parsed.day <= 20:
                return 0.8
            if "late" in q and parsed.day >= 21:
                return 0.8
            return 0.55
        if str(parsed.year) in q:
            return 0.25
    return 0.0


class DualEncoderRetriever:
    def __init__(
        self,
        store: DualEncoderMemoryStore,
        text_encoder: Any,
        vision_encoder: Any,
    ):
        self.store = store
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder

    def retrieve(
        self,
        query: str,
        question_image: Optional[Any] = None,
        top_k_text: int = 20,
        top_k_image: int = 20,
        top_k_bm25: int = 20,
        rerank_top_k: int = 10,
        rrf_k: int = 60,
    ) -> RetrievalResult:
        text_hits = self._text_route(query, top_k_text)
        image_hits = self._image_route(query, question_image, top_k_image)
        bm25_hits = self._bm25_route(query, top_k_bm25)
        ranked = self._rrf_fuse(
            query,
            text_hits=text_hits,
            image_hits=image_hits,
            bm25_hits=bm25_hits,
            top_k=rerank_top_k,
            rrf_k=rrf_k,
        )
        return RetrievalResult(
            query=query,
            text_hits=text_hits,
            image_hits=image_hits,
            bm25_hits=bm25_hits,
            ranked_memories=ranked,
        )

    def _text_route(self, query: str, top_k: int) -> List[RouteHit]:
        if top_k <= 0:
            return []
        embedding = self.text_encoder.encode(query)
        hits = []
        for rank, (memory_id, score, _row_id) in enumerate(
            self.store.search_text(embedding, top_k),
            start=1,
        ):
            hits.append(
                RouteHit(
                    route="text",
                    memory_id=memory_id,
                    score=score,
                    rank=rank,
                )
            )
        return hits

    def _memory_text(self, memory: Any) -> str:
        return " ".join(
            [
                memory.text,
                " ".join(image.caption for image in memory.images),
                memory.date,
                memory.session_id,
                memory.turn_id,
            ]
        )

    def _bm25_route(self, query: str, top_k: int) -> List[RouteHit]:
        if top_k <= 0:
            return []
        query_terms = retrieval_tokens(query)
        if not query_terms:
            return []
        memories = self.store.iter_memories()
        if not memories:
            return []

        doc_tokens: Dict[str, List[str]] = {}
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for memory in memories:
            tokens = retrieval_tokens(self._memory_text(memory))
            doc_tokens[memory.memory_id] = tokens
            total_length += len(tokens)
            document_frequency.update(set(tokens))

        avg_doc_length = total_length / len(memories) if memories else 0.0
        if avg_doc_length <= 0:
            return []

        k1 = 1.5
        b = 0.75
        query_counter = Counter(query_terms)
        scored = []
        total_docs = len(memories)
        for memory in memories:
            tokens = doc_tokens.get(memory.memory_id) or []
            if not tokens:
                continue
            frequencies = Counter(tokens)
            doc_length = len(tokens)
            score = 0.0
            for term, query_count in query_counter.items():
                tf = frequencies.get(term, 0)
                if tf <= 0:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = tf + k1 * (1.0 - b + b * doc_length / avg_doc_length)
                score += query_count * idf * (tf * (k1 + 1.0)) / denominator
            if score > 0:
                scored.append((memory.memory_id, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            RouteHit(route="bm25", memory_id=memory_id, score=score, rank=rank)
            for rank, (memory_id, score) in enumerate(scored[:top_k], start=1)
        ]

    def _image_route(
        self,
        query: str,
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
        for rank, (image_row_id, memory_id, image_id, score, _row_id) in enumerate(
            self.store.search_image(embedding, top_k),
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

    def _route_family(self, hit: RouteHit) -> str:
        if hit.route.startswith("image"):
            return "image"
        return hit.route

    def _rrf_fuse(
        self,
        query: str,
        text_hits: List[RouteHit],
        image_hits: List[RouteHit],
        bm25_hits: List[RouteHit],
        top_k: int,
        rrf_k: int,
    ) -> List[RankedMemory]:
        hits_by_memory: Dict[str, List[RouteHit]] = defaultdict(list)
        for hit in [*text_hits, *image_hits, *bm25_hits]:
            hits_by_memory[hit.memory_id].append(hit)

        ranked: List[RankedMemory] = []
        for memory_id, hits in hits_by_memory.items():
            memory = self.store.get_memory(memory_id)
            if not memory:
                continue
            text_score = max((hit.score for hit in hits if hit.route == "text"), default=0.0)
            image_score = max((hit.score for hit in hits if hit.route.startswith("image")), default=0.0)
            bm25_score = max((hit.score for hit in hits if hit.route == "bm25"), default=0.0)
            best_rank_by_route: Dict[str, int] = {}
            for hit in hits:
                family = self._route_family(hit)
                best_rank_by_route[family] = min(best_rank_by_route.get(family, 10**6), hit.rank)
            rrf_score = sum(
                1.0 / (max(float(rrf_k), 0.0) + float(rank))
                for rank in best_rank_by_route.values()
            )
            doc_text = self._memory_text(memory)
            lexical_score = lexical_overlap(query, doc_text)
            date_score = date_match_score(query, memory.date)
            ranked.append(
                RankedMemory(
                    memory=memory,
                    score=rrf_score,
                    text_score=text_score,
                    image_score=image_score,
                    bm25_score=bm25_score,
                    lexical_score=lexical_score,
                    date_score=date_score,
                    route_bonus=rrf_score,
                    route_hits=sorted(hits, key=lambda item: (item.rank, item.route)),
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: max(top_k, 0)]
