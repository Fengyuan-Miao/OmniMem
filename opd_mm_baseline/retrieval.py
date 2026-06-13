"""Hidden memory store and generic retrieval used by the tool executor."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np

from .models import MemoryRecord, PoolItem


TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
QUERY_SEGMENT_PATTERN = re.compile(r"[\n\r]+|(?<=[.!?。！？])\s+")


def tokenize(text: Any) -> List[str]:
    value = str(text or "").lower()
    tokens = TOKEN_PATTERN.findall(value)
    cjk = [token for token in tokens if "\u4e00" <= token <= "\u9fff"]
    if len(cjk) > 1:
        tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


def query_variants(query: str, limit: int = 6) -> List[str]:
    """Build generic full-query and clause-level retrieval views."""
    full = str(query or "").strip()
    values = [full] if full else []
    for segment in QUERY_SEGMENT_PATTERN.split(full):
        value = segment.strip(" \t:-")
        if len(tokenize(value)) < 3 or value in values:
            continue
        values.append(value)
        if len(values) >= limit:
            break
    return values


def normalize_scores(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: (1.0 if high > 0 else 0.0) for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class DenseEncoder(Protocol):
    def encode(self, text: str) -> List[float]:
        ...

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        ...


class VisionEncoder(Protocol):
    def encode_image(self, image: Any) -> List[float]:
        ...

    def encode_images(self, images: List[Any]) -> List[List[float]]:
        ...

    def encode_text(self, text: str) -> List[float]:
        ...


class HiddenMemoryStore:
    """Memory collection visible only to the executor."""

    def __init__(
        self,
        records: Iterable[MemoryRecord],
        dense_encoder: Optional[DenseEncoder] = None,
        vision_encoder: Optional[VisionEncoder] = None,
    ):
        self._records = list(records)
        self._dense_encoder = dense_encoder
        self._vision_encoder = vision_encoder
        self._dense_cache: Dict[str, np.ndarray] = {}
        self._dense_prepared = False
        self._vision_cache: Dict[str, np.ndarray] = {}
        self._vision_prepared = False

    def initial_pool(self) -> List[PoolItem]:
        return [PoolItem(memory=record) for record in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def abstract_support_profile(self, turn_ids: Iterable[str]) -> Dict[str, Any]:
        """Return training-only support metadata without content or identifiers."""
        targets = set(turn_ids)
        matches = [record for record in self._records if record.turn_id in targets]
        timestamps = sorted(record.timestamp for record in matches if record.timestamp)
        return {
            "support_count": len(matches),
            "modalities": sorted({record.modality for record in matches}),
            "authors": sorted({record.author for record in matches}),
            "source_types": sorted({record.source_type for record in matches}),
            "earliest_timestamp": timestamps[0] if timestamps else "",
            "latest_timestamp": timestamps[-1] if timestamps else "",
            "has_raw_media": any(bool(record.raw_pointer) for record in matches),
        }

    def oracle_retrieval_profile(
        self,
        query: str,
        turn_ids: Iterable[str],
        question_image: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Report gold-support ranks without exposing support IDs or content."""
        targets = set(turn_ids)
        profile: Dict[str, Any] = {}
        retriever = TurnAwareHybridRetriever()
        for method in ("bm25", "dense", "hybrid"):
            ranked = retriever.retrieve(
                self.initial_pool(),
                query=query,
                store=self,
                method=method,
                top_k=max(1, len(self._records)),
                question_image=question_image,
            )
            ranks_by_turn: Dict[str, List[int]] = {
                turn_id: [] for turn_id in targets
            }
            turn_ranks: Dict[str, int] = {}
            for item in ranked:
                turn_ranks.setdefault(item.memory.turn_id, len(turn_ranks) + 1)
                if item.memory.turn_id in ranks_by_turn:
                    ranks_by_turn[item.memory.turn_id].append(
                        turn_ranks[item.memory.turn_id]
                    )
            per_turn_best_ranks = sorted(
                min(ranks)
                for ranks in ranks_by_turn.values()
                if ranks
            )
            ranks = sorted(
                rank
                for values in ranks_by_turn.values()
                for rank in values
            )
            best_rank = min(ranks) if ranks else None
            all_records_rank = max(ranks) if ranks else None
            all_turns_rank = (
                max(per_turn_best_ranks)
                if len(per_turn_best_ranks) == len(targets) and targets
                else None
            )
            profile[method] = {
                "best_support_rank": best_rank,
                "minimum_top_k_for_any_support": best_rank,
                "minimum_top_k_for_all_support_turns": all_turns_rank,
                "minimum_top_k_for_all_support_records": all_records_rank,
                "support_records_found": len(ranks),
                "support_turns_found": len(per_turn_best_ranks),
                "support_turn_count": len(targets),
                "support_recall_at_1": bool(best_rank and best_rank <= 1),
                "support_recall_at_3": bool(best_rank and best_rank <= 3),
                "support_recall_at_5": bool(best_rank and best_rank <= 5),
                "support_recall_at_10": bool(best_rank and best_rank <= 10),
                "support_recall_at_20": bool(best_rank and best_rank <= 20),
            }
        return profile

    @staticmethod
    def oracle_action_advice(
        retrieval_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert support ranks into content-free executable guidance."""
        options = []
        method_preference = {"bm25": 0, "dense": 1, "hybrid": 2}
        for method, values in retrieval_profile.items():
            top_k = values.get("minimum_top_k_for_all_support_records")
            objective = "all_support_records"
            if top_k is None:
                top_k = values.get("minimum_top_k_for_all_support_turns")
                objective = "all_support_turns"
            if top_k is None:
                top_k = values.get("minimum_top_k_for_any_support")
                objective = "any_support"
            if not isinstance(top_k, int) or top_k <= 0:
                continue
            options.append(
                {
                    "method": method,
                    "minimum_top_k": top_k,
                    "verified_objective": objective,
                    "must_read_after_retrieval": True,
                    "do_not_apply_smaller_topk_after_retrieval": True,
                }
            )
        options.sort(
            key=lambda item: (
                item["verified_objective"] != "all_support_records",
                item["verified_objective"] != "all_support_turns",
                item["minimum_top_k"],
                method_preference.get(item["method"], 99),
            )
        )
        return {
            "recommended": options[0] if options else None,
            "alternatives": options[1:],
            "trajectory_shape": ["RETRIEVE", "READ", "STOP"],
            "note": (
                "These are training-only counterfactual action requirements. "
                "They expose neither support identifiers nor support content."
            ),
        }

    def dense_vector(self, record: MemoryRecord) -> Optional[np.ndarray]:
        if self._dense_encoder is None:
            return None
        self._prepare_dense_cache()
        return self._dense_cache.get(record.memory_id)

    def _prepare_dense_cache(self) -> None:
        if self._dense_prepared or self._dense_encoder is None:
            return
        texts = [record.searchable_text() for record in self._records]
        if hasattr(self._dense_encoder, "encode_batch"):
            vectors = self._dense_encoder.encode_batch(texts)
        else:
            vectors = [self._dense_encoder.encode(text) for text in texts]
        for record, values in zip(self._records, vectors):
            vector = np.asarray(values, dtype="float32")
            norm = float(np.linalg.norm(vector))
            self._dense_cache[record.memory_id] = vector / norm if norm > 0 else vector
        self._dense_prepared = True

    def query_vector(self, query: str) -> Optional[np.ndarray]:
        vectors = self.query_vectors(query)
        return vectors[0] if vectors else None

    def query_vectors(self, query: str) -> List[np.ndarray]:
        if self._dense_encoder is None:
            return []
        variants = query_variants(query)
        if not variants:
            return []
        if hasattr(self._dense_encoder, "encode_batch"):
            values = self._dense_encoder.encode_batch(variants)
        else:
            values = [
                self._dense_encoder.encode(value) for value in variants
            ]
        vectors = []
        for embedding in values:
            vector = np.asarray(embedding, dtype="float32")
            norm = float(np.linalg.norm(vector))
            vectors.append(vector / norm if norm > 0 else vector)
        return vectors

    def vision_vector(self, record: MemoryRecord) -> Optional[np.ndarray]:
        if self._vision_encoder is None or not record.raw_pointer:
            return None
        self._prepare_vision_cache()
        return self._vision_cache.get(record.memory_id)

    def _prepare_vision_cache(self) -> None:
        if self._vision_prepared or self._vision_encoder is None:
            return
        records = [record for record in self._records if record.raw_pointer]
        paths = [record.raw_pointer for record in records]
        if hasattr(self._vision_encoder, "encode_images"):
            vectors = self._vision_encoder.encode_images(paths)
        else:
            vectors = [
                self._vision_encoder.encode_image(path) for path in paths
            ]
        for record, values in zip(records, vectors):
            vector = np.asarray(values, dtype="float32")
            norm = float(np.linalg.norm(vector))
            self._vision_cache[record.memory_id] = (
                vector / norm if norm > 0 else vector
            )
        self._vision_prepared = True

    def vision_query_vector(
        self,
        query: str,
        question_image: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        if self._vision_encoder is None:
            return None
        values = (
            self._vision_encoder.encode_image(question_image)
            if question_image
            else self._vision_encoder.encode_text(query)
        )
        vector = np.asarray(values, dtype="float32")
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector


class HybridRetriever:
    """BM25, dense, or normalized hybrid ranking over the current pool."""

    def __init__(self, hybrid_alpha: float = 0.5):
        self.hybrid_alpha = max(0.0, min(1.0, float(hybrid_alpha)))

    def retrieve(
        self,
        pool: List[PoolItem],
        query: str,
        store: HiddenMemoryStore,
        method: str = "hybrid",
        top_k: int = 5,
        question_image: Optional[str] = None,
    ) -> List[PoolItem]:
        if not pool or top_k <= 0:
            return []
        bm25 = self._bm25_scores(pool, query)
        dense = self._dense_scores(pool, query, store)
        if method == "bm25":
            scores = bm25
        elif method == "dense":
            scores = dense
        else:
            sparse_norm = normalize_scores(bm25)
            dense_norm = normalize_scores(dense)
            scores = {
                item.memory.memory_id: (
                    self.hybrid_alpha * dense_norm.get(item.memory.memory_id, 0.0)
                    + (1.0 - self.hybrid_alpha)
                    * sparse_norm.get(item.memory.memory_id, 0.0)
                )
                for item in pool
            }
        ranked = [
            PoolItem(item.memory, float(scores.get(item.memory.memory_id, 0.0)))
            for item in pool
        ]
        ranked.sort(
            key=lambda item: (
                -item.score,
                item.memory.timestamp,
                item.memory.turn_id,
                item.memory.memory_id,
            )
        )
        return ranked[: min(top_k, len(ranked))]

    @staticmethod
    def _bm25_scores(pool: List[PoolItem], query: str) -> Dict[str, float]:
        query_terms = tokenize(query)
        documents = {
            item.memory.memory_id: tokenize(item.memory.searchable_text())
            for item in pool
        }
        if not query_terms or not documents:
            return {item.memory.memory_id: 0.0 for item in pool}
        document_frequency: Counter[str] = Counter()
        for tokens in documents.values():
            document_frequency.update(set(tokens))
        avg_length = sum(len(tokens) for tokens in documents.values()) / len(documents)
        k1 = 1.5
        b = 0.75
        scores: Dict[str, float] = {}
        query_counts = Counter(query_terms)
        total_docs = len(documents)
        for memory_id, tokens in documents.items():
            frequencies = Counter(tokens)
            score = 0.0
            for term, query_count in query_counts.items():
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = tf + k1 * (
                    1.0 - b + b * len(tokens) / max(avg_length, 1.0)
                )
                score += query_count * idf * tf * (k1 + 1.0) / denominator
            scores[memory_id] = score
        return scores

    @staticmethod
    def _dense_scores(
        pool: List[PoolItem],
        query: str,
        store: HiddenMemoryStore,
    ) -> Dict[str, float]:
        query_vectors = store.query_vectors(query)
        if not query_vectors:
            return {item.memory.memory_id: 0.0 for item in pool}
        scores = {}
        for item in pool:
            vector = store.dense_vector(item.memory)
            compatible = [
                query_vector
                for query_vector in query_vectors
                if vector is not None and vector.size == query_vector.size
            ]
            if vector is None or not compatible:
                scores[item.memory.memory_id] = 0.0
            else:
                scores[item.memory.memory_id] = max(
                    float(np.dot(query_vector, vector))
                    for query_vector in compatible
                )
        return scores

    @staticmethod
    def _vision_scores(
        pool: List[PoolItem],
        query: str,
        store: HiddenMemoryStore,
        question_image: Optional[str],
    ) -> Dict[str, float]:
        query_vector = store.vision_query_vector(query, question_image)
        if query_vector is None or query_vector.size == 0:
            return {item.memory.memory_id: 0.0 for item in pool}
        scores = {}
        for item in pool:
            vector = store.vision_vector(item.memory)
            if vector is None or vector.size != query_vector.size:
                scores[item.memory.memory_id] = 0.0
            else:
                scores[item.memory.memory_id] = float(
                    np.dot(query_vector, vector)
                )
        return scores


class TurnAwareHybridRetriever(HybridRetriever):
    """Rank dialogue turns, then return all text/image records in each turn."""

    def __init__(
        self,
        hybrid_alpha: float = 0.5,
        context_window: int = 1,
        context_decay: float = 0.9,
    ):
        super().__init__(hybrid_alpha)
        self.context_window = max(0, int(context_window))
        self.context_decay = max(0.0, min(1.0, float(context_decay)))

    def retrieve(
        self,
        pool: List[PoolItem],
        query: str,
        store: HiddenMemoryStore,
        method: str = "hybrid",
        top_k: int = 5,
        question_image: Optional[str] = None,
    ) -> List[PoolItem]:
        if not pool or top_k <= 0:
            return []
        bm25_record = self._bm25_scores(pool, query)
        dense_record = self._dense_scores(pool, query, store)
        vision_record = self._vision_scores(
            pool,
            query,
            store,
            question_image,
        )
        groups: Dict[str, List[PoolItem]] = defaultdict(list)
        for item in pool:
            groups[item.memory.turn_id].append(item)

        def aggregate(values: Dict[str, float]) -> Dict[str, float]:
            return {
                turn_id: max(
                    values.get(item.memory.memory_id, 0.0)
                    for item in items
                )
                for turn_id, items in groups.items()
            }

        bm25 = aggregate(bm25_record)
        dense = aggregate(dense_record)
        vision = aggregate(vision_record)
        if method == "bm25":
            scores = bm25
        elif method == "dense":
            scores = dense
        else:
            sparse_norm = normalize_scores(bm25)
            dense_norm = normalize_scores(dense)
            vision_norm = normalize_scores(vision)
            has_vision = any(value != 0.0 for value in vision.values())
            if has_vision and question_image:
                vision_weight = 0.5
            elif has_vision:
                vision_weight = 0.2
            else:
                vision_weight = 0.0
            text_weight = 1.0 - vision_weight
            weights = (
                text_weight * (1.0 - self.hybrid_alpha),
                text_weight * self.hybrid_alpha,
                vision_weight,
            )
            scores = {
                turn_id: (
                    weights[0] * sparse_norm.get(turn_id, 0.0)
                    + weights[1] * dense_norm.get(turn_id, 0.0)
                    + weights[2] * vision_norm.get(turn_id, 0.0)
                )
                for turn_id in groups
            }
        scores = self._propagate_local_context(groups, scores)

        ranked_turns = sorted(
            groups,
            key=lambda turn_id: (
                -scores.get(turn_id, 0.0),
                min(item.memory.timestamp for item in groups[turn_id]),
                turn_id,
            ),
        )[: min(top_k, len(groups))]
        ranked: List[PoolItem] = []
        for turn_id in ranked_turns:
            turn_score = float(scores.get(turn_id, 0.0))
            ranked.extend(
                PoolItem(item.memory, turn_score)
                for item in sorted(
                    groups[turn_id],
                    key=lambda value: (
                        value.memory.timestamp,
                        value.memory.memory_id,
                    ),
                )
            )
        return ranked

    def _propagate_local_context(
        self,
        groups: Dict[str, List[PoolItem]],
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        if self.context_window <= 0:
            return scores
        positions: Dict[tuple[str, int], str] = {}
        for turn_id, items in groups.items():
            memory = items[0].memory
            session_id = str(memory.metadata.get("session_id") or "")
            turn_index = memory.metadata.get("turn_index")
            if session_id and isinstance(turn_index, int):
                positions[(session_id, turn_index)] = turn_id
        propagated = dict(scores)
        for (session_id, turn_index), turn_id in positions.items():
            best = scores.get(turn_id, 0.0)
            for distance in range(1, self.context_window + 1):
                weight = self.context_decay**distance
                for neighbor_index in (
                    turn_index - distance,
                    turn_index + distance,
                ):
                    neighbor = positions.get((session_id, neighbor_index))
                    if neighbor is not None:
                        best = max(
                            best,
                            scores.get(neighbor, 0.0) * weight,
                        )
            propagated[turn_id] = best
        return propagated
