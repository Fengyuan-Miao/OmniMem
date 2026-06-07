"""Top-k retrieval over one unified GME embedding per memory entry."""

from __future__ import annotations

from typing import Any, List, Optional

from .models import GmeRetrievalResult, GmeRetrievedEntry
from .store import GmeMemoryStore


class GmeMemoryRetriever:
    def __init__(self, store: GmeMemoryStore, encoder: Any):
        self.store = store
        self.encoder = encoder

    def retrieve(
        self,
        query: str,
        question_image: Optional[str] = None,
        top_k: int = 10,
    ) -> GmeRetrievalResult:
        embedding, query_mode = self.encoder.encode_query(query, question_image=question_image)
        entries: List[GmeRetrievedEntry] = []
        for rank, (memory_id, score, _row_id, mode, image_id) in enumerate(
            self.store.search_entries(embedding, top_k),
            start=1,
        ):
            memory = self.store.get_memory(memory_id)
            if not memory:
                continue
            matched_image_ids = [image_id] if image_id else []
            entries.append(
                GmeRetrievedEntry(
                    memory=memory,
                    score=score,
                    rank=rank,
                    embedding_mode=mode,
                    matched_image_ids=matched_image_ids,
                )
            )
        return GmeRetrievalResult(
            query=query,
            query_embedding_mode=query_mode,
            entries=entries,
        )
