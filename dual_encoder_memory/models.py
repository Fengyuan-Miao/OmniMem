"""Data models for the dual-encoder unified memory index."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImagePointer:
    image_id: str
    path: str
    caption: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_row_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_row_id": self.image_row_id,
            "image_id": self.image_id,
            "path": self.path,
            "caption": self.caption,
            "metadata": self.metadata,
        }


@dataclass
class UnifiedMemoryRecord:
    memory_id: str
    text: str
    session_id: str = ""
    turn_id: str = ""
    date: str = ""
    images: List[ImagePointer] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "date": self.date,
            "images": [image.to_dict() for image in self.images],
            "metadata": self.metadata,
        }


@dataclass
class RouteHit:
    route: str
    memory_id: str
    score: float
    rank: int
    image_row_id: Optional[int] = None
    image_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "memory_id": self.memory_id,
            "score": self.score,
            "rank": self.rank,
            "image_row_id": self.image_row_id,
            "image_id": self.image_id,
        }


@dataclass
class RankedMemory:
    memory: UnifiedMemoryRecord
    score: float
    text_score: float = 0.0
    image_score: float = 0.0
    bm25_score: float = 0.0
    lexical_score: float = 0.0
    date_score: float = 0.0
    route_bonus: float = 0.0
    route_hits: List[RouteHit] = field(default_factory=list)

    def matched_image_ids(self) -> List[str]:
        ids: List[str] = []
        for hit in self.route_hits:
            if hit.image_id and hit.image_id not in ids:
                ids.append(hit.image_id)
        return ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "text_score": self.text_score,
            "image_score": self.image_score,
            "bm25_score": self.bm25_score,
            "lexical_score": self.lexical_score,
            "date_score": self.date_score,
            "route_bonus": self.route_bonus,
            "matched_image_ids": self.matched_image_ids(),
            "route_hits": [hit.to_dict() for hit in self.route_hits],
        }


@dataclass
class RetrievalResult:
    query: str
    text_hits: List[RouteHit]
    image_hits: List[RouteHit]
    bm25_hits: List[RouteHit]
    ranked_memories: List[RankedMemory]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "text_hits": [hit.to_dict() for hit in self.text_hits],
            "image_hits": [hit.to_dict() for hit in self.image_hits],
            "bm25_hits": [hit.to_dict() for hit in self.bm25_hits],
            "ranked_memories": [item.to_dict() for item in self.ranked_memories],
        }
