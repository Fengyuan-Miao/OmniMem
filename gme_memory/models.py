"""Data models for GME unified-entry memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GmeImagePointer:
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
class GmeMemoryRecord:
    memory_id: str
    text: str
    session_id: str = ""
    turn_id: str = ""
    date: str = ""
    images: List[GmeImagePointer] = field(default_factory=list)
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
class GmeEntryEmbedding:
    memory_id: str
    vector: List[float]
    embedding_mode: str
    image_id: str = ""
    image_path: str = ""


@dataclass
class GmeRetrievedEntry:
    memory: GmeMemoryRecord
    score: float
    rank: int
    embedding_mode: str = ""
    matched_image_ids: List[str] = field(default_factory=list)

    def retrieved_image_ids(self) -> List[str]:
        ids: List[str] = []
        for image in self.memory.images:
            if image.image_id and image.image_id not in ids:
                ids.append(image.image_id)
        return ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "embedding_mode": self.embedding_mode,
            "matched_image_ids": self.matched_image_ids,
            "retrieved_image_ids": self.retrieved_image_ids(),
        }


@dataclass
class GmeRetrievalResult:
    query: str
    query_embedding_mode: str
    entries: List[GmeRetrievedEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_embedding_mode": self.query_embedding_mode,
            "entries": [entry.to_dict() for entry in self.entries],
        }
