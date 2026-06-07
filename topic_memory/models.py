"""Data models for topic-gated memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TopicRecord:
    topic_id: str
    summary: str
    turn_count: int = 0
    created_sequence: int = 0
    updated_sequence: int = 0
    first_date: str = ""
    last_date: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "summary": self.summary,
            "turn_count": self.turn_count,
            "created_sequence": self.created_sequence,
            "updated_sequence": self.updated_sequence,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "metadata": self.metadata,
        }


@dataclass
class TopicAssignment:
    memory_id: str
    topic_id: str
    action: str
    summary: str
    candidate_topic_ids: List[str] = field(default_factory=list)
    user_query: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "topic_id": self.topic_id,
            "action": self.action,
            "summary": self.summary,
            "candidate_topic_ids": self.candidate_topic_ids,
            "user_query": self.user_query,
            "raw_response": self.raw_response,
            "error": self.error,
        }


@dataclass
class TopicRouteDecision:
    use_memory: bool
    direct_answer: str = ""
    topics: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    reason: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_memory": self.use_memory,
            "direct_answer": self.direct_answer,
            "topics": self.topics,
            "modalities": self.modalities,
            "reason": self.reason,
            "raw_response": self.raw_response,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }
