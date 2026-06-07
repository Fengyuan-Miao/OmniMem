"""Topic-gated multimodal memory for RG-OmniMem experiments."""

from .context import build_ordered_topic_evidence_context, build_topic_index_context
from .models import TopicAssignment, TopicRecord, TopicRouteDecision
from .retriever import TopicScopedRetriever
from .store import TopicMemoryStore
from .topic_builder import TopicBuilder

__all__ = [
    "TopicAssignment",
    "TopicBuilder",
    "TopicMemoryStore",
    "TopicRecord",
    "TopicRouteDecision",
    "TopicScopedRetriever",
    "build_ordered_topic_evidence_context",
    "build_topic_index_context",
]
