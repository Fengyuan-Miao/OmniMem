"""Prompt context builders for topic-gated retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from dual_encoder_memory.models import RankedMemory

from .models import TopicRecord
from .store import TopicMemoryStore


def _shorten(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def build_topic_index_context(topics: Sequence[TopicRecord]) -> str:
    if not topics:
        return "No memory topics are available."
    lines = ["Available memory topics:"]
    for topic in topics:
        meta = ", ".join(
            part
            for part in [
                f"turns={topic.turn_count}",
                f"range={topic.first_date}..{topic.last_date}" if topic.first_date or topic.last_date else "",
            ]
            if part
        )
        lines.append(f"- {topic.topic_id} ({meta}): {topic.summary}")
    return "\n".join(lines)


def build_ordered_topic_evidence_context(
    store: TopicMemoryStore,
    selected_topics: Sequence[TopicRecord],
    ranked_memories: Sequence[RankedMemory],
    memory_limit: int,
    text_chars: int = 1400,
    caption_chars: int = 300,
) -> str:
    if not selected_topics:
        return "No selected topics."
    ranked_by_memory: Dict[str, RankedMemory] = {
        ranked.memory.memory_id: ranked for ranked in ranked_memories[: max(memory_limit, 0)]
    }
    selected_ids = list(ranked_by_memory)
    lines: List[str] = ["Topic-scoped ordered memory evidence:"]
    for topic in selected_topics:
        lines.append(f"\nTopic {topic.topic_id}: {topic.summary}")
        memories = store.ordered_memories_for_topic_ids([topic.topic_id], selected_ids)
        if not memories:
            lines.append("  No retrieved turns in this topic.")
            continue
        for memory in memories:
            ranked = ranked_by_memory.get(memory.memory_id)
            score_meta = ""
            if ranked is not None:
                score_meta = (
                    f", score={ranked.score:.3f}, text={ranked.text_score:.3f}, "
                    f"image={ranked.image_score:.3f}"
                )
            lines.append(
                f"  Turn {memory.turn_id} (date={memory.date}, session={memory.session_id}{score_meta})"
            )
            lines.append(_shorten(memory.text, text_chars))
            matched_ids = set(ranked.matched_image_ids() if ranked is not None else [])
            for image in memory.images:
                marker = " matched" if image.image_id in matched_ids else ""
                if image.caption:
                    lines.append(
                        f"    Image {image.image_id}{marker}: "
                        f"caption={_shorten(image.caption, caption_chars)}"
                    )
                elif image.image_id:
                    lines.append(f"    Image {image.image_id}{marker}")
            if ranked is not None and ranked.route_hits:
                route_parts = []
                for hit in ranked.route_hits:
                    suffix = f":{hit.image_id}" if hit.image_id else ""
                    route_parts.append(f"{hit.route}{suffix}@{hit.rank}={hit.score:.3f}")
                lines.append("    Routes: " + "; ".join(route_parts))
    return "\n".join(lines)
