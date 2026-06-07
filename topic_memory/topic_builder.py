"""LLM-assisted topic assignment and summary maintenance."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Any, Dict, List, Sequence

from dual_encoder_memory import UnifiedMemoryRecord

from .models import TopicAssignment, TopicRecord
from .store import TopicMemoryStore


MAX_TOPIC_SUMMARY_CHARS = 220


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : index + 1])
    raise ValueError("unterminated JSON object")


def concise_fallback_summary(user_query: str) -> str:
    text = re.sub(r"\s+", " ", str(user_query or "")).strip()
    if not text:
        return "User asked about an unspecified topic."
    sentence = re.split(r"(?<=[.!?])\s+", text)[0]
    return sentence[:220].rstrip()


SUMMARY_STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "discusses",
    "discussing",
    "explores",
    "exploring",
    "for",
    "from",
    "includes",
    "including",
    "into",
    "now",
    "the",
    "their",
    "topic",
    "user",
    "with",
}


def summary_terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {
        token
        for token in tokens
        if len(token) > 3 and token not in SUMMARY_STOPWORDS
    }


def clip_at_word_boundary(text: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    clipped = value[: max(0, limit - 1)].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()
    return clipped.rstrip(" ,;:-")


class TopicLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        max_tokens: int = 256,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

    def complete_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'ollama'}",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = str(body["choices"][0]["message"].get("content") or "")
        return extract_json_object(content)


class TopicBuilder:
    def __init__(
        self,
        store: TopicMemoryStore,
        text_encoder: Any,
        llm_client: TopicLLMClient,
        match_top_k: int = 12,
    ):
        self.store = store
        self.text_encoder = text_encoder
        self.llm_client = llm_client
        self.match_top_k = match_top_k

    def assign_turn(
        self,
        user_query: str,
        memory: UnifiedMemoryRecord,
    ) -> TopicAssignment:
        sequence = self.store.next_sequence()
        query_embedding = self.text_encoder.encode(user_query)
        candidates = self.store.candidate_topics(query_embedding, self.match_top_k)
        candidate_topics = [topic for topic, _score in candidates]
        candidate_ids = [topic.topic_id for topic in candidate_topics]
        messages = self._assignment_messages(user_query, candidate_topics)
        raw: Dict[str, Any] = {}
        error = ""
        try:
            raw = self.llm_client.complete_json(messages)
            action = str(raw.get("action") or "").lower().strip()
            topic_id = str(raw.get("topic_id") or "").strip()
            summary = str(raw.get("topic_summary") or raw.get("summary") or "").strip()
            if action not in {"merge", "new"}:
                raise ValueError(f"invalid topic action: {action}")
            if action == "merge" and topic_id not in candidate_ids:
                raise ValueError(f"invalid merge topic_id: {topic_id}")
            if not summary:
                raise ValueError("empty topic summary")
        except Exception as exc:
            action = "new"
            topic_id = ""
            summary = concise_fallback_summary(user_query)
            raw = raw or {}
            error = str(exc)

        if action == "merge":
            current = self.store.get_topic(topic_id)
            summary = self._merge_summary(
                current.summary if current is not None else "",
                summary,
            )
            summary = self._limit_summary(summary)
            summary_embedding = self.text_encoder.encode(summary)
            topic = self.store.update_topic(
                topic_id,
                summary,
                summary_embedding,
                sequence=sequence,
                date=memory.date,
            )
        else:
            summary = self._limit_summary(summary)
            summary_embedding = self.text_encoder.encode(summary)
            topic = self.store.create_topic(
                summary,
                summary_embedding,
                sequence=sequence,
                date=memory.date,
            )
            topic_id = topic.topic_id
            action = "new"
        self.store.add_turn_to_topic(topic_id, memory, sequence)
        self.store.record_topic_event(
            topic_id=topic_id,
            memory_id=memory.memory_id,
            action=action,
            user_query=user_query,
            candidate_topic_ids=candidate_ids,
            response=raw,
            error=error,
        )
        return TopicAssignment(
            memory_id=memory.memory_id,
            topic_id=topic_id,
            action=action,
            summary=topic.summary,
            candidate_topic_ids=candidate_ids,
            user_query=user_query,
            raw_response=raw,
            error=error,
        )

    @staticmethod
    def _limit_summary(summary: str) -> str:
        value = re.sub(r"\s+", " ", str(summary or "")).strip()
        sentences = re.split(r"(?<=[.!?])\s+", value)
        value = sentences[0].strip() if sentences else value
        value = clip_at_word_boundary(value, MAX_TOPIC_SUMMARY_CHARS)
        return value.rstrip(".") + "." if value else ""

    @staticmethod
    def _merge_summary(existing: str, proposed: str) -> str:
        old = re.sub(r"\s+", " ", str(existing or "")).strip()
        new = re.sub(r"\s+", " ", str(proposed or "")).strip()
        if not old:
            return new
        if not new:
            return old
        old_l = old.lower().rstrip(".")
        new_l = new.lower().rstrip(".")
        if old_l in new_l:
            return new
        if new_l in old_l:
            return old

        old_terms = summary_terms(old)
        new_terms = summary_terms(new)
        if old_terms:
            common = old_terms & new_terms
            preserved = len(common) / max(len(old_terms), 1)
            if preserved >= 0.35 or len(common) >= 4:
                return new

        old_part = clip_at_word_boundary(old.rstrip("."), 110)
        new_part = clip_at_word_boundary(new.rstrip("."), 100)
        return f"{old_part}; also includes {new_part}."

    @staticmethod
    def _assignment_messages(
        user_query: str,
        candidates: Sequence[TopicRecord],
    ) -> List[Dict[str, str]]:
        topic_lines = []
        for topic in candidates:
            topic_lines.append(
                f"- {topic.topic_id}: {topic.summary} "
                f"(turns={topic.turn_count}, range={topic.first_date}..{topic.last_date})"
            )
        topic_index = "\n".join(topic_lines) if topic_lines else "No existing topics."
        system_prompt = """You are a topic organizer for a multimodal long-term memory system.

Your job is to keep topics FINE-GRAINED, STABLE, ADDITIVE, and CONCISE.

Definitions:
- A topic is a coherent user thread: one goal, entity set, object set, place/event, project, image collection, or decision process.
- Do not use broad domains as topics when the user is actually moving across distinct subthreads. "AI", "pets", "travel", "fashion", or "education" are usually too broad.
- A good topic summary is a compact abstraction, not an event log. It should be specific enough that a router can later choose it without reading all turns.

Decision rules:
1. Use only the current user query and the candidate topic summaries. Never use assistant text directly, because it is not provided to you.
2. Prefer `new` when the query introduces a different entity, object, image collection, location, event, project, decision, or date-bounded activity.
3. Prefer `new` when two threads share only a broad domain. Do not merge just because both are about pets, AI, art, travel, food, school, shopping, or home life.
4. Prefer `new` when uncertain. A few extra fine topics are better than one coarse mixed topic.
5. Use `merge` only when the new query is clearly a continuation of the same concrete thread.
6. When merging, DO NOT overwrite the topic with just the newest turn. Preserve the stable topic scope and add only the important new facet.
7. For merge summaries, write additive summaries: existing scope first, then the new facet. Do not drop old entities, old goals, or old constraints.
8. The `topic_summary` MUST be one short sentence, ideally 14 to 26 words and under 180 characters.
9. Summaries must be conceptual. Do not list every turn, date, image, seminar, example, side comment, or latest event.
10. Avoid summaries that collapse to a single latest example, brand, object, or date unless that detail defines the whole thread.

Fine-grained boundaries:
- Same pet adoption/care thread -> merge.
- Different pet, breed encounter, outfit shopping thread, or park event -> new.
- Same trip planning thread -> merge.
- Different destination, museum visit, restaurant search, or gear purchase -> new.
- Same research/project thread -> merge.
- Different paper, lecture, experiment, dataset, or job/career decision -> new.
- Same product comparison or purchase decision -> merge.
- Different product category or unrelated shopping session -> new.

Output format:
Return only valid JSON with exactly these keys:
{
  "action": "merge" or "new",
  "topic_id": "existing topic id when action is merge, otherwise empty",
  "topic_summary": "one concise sentence summarizing the stable topic scope"
}

Examples:

Example 1 - merge same concrete thread:
Candidate topics:
- T001: The user is comparing autonomous vehicle safety, reliability, sensors, and traffic automation.
New user query:
Can we add how real-time road data helps autonomous cars avoid risky turns?
Correct decision:
{
  "action": "merge",
  "topic_id": "T001",
  "topic_summary": "The user compares autonomous vehicle safety, sensors, traffic automation, and real-time road data for safer decisions."
}

Example 2 - new despite same broad domain:
Candidate topics:
- T001: The user is discussing autonomous vehicles and safety.
New user query:
Can AI help with hospital robots that deliver medicine to patients?
Correct decision:
{
  "action": "new",
  "topic_id": "",
  "topic_summary": "The user discusses healthcare robots for hospital delivery tasks and patient support."
}

Example 3 - merge with preserved older details:
Candidate topics:
- T002: The user is planning Maltese dog adoption, including breed temperament, grooming, diet, and home care.
New user query:
Now I'm comparing Maltese grooming and diet with other small dogs before adopting one.
Correct decision:
{
  "action": "merge",
  "topic_id": "T002",
  "topic_summary": "The user plans Maltese dog adoption, covering temperament, grooming, diet, care, and comparisons with other small dogs."
}

Example 4 - new pet subthread, not a coarse pet topic:
Candidate topics:
- T002: The user is planning Maltese dog adoption, including temperament, grooming, diet, and home care.
New user query:
At the park we met a Papillon with butterfly-like ears; I want to remember that encounter.
Correct decision:
{
  "action": "new",
  "topic_id": "",
  "topic_summary": "The user remembers a park encounter with a Papillon dog and its butterfly-like ears."
}

Example 5 - merge must not overwrite:
Candidate topics:
- T003: The user is choosing dresses for an event, comparing floral patterns, sleeve length, comfort, and daily wear.
New user query:
I finally ordered the Abstract Floral Babydoll Dress because it is comfortable and versatile.
Correct decision:
{
  "action": "merge",
  "topic_id": "T003",
  "topic_summary": "The user chooses event dresses, comparing floral patterns, comfort, daily wear, and the final Babydoll Dress order."
}
"""
        user_prompt = f"""Candidate topics:
{topic_index}

New user query:
{user_query}
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
