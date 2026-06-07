"""Persistent stores and lightweight indexes for SVI-OmniMem."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import (
    QUERY_VERIFIED,
    StructuredVisualCard,
    VerifiedVisualFact,
)
from .utils import normalize_text, tokenize


def _match_tags(card: StructuredVisualCard, tags_filter: Optional[List[str]]) -> bool:
    if not tags_filter:
        return True
    tags = {normalize_text(tag) for tag in card.tags}
    return any(normalize_text(tag) in tags for tag in tags_filter)


def _match_time(
    observed_at: Optional[str],
    time_range: Optional[Tuple[Any, Any]],
) -> bool:
    if not time_range or not observed_at:
        return True
    start, end = time_range
    value = str(observed_at)
    if start is not None and value < str(start):
        return False
    if end is not None and value > str(end):
        return False
    return True


def _field_score(query_terms: Iterable[str], field_text: str) -> float:
    query_tokens = {normalize_text(term) for term in query_terms if normalize_text(term)}
    if not query_tokens:
        return 0.0

    field_norm = normalize_text(field_text)
    field_tokens = set(tokenize(field_norm))
    hits = 0
    for term in query_tokens:
        if not term:
            continue
        if term in field_norm or term in field_tokens:
            hits += 1
    return hits / max(len(query_tokens), 1)


def _card_date_text(observed_at: Optional[str]) -> str:
    if not observed_at:
        return ""
    value = str(observed_at).strip()
    if not value:
        return ""
    parsed = None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(value[:10], "%Y-%m-%d")
        except ValueError:
            return value
    if parsed is None:
        return ""
    return " ".join(
        part
        for part in [
            parsed.strftime("%B"),
            parsed.strftime("%b"),
            str(parsed.year),
            str(parsed.month),
            str(parsed.day),
        ]
        if part
    )


class StructuredVisualStore:
    """JSONL-backed store for StructuredVisualCard records.

    The store maintains compact in-memory inverted indexes. The card is used
    only as a retrieval hint; raw image verification remains a separate step.
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "structured_visual_cards.jsonl"
        self._lock = threading.RLock()
        self._cards_by_id: Dict[str, StructuredVisualCard] = {}
        self._cards_by_image: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self._lock:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    card = StructuredVisualCard.from_dict(json.loads(line))
                except Exception:
                    continue
                self._cards_by_id[card.card_id] = card
                self._cards_by_image[card.image_mau_id] = card.card_id

    def append(self, card: StructuredVisualCard) -> str:
        with self._lock:
            self._cards_by_id[card.card_id] = card
            self._cards_by_image[card.image_mau_id] = card.card_id
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(card.to_dict(), ensure_ascii=False) + "\n")
        return card.card_id

    def get_by_card_id(self, card_id: str) -> Optional[StructuredVisualCard]:
        return self._cards_by_id.get(card_id)

    def get_by_image_mau_id(self, image_mau_id: str) -> Optional[StructuredVisualCard]:
        card_id = self._cards_by_image.get(image_mau_id)
        return self._cards_by_id.get(card_id) if card_id else None

    def all_cards(self) -> List[StructuredVisualCard]:
        return list(self._cards_by_id.values())

    def count(self) -> int:
        return len(self._cards_by_id)

    def filtered_cards(
        self,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[StructuredVisualCard]:
        return list(self._iter_filtered(tags_filter, time_range))

    def _iter_filtered(
        self,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> Iterable[StructuredVisualCard]:
        for card in self._cards_by_id.values():
            if not _match_tags(card, tags_filter):
                continue
            if not _match_time(card.observed_at, time_range):
                continue
            yield card

    def search_entity_alias(
        self,
        entity_terms: List[str],
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[Tuple[StructuredVisualCard, float, str]]:
        results: List[Tuple[StructuredVisualCard, float, str]] = []
        for card in self._iter_filtered(tags_filter, time_range):
            best_score = 0.0
            best_field = ""
            for anchor in card.retrieval_anchors:
                field_text = " ".join(
                    [
                        anchor.category,
                    ]
                )
                score = _field_score(entity_terms, field_text)
                if score > best_score:
                    best_score = score
                    best_field = f"anchor:{anchor.anchor_id}:{field_text}"
            caption_score = _field_score(entity_terms, card.global_caption)
            if caption_score > best_score:
                best_score = caption_score * 0.7
                best_field = f"caption:{card.global_caption}"
            if best_score > 0:
                results.append((card, best_score, best_field))
        return sorted(results, key=lambda item: item[1], reverse=True)

    def search_attribute(
        self,
        entity_terms: List[str],
        attribute_terms: List[str],
        value_terms: Optional[List[str]] = None,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[Tuple[StructuredVisualCard, float, str]]:
        value_terms = value_terms or []
        results: List[Tuple[StructuredVisualCard, float, str]] = []
        for card in self._iter_filtered(tags_filter, time_range):
            best_score = 0.0
            best_field = ""
            for anchor in card.retrieval_anchors:
                entity_score = _field_score(entity_terms, " ".join(anchor.all_names()))
                for attr_name, attr in anchor.salient_attributes.items():
                    attr_field = f"{attr_name} {attr.value}"
                    attr_score = _field_score(attribute_terms, attr_field)
                    value_score = _field_score(value_terms, attr.value) if value_terms else 0
                    score = (entity_score * 0.45) + (attr_score * 0.40)
                    if value_terms:
                        score += value_score * 0.15
                    elif attr_score > 0:
                        score += 0.10
                    if score > best_score:
                        best_score = score
                        best_field = f"attribute:{anchor.anchor_id}:{attr_name}={attr.value}"
            if best_score > 0:
                results.append((card, best_score, best_field))
        return sorted(results, key=lambda item: item[1], reverse=True)

    def search_ocr(
        self,
        terms: List[str],
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[Tuple[StructuredVisualCard, float, str]]:
        results: List[Tuple[StructuredVisualCard, float, str]] = []
        for card in self._iter_filtered(tags_filter, time_range):
            best_score = 0.0
            best_field = ""
            for obs in card.ocr_observations:
                field_text = " ".join([obs.text, obs.context or ""])
                score = _field_score(terms, field_text)
                if score > best_score:
                    best_score = score
                    best_field = f"ocr:{field_text}"
            if best_score > 0:
                results.append((card, best_score, best_field))
        return sorted(results, key=lambda item: item[1], reverse=True)

    def search_caption(
        self,
        query: str,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[Tuple[StructuredVisualCard, float, str]]:
        terms = tokenize(query)
        results: List[Tuple[StructuredVisualCard, float, str]] = []
        for card in self._iter_filtered(tags_filter, time_range):
            mirror = card.to_mirror_text()
            score = max(
                _field_score(terms, card.global_caption),
                _field_score(terms, mirror) * 0.8,
            )
            if score > 0:
                results.append((card, score, f"caption:{card.global_caption}"))
        return sorted(results, key=lambda item: item[1], reverse=True)

    def search_all_text(
        self,
        query: str,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[Tuple[StructuredVisualCard, float, str]]:
        """Search all card text without query-type or benchmark-specific routing."""
        visual_query = query.startswith("VISUAL_COMPARE::")
        if query.startswith("VISUAL_COMPARE::") or query.startswith("IMAGE_RECALL::"):
            query = query.split("\n", 1)[1] if "\n" in query else ""
        terms = tokenize(query)
        results: List[Tuple[StructuredVisualCard, float, str]] = []
        for card in self._iter_filtered(tags_filter, time_range):
            date_text = _card_date_text(card.observed_at)
            fields = [
                ("global_caption", card.global_caption),
                ("mirror_text", card.to_mirror_text()),
                ("source_text_context", card.source_text_context),
                ("tags", " ".join(card.tags)),
                ("source", " ".join([card.session_id or "", card.turn_id or ""])),
            ]
            best_score = 0.0
            best_field = ""
            for field_name, field_text in fields:
                score = _field_score(terms, field_text)
                if field_name == "global_caption":
                    score *= 1.35 if visual_query else 1.25
                elif field_name == "mirror_text":
                    score *= 1.05 if visual_query else 1.00
                elif field_name == "source_text_context":
                    score *= 0.20 if visual_query else 0.70
                else:
                    score *= 0.25 if visual_query else 0.45
                if score > best_score:
                    best_score = score
                    best_field = f"{field_name}:{field_text[:160]}"
            if date_text:
                date_score = _field_score(terms, date_text)
                if date_score > 0:
                    best_score += date_score * (0.70 if visual_query else 0.50)
                    if not best_field:
                        best_field = f"date_text:{date_text[:160]}"
            if best_score > 0:
                results.append((card, best_score, best_field))
        return sorted(results, key=lambda item: item[1], reverse=True)

    def get_recent(
        self,
        limit: int = 3,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[StructuredVisualCard]:
        cards = list(self._iter_filtered(tags_filter, time_range))
        return sorted(cards, key=lambda card: card.observed_at, reverse=True)[:limit]

    def get_temporal_sequence(
        self,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[StructuredVisualCard]:
        return sorted(
            self._iter_filtered(tags_filter, time_range),
            key=lambda card: card.observed_at,
        )

    def get_same_session(
        self,
        session_id: Optional[str],
        limit: int = 3,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[StructuredVisualCard]:
        if not session_id:
            return []
        cards = [
            card
            for card in self._iter_filtered(tags_filter, time_range)
            if card.session_id == session_id
        ]
        return sorted(cards, key=lambda card: card.observed_at, reverse=True)[:limit]


class VerifiedFactStore:
    """JSONL-backed store for query-verified visual facts."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "verified_visual_facts.jsonl"
        self._lock = threading.RLock()
        self._facts_by_id: Dict[str, VerifiedVisualFact] = {}
        self._key_index: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self._lock:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    fact = VerifiedVisualFact.from_dict(json.loads(line))
                except Exception:
                    continue
                self._facts_by_id[fact.fact_id] = fact
                self._key_index[fact.key()] = fact.fact_id

    def append(self, fact: VerifiedVisualFact, deduplicate: bool = True) -> str:
        with self._lock:
            if deduplicate and fact.key() in self._key_index:
                return self._key_index[fact.key()]
            self._facts_by_id[fact.fact_id] = fact
            self._key_index[fact.key()] = fact.fact_id
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")
        return fact.fact_id

    def get(self, fact_id: str) -> Optional[VerifiedVisualFact]:
        return self._facts_by_id.get(fact_id)

    def all_facts(self) -> List[VerifiedVisualFact]:
        return list(self._facts_by_id.values())

    def count(self) -> int:
        return len(self._facts_by_id)

    def search(
        self,
        terms: List[str],
        query_type: Optional[str] = None,
        tags_filter: Optional[List[str]] = None,
        time_range: Optional[Tuple[Any, Any]] = None,
    ) -> List[Tuple[VerifiedVisualFact, float, str]]:
        del tags_filter
        results: List[Tuple[VerifiedVisualFact, float, str]] = []
        for fact in self._facts_by_id.values():
            if fact.status != QUERY_VERIFIED:
                continue
            if query_type and fact.query_type != query_type:
                continue
            if not _match_time(fact.observation_time, time_range):
                continue
            field_text = " ".join(
                [
                    fact.subject,
                    fact.predicate,
                    fact.value,
                    fact.evidence_description,
                ]
            )
            score = _field_score(terms, field_text) * max(fact.confidence, 0.4)
            if score > 0:
                results.append((fact, score, f"verified_fact:{fact.fact_id}"))
        return sorted(results, key=lambda item: item[1], reverse=True)

    def conflicting_after(
        self,
        subject: str,
        predicate: str,
        value: str,
        observation_time: str,
    ) -> bool:
        subject_norm = normalize_text(subject)
        predicate_norm = normalize_text(predicate)
        value_norm = normalize_text(value)
        for fact in self._facts_by_id.values():
            if fact.status != QUERY_VERIFIED:
                continue
            if normalize_text(fact.subject) != subject_norm:
                continue
            if normalize_text(fact.predicate) != predicate_norm:
                continue
            if normalize_text(fact.value) == value_norm:
                continue
            if str(fact.observation_time) >= str(observation_time):
                return True
        return False
