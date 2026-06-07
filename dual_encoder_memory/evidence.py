"""Deterministic evidence organization for dual-encoder retrieval results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .models import ImagePointer, RankedMemory, RetrievalResult, RouteHit


def _shorten(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _turn_number(turn_id: str) -> Optional[int]:
    numbers = re.findall(r"\d+", str(turn_id or ""))
    return int(numbers[-1]) if numbers else None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique_route_hits(hits: Iterable[RouteHit]) -> List[RouteHit]:
    seen = set()
    output: List[RouteHit] = []
    for hit in sorted(hits, key=lambda item: (item.rank, item.route, item.image_id)):
        key = (hit.route, hit.memory_id, hit.image_row_id, hit.image_id)
        if key in seen:
            continue
        seen.add(key)
        output.append(hit)
    return output


@dataclass
class EvidenceAtom:
    """One retrieved memory normalized into a prompt-ready evidence unit."""

    evidence_id: str
    memory_id: str
    rank: int
    score: float
    text_score: float = 0.0
    image_score: float = 0.0
    lexical_score: float = 0.0
    date_score: float = 0.0
    session_id: str = ""
    turn_id: str = ""
    turn_number: Optional[int] = None
    date: str = ""
    manual_observed_at: str = ""
    session_index: Optional[int] = None
    turn_index: Optional[int] = None
    global_turn_index: Optional[int] = None
    text: str = ""
    images: List[ImagePointer] = field(default_factory=list)
    route_hits: List[RouteHit] = field(default_factory=list)

    @classmethod
    def from_ranked(cls, ranked: RankedMemory, rank: int) -> "EvidenceAtom":
        memory = ranked.memory
        return cls(
            evidence_id=f"E{rank}",
            memory_id=memory.memory_id,
            rank=rank,
            score=ranked.score,
            text_score=ranked.text_score,
            image_score=ranked.image_score,
            lexical_score=ranked.lexical_score,
            date_score=ranked.date_score,
            session_id=memory.session_id,
            turn_id=memory.turn_id,
            turn_number=_turn_number(memory.turn_id),
            date=memory.date,
            manual_observed_at=str(memory.metadata.get("manual_observed_at") or ""),
            session_index=_coerce_int(memory.metadata.get("session_index")),
            turn_index=_coerce_int(memory.metadata.get("turn_index")),
            global_turn_index=_coerce_int(memory.metadata.get("global_turn_index")),
            text=memory.text,
            images=list(memory.images),
            route_hits=_unique_route_hits(ranked.route_hits),
        )

    def matched_image_ids(self) -> List[str]:
        ids: List[str] = []
        for hit in self.route_hits:
            if hit.image_id and hit.image_id not in ids:
                ids.append(hit.image_id)
        return ids

    def route_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "route": hit.route,
                "rank": hit.rank,
                "score": hit.score,
                "image_id": hit.image_id,
            }
            for hit in self.route_hits
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "memory_id": self.memory_id,
            "rank": self.rank,
            "score": self.score,
            "text_score": self.text_score,
            "image_score": self.image_score,
            "lexical_score": self.lexical_score,
            "date_score": self.date_score,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "turn_number": self.turn_number,
            "date": self.date,
            "manual_observed_at": self.manual_observed_at,
            "session_index": self.session_index,
            "turn_index": self.turn_index,
            "global_turn_index": self.global_turn_index,
            "text": self.text,
            "images": [image.to_dict() for image in self.images],
            "matched_image_ids": self.matched_image_ids(),
            "route_hits": self.route_summary(),
        }

    def to_prompt_block(self, text_chars: int, caption_chars: int) -> str:
        lines = [f"Evidence {self.rank}:"]
        source_parts = [
            f"time/order={self.manual_observed_at}" if self.manual_observed_at else "",
            f"date={self.date}" if self.date else "",
            f"session={self.session_id}" if self.session_id else "",
            f"turn={self.turn_id}" if self.turn_id else "",
        ]
        source = ", ".join(part for part in source_parts if part)
        if source:
            lines.append(f"Source: {source}")
        if self.text:
            lines.append("Text:")
            lines.append(_shorten(self.text, text_chars))
        matched_ids = set(self.matched_image_ids())
        image_lines = []
        for image in self.images:
            marker = " (selected visual evidence)" if image.image_id in matched_ids else ""
            image_id = image.image_id or "unknown-image"
            if image.caption:
                image_lines.append(f"- {image_id}{marker}: {_shorten(image.caption, caption_chars)}")
            else:
                image_lines.append(f"- {image_id}{marker}")
        if image_lines:
            lines.append("Images:")
            lines.extend(image_lines)
        return "\n".join(lines)


@dataclass
class EvidenceGroup:
    """A small chronological bundle of nearby evidence atoms."""

    group_id: str
    atoms: List[EvidenceAtom] = field(default_factory=list)

    @property
    def score(self) -> float:
        return max((atom.score for atom in self.atoms), default=0.0)

    @property
    def first_rank(self) -> int:
        return min((atom.rank for atom in self.atoms), default=10**6)

    @property
    def session_id(self) -> str:
        return self.atoms[0].session_id if self.atoms else ""

    @property
    def date(self) -> str:
        return self.atoms[0].date if self.atoms else ""

    def can_merge(self, atom: EvidenceAtom, neighbor_turn_window: int) -> bool:
        if not self.atoms:
            return True
        if atom.session_id != self.session_id or atom.date != self.date:
            return False
        if atom.turn_number is None:
            return False
        turn_numbers = [item.turn_number for item in self.atoms if item.turn_number is not None]
        if not turn_numbers:
            return False
        return min(abs(atom.turn_number - turn) for turn in turn_numbers) <= neighbor_turn_window

    def add(self, atom: EvidenceAtom) -> None:
        self.atoms.append(atom)
        self.atoms.sort(
            key=lambda item: (
                item.global_turn_index is None,
                item.global_turn_index or 10**9,
                item.turn_number is None,
                item.turn_number or 10**9,
                item.rank,
            )
        )

    def image_ids(self) -> List[str]:
        ids: List[str] = []
        for atom in self.atoms:
            for image in atom.images:
                if image.image_id and image.image_id not in ids:
                    ids.append(image.image_id)
        return ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "score": self.score,
            "first_rank": self.first_rank,
            "session_id": self.session_id,
            "date": self.date,
            "image_ids": self.image_ids(),
            "atoms": [atom.to_dict() for atom in self.atoms],
        }

    def to_prompt_block(self, text_chars: int, caption_chars: int) -> str:
        turns = [atom.turn_id for atom in self.atoms if atom.turn_id]
        turn_text = ""
        if turns:
            turn_text = turns[0] if len(turns) == 1 else f"{turns[0]} to {turns[-1]}"
        times = [atom.manual_observed_at for atom in self.atoms if atom.manual_observed_at]
        time_text = ""
        if times:
            time_text = times[0] if len(times) == 1 else f"{times[0]} to {times[-1]}"
        meta = ", ".join(
            part
            for part in [
                f"time/order={time_text}" if time_text else "",
                f"date={self.date}" if self.date else "",
                f"session={self.session_id}" if self.session_id else "",
                f"turns={turn_text}" if turn_text else "",
            ]
            if part
        )
        heading = f"Evidence group {self.group_id}"
        if meta:
            heading += f" ({meta})"
        lines = [heading]
        for atom in self.atoms:
            lines.append(atom.to_prompt_block(text_chars=text_chars, caption_chars=caption_chars))
        return "\n".join(lines)


@dataclass
class EvidenceSet:
    query: str
    atoms: List[EvidenceAtom]
    groups: List[EvidenceGroup]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "groups": [group.to_dict() for group in self.groups],
        }

    def to_prompt_context(
        self,
        group_limit: int,
        text_chars: int = 1400,
        caption_chars: int = 300,
    ) -> str:
        lines = ["Organized retrieved evidence:"]
        for group in self.groups[: max(0, group_limit)]:
            lines.append(group.to_prompt_block(text_chars=text_chars, caption_chars=caption_chars))
        if len(lines) == 1:
            lines.append("No retrieved evidence.")
        return "\n\n".join(lines)


class EvidenceOrganizer:
    """Build evidence atoms and nearby-turn groups without model calls."""

    def __init__(self, neighbor_turn_window: int = 1):
        self.neighbor_turn_window = max(0, int(neighbor_turn_window))

    def organize(
        self,
        retrieval: RetrievalResult,
        max_atoms: Optional[int] = None,
    ) -> EvidenceSet:
        ranked = retrieval.ranked_memories
        if max_atoms is not None:
            ranked = ranked[: max(0, max_atoms)]
        atoms = [
            EvidenceAtom.from_ranked(ranked_memory, rank=index)
            for index, ranked_memory in enumerate(ranked, start=1)
        ]
        groups = self._group_atoms(atoms)
        return EvidenceSet(query=retrieval.query, atoms=atoms, groups=groups)

    def _group_atoms(self, atoms: List[EvidenceAtom]) -> List[EvidenceGroup]:
        groups: List[EvidenceGroup] = []
        chronological = sorted(
            atoms,
            key=lambda atom: (
                atom.global_turn_index is None,
                atom.global_turn_index or 10**9,
                atom.date,
                atom.session_id,
                atom.turn_number is None,
                atom.turn_number or 10**9,
                atom.rank,
            ),
        )
        for atom in chronological:
            target = None
            for group in groups:
                if group.can_merge(atom, self.neighbor_turn_window):
                    target = group
                    break
            if target is None:
                target = EvidenceGroup(group_id=f"G{len(groups) + 1}")
                groups.append(target)
            target.add(atom)
        groups.sort(key=lambda group: (group.first_rank, -group.score))
        for index, group in enumerate(groups, start=1):
            group.group_id = f"G{index}"
        return groups
