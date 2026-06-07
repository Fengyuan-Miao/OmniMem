"""SQLite topic layer on top of the dual-encoder memory store."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dual_encoder_memory import ImagePointer, UnifiedMemoryRecord
from dual_encoder_memory.store import (
    DualEncoderMemoryStore,
    blob_to_vector,
    normalize_vector,
    vector_to_blob,
)

from .models import TopicRecord


def turn_number(turn_id: str) -> int:
    values = re.findall(r"\d+", str(turn_id or ""))
    return int(values[-1]) if values else 0


class TopicMemoryStore:
    """Add topics and strict topic-scoped vector search to DualEncoderMemoryStore."""

    def __init__(self, storage_dir: str | Path):
        self.base = DualEncoderMemoryStore(storage_dir)
        self.conn = self.base.conn
        self._init_topic_schema()

    @property
    def storage_dir(self) -> Path:
        return self.base.storage_dir

    def close(self) -> None:
        self.base.close()

    def _init_topic_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists topics (
                topic_id text primary key,
                summary text not null,
                summary_dim integer,
                summary_embedding_blob blob,
                turn_count integer not null default 0,
                created_sequence integer not null default 0,
                updated_sequence integer not null default 0,
                first_date text,
                last_date text,
                metadata_json text not null default '{}'
            );

            create table if not exists topic_turns (
                topic_id text not null,
                memory_id text not null,
                session_id text,
                turn_id text,
                date text,
                turn_number integer not null default 0,
                sequence integer not null default 0,
                primary key(topic_id, memory_id),
                foreign key(topic_id) references topics(topic_id),
                foreign key(memory_id) references memories(memory_id)
            );

            create table if not exists topic_events (
                event_id integer primary key autoincrement,
                topic_id text,
                memory_id text,
                action text,
                user_query text,
                candidate_topic_ids_json text not null default '[]',
                llm_response_json text not null default '{}',
                error text,
                created_at text default current_timestamp
            );
            """
        )
        self.conn.commit()

    def add_memory(
        self,
        record: UnifiedMemoryRecord,
        text_embedding: Optional[Iterable[float]] = None,
        image_embeddings: Optional[List[Tuple[ImagePointer, Iterable[float]]]] = None,
    ) -> UnifiedMemoryRecord:
        return self.base.add_memory(
            record,
            text_embedding=text_embedding,
            image_embeddings=image_embeddings,
        )

    def save_indexes(self) -> None:
        self.base.save_indexes()

    def next_sequence(self) -> int:
        row = self.conn.execute(
            "select coalesce(max(sequence), 0) + 1 as next_id from topic_turns"
        ).fetchone()
        return int(row["next_id"])

    def next_topic_id(self) -> str:
        row = self.conn.execute("select count(*) as n from topics").fetchone()
        return f"T{int(row['n']) + 1:03d}"

    def create_topic(
        self,
        summary: str,
        summary_embedding: Iterable[float],
        sequence: int,
        date: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TopicRecord:
        topic_id = self.next_topic_id()
        record = TopicRecord(
            topic_id=topic_id,
            summary=str(summary or "").strip(),
            turn_count=0,
            created_sequence=sequence,
            updated_sequence=sequence,
            first_date=date,
            last_date=date,
            metadata=metadata or {},
        )
        self.upsert_topic(record, summary_embedding)
        return record

    def update_topic(
        self,
        topic_id: str,
        summary: str,
        summary_embedding: Iterable[float],
        sequence: int,
        date: str = "",
    ) -> TopicRecord:
        current = self.get_topic(topic_id)
        if current is None:
            current = TopicRecord(topic_id=topic_id, summary=str(summary or ""))
        current.summary = str(summary or current.summary).strip()
        current.updated_sequence = sequence
        if date:
            current.first_date = current.first_date or date
            current.last_date = max(current.last_date or date, date)
        self.upsert_topic(current, summary_embedding)
        return current

    def upsert_topic(
        self,
        topic: TopicRecord,
        summary_embedding: Optional[Iterable[float]] = None,
    ) -> None:
        vector = normalize_vector(summary_embedding or [])
        dim = int(vector.size) if vector.size else None
        blob = vector_to_blob(vector) if vector.size else None
        self.conn.execute(
            """
            insert or replace into topics
            (topic_id, summary, summary_dim, summary_embedding_blob, turn_count,
             created_sequence, updated_sequence, first_date, last_date, metadata_json)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic.topic_id,
                topic.summary,
                dim,
                blob,
                topic.turn_count,
                topic.created_sequence,
                topic.updated_sequence,
                topic.first_date,
                topic.last_date,
                json.dumps(topic.metadata, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def add_turn_to_topic(
        self,
        topic_id: str,
        record: UnifiedMemoryRecord,
        sequence: int,
    ) -> None:
        self.conn.execute(
            """
            insert or replace into topic_turns
            (topic_id, memory_id, session_id, turn_id, date, turn_number, sequence)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                record.memory_id,
                record.session_id,
                record.turn_id,
                record.date,
                turn_number(record.turn_id),
                sequence,
            ),
        )
        row = self.conn.execute(
            """
            select count(*) as n, min(date) as first_date, max(date) as last_date,
                   max(sequence) as updated_sequence
            from topic_turns where topic_id = ?
            """,
            (topic_id,),
        ).fetchone()
        self.conn.execute(
            """
            update topics
            set turn_count = ?, first_date = ?, last_date = ?, updated_sequence = ?
            where topic_id = ?
            """,
            (
                int(row["n"]),
                str(row["first_date"] or ""),
                str(row["last_date"] or ""),
                int(row["updated_sequence"] or sequence),
                topic_id,
            ),
        )
        self.conn.commit()

    def record_topic_event(
        self,
        topic_id: str,
        memory_id: str,
        action: str,
        user_query: str,
        candidate_topic_ids: Sequence[str],
        response: Dict[str, Any],
        error: str = "",
    ) -> None:
        self.conn.execute(
            """
            insert into topic_events
            (topic_id, memory_id, action, user_query, candidate_topic_ids_json,
             llm_response_json, error)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                memory_id,
                action,
                user_query,
                json.dumps(list(candidate_topic_ids), ensure_ascii=False),
                json.dumps(response or {}, ensure_ascii=False),
                error,
            ),
        )
        self.conn.commit()

    def get_topic(self, topic_id: str) -> Optional[TopicRecord]:
        row = self.conn.execute(
            "select * from topics where topic_id = ?",
            (topic_id,),
        ).fetchone()
        return self._row_to_topic(row) if row else None

    def list_topics(self) -> List[TopicRecord]:
        rows = self.conn.execute(
            "select * from topics order by created_sequence, topic_id"
        ).fetchall()
        return [self._row_to_topic(row) for row in rows]

    def get_topics(self, topic_ids: Sequence[str]) -> List[TopicRecord]:
        wanted = set(topic_ids)
        return [topic for topic in self.list_topics() if topic.topic_id in wanted]

    def candidate_topics(
        self,
        query_embedding: Iterable[float],
        top_k: int,
    ) -> List[Tuple[TopicRecord, float]]:
        topics = []
        query = normalize_vector(query_embedding)
        if query.size == 0 or top_k <= 0:
            return []
        rows = self.conn.execute(
            """
            select * from topics
            where summary_embedding_blob is not null and summary_dim is not null
            """
        ).fetchall()
        for row in rows:
            vector = blob_to_vector(row["summary_embedding_blob"], int(row["summary_dim"]))
            if vector.size != query.size:
                continue
            topics.append((self._row_to_topic(row), float(np.dot(query, vector))))
        topics.sort(key=lambda item: item[1], reverse=True)
        return topics[:top_k]

    def search_text_in_topics(
        self,
        query_embedding: Iterable[float],
        topic_ids: Sequence[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        query = normalize_vector(query_embedding)
        if query.size == 0 or not topic_ids or top_k <= 0:
            return []
        placeholders = ",".join("?" for _ in topic_ids)
        rows = self.conn.execute(
            f"""
            select e.memory_id, e.dim, e.vector_blob
            from text_embeddings e
            join topic_turns tt on tt.memory_id = e.memory_id
            where tt.topic_id in ({placeholders})
            """,
            tuple(topic_ids),
        ).fetchall()
        hits = []
        seen = set()
        for row in rows:
            memory_id = str(row["memory_id"])
            if memory_id in seen:
                continue
            vector = blob_to_vector(row["vector_blob"], int(row["dim"]))
            if vector.size != query.size:
                continue
            seen.add(memory_id)
            hits.append((memory_id, float(np.dot(query, vector))))
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:top_k]

    def search_image_in_topics(
        self,
        query_embedding: Iterable[float],
        topic_ids: Sequence[str],
        top_k: int,
    ) -> List[Tuple[int, str, str, float]]:
        query = normalize_vector(query_embedding)
        if query.size == 0 or not topic_ids or top_k <= 0:
            return []
        placeholders = ",".join("?" for _ in topic_ids)
        rows = self.conn.execute(
            f"""
            select e.image_row_id, e.memory_id, e.dim, e.vector_blob, i.image_id
            from image_embeddings e
            join topic_turns tt on tt.memory_id = e.memory_id
            join images i on i.image_row_id = e.image_row_id
            where tt.topic_id in ({placeholders})
            """,
            tuple(topic_ids),
        ).fetchall()
        hits = []
        seen = set()
        for row in rows:
            image_row_id = int(row["image_row_id"])
            if image_row_id in seen:
                continue
            vector = blob_to_vector(row["vector_blob"], int(row["dim"]))
            if vector.size != query.size:
                continue
            seen.add(image_row_id)
            hits.append(
                (
                    image_row_id,
                    str(row["memory_id"]),
                    str(row["image_id"] or ""),
                    float(np.dot(query, vector)),
                )
            )
        hits.sort(key=lambda item: item[3], reverse=True)
        return hits[:top_k]

    def get_memory(self, memory_id: str) -> Optional[UnifiedMemoryRecord]:
        return self.base.get_memory(memory_id)

    def ordered_memories_for_topic_ids(
        self,
        topic_ids: Sequence[str],
        memory_ids: Optional[Sequence[str]] = None,
    ) -> List[UnifiedMemoryRecord]:
        if not topic_ids:
            return []
        placeholders = ",".join("?" for _ in topic_ids)
        params: List[Any] = list(topic_ids)
        extra = ""
        if memory_ids is not None:
            if not memory_ids:
                return []
            mem_placeholders = ",".join("?" for _ in memory_ids)
            extra = f" and memory_id in ({mem_placeholders})"
            params.extend(memory_ids)
        rows = self.conn.execute(
            f"""
            select distinct memory_id, date, session_id, turn_number, sequence
            from topic_turns
            where topic_id in ({placeholders}){extra}
            order by date, session_id, turn_number, sequence
            """,
            tuple(params),
        ).fetchall()
        memories = []
        for row in rows:
            memory = self.get_memory(str(row["memory_id"]))
            if memory is not None:
                memories.append(memory)
        return memories

    def topic_id_for_memory(self, memory_id: str) -> str:
        row = self.conn.execute(
            "select topic_id from topic_turns where memory_id = ? order by sequence limit 1",
            (memory_id,),
        ).fetchone()
        return str(row["topic_id"]) if row else ""

    def dump_topics_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for topic in self.list_topics():
                f.write(json.dumps(topic.to_dict(), ensure_ascii=False) + "\n")

    def dump_topic_assignments_jsonl(self, path: Path) -> None:
        rows = self.conn.execute("select * from topic_events order by event_id").fetchall()
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(
                    json.dumps(
                        {
                            "event_id": int(row["event_id"]),
                            "topic_id": row["topic_id"],
                            "memory_id": row["memory_id"],
                            "action": row["action"],
                            "user_query": row["user_query"],
                            "candidate_topic_ids": json.loads(row["candidate_topic_ids_json"] or "[]"),
                            "llm_response": json.loads(row["llm_response_json"] or "{}"),
                            "error": row["error"],
                            "created_at": row["created_at"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def stats(self) -> Dict[str, Any]:
        topics = self.list_topics()
        turn_counts = [topic.turn_count for topic in topics]
        avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0.0
        return {
            **self.base.stats(),
            "topic_count": len(topics),
            "avg_turns_per_topic": avg_turns,
        }

    @staticmethod
    def _row_to_topic(row: Any) -> TopicRecord:
        return TopicRecord(
            topic_id=str(row["topic_id"]),
            summary=str(row["summary"] or ""),
            turn_count=int(row["turn_count"] or 0),
            created_sequence=int(row["created_sequence"] or 0),
            updated_sequence=int(row["updated_sequence"] or 0),
            first_date=str(row["first_date"] or ""),
            last_date=str(row["last_date"] or ""),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
