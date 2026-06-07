"""SQLite-backed GME memory store with one FAISS index per entry."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .encoders import l2_normalize
from .models import GmeImagePointer, GmeMemoryRecord


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="float32").tobytes()


def blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    if not blob:
        return np.asarray([], dtype="float32")
    arr = np.frombuffer(blob, dtype="float32")
    if dim and arr.size != dim:
        return np.asarray([], dtype="float32")
    return arr.astype("float32", copy=True)


def normalize_vector(vector: Iterable[float]) -> np.ndarray:
    return np.asarray(l2_normalize(vector), dtype="float32")


class GmeMemoryStore:
    """Persist unified memory entries and accelerate search with FAISS."""

    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.storage_dir / "gme_memory.sqlite"
        self.entry_index_path = self.storage_dir / "faiss_entry.index"
        self.conn = sqlite3.connect(str(self.sqlite_path))
        self.conn.row_factory = sqlite3.Row
        self._faiss_entry = None
        self._init_schema()
        self._load_or_rebuild_index()

    def close(self) -> None:
        self.save_index()
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists memories (
                memory_id text primary key,
                text text not null,
                session_id text,
                turn_id text,
                date text,
                metadata_json text not null default '{}'
            );

            create table if not exists images (
                image_row_id integer primary key autoincrement,
                memory_id text not null,
                image_id text,
                path text not null,
                caption text,
                metadata_json text not null default '{}',
                foreign key(memory_id) references memories(memory_id)
            );

            create table if not exists entry_embeddings (
                memory_id text primary key,
                dim integer not null,
                vector_blob blob not null,
                faiss_row_id integer not null unique,
                embedding_mode text not null,
                embedding_image_row_id integer,
                embedding_image_id text,
                foreign key(memory_id) references memories(memory_id)
            );

            create table if not exists index_meta (
                key text primary key,
                value text
            );
            """
        )
        self.conn.commit()

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "insert or replace into index_meta(key, value) values (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("select value from index_meta where key = ?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def add_memory(
        self,
        record: GmeMemoryRecord,
        entry_embedding: Optional[Iterable[float]] = None,
        embedding_mode: str = "text",
        embedding_image_id: str = "",
    ) -> GmeMemoryRecord:
        self.conn.execute(
            """
            insert or replace into memories
            (memory_id, text, session_id, turn_id, date, metadata_json)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                record.memory_id,
                record.text,
                record.session_id,
                record.turn_id,
                record.date,
                json.dumps(record.metadata, ensure_ascii=False),
            ),
        )
        self.conn.execute("delete from images where memory_id = ?", (record.memory_id,))
        stored_images: List[GmeImagePointer] = []
        for image in record.images:
            cursor = self.conn.execute(
                """
                insert into images(memory_id, image_id, path, caption, metadata_json)
                values (?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    image.image_id,
                    image.path,
                    image.caption,
                    json.dumps(image.metadata, ensure_ascii=False),
                ),
            )
            image.image_row_id = int(cursor.lastrowid)
            stored_images.append(image)
        record.images = stored_images
        if entry_embedding is not None:
            self.add_entry_embedding(
                memory_id=record.memory_id,
                embedding=entry_embedding,
                embedding_mode=embedding_mode,
                embedding_image_id=embedding_image_id,
            )
        self.conn.commit()
        return record

    def add_entry_embedding(
        self,
        memory_id: str,
        embedding: Iterable[float],
        embedding_mode: str = "text",
        embedding_image_id: str = "",
    ) -> None:
        vector = normalize_vector(embedding)
        if vector.size == 0:
            return
        dim = int(vector.size)
        row = self.conn.execute(
            """
            select image_row_id from images
            where memory_id = ? and image_id = ?
            order by image_row_id limit 1
            """,
            (memory_id, embedding_image_id),
        ).fetchone()
        embedding_image_row_id = int(row["image_row_id"]) if row else None
        faiss_row_id = self._next_faiss_row_id()
        self.conn.execute(
            """
            insert or replace into entry_embeddings
            (memory_id, dim, vector_blob, faiss_row_id, embedding_mode,
             embedding_image_row_id, embedding_image_id)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                dim,
                vector_to_blob(vector),
                faiss_row_id,
                embedding_mode,
                embedding_image_row_id,
                embedding_image_id,
            ),
        )
        self._add_to_faiss(vector)
        self.set_meta("entry_dim", dim)

    def _next_faiss_row_id(self) -> int:
        row = self.conn.execute(
            "select coalesce(max(faiss_row_id), -1) + 1 as next_id from entry_embeddings"
        ).fetchone()
        return int(row["next_id"])

    def _load_or_rebuild_index(self) -> None:
        if not self._try_load_faiss():
            self.rebuild_faiss_index()

    def _try_load_faiss(self) -> bool:
        try:
            import faiss
        except Exception:
            return False
        if not self.entry_index_path.exists():
            return False
        try:
            index = faiss.read_index(str(self.entry_index_path))
        except Exception:
            return False
        if index.ntotal != self.count_entry_embeddings():
            return False
        expected_dim = self.get_meta("entry_dim")
        if expected_dim and index.d != int(expected_dim):
            return False
        self._faiss_entry = index
        return True

    def rebuild_faiss_index(self) -> None:
        import faiss

        rows = self.conn.execute(
            "select dim, vector_blob from entry_embeddings order by faiss_row_id asc"
        ).fetchall()
        dim = int(rows[0]["dim"]) if rows else 1
        index = faiss.IndexFlatIP(dim)
        if rows:
            vectors = np.vstack(
                [blob_to_vector(row["vector_blob"], int(row["dim"])) for row in rows]
            ).astype("float32")
            if vectors.size:
                index.add(vectors)
        self._faiss_entry = index
        self.save_index()

    def _add_to_faiss(self, vector: np.ndarray) -> None:
        import faiss

        if self._faiss_entry is None or self._faiss_entry.d != int(vector.size):
            self._faiss_entry = faiss.IndexFlatIP(int(vector.size))
            self.rebuild_faiss_index()
            return
        self._faiss_entry.add(vector.reshape(1, -1).astype("float32"))

    def save_index(self) -> None:
        try:
            import faiss
        except Exception:
            return
        if self._faiss_entry is not None:
            faiss.write_index(self._faiss_entry, str(self.entry_index_path))

    def search_entries(self, query_embedding: Iterable[float], top_k: int) -> List[Tuple[str, float, int, str, str]]:
        vector = normalize_vector(query_embedding)
        if vector.size == 0 or top_k <= 0 or self._faiss_entry is None:
            return []
        scores, row_ids = self._faiss_entry.search(
            vector.reshape(1, -1).astype("float32"),
            min(top_k, max(1, self._faiss_entry.ntotal)),
        )
        hits: List[Tuple[str, float, int, str, str]] = []
        for row_id, score in zip(row_ids[0], scores[0]):
            if row_id < 0:
                continue
            row = self.conn.execute(
                """
                select memory_id, embedding_mode, embedding_image_id
                from entry_embeddings where faiss_row_id = ?
                """,
                (int(row_id),),
            ).fetchone()
            if row:
                hits.append(
                    (
                        str(row["memory_id"]),
                        float(score),
                        int(row_id),
                        str(row["embedding_mode"] or ""),
                        str(row["embedding_image_id"] or ""),
                    )
                )
        return hits

    def get_memory(self, memory_id: str) -> Optional[GmeMemoryRecord]:
        row = self.conn.execute("select * from memories where memory_id = ?", (memory_id,)).fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    def _row_to_memory(self, row: sqlite3.Row) -> GmeMemoryRecord:
        image_rows = self.conn.execute(
            "select * from images where memory_id = ? order by image_row_id",
            (row["memory_id"],),
        ).fetchall()
        images = [
            GmeImagePointer(
                image_id=str(image["image_id"] or ""),
                path=str(image["path"] or ""),
                caption=str(image["caption"] or ""),
                metadata=json.loads(image["metadata_json"] or "{}"),
                image_row_id=int(image["image_row_id"]),
            )
            for image in image_rows
        ]
        return GmeMemoryRecord(
            memory_id=str(row["memory_id"]),
            text=str(row["text"] or ""),
            session_id=str(row["session_id"] or ""),
            turn_id=str(row["turn_id"] or ""),
            date=str(row["date"] or ""),
            images=images,
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def count_memories(self) -> int:
        row = self.conn.execute("select count(*) as n from memories").fetchone()
        return int(row["n"])

    def count_images(self) -> int:
        row = self.conn.execute("select count(*) as n from images").fetchone()
        return int(row["n"])

    def count_entry_embeddings(self) -> int:
        row = self.conn.execute("select count(*) as n from entry_embeddings").fetchone()
        return int(row["n"])

    def stats(self) -> Dict[str, Any]:
        return {
            "memories": self.count_memories(),
            "images": self.count_images(),
            "entry_embeddings": self.count_entry_embeddings(),
            "entry_dim": self.get_meta("entry_dim"),
        }
