"""SQLite-backed unified memory store with FAISS vector indexes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .models import ImagePointer, UnifiedMemoryRecord


def normalize_vector(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vector), dtype="float32")
    if arr.ndim != 1 or arr.size == 0:
        return np.asarray([], dtype="float32")
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return arr
    return arr / norm


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="float32").tobytes()


def blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    if not blob:
        return np.asarray([], dtype="float32")
    arr = np.frombuffer(blob, dtype="float32")
    if dim and arr.size != dim:
        return np.asarray([], dtype="float32")
    return arr.astype("float32", copy=True)


class DualEncoderMemoryStore:
    """Persist memories in SQLite and accelerate vector search with FAISS."""

    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.storage_dir / "dual_encoder_memory.sqlite"
        self.text_index_path = self.storage_dir / "faiss_text.index"
        self.image_index_path = self.storage_dir / "faiss_image.index"
        self.conn = sqlite3.connect(str(self.sqlite_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._faiss_text = None
        self._faiss_image = None
        self._load_or_rebuild_indexes()

    def close(self) -> None:
        self.save_indexes()
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

            create table if not exists text_embeddings (
                memory_id text primary key,
                dim integer not null,
                vector_blob blob not null,
                faiss_row_id integer not null unique,
                foreign key(memory_id) references memories(memory_id)
            );

            create table if not exists image_embeddings (
                image_row_id integer primary key,
                memory_id text not null,
                dim integer not null,
                vector_blob blob not null,
                faiss_row_id integer not null unique,
                foreign key(image_row_id) references images(image_row_id),
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
        row = self.conn.execute(
            "select value from index_meta where key = ?",
            (key,),
        ).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def add_memory(
        self,
        record: UnifiedMemoryRecord,
        text_embedding: Optional[Iterable[float]] = None,
        image_embeddings: Optional[List[Tuple[ImagePointer, Iterable[float]]]] = None,
    ) -> UnifiedMemoryRecord:
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
        stored_images: List[ImagePointer] = []
        for image in record.images:
            row_id = image.image_row_id
            if row_id is None:
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
                row_id = int(cursor.lastrowid)
            else:
                self.conn.execute(
                    """
                    insert or replace into images
                    (image_row_id, memory_id, image_id, path, caption, metadata_json)
                    values (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_id,
                        record.memory_id,
                        image.image_id,
                        image.path,
                        image.caption,
                        json.dumps(image.metadata, ensure_ascii=False),
                    ),
                )
            image.image_row_id = row_id
            stored_images.append(image)
        record.images = stored_images

        if text_embedding is not None:
            self.add_text_embedding(record.memory_id, text_embedding)
        for image, embedding in image_embeddings or []:
            if image.image_row_id is None:
                continue
            self.add_image_embedding(image.image_row_id, record.memory_id, embedding)

        self.conn.commit()
        return record

    def add_text_embedding(self, memory_id: str, embedding: Iterable[float]) -> None:
        vector = normalize_vector(embedding)
        if vector.size == 0:
            return
        dim = int(vector.size)
        faiss_row_id = self._next_faiss_row_id("text_embeddings")
        self.conn.execute(
            """
            insert or replace into text_embeddings(memory_id, dim, vector_blob, faiss_row_id)
            values (?, ?, ?, ?)
            """,
            (memory_id, dim, vector_to_blob(vector), faiss_row_id),
        )
        self._add_to_faiss("text", vector)
        self.set_meta("text_dim", dim)

    def add_image_embedding(
        self,
        image_row_id: int,
        memory_id: str,
        embedding: Iterable[float],
    ) -> None:
        vector = normalize_vector(embedding)
        if vector.size == 0:
            return
        dim = int(vector.size)
        faiss_row_id = self._next_faiss_row_id("image_embeddings")
        self.conn.execute(
            """
            insert or replace into image_embeddings
            (image_row_id, memory_id, dim, vector_blob, faiss_row_id)
            values (?, ?, ?, ?, ?)
            """,
            (image_row_id, memory_id, dim, vector_to_blob(vector), faiss_row_id),
        )
        self._add_to_faiss("image", vector)
        self.set_meta("image_dim", dim)

    def _next_faiss_row_id(self, table: str) -> int:
        row = self.conn.execute(
            f"select coalesce(max(faiss_row_id), -1) + 1 as next_id from {table}"
        ).fetchone()
        return int(row["next_id"])

    def _load_or_rebuild_indexes(self) -> None:
        if not self._try_load_faiss():
            self.rebuild_faiss_indexes()

    def _try_load_faiss(self) -> bool:
        try:
            import faiss
        except Exception:
            return False
        if not self.text_index_path.exists() or not self.image_index_path.exists():
            return False
        try:
            text_index = faiss.read_index(str(self.text_index_path))
            image_index = faiss.read_index(str(self.image_index_path))
        except Exception:
            return False
        text_count = self.count_text_embeddings()
        image_count = self.count_image_embeddings()
        if text_index.ntotal != text_count or image_index.ntotal != image_count:
            return False
        self._faiss_text = text_index
        self._faiss_image = image_index
        return True

    def rebuild_faiss_indexes(self) -> None:
        self._faiss_text = self._build_faiss_from_table("text_embeddings")
        self._faiss_image = self._build_faiss_from_table("image_embeddings")
        self.save_indexes()

    def _build_faiss_from_table(self, table: str):
        import faiss

        rows = self.conn.execute(
            f"select dim, vector_blob from {table} order by faiss_row_id asc"
        ).fetchall()
        dim = int(rows[0]["dim"]) if rows else 1
        index = faiss.IndexFlatIP(dim)
        if rows:
            vectors = np.vstack(
                [blob_to_vector(row["vector_blob"], int(row["dim"])) for row in rows]
            ).astype("float32")
            if vectors.size:
                index.add(vectors)
        return index

    def _add_to_faiss(self, route: str, vector: np.ndarray) -> None:
        import faiss

        target = "_faiss_text" if route == "text" else "_faiss_image"
        index = getattr(self, target)
        if index is None or index.d != int(vector.size):
            index = faiss.IndexFlatIP(int(vector.size))
            setattr(self, target, index)
            self.rebuild_faiss_indexes()
            return
        index.add(vector.reshape(1, -1).astype("float32"))

    def save_indexes(self) -> None:
        try:
            import faiss
        except Exception:
            return
        if self._faiss_text is not None:
            faiss.write_index(self._faiss_text, str(self.text_index_path))
        if self._faiss_image is not None:
            faiss.write_index(self._faiss_image, str(self.image_index_path))

    def search_text(self, query_embedding: Iterable[float], top_k: int) -> List[Tuple[str, float, int]]:
        vector = normalize_vector(query_embedding)
        if vector.size == 0 or top_k <= 0 or self._faiss_text is None:
            return []
        scores, row_ids = self._faiss_text.search(
            vector.reshape(1, -1).astype("float32"),
            min(top_k, max(1, self._faiss_text.ntotal)),
        )
        hits: List[Tuple[str, float, int]] = []
        for row_id, score in zip(row_ids[0], scores[0]):
            if row_id < 0:
                continue
            row = self.conn.execute(
                "select memory_id from text_embeddings where faiss_row_id = ?",
                (int(row_id),),
            ).fetchone()
            if row:
                hits.append((str(row["memory_id"]), float(score), int(row_id)))
        return hits

    def search_image(
        self,
        query_embedding: Iterable[float],
        top_k: int,
    ) -> List[Tuple[int, str, str, float, int]]:
        vector = normalize_vector(query_embedding)
        if vector.size == 0 or top_k <= 0 or self._faiss_image is None:
            return []
        scores, row_ids = self._faiss_image.search(
            vector.reshape(1, -1).astype("float32"),
            min(top_k, max(1, self._faiss_image.ntotal)),
        )
        hits: List[Tuple[int, str, str, float, int]] = []
        for row_id, score in zip(row_ids[0], scores[0]):
            if row_id < 0:
                continue
            row = self.conn.execute(
                """
                select e.image_row_id, e.memory_id, i.image_id
                from image_embeddings e
                join images i on i.image_row_id = e.image_row_id
                where e.faiss_row_id = ?
                """,
                (int(row_id),),
            ).fetchone()
            if row:
                hits.append(
                    (
                        int(row["image_row_id"]),
                        str(row["memory_id"]),
                        str(row["image_id"] or ""),
                        float(score),
                        int(row_id),
                    )
                )
        return hits

    def get_memory(self, memory_id: str) -> Optional[UnifiedMemoryRecord]:
        row = self.conn.execute(
            "select * from memories where memory_id = ?",
            (memory_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    def iter_memories(self) -> List[UnifiedMemoryRecord]:
        rows = self.conn.execute("select * from memories order by date, session_id, turn_id").fetchall()
        return [self._row_to_memory(row) for row in rows]

    def _row_to_memory(self, row: sqlite3.Row) -> UnifiedMemoryRecord:
        image_rows = self.conn.execute(
            "select * from images where memory_id = ? order by image_row_id",
            (row["memory_id"],),
        ).fetchall()
        images = [
            ImagePointer(
                image_id=str(image["image_id"] or ""),
                path=str(image["path"] or ""),
                caption=str(image["caption"] or ""),
                metadata=json.loads(image["metadata_json"] or "{}"),
                image_row_id=int(image["image_row_id"]),
            )
            for image in image_rows
        ]
        return UnifiedMemoryRecord(
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

    def count_text_embeddings(self) -> int:
        row = self.conn.execute("select count(*) as n from text_embeddings").fetchone()
        return int(row["n"])

    def count_image_embeddings(self) -> int:
        row = self.conn.execute("select count(*) as n from image_embeddings").fetchone()
        return int(row["n"])

    def stats(self) -> Dict[str, Any]:
        return {
            "memories": self.count_memories(),
            "images": self.count_images(),
            "text_embeddings": self.count_text_embeddings(),
            "image_embeddings": self.count_image_embeddings(),
            "text_dim": self.get_meta("text_dim"),
            "image_dim": self.get_meta("image_dim"),
        }
