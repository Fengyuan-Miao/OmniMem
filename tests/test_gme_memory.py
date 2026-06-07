from pathlib import Path

from gme_memory import (
    GmeImagePointer,
    GmeMemoryRecord,
    GmeMemoryRetriever,
    GmeMemoryStore,
    select_entry_embedding_image,
)


class FakeGmeEncoder:
    def __init__(self):
        self.query_calls = []

    def encode_query(self, query, question_image=None):
        self.query_calls.append((query, question_image))
        text = str(query).lower()
        if question_image:
            return [0.0, 1.0, 0.0], "query_image_text_pair"
        if "cat" in text:
            return [1.0, 0.0, 0.0], "query_text"
        return [0.0, 1.0, 0.0], "query_text"


def test_gme_store_reopens_and_rebuilds_single_entry_index(tmp_path):
    store = GmeMemoryStore(tmp_path)
    record = GmeMemoryRecord(
        memory_id="D1:D1:1",
        text="User discussed a cat photo.",
        session_id="D1",
        turn_id="D1:1",
        date="2024-07-21",
        images=[
            GmeImagePointer(
                image_id="D1:IMG_001",
                path=str(tmp_path / "cat.jpg"),
                caption="A cat on a sofa",
            )
        ],
    )
    store.add_memory(
        record,
        entry_embedding=[1.0, 0.0, 0.0],
        embedding_mode="image_text_pair",
        embedding_image_id="D1:IMG_001",
    )
    store.close()

    reopened = GmeMemoryStore(tmp_path)
    hits = reopened.search_entries([1.0, 0.0, 0.0], top_k=1)
    assert hits[0][0] == "D1:D1:1"
    assert hits[0][3] == "image_text_pair"
    memory = reopened.get_memory("D1:D1:1")
    assert memory is not None
    assert memory.text == "User discussed a cat photo."
    assert memory.images[0].image_id == "D1:IMG_001"

    Path(tmp_path / "faiss_entry.index").unlink()
    reopened.close()
    rebuilt = GmeMemoryStore(tmp_path)
    assert rebuilt.search_entries([1.0, 0.0, 0.0], top_k=1)[0][0] == "D1:D1:1"
    rebuilt.close()


def test_multi_image_entry_selects_first_valid_image_but_returns_all_images(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    record = GmeMemoryRecord(
        memory_id="D1:D1:2",
        text="Two images were shared.",
        images=[
            GmeImagePointer("D1:IMG_001", str(first), "first image"),
            GmeImagePointer("D1:IMG_002", str(second), "second image"),
        ],
    )
    selected_path, selected_id = select_entry_embedding_image(record)
    assert selected_path == str(first)
    assert selected_id == "D1:IMG_001"

    store = GmeMemoryStore(tmp_path / "store")
    store.add_memory(
        record,
        entry_embedding=[1.0, 0.0, 0.0],
        embedding_mode="image_text_pair",
        embedding_image_id=selected_id,
    )
    result = GmeMemoryRetriever(store, FakeGmeEncoder()).retrieve("cat", top_k=1)
    assert result.entries[0].matched_image_ids == ["D1:IMG_001"]
    assert result.entries[0].retrieved_image_ids() == ["D1:IMG_001", "D1:IMG_002"]
    store.close()


def test_query_with_image_uses_fused_query_path(tmp_path):
    question_image = tmp_path / "question.jpg"
    question_image.write_bytes(b"question")
    store = GmeMemoryStore(tmp_path / "store")
    store.add_memory(
        GmeMemoryRecord(
            memory_id="D1:D1:1",
            text="Robot arm memory.",
            images=[GmeImagePointer("D1:IMG_010", str(question_image), "robot")],
        ),
        entry_embedding=[0.0, 1.0, 0.0],
        embedding_mode="image_text_pair",
        embedding_image_id="D1:IMG_010",
    )
    encoder = FakeGmeEncoder()
    result = GmeMemoryRetriever(store, encoder).retrieve(
        "What is shown in the attached image?",
        question_image=str(question_image),
        top_k=1,
    )
    assert encoder.query_calls == [("What is shown in the attached image?", str(question_image))]
    assert result.query_embedding_mode == "query_image_text_pair"
    assert result.entries[0].memory.memory_id == "D1:D1:1"
    store.close()
