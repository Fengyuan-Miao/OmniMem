from pathlib import Path

from dual_encoder_memory import (
    DualEncoderMemoryStore,
    DualEncoderRetriever,
    EvidenceOrganizer,
    ImagePointer,
    UnifiedMemoryRecord,
)


class FakeTextEncoder:
    def encode(self, text):
        if "cat" in text.lower():
            return [1.0, 0.0, 0.0]
        if "robot" in text.lower():
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class FakeVisionEncoder:
    def encode_text(self, text):
        if "cat" in text.lower():
            return [1.0, 0.0]
        if "robot" in text.lower():
            return [0.0, 1.0]
        return [0.5, 0.5]

    def encode_image(self, image):
        value = str(image).lower()
        if "cat" in value:
            return [1.0, 0.0]
        if "robot" in value:
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_sqlite_faiss_reopen_and_rebuild(tmp_path):
    store = DualEncoderMemoryStore(tmp_path)
    record = UnifiedMemoryRecord(
        memory_id="D1:D1:1",
        text="User discussed a cat photo.",
        session_id="D1",
        turn_id="D1:1",
        date="2024-07-21",
        images=[
            ImagePointer(
                image_id="D1:IMG_001",
                path=str(tmp_path / "cat.jpg"),
                caption="A cat on a sofa",
            )
        ],
    )
    stored = store.add_memory(record, text_embedding=[1.0, 0.0, 0.0])
    store.add_memory(stored, image_embeddings=[(stored.images[0], [1.0, 0.0])])
    store.close()

    reopened = DualEncoderMemoryStore(tmp_path)
    assert reopened.count_memories() == 1
    assert reopened.count_images() == 1
    assert reopened.search_text([1.0, 0.0, 0.0], 1)[0][0] == "D1:D1:1"
    assert reopened.search_image([1.0, 0.0], 1)[0][2] == "D1:IMG_001"

    Path(tmp_path / "faiss_text.index").unlink()
    Path(tmp_path / "faiss_image.index").unlink()
    reopened.close()
    rebuilt = DualEncoderMemoryStore(tmp_path)
    assert rebuilt.search_text([1.0, 0.0, 0.0], 1)[0][0] == "D1:D1:1"
    assert rebuilt.search_image([1.0, 0.0], 1)[0][2] == "D1:IMG_001"
    rebuilt.close()


def test_dual_routes_map_back_to_same_memory(tmp_path):
    store = DualEncoderMemoryStore(tmp_path)
    cat = UnifiedMemoryRecord(
        memory_id="D1:D1:1",
        text="A memory about Lena's cat.",
        session_id="D1",
        turn_id="D1:1",
        date="2024-07-21",
        images=[ImagePointer("D1:IMG_001", str(tmp_path / "cat.jpg"), "cat")],
    )
    robot = UnifiedMemoryRecord(
        memory_id="D1:D1:2",
        text="A memory about a robot arm.",
        session_id="D1",
        turn_id="D1:2",
        date="2024-07-22",
        images=[ImagePointer("D1:IMG_002", str(tmp_path / "robot.jpg"), "robot")],
    )
    cat = store.add_memory(cat, text_embedding=[1.0, 0.0, 0.0])
    robot = store.add_memory(robot, text_embedding=[0.0, 1.0, 0.0])
    store.add_memory(cat, image_embeddings=[(cat.images[0], [1.0, 0.0])])
    store.add_memory(robot, image_embeddings=[(robot.images[0], [0.0, 1.0])])

    retriever = DualEncoderRetriever(store, FakeTextEncoder(), FakeVisionEncoder())
    result = retriever.retrieve("Which image is about the cat?", top_k_text=2, top_k_image=2)
    assert result.ranked_memories[0].memory.memory_id == "D1:D1:1"
    assert result.ranked_memories[0].matched_image_ids() == ["D1:IMG_001"]
    assert {hit.route for hit in result.ranked_memories[0].route_hits} == {
        "text",
        "image_by_text",
        "bm25",
    }
    assert result.to_dict()["bm25_hits"]
    assert result.to_dict()["ranked_memories"][0]["route_hits"]
    store.close()


def test_bm25_route_can_recall_text_memory_without_dense_routes(tmp_path):
    store = DualEncoderMemoryStore(tmp_path)
    first = UnifiedMemoryRecord(
        memory_id="D1:D1:1",
        text="Lena discussed a rare papillon dog at the park.",
        session_id="D1",
        turn_id="D1:1",
        date="2024-07-21",
    )
    second = UnifiedMemoryRecord(
        memory_id="D1:D1:2",
        text="A memory about a robot arm.",
        session_id="D1",
        turn_id="D1:2",
        date="2024-07-22",
    )
    store.add_memory(first, text_embedding=[0.0, 1.0, 0.0])
    store.add_memory(second, text_embedding=[1.0, 0.0, 0.0])

    retriever = DualEncoderRetriever(store, FakeTextEncoder(), FakeVisionEncoder())
    result = retriever.retrieve(
        "Which turn mentioned papillon?",
        top_k_text=0,
        top_k_image=0,
        top_k_bm25=2,
        rerank_top_k=1,
    )

    assert result.bm25_hits[0].memory_id == "D1:D1:1"
    assert result.ranked_memories[0].memory.memory_id == "D1:D1:1"
    assert result.ranked_memories[0].route_hits[0].route == "bm25"
    store.close()


def test_evidence_organizer_groups_nearby_turns_without_paths(tmp_path):
    store = DualEncoderMemoryStore(tmp_path)
    first = UnifiedMemoryRecord(
        memory_id="D1:D1:1",
        text="User adopted a cat named Miso.",
        session_id="D1",
        turn_id="D1:1",
        date="2024-07-21",
        images=[ImagePointer("D1:IMG_001", str(tmp_path / "cat.jpg"), "cat on a sofa")],
    )
    second = UnifiedMemoryRecord(
        memory_id="D1:D1:2",
        text="Assistant discussed cat care.",
        session_id="D1",
        turn_id="D1:2",
        date="2024-07-21",
    )
    first = store.add_memory(first, text_embedding=[1.0, 0.0, 0.0])
    second = store.add_memory(second, text_embedding=[1.0, 0.0, 0.0])
    store.add_memory(first, image_embeddings=[(first.images[0], [1.0, 0.0])])

    retriever = DualEncoderRetriever(store, FakeTextEncoder(), FakeVisionEncoder())
    result = retriever.retrieve("cat", top_k_text=2, top_k_image=1, rerank_top_k=2)
    evidence = EvidenceOrganizer(neighbor_turn_window=1).organize(result)

    assert len(evidence.atoms) == 2
    assert len(evidence.groups) == 1
    prompt_context = evidence.to_prompt_context(group_limit=3)
    assert "D1:IMG_001" in prompt_context
    assert str(tmp_path / "cat.jpg") not in prompt_context
    store.close()
