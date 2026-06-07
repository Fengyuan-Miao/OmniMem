from dual_encoder_memory import ImagePointer, UnifiedMemoryRecord
from topic_memory import TopicBuilder, TopicMemoryStore, TopicScopedRetriever
from topic_memory.context import build_ordered_topic_evidence_context
from topic_memory.topic_builder import MAX_TOPIC_SUMMARY_CHARS


class FakeTextEncoder:
    def encode(self, text):
        value = str(text or "").lower()
        if "cat" in value:
            return [1.0, 0.0, 0.0]
        if "robot" in value:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class FakeVisionEncoder:
    def encode_text(self, text):
        value = str(text or "").lower()
        if "cat" in value:
            return [1.0, 0.0]
        if "robot" in value:
            return [0.0, 1.0]
        return [0.5, 0.5]

    def encode_image(self, image):
        return self.encode_text(image)


class FakeTopicLLM:
    def __init__(self, response):
        self.response = response
        self.messages = []

    def complete_json(self, messages):
        self.messages.append(messages)
        return dict(self.response)


def test_topic_assignment_uses_user_query_only(tmp_path):
    store = TopicMemoryStore(tmp_path)
    record = UnifiedMemoryRecord(
        memory_id="D1:D1:1",
        text="User: cats\nAssistant: secret assistant content",
        session_id="D1",
        turn_id="D1:1",
        date="2024-01-01",
    )
    store.add_memory(record, text_embedding=[1.0, 0.0, 0.0])
    llm = FakeTopicLLM(
        {
            "action": "new",
            "topic_id": "",
            "topic_summary": "The user discusses cats.",
        }
    )
    builder = TopicBuilder(store, FakeTextEncoder(), llm, match_top_k=12)
    assignment = builder.assign_turn("cats", record)

    assert assignment.topic_id == "T001"
    system = llm.messages[0][0]["content"]
    user = llm.messages[0][1]["content"]
    assert "cats" in user
    assert "secret assistant content" not in system
    assert "DO NOT overwrite" in system
    assert store.list_topics()[0].summary == "The user discusses cats."
    store.close()


def test_topic_builder_fallback_new_topic_on_invalid_json(tmp_path):
    store = TopicMemoryStore(tmp_path)
    record = UnifiedMemoryRecord(
        memory_id="D1:D1:1",
        text="User: cats",
        session_id="D1",
        turn_id="D1:1",
        date="2024-01-01",
    )
    store.add_memory(record, text_embedding=[1.0, 0.0, 0.0])
    builder = TopicBuilder(store, FakeTextEncoder(), FakeTopicLLM({}), match_top_k=12)
    assignment = builder.assign_turn("cats and kittens", record)

    assert assignment.action == "new"
    assert assignment.error
    assert assignment.topic_id == "T001"
    store.close()


def test_topic_merge_preserves_existing_summary(tmp_path):
    store = TopicMemoryStore(tmp_path)
    existing = store.create_topic(
        "The user is planning cat adoption, including shelter visits and vet preparation.",
        [1.0, 0.0, 0.0],
        sequence=1,
        date="2024-01-01",
    )
    record = UnifiedMemoryRecord(
        memory_id="D1:D1:2",
        text="User: cat food planning",
        session_id="D1",
        turn_id="D1:2",
        date="2024-01-02",
    )
    store.add_memory(record, text_embedding=[1.0, 0.0, 0.0])
    llm = FakeTopicLLM(
        {
            "action": "merge",
            "topic_id": existing.topic_id,
            "topic_summary": "The user is comparing cat food options.",
        }
    )
    builder = TopicBuilder(store, FakeTextEncoder(), llm, match_top_k=12)
    assignment = builder.assign_turn("cat food planning", record)

    summary = store.get_topic(existing.topic_id).summary
    assert assignment.action == "merge"
    assert "adoption" in summary
    assert "vet preparation" in summary
    assert "cat food" in summary
    store.close()


def test_topic_summary_limit_uses_word_boundary():
    summary = TopicBuilder._limit_summary(
        "The user is discussing a very long topic about animal behavior, "
        "vocal communication, dog training, seminars, applications, "
        "and many other details that should not end with a broken word."
    )
    assert len(summary) <= MAX_TOPIC_SUMMARY_CHARS + 1
    assert summary.endswith(".")
    assert not summary.endswith(" .")
    assert summary.split()[-1] != "brok."


def test_topic_merge_uses_concise_update_when_overlap_is_clear():
    merged = TopicBuilder._merge_summary(
        "The user explores Life Sciences, dog communication, Maltese adoption plans, diet, training, and Lumi's park.",
        "The user explores Life Sciences, dog communication, Maltese adoption plans, Lumi's park experiences, and photos with Amy's Cairn Terrier.",
    )
    assert "also includes" not in merged
    assert "photos with Amy" in merged


def test_topic_scoped_retrieval_and_ordered_context(tmp_path):
    store = TopicMemoryStore(tmp_path)
    cat_image = tmp_path / "cat.jpg"
    robot_image = tmp_path / "robot.jpg"
    cat_image.write_bytes(b"cat")
    robot_image.write_bytes(b"robot")
    cat = UnifiedMemoryRecord(
        memory_id="D1:D1:2",
        text="User discussed a cat photo.",
        session_id="D1",
        turn_id="D1:2",
        date="2024-01-01",
        images=[ImagePointer("D1:IMG_001", str(cat_image), "cat")],
    )
    robot = UnifiedMemoryRecord(
        memory_id="D1:D1:1",
        text="User discussed a robot arm.",
        session_id="D1",
        turn_id="D1:1",
        date="2024-01-01",
        images=[ImagePointer("D1:IMG_002", str(robot_image), "robot")],
    )
    cat = store.add_memory(cat, text_embedding=[1.0, 0.0, 0.0])
    robot = store.add_memory(robot, text_embedding=[0.0, 1.0, 0.0])
    store.add_memory(cat, image_embeddings=[(cat.images[0], [1.0, 0.0])])
    store.add_memory(robot, image_embeddings=[(robot.images[0], [0.0, 1.0])])
    cat_topic = store.create_topic("cat topic", [1.0, 0.0, 0.0], 1, "2024-01-01")
    robot_topic = store.create_topic("robot topic", [0.0, 1.0, 0.0], 2, "2024-01-01")
    store.add_turn_to_topic(cat_topic.topic_id, cat, 1)
    store.add_turn_to_topic(robot_topic.topic_id, robot, 2)

    retriever = TopicScopedRetriever(store, FakeTextEncoder(), FakeVisionEncoder())
    text_only = retriever.retrieve(
        "cat",
        topic_ids=[cat_topic.topic_id],
        modalities=["text"],
        top_k_text=5,
        top_k_image=5,
    )
    assert text_only.ranked_memories[0].memory.memory_id == "D1:D1:2"
    assert text_only.image_hits == []

    image_only = retriever.retrieve(
        "cat",
        topic_ids=[cat_topic.topic_id],
        modalities=["image"],
        top_k_text=5,
        top_k_image=5,
    )
    assert image_only.text_hits == []
    assert image_only.image_hits[0].image_id == "D1:IMG_001"
    assert all(item.memory.memory_id != "D1:D1:1" for item in image_only.ranked_memories)

    context = build_ordered_topic_evidence_context(
        store,
        [cat_topic],
        image_only.ranked_memories,
        memory_limit=5,
    )
    assert "Turn D1:2" in context
    assert "D1:IMG_001" in context
    assert str(cat_image) not in context
    store.close()
