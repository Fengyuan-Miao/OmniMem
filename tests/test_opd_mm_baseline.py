import json
from pathlib import Path

import pytest

from opd_mm_baseline.clients import (
    ChatAnswerModel,
    ChatHindsightTeacher,
    OpenAICompatibleClient,
    extract_json_array,
)
from opd_mm_baseline.executor import ToolExecutor
from opd_mm_baseline.memgallery import build_scenario_store, scenario_samples
from opd_mm_baseline.memgallery_pipeline import select_sft_example
from opd_mm_baseline.memeye import (
    build_scenario_store as build_memeye_scenario_store,
    normalize_memeye_dialog_image_paths,
    normalize_memeye_image_path,
    scenario_samples as memeye_scenario_samples,
)
from opd_mm_baseline.models import (
    EvidenceItem,
    MemoryRecord,
    OPDSample,
    PolicyOutput,
    ToolAction,
)
from opd_mm_baseline.retrieval import (
    HiddenMemoryStore,
    HybridRetriever,
    TurnAwareHybridRetriever,
)
from opd_mm_baseline.schema import (
    RETRIEVAL_METHODS,
    TrajectoryValidationError,
    TrajectoryValidator,
)
from opd_mm_baseline.training import OnPolicyDistiller


class FakeDenseEncoder:
    def encode(self, text):
        value = str(text).lower()
        if "receipt" in value or "收据" in value:
            return [1.0, 0.0]
        if "research" in value or "研究" in value:
            return [0.0, 1.0]
        return [0.5, 0.5]


class FakeVisionEncoder:
    @staticmethod
    def _value(image):
        return [1.0, 0.0] if "support" in str(image) else [0.0, 1.0]

    def encode_image(self, image):
        return self._value(image)

    def encode_images(self, images):
        return [self._value(image) for image in images]

    def encode_text(self, text):
        return [0.0, 1.0]


class FakePolicy:
    def __init__(self, actions):
        self.actions = actions

    def generate_trace(self, query):
        return PolicyOutput(actions=list(self.actions))


class FakeTeacher:
    def correct(
        self,
        query,
        gold_answer,
        student_policy,
        student_answer,
        correct,
        execution=None,
        privileged_context=None,
    ):
        return PolicyOutput(actions=list(student_policy.actions))


class FeedbackTeacher:
    privilege_mode = "oracle-feedback"

    def __init__(self, revised_actions):
        self.revised_actions = revised_actions
        self.feedback = []

    def correct(
        self,
        query,
        gold_answer,
        student_policy,
        student_answer,
        correct,
        execution=None,
        privileged_context=None,
    ):
        return PolicyOutput(actions=list(student_policy.actions))

    def revise(
        self,
        query,
        gold_answer,
        student_policy,
        student_answer,
        correct,
        previous_policy,
        replay_feedback,
        attempt_index,
        execution=None,
        privileged_context=None,
    ):
        self.feedback.append(replay_feedback)
        return PolicyOutput(actions=list(self.revised_actions))


class FakeAnswer:
    def answer(self, query, evidence, question_image=None):
        return "ICLR receipt"


class FakeJudge:
    def evaluate(self, query, prediction, gold_answer):
        return prediction == gold_answer, float(prediction == gold_answer), "fake"


class CapturingClient:
    def __init__(self, response):
        self.response = response
        self.messages = []

    def complete(self, messages, max_tokens=512, temperature=0.0):
        self.messages.append(messages)
        return self.response


class CapturingRawInspector:
    def __init__(self):
        self.calls = []

    def inspect(
        self,
        image_path,
        query,
        question_image=None,
        text_context=None,
    ):
        self.calls.append(
            {
                "image_path": image_path,
                "query": query,
                "question_image": question_image,
                "text_context": text_context,
            }
        )
        return "same visual type; text label says Cairn Terrier"


def test_openai_client_filters_local_extensions_for_api(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode("utf-8")

    class FakeOpener:
        def open(self, request, timeout=0):
            requests.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *_args, **_kwargs: FakeOpener(),
    )
    api_client = OpenAICompatibleClient(
        "https://api.example.com/v1",
        "model",
        "key",
        service_mode="api",
    )
    api_client.complete(
        [{"role": "user", "content": "hello"}],
        extra_body={
            "thinking_token_budget": 128,
            "chat_template_kwargs": {"enable_thinking": False},
            "foo": "bar",
        },
        prefill_assistant="</think>\n\n",
    )
    local_client = OpenAICompatibleClient(
        "http://127.0.0.1:11438/v1",
        "model",
        "key",
        service_mode="local",
    )
    local_client.complete(
        [{"role": "user", "content": "hello"}],
        extra_body={
            "thinking_token_budget": 128,
            "chat_template_kwargs": {"enable_thinking": False},
            "foo": "bar",
        },
        prefill_assistant="</think>\n\n",
    )

    assert requests[0]["foo"] == "bar"
    assert "thinking_token_budget" not in requests[0]
    assert "chat_template_kwargs" not in requests[0]
    assert "add_generation_prompt" not in requests[0]
    assert requests[0]["messages"][-1]["role"] == "user"

    assert requests[1]["thinking_token_budget"] == 128
    assert requests[1]["chat_template_kwargs"]["enable_thinking"] is False
    assert requests[1]["add_generation_prompt"] is False
    assert requests[1]["continue_final_message"] is True
    assert requests[1]["messages"][-1]["role"] == "assistant"


def test_validator_rejects_memory_ids_and_custom_search_queries():
    validator = TrajectoryValidator()
    assert "vision" in RETRIEVAL_METHODS
    assert "bm25|dense|vision|hybrid" in validator.schema_text()
    actions = validator.validate(
        [
            {"tool": "RETRIEVE", "method": "vision", "top_k": 3},
            {"tool": "STOP"},
        ]
    )
    assert actions[0].arguments["method"] == "vision"
    with pytest.raises(TrajectoryValidationError, match="forbidden"):
        validator.validate(
            [
                {
                    "tool": "RETRIEVE",
                    "method": "hybrid",
                    "top_k": 5,
                    "query": "hidden answer",
                },
                {"tool": "STOP"},
            ]
        )
    with pytest.raises(TrajectoryValidationError, match="memory IDs"):
        validator.validate(
            [
                {
                    "tool": "FILTER",
                    "field": "status",
                    "op": "eq",
                    "value": "m_0012",
                },
                {"tool": "STOP"},
            ]
        )


def test_validator_hides_unavailable_raw_inspection_tool():
    validator = TrajectoryValidator(allow_inspect_raw=False)
    assert "INSPECT_RAW" not in validator.schema_text()
    with pytest.raises(TrajectoryValidationError, match="unavailable"):
        validator.validate(
            [
                {
                    "tool": "INSPECT_RAW",
                    "target": "current_pool",
                    "instruction": "answer_query_related_visual_details",
                },
                {"tool": "STOP"},
            ]
        )


def test_policy_parser_repairs_common_stop_syntax():
    values = extract_json_array(
        '[{"tool":"READ","fields":["summary"]},{"tool":"STOP": {}}]'
    )
    assert values[-1] == {"tool": "STOP"}


def test_executor_composes_generic_tools_to_read_latest_user_image():
    records = [
        MemoryRecord(
            memory_id="internal-old",
            turn_id="D1:2",
            timestamp="2026-06-12T10:20:00Z",
            author="user",
            modality="image",
            source_type="uploaded_image",
            summary="old image",
        ),
        MemoryRecord(
            memory_id="internal-new",
            turn_id="D1:10",
            timestamp="2026-06-12T10:35:00Z",
            author="user",
            modality="image",
            source_type="uploaded_image",
            summary="ICLR receipt",
            content="registration total $450",
        ),
        MemoryRecord(
            memory_id="internal-text",
            turn_id="D1:11",
            timestamp="2026-06-12T10:36:00Z",
            author="assistant",
            modality="text",
            source_type="conversation",
            content="unrelated",
        ),
    ]
    trace = [
        ToolAction("FILTER", {"field": "modality", "op": "eq", "value": "image"}),
        ToolAction("FILTER", {"field": "author", "op": "eq", "value": "user"}),
        ToolAction("SORT", {"field": "timestamp", "order": "desc"}),
        ToolAction("TOPK", {"k": 1}),
        ToolAction("READ", {"fields": ["summary", "content", "timestamp"]}),
        ToolAction("STOP"),
    ]
    result = ToolExecutor().run(trace, "What was my last image?", HiddenMemoryStore(records))
    assert result.stopped is True
    assert result.final_memory_ids == ["internal-new"]
    assert result.evidence[0].fields["summary"] == "ICLR receipt"
    assert result.evidence[0].fields["content"] == "registration total $450"


def test_retrieve_operates_only_on_filtered_current_pool():
    records = [
        MemoryRecord(
            "text-receipt",
            "D1:1",
            "2026-01-01T00:00:00Z",
            "user",
            "text",
            "conversation",
            content="receipt discussion",
        ),
        MemoryRecord(
            "image-receipt",
            "D1:2",
            "2026-01-01T00:00:10Z",
            "user",
            "image",
            "uploaded_image",
            summary="ICLR registration receipt amount 450 dollars",
        ),
        MemoryRecord(
            "image-dog",
            "D1:3",
            "2026-01-01T00:00:20Z",
            "user",
            "image",
            "uploaded_image",
            summary="a dog in a park",
        ),
    ]
    store = HiddenMemoryStore(records, dense_encoder=FakeDenseEncoder())
    trace = [
        ToolAction("FILTER", {"field": "modality", "op": "eq", "value": "image"}),
        ToolAction("RETRIEVE", {"method": "hybrid", "top_k": 1}),
        ToolAction("READ", {"fields": ["summary"]}),
        ToolAction("STOP"),
    ]
    result = ToolExecutor(HybridRetriever()).run(
        trace,
        "What amount was on the receipt?",
        store,
    )
    assert result.final_memory_ids == ["image-receipt"]
    assert result.evidence[0].fields["summary"].startswith("ICLR")


def test_turn_aware_retrieval_returns_text_and_image_from_same_memory_entry():
    records = [
        MemoryRecord(
            "D1:1:turn",
            "D1:1",
            "2026-01-01T00:00:00Z",
            "user",
            "text",
            "conversation",
            summary="Sagrada Familia architecture",
        ),
        MemoryRecord(
            "D1:1:image:1",
            "D1:1",
            "2026-01-01T00:00:01Z",
            "user",
            "image",
            "uploaded_image",
            summary="generic stone building",
            raw_pointer="/tmp/support.jpg",
        ),
        MemoryRecord(
            "D1:2:turn",
            "D1:2",
            "2026-01-01T00:00:10Z",
            "user",
            "text",
            "conversation",
            summary="unrelated memory",
        ),
    ]
    store = HiddenMemoryStore(records)
    result = TurnAwareHybridRetriever().retrieve(
        store.initial_pool(),
        "Sagrada Familia",
        store,
        method="bm25",
        top_k=1,
    )
    assert [item.memory.memory_id for item in result] == [
        "D1:1:turn",
        "D1:1:image:1",
    ]


def test_question_image_vision_route_ranks_and_expands_matching_turn():
    records = [
        MemoryRecord(
            "D1:1:turn",
            "D1:1",
            "2026-01-01T00:00:00Z",
            "user",
            "text",
            "conversation",
            summary="matching visual memory",
        ),
        MemoryRecord(
            "D1:1:image:1",
            "D1:1",
            "2026-01-01T00:00:01Z",
            "user",
            "image",
            "uploaded_image",
            raw_pointer="/tmp/support-memory.jpg",
        ),
        MemoryRecord(
            "D1:2:image:1",
            "D1:2",
            "2026-01-01T00:00:10Z",
            "user",
            "image",
            "uploaded_image",
            raw_pointer="/tmp/distractor.jpg",
        ),
    ]
    store = HiddenMemoryStore(records, vision_encoder=FakeVisionEncoder())
    result = TurnAwareHybridRetriever().retrieve(
        store.initial_pool(),
        "What is in this picture?",
        store,
        method="vision",
        top_k=1,
        question_image="/tmp/support-query.jpg",
    )
    assert [item.memory.memory_id for item in result] == [
        "D1:1:turn",
        "D1:1:image:1",
    ]


def test_inspect_raw_receives_question_image_and_turn_text_context():
    records = [
        MemoryRecord(
            "D1:1:turn",
            "D1:1",
            "2026-01-01T00:00:00Z",
            "user",
            "text",
            "conversation",
            summary="Amy has a Cairn Terrier.",
            metadata={"session_date": "2026-01-01"},
        ),
        MemoryRecord(
            "D1:1:image:1",
            "D1:1",
            "2026-01-01T00:00:01Z",
            "user",
            "image",
            "uploaded_image",
            raw_pointer="/tmp/support-memory.jpg",
            metadata={"session_date": "2026-01-01"},
        ),
    ]
    inspector = CapturingRawInspector()
    result = ToolExecutor(
        raw_inspector=inspector,
        max_raw_inspections=1,
    ).run(
        [
            ToolAction("TOPK", {"k": 1}),
            ToolAction(
                "INSPECT_RAW",
                {"target": "current_pool", "instruction": "answer_query_related_visual_details"},
            ),
            ToolAction("STOP"),
        ],
        "What breed is the dog in the attached image?",
        HiddenMemoryStore(records),
        question_image="/tmp/question.jpg",
    )
    assert result.raw_inspection_calls == 1
    assert inspector.calls[0]["question_image"] == "/tmp/question.jpg"
    assert "Cairn Terrier" in inspector.calls[0]["text_context"]
    assert result.evidence[0].fields["linked_text_context"].startswith("Amy")
    assert result.evidence[0].fields["session_date"] == "2026-01-01"


def test_read_includes_session_date_even_when_not_requested():
    records = [
        MemoryRecord(
            "D1:1:turn",
            "D1:1",
            "2024-05-23T00:02:10Z",
            "user",
            "text",
            "conversation",
            summary="I adopted a dog yesterday.",
            metadata={"session_date": "2024-05-23"},
        )
    ]
    result = ToolExecutor().run(
        [
            ToolAction("TOPK", {"k": 1}),
            ToolAction("READ", {"fields": ["timestamp", "summary"]}),
            ToolAction("STOP"),
        ],
        "When was the dog adopted?",
        HiddenMemoryStore(records),
    )
    assert result.evidence[0].fields["session_date"] == "2024-05-23"


def test_answer_model_labels_memory_images_in_multimodal_prompt(tmp_path):
    question = tmp_path / "question.jpg"
    memory = tmp_path / "memory.jpg"
    question.write_bytes(b"fake-question")
    memory.write_bytes(b"fake-memory")
    client = CapturingClient("Cairn Terrier")
    model = ChatAnswerModel(client, max_images=1)
    answer = model.answer(
        "What breed is the dog?",
        [
            EvidenceItem(
                "D1:1:image:1",
                {
                    "turn_id": "D1:1",
                    "summary": "Amy has a Cairn Terrier.",
                    "image_label": "turn=D1:1; context=Amy has a Cairn Terrier.",
                    "raw_pointer": str(memory),
                },
            )
        ],
        question_image=str(question),
    )
    assert answer == "Cairn Terrier"
    content = client.messages[0][0]["content"]
    text_parts = [
        item["text"] for item in content if item.get("type") == "text"
    ]
    assert any("Question image" in text for text in text_parts)
    assert any("Memory image 1" in text for text in text_parts)
    assert any("Cairn Terrier" in text for text in text_parts)


def test_answer_model_prompt_mentions_relative_time_session_date(tmp_path):
    client = CapturingClient("2024-05-22")
    model = ChatAnswerModel(client, max_images=0)
    model.answer(
        "When did Lena adopt her dog?",
        [
            EvidenceItem(
                "D2:1:turn",
                {
                    "timestamp": "2024-05-23T00:02:10Z",
                    "session_date": "2024-05-23",
                    "content": "I finally adopted a Maltese dog yesterday!",
                },
            )
        ],
    )
    prompt = client.messages[0][0]["content"]
    assert "relative time expressions" in prompt
    assert "session_date" in prompt
    assert "2024-05-23" in prompt


def test_topk_preserves_all_records_in_selected_turn():
    records = [
        MemoryRecord(
            "D1:1:turn",
            "D1:1",
            "2026-01-01T00:00:00Z",
            "user",
            "text",
            "conversation",
        ),
        MemoryRecord(
            "D1:1:image:1",
            "D1:1",
            "2026-01-01T00:00:01Z",
            "user",
            "image",
            "uploaded_image",
        ),
        MemoryRecord(
            "D1:2:turn",
            "D1:2",
            "2026-01-01T00:00:10Z",
            "user",
            "text",
            "conversation",
        ),
    ]
    result = ToolExecutor().run(
        [ToolAction("TOPK", {"k": 1}), ToolAction("STOP")],
        "query",
        HiddenMemoryStore(records),
    )
    assert result.final_memory_ids == ["D1:1:turn", "D1:1:image:1"]


def test_teacher_prompt_contains_no_store_or_executor_evidence():
    response = json.dumps(
        [
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 5},
            {"tool": "READ", "fields": ["summary", "content"]},
            {"tool": "STOP"},
        ]
    )
    client = CapturingClient(response)
    teacher = ChatHindsightTeacher(client, privilege_mode="minimal")
    teacher.correct(
        query="What was my research topic?",
        gold_answer="Cancer response prediction",
        student_policy=PolicyOutput(
            actions=[ToolAction("STOP")],
        ),
        student_answer="Not mentioned.",
        correct=False,
    )
    prompt = client.messages[0][0]["content"]
    assert "Cancer response prediction" in prompt
    assert "full memory store" in prompt
    assert "internal-secret-memory" not in prompt
    assert "retrieved evidence:" not in prompt.lower()


def test_diagnostic_teacher_sees_only_observed_evidence_without_internal_ids():
    response = json.dumps(
        [
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 5},
            {"tool": "READ", "fields": ["summary"]},
            {"tool": "STOP"},
        ]
    )
    client = CapturingClient(response)
    teacher = ChatHindsightTeacher(client, privilege_mode="diagnostic")
    execution = ToolExecutor().run(
        [
            ToolAction("READ", {"fields": ["summary", "raw_pointer"]}),
            ToolAction("STOP"),
        ],
        "query",
        HiddenMemoryStore(
            [
                MemoryRecord(
                    "internal-secret-memory",
                    "D1:1",
                    "2024-01-01T00:00:00Z",
                    "user",
                    "text",
                    "conversation",
                    summary="visible observation",
                    raw_pointer="/secret/path.jpg",
                )
            ]
        ),
    )
    teacher.correct(
        "query",
        "gold",
        PolicyOutput([ToolAction("STOP")]),
        "wrong",
        False,
        execution=execution,
    )
    prompt = client.messages[0][0]["content"]
    assert "visible observation" in prompt
    assert "internal-secret-memory" not in prompt
    assert "/secret/path.jpg" not in prompt


def test_oracle_feedback_hides_ranks_action_advice_and_exact_timestamps():
    response = json.dumps(
        [
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 5},
            {"tool": "READ", "fields": ["summary"]},
            {"tool": "STOP"},
        ]
    )
    client = CapturingClient(response)
    teacher = ChatHindsightTeacher(client, privilege_mode="oracle-feedback")
    teacher.correct(
        "query",
        "gold",
        PolicyOutput([ToolAction("STOP")]),
        "wrong",
        False,
        privileged_context={
            "support_count": 2,
            "modalities": ["image", "text"],
            "authors": ["user"],
            "source_types": ["conversation", "uploaded_image"],
            "has_raw_media": True,
            "earliest_timestamp": "2024-01-01T00:00:00Z",
            "retrieval_ranks_for_original_query": {
                "dense": {"best_support_rank": 47}
            },
            "verified_action_advice": {
                "recommended": {"method": "dense", "minimum_top_k": 47}
            },
        },
    )
    prompt = client.messages[0][0]["content"]
    assert '"modalities": ["image", "text"]' in prompt
    assert "best_support_rank" not in prompt
    assert "minimum_top_k" not in prompt
    assert "2024-01-01T00:00:00Z" not in prompt


def test_opd_rollout_produces_query_to_corrected_trace_sft_pair():
    actions = [
        ToolAction("RETRIEVE", {"method": "bm25", "top_k": 1}),
        ToolAction("READ", {"fields": ["summary"]}),
        ToolAction("STOP"),
    ]
    sample = OPDSample(
        sample_id="sample-1",
        query="What was the receipt?",
        gold_answer="ICLR receipt",
        memory_store=HiddenMemoryStore(
            [
                MemoryRecord(
                    "secret-1",
                    "D1:1",
                    "2026-01-01T00:00:00Z",
                    "user",
                    "image",
                    "uploaded_image",
                    summary="ICLR receipt",
                )
            ]
        ),
    )
    rollout = OnPolicyDistiller(
        student=FakePolicy(actions),
        teacher=FakeTeacher(),
        executor=ToolExecutor(),
        answer_model=FakeAnswer(),
        judge=FakeJudge(),
    ).rollout(sample)
    assert rollout.correct is True
    assert rollout.sft_example.input.find("secret-1") == -1
    assert json.loads(rollout.sft_example.target)[0]["tool"] == "RETRIEVE"


def test_oracle_action_advisor_replaces_teacher_trace_that_drops_support():
    student_actions = [
        ToolAction("RETRIEVE", {"method": "bm25", "top_k": 1}),
        ToolAction("READ", {"fields": ["summary"]}),
        ToolAction("STOP"),
    ]
    store = HiddenMemoryStore(
        [
            MemoryRecord(
                "D1:1:turn",
                "D1:1",
                "2026-01-01T00:00:00Z",
                "user",
                "text",
                "conversation",
                summary="receipt",
            ),
            MemoryRecord(
                "D1:2:turn",
                "D1:2",
                "2026-01-01T00:00:10Z",
                "user",
                "text",
                "conversation",
                summary="receipt target target target target",
            ),
        ]
    )
    sample = OPDSample(
        sample_id="sample-oracle",
        query="receipt",
        gold_answer="target",
        memory_store=store,
        metadata={
            "gold_clue_turn_ids": ["D1:2"],
            "teacher_privileged_context": {
                "verified_action_advice": {
                    "recommended": {
                        "method": "bm25",
                        "minimum_top_k": 2,
                        "verified_objective": "all_support_turns",
                    }
                }
            },
        },
    )
    rollout = OnPolicyDistiller(
        student=FakePolicy(student_actions),
        teacher=FakeTeacher(),
        executor=ToolExecutor(validator=TrajectoryValidator(max_top_k=10)),
        answer_model=FakeAnswer(),
        judge=FakeJudge(),
    ).rollout(sample)
    assert rollout.metadata["teacher_selection_source"] == "oracle_action_advisor"
    assert rollout.teacher_policy.actions[0].arguments["top_k"] == 2
    assert rollout.teacher_candidate_diagnostics[0]["support_hit_count"] == 0
    assert rollout.teacher_candidate_diagnostics[1]["support_hit_count"] == 1
    assert rollout.teacher_candidate_diagnostics[1]["selected"] is True


def test_oracle_feedback_revision_is_selected_by_hidden_support_replay():
    initial_actions = [
        ToolAction("RETRIEVE", {"method": "bm25", "top_k": 1}),
        ToolAction("READ", {"fields": ["summary"]}),
        ToolAction("STOP"),
    ]
    revised_actions = [
        ToolAction("RETRIEVE", {"method": "bm25", "top_k": 2}),
        ToolAction("READ", {"fields": ["summary"]}),
        ToolAction("STOP"),
    ]
    store = HiddenMemoryStore(
        [
            MemoryRecord(
                "D1:1:turn",
                "D1:1",
                "2026-01-01T00:00:00Z",
                "user",
                "text",
                "conversation",
                summary="receipt",
            ),
            MemoryRecord(
                "D1:2:turn",
                "D1:2",
                "2026-01-01T00:00:10Z",
                "user",
                "text",
                "conversation",
                summary="receipt target target target",
            ),
        ]
    )
    teacher = FeedbackTeacher(revised_actions)
    rollout = OnPolicyDistiller(
        student=FakePolicy(initial_actions),
        teacher=teacher,
        executor=ToolExecutor(validator=TrajectoryValidator(max_top_k=10)),
        answer_model=FakeAnswer(),
        judge=FakeJudge(),
        teacher_feedback_rounds=2,
        teacher_evidence_budget=2,
    ).rollout(
        OPDSample(
            sample_id="feedback",
            query="receipt",
            gold_answer="target",
            memory_store=store,
            metadata={
                "gold_clue_turn_ids": ["D1:2"],
                "teacher_privileged_context": {
                    "support_count": 1,
                    "modalities": ["text"],
                },
            },
        )
    )
    assert rollout.metadata["teacher_selection_source"] == (
        "llm_teacher_feedback_1"
    )
    assert len(rollout.teacher_candidate_diagnostics) == 2
    assert rollout.teacher_candidate_diagnostics[0]["support_record_hit_count"] == 0
    assert rollout.teacher_candidate_diagnostics[1]["support_record_hit_count"] == 1
    assert teacher.feedback[0]["support_records_covered"] == 0
    assert "best_support_rank" not in json.dumps(teacher.feedback)


def test_memgallery_adapter_splits_text_and_image_memories(tmp_path):
    image_dir = tmp_path / "data" / "image"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "image.jpg"
    image_path.write_bytes(b"image")
    data = {
        "multi_session_dialogues": [
            {
                "session_id": "D1",
                "date": "2024-01-01",
                "dialogues": [
                    {
                        "round": "D1:1",
                        "user": "I uploaded a receipt.",
                        "assistant": "I can see it.",
                        "image_id": ["D1:IMG_001"],
                        "input_image": ["../image/image.jpg"],
                        "image_caption": ["A registration receipt."],
                    }
                ],
            }
        ],
        "human-annotated QAs": [
            {
                "question": "Which image was the receipt?",
                "answer": "D1:IMG_001",
                "point": "VS",
                "clue": ["D1:1"],
            }
        ],
    }
    store, records = build_scenario_store(data, tmp_path)
    samples = scenario_samples(
        data,
        store,
        tmp_path,
        "fixture",
        include_oracle_profile=True,
    )
    assert [record.modality for record in records] == ["text", "image"]
    assert "User: I uploaded a receipt." in records[0].content
    assert "Assistant: I can see it." in records[0].content
    assert records[-1].metadata["public_image_id"] == "D1:IMG_001"
    assert samples[0].metadata["gold_image_ids"] == ["D1:IMG_001"]
    assert samples[0].metadata["gold_clue_turn_ids"] == ["D1:1"]
    ranks = samples[0].metadata["teacher_privileged_context"][
        "retrieval_ranks_for_original_query"
    ]
    assert set(ranks) == {"bm25", "dense", "hybrid"}
    advice = samples[0].metadata["teacher_privileged_context"][
        "verified_action_advice"
    ]
    assert advice["recommended"]["minimum_top_k"] >= 1
    assert advice["trajectory_shape"] == ["RETRIEVE", "STOP"]


def test_memeye_paths_are_normalized_to_memgallery_image_format(tmp_path):
    image_dir = tmp_path / "data" / "image" / "Brand_Memory_Test"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "CocaCola_1.png"
    image_path.write_bytes(b"image")
    avatar_dir = tmp_path / "data" / "image" / "Social_Chat_Memory_Test" / "avatars"
    avatar_dir.mkdir(parents=True)
    question_image = avatar_dir / "P01_marcus.png"
    question_image.write_bytes(b"avatar")
    data = {
        "multi_session_dialogues": [
            {
                "session_id": "S1",
                "date": "2024-01-01",
                "dialogues": [
                    {
                        "round": "S1:1",
                        "user": "I saw a Coca-Cola ad.",
                        "assistant": "Noted.",
                        "image_id": ["CC1:IMG_001"],
                        "input_image": ["Brand_Memory_Test/CocaCola_1.png"],
                        "image_caption": ["A red Coca-Cola campaign image."],
                    }
                ],
            }
        ],
        "human-annotated QAs": [
            {
                "question_id": "Q1",
                "question": "Which brand was shown?",
                "answer": "Coca-Cola",
                "question_image": "Social_Chat_Memory_Test/avatars/P01_marcus.png",
            }
        ],
    }

    assert normalize_memeye_image_path("Brand_Memory_Test/CocaCola_1.png") == (
        "../image/Brand_Memory_Test/CocaCola_1.png"
    )
    normalized_data = normalize_memeye_dialog_image_paths(json.loads(json.dumps(data)))
    assert normalized_data["multi_session_dialogues"][0]["dialogues"][0][
        "input_image"
    ] == ["../image/Brand_Memory_Test/CocaCola_1.png"]
    assert normalized_data["human-annotated QAs"][0]["question_image"] == (
        "../image/Social_Chat_Memory_Test/avatars/P01_marcus.png"
    )

    store, records = build_memeye_scenario_store(data, tmp_path)
    samples = memeye_scenario_samples(data, store, tmp_path, "Brand_Memory_Test_Open")
    assert [record.modality for record in records] == ["text", "image"]
    assert records[-1].metadata["relative_path"] == (
        "../image/Brand_Memory_Test/CocaCola_1.png"
    )
    assert records[-1].metadata["source_relative_path"] == (
        "Brand_Memory_Test/CocaCola_1.png"
    )
    assert records[-1].raw_pointer == str(image_path.resolve())
    assert samples[0].metadata["question_image"] == str(question_image.resolve())


def test_support_verified_sft_filter_rejects_unverified_corrections():
    base = {
        "correct": False,
        "teacher_policy": {"error": ""},
        "teacher_execution": {"error": ""},
    }
    assert select_sft_example(
        {**base, "teacher_evidence_clue_recall_any": True},
        "support-verified",
    )
    assert not select_sft_example(
        {**base, "teacher_evidence_clue_recall_any": False},
        "support-verified",
    )
    assert select_sft_example(base, "valid")
