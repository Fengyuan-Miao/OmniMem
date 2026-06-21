from __future__ import annotations

import json
from pathlib import Path

import pytest

from opd_mm_baseline.interactive import (
    ChatGoldEvidenceVerifier,
    ChatInteractivePlanner,
    ExecutorSession,
    InteractiveActionValidator,
    InteractiveDecision,
    InteractivePolicyRunner,
    InteractiveTeacherSearch,
    InteractiveValidationError,
    AnswerValidationResult,
    StrictAnswerValidator,
    VerificationResult,
    build_compact_planner_prompt,
    build_online_policy_prompt,
    build_simple_student_policy_prompt,
    fallback_action_chunks,
)
from opd_mm_baseline.clients import ChatAnswerJudge
from opd_mm_baseline.build_opd_dataset import (
    correction_to_dataset_row,
    split_name,
)
from opd_mm_baseline.opd_online_train import render_command
from opd_mm_baseline.memgallery_interactive_pipeline import (
    _select_sft_trajectory,
)
from opd_mm_baseline.models import (
    EvidenceItem,
    MemoryRecord,
    OPDSample,
    SFTExample,
    ToolAction,
)
from opd_mm_baseline.online import (
    OnlineCorrection,
    OnlineDistillationBuffer,
    OnlineSampleResult,
    OnlineSelfDistiller,
    _is_trainable_teacher_path,
)
from opd_mm_baseline.retrieval import HiddenMemoryStore, TurnAwareHybridRetriever


def make_store() -> HiddenMemoryStore:
    return HiddenMemoryStore(
        [
            MemoryRecord(
                memory_id="D1:1:turn",
                turn_id="D1:1",
                timestamp="2026-01-01T00:00:00Z",
                author="user",
                modality="text",
                source_type="conversation",
                summary="Alice discussed apples.",
                content="User: Alice discussed apples.",
                metadata={"session_id": "D1", "turn_index": 1},
            ),
            MemoryRecord(
                memory_id="D1:2:turn",
                turn_id="D1:2",
                timestamp="2026-01-01T00:00:10Z",
                author="user",
                modality="text",
                source_type="conversation",
                summary="Bob selected the banana project.",
                content="User: Bob selected the banana project.",
                metadata={"session_id": "D1", "turn_index": 2},
            ),
            MemoryRecord(
                memory_id="D1:3:turn",
                turn_id="D1:3",
                timestamp="2026-01-01T00:00:20Z",
                author="assistant",
                modality="text",
                source_type="conversation",
                summary="The group discussed cherries.",
                content="Assistant: The group discussed cherries.",
                metadata={"session_id": "D1", "turn_index": 3},
            ),
        ]
    )


def test_interactive_validator_allows_rewrite_and_blocks_dead_actions() -> None:
    validator = InteractiveActionValidator(max_chunk_actions=3, max_top_k=20)
    actions = validator.validate(
        [
            {
                "tool": "RETRIEVE",
                "method": "hybrid",
                "top_k": 5,
                "query": "banana project",
            },
            {"tool": "READ", "fields": ["summary", "content"]},
        ]
    )
    assert actions[0].arguments["query"] == "banana project"

    with pytest.raises(InteractiveValidationError):
        validator.validate(
            [
                {"tool": "READ", "fields": ["summary"]},
                {"tool": "TOPK", "k": 2},
            ]
        )
    with pytest.raises(InteractiveValidationError):
        validator.validate(
            [
                {
                    "tool": "RETRIEVE",
                    "query": "memory-123",
                    "top_k": 2,
                }
            ]
        )


def test_interactive_validator_repairs_harmless_schema_drift() -> None:
    validator = InteractiveActionValidator()
    actions = validator.repair(
        [
            {
                "tool": "READ",
                "fields": ["content", "summary"],
                "limit": 2,
                "pool_record_id": 1,
            },
            {"tool": "STOP", "reason": "done"},
        ]
    )
    assert actions == [
        ToolAction("READ", {"fields": ["content", "summary"]}),
        ToolAction("STOP"),
    ]


def test_atomic_validator_repairs_to_first_executable_action() -> None:
    validator = InteractiveActionValidator(max_chunk_actions=1)
    with pytest.raises(InteractiveValidationError):
        validator.validate(
            [
                {"tool": "RETRIEVE", "method": "bm25", "top_k": 2},
                {"tool": "READ", "fields": ["summary"]},
            ]
        )

    actions = validator.repair(
        [
            {"tool": "RETRIEVE", "method": "bm25", "top_k": 2},
            {"tool": "READ", "fields": ["summary"]},
        ]
    )
    assert actions == [
        ToolAction("RETRIEVE", {"method": "bm25", "top_k": 2})
    ]


def test_atomic_fallback_returns_single_actions_by_state() -> None:
    validator = InteractiveActionValidator(
        max_chunk_actions=1,
        allow_inspect_raw=True,
    )
    session = ExecutorSession(
        query="Which project did Bob select?",
        memory_store=make_store(),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
    )
    first = fallback_action_chunks(session.observation(), validator)[0]
    assert [action.tool for action in first] == ["RETRIEVE"]

    session.execute_chunk(first)
    second = fallback_action_chunks(session.observation(), validator)[0]
    assert [action.tool for action in second] == ["READ"]

    session.execute_chunk(second)
    third = fallback_action_chunks(session.observation(), validator)[0]
    assert [action.tool for action in third] == ["INSPECT_RAW"]


def test_executor_observation_is_hidden_until_pool_selection() -> None:
    validator = InteractiveActionValidator()
    session = ExecutorSession(
        query="Which project did Bob select?",
        memory_store=make_store(),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
    )
    initial = session.observation()
    assert initial.pool_record_count == 3
    assert initial.candidate_previews == []

    after = session.execute_chunk(
        [
            {
                "tool": "RETRIEVE",
                "method": "bm25",
                "top_k": 1,
                "query": "Bob banana project",
            },
            {"tool": "READ", "fields": ["summary", "content", "turn_id"]},
        ]
    )
    assert after.pool_turn_count == 1
    assert after.evidence_count == 1
    assert after.new_evidence_count == 1
    assert after.last_retrieval_signature == {
        "method": "bm25",
        "top_k": 1,
        "query": "Bob banana project",
        "scope": "all",
    }
    assert "banana" in str(after.evidence_previews).lower()

    persisted = session.observation()
    assert persisted.new_evidence_count == 1
    assert (
        persisted.last_retrieval_signature
        == after.last_retrieval_signature
    )

    repeated = session.execute_chunk(
        [
            {
                "tool": "RETRIEVE",
                "method": "bm25",
                "top_k": 1,
                "query": "Bob banana project",
            },
            {"tool": "READ", "fields": ["summary", "content", "turn_id"]},
        ]
    )
    assert repeated.new_evidence_count == 0
    assert (
        repeated.last_retrieval_signature
        == after.last_retrieval_signature
    )

    cloned = session.clone().observation()
    assert cloned.new_evidence_count == 0
    assert cloned.last_retrieval_signature == after.last_retrieval_signature
    planner_prompt = build_compact_planner_prompt(
        query=session.query,
        history=session.history,
        observation=cloned,
        allow_inspect_raw=False,
        candidate_count=1,
    )
    assert '"new": 0' in planner_prompt
    assert '"query": "Bob banana project"' in planner_prompt


def test_atomic_policy_runner_reobserves_after_each_action() -> None:
    class AtomicPlanner:
        def __init__(self) -> None:
            self.calls = 0
            self.observations = []
            self.last_raw_response = ""

        def propose(
            self,
            query,
            history,
            observation,
            candidate_count=1,
            privileged_feedback=None,
        ):
            self.calls += 1
            self.observations.append(observation)
            if not observation.candidate_previews:
                return [
                    [
                        ToolAction(
                            "RETRIEVE",
                            {
                                "method": "bm25",
                                "top_k": 1,
                                "query": "banana",
                            },
                        )
                    ]
                ]
            if observation.evidence_count == 0:
                return [[ToolAction("READ", {"fields": ["summary"]})]]
            return [[ToolAction("STOP")]]

    planner = AtomicPlanner()
    validator = InteractiveActionValidator(max_chunk_actions=1)
    runner = InteractivePolicyRunner(
        planner=planner,
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
        max_rounds=3,
        max_actions=3,
    )
    result = runner.run("Which project did Bob select?", make_store())
    assert [action.tool for action in result.actions] == [
        "RETRIEVE",
        "READ",
        "STOP",
    ]
    assert len(planner.observations) == 3
    assert planner.observations[0].candidate_previews == []
    assert planner.observations[1].candidate_previews
    assert planner.observations[1].evidence_count == 0
    assert planner.observations[2].evidence_count > 0


def test_online_policy_prompt_uses_compact_observation() -> None:
    validator = InteractiveActionValidator()
    session = ExecutorSession(
        query="Which project did Bob select?",
        memory_store=make_store(),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
    )
    session.execute_chunk(
        [
            {
                "tool": "RETRIEVE",
                "method": "bm25",
                "top_k": 2,
                "query": "Bob banana project",
            },
            {
                "tool": "READ",
                "fields": [
                    "summary",
                    "content",
                    "timestamp",
                    "session_date",
                    "turn_id",
                    "raw_pointer",
                ],
            },
        ]
    )
    prompt = build_online_policy_prompt(
        "Which project did Bob select?",
        session.history,
        session.observation(),
        validator.schema_text(),
    )
    assert "banana" in prompt.lower()
    assert '"last_retrieval_signature"' in prompt
    assert '"new_evidence_count":' in prompt
    observation_json = prompt.split("Current executor observation:\n", 1)[1]
    assert "raw_pointer" not in observation_json
    assert "memory_id" not in observation_json
    assert len(prompt) < 4500


def test_simple_student_prompt_only_describes_tools_and_online_state() -> None:
    validator = InteractiveActionValidator(allow_inspect_raw=True)
    session = ExecutorSession(
        query="Which project did Bob select?",
        memory_store=make_store(),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
    )
    session.execute_chunk(
        [{"tool": "RETRIEVE", "method": "bm25", "top_k": 1}]
    )
    prompt = build_simple_student_policy_prompt(
        "Which project did Bob select?",
        session.history,
        session.observation(),
        validator.schema_text(),
    )
    assert "Available tools:" in prompt
    assert "RETRIEVE(" in prompt
    assert "INSPECT_RAW(" in prompt
    assert "Return only a JSON array" in prompt
    assert '"reflection"' not in prompt
    assert "teacher" not in prompt.lower()
    assert "gold answer" not in prompt.lower()
    assert "feedback" not in prompt.lower()
    assert "memory_id" not in prompt


def test_expand_neighbors_uses_turn_order() -> None:
    session = ExecutorSession(
        query="banana",
        memory_store=make_store(),
        validator=InteractiveActionValidator(),
        retriever=TurnAwareHybridRetriever(context_window=0),
    )
    session.execute_chunk(
        [{"tool": "RETRIEVE", "method": "bm25", "top_k": 1}]
    )
    after = session.execute_chunk([{"tool": "EXPAND_NEIGHBORS", "window": 1}])
    assert after.pool_turn_count == 3


class FakePlanner:
    def __init__(self) -> None:
        self.calls = 0
        self.last_raw_response = ""
        self.feedback = []

    def propose(
        self,
        query,
        history,
        observation,
        candidate_count=3,
        privileged_feedback=None,
    ):
        self.calls += 1
        self.feedback.append(privileged_feedback)
        if history:
            return [[ToolAction("STOP")]]
        return [
            [
                ToolAction(
                    "RETRIEVE",
                    {"method": "bm25", "top_k": 1, "query": "apples"},
                ),
                ToolAction("READ", {"fields": ["summary", "content"]}),
            ],
            [
                ToolAction(
                    "RETRIEVE",
                    {
                        "method": "bm25",
                        "top_k": 1,
                        "query": "Bob banana project",
                    },
                ),
                ToolAction("READ", {"fields": ["summary", "content"]}),
            ],
        ][:candidate_count]


class FakeGoldVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, query, gold_answer, evidence):
        self.calls += 1
        text = str([item.fields for item in evidence]).lower()
        answerable = "banana" in text
        return VerificationResult(
            answerable=answerable,
            relevance=1.0 if answerable else 0.2,
            completeness=1.0 if answerable else 0.1,
            redundancy=0.0,
            reason=f"gold was {gold_answer}",
        )


class ExplodingVerifier:
    calls = 0

    def evaluate(self, query, gold_answer, evidence):
        self.calls += 1
        raise AssertionError("teacher search should not call verifier")


def test_teacher_search_selects_answerable_branch_without_sft_privilege() -> None:
    planner = FakePlanner()
    verifier = ExplodingVerifier()
    validator = InteractiveActionValidator()
    answer_validator = StrictAnswerValidator(
        EvidenceEchoAnswer(),
        ContainsBananaJudge(),
    )
    result = InteractiveTeacherSearch(
        planner=planner,
        verifier=verifier,
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
        max_rounds=2,
        beam_size=2,
        candidates_per_node=2,
        answer_validator=answer_validator,
    ).search(
        query="Which project did Bob select?",
        gold_answer="SECRET_GOLD banana",
        memory_store=make_store(),
    )
    assert result.verification.answerable
    assert result.verifier_calls == 0
    assert verifier.calls == 0
    assert result.answer_validation is not None
    assert result.answer_validation.correct
    assert any(
        action.arguments.get("query") == "Bob banana project"
        for action in result.actions
        if action.tool == "RETRIEVE"
    )
    examples = result.sft_examples(
        "sample",
        "Which project did Bob select?",
        validator.schema_text(),
    )
    assert examples
    assert all("SECRET_GOLD" not in example.input for example in examples)
    assert all("gold was" not in example.input for example in examples)
    assert all("continue_required" not in example.input for example in examples)
    assert examples[0].metadata["evidence_count_after"] >= 1
    serialized = examples[0].to_dict()
    assert set(serialized) == {"sample_id", "input", "target", "round_index"}
    assert "metadata" not in serialized
    assert "teacher_reflection" not in json.dumps(serialized)
    assert "privileged_feedback" not in json.dumps(serialized)


def test_online_policy_runner_never_requests_privileged_feedback() -> None:
    planner = FakePlanner()
    result = InteractivePolicyRunner(
        planner=planner,
        validator=InteractiveActionValidator(),
        retriever=TurnAwareHybridRetriever(context_window=0),
        max_rounds=2,
    ).run(
        query="Which project did Bob select?",
        memory_store=make_store(),
    )
    assert result.execution.stopped
    assert planner.feedback == [None, None]


class CapturingClient:
    def __init__(self, response: str | None = None) -> None:
        self.prompt = ""
        self.extra_body = None
        self.response = response or (
            '{"candidates":[[{"tool":"RETRIEVE","method":"bm25",'
            '"top_k":2,"query":"banana"}]]}'
        )

    def complete(
        self,
        messages,
        max_tokens=512,
        temperature=0.0,
        extra_body=None,
    ):
        self.prompt = "\n".join(str(message["content"]) for message in messages)
        self.extra_body = extra_body
        return self.response


class CapturingInspectRawClient(CapturingClient):
    def complete(
        self,
        messages,
        max_tokens=512,
        temperature=0.0,
        extra_body=None,
    ):
        self.prompt = "\n".join(str(message["content"]) for message in messages)
        self.extra_body = extra_body
        return (
            '{"candidates":[{"next_tool":"INSPECT_RAW",'
            '"expected_gain":"raw visual details",'
            '"actions":[{"tool":"INSPECT_RAW",'
            '"target":"current_pool",'
            '"instruction":"answer_query_related_visual_details"}]}]}'
        )


def test_chat_planner_prompt_has_no_gold_interface() -> None:
    client = CapturingClient()
    validator = InteractiveActionValidator()
    planner = ChatInteractivePlanner(client, validator)
    session = ExecutorSession(
        query="Find the selected project",
        memory_store=make_store(),
        validator=validator,
    )
    planner.propose(
        query="Find the selected project",
        history=[],
        observation=session.observation(),
        privileged_feedback={"completeness": "low"},
    )
    assert "SECRET_GOLD" not in client.prompt
    assert "gold answer" not in client.prompt.lower()
    assert "memory_id" not in client.prompt
    assert "it does NOT add answer evidence" in client.prompt
    assert "bm25|dense|vision|hybrid" in client.prompt
    assert "Analyze the query before choosing RETRIEVE.method" in client.prompt
    assert "Use vision for SigLIP visual search" in client.prompt
    assert "RETRIEVE alone cannot support an answer" in client.prompt
    assert "read or inspect them instead of" in client.prompt


def test_chat_planner_passes_thinking_token_budget() -> None:
    client = CapturingClient()
    validator = InteractiveActionValidator()
    planner = ChatInteractivePlanner(
        client,
        validator,
        thinking_token_budget=128,
    )
    session = ExecutorSession(
        query="Find the selected project",
        memory_store=make_store(),
        validator=validator,
    )
    planner.propose(
        query="Find the selected project",
        history=[],
        observation=session.observation(),
    )
    assert client.extra_body == {"thinking_token_budget": 128}


def test_chat_planner_student_simple_uses_action_array_prompt() -> None:
    client = CapturingClient(
        '[{"tool":"RETRIEVE","method":"bm25","top_k":1,"scope":"all"}]'
    )
    validator = InteractiveActionValidator()
    planner = ChatInteractivePlanner(
        client,
        validator,
        prompt_mode="student_simple",
    )
    session = ExecutorSession(
        query="Find the selected project",
        memory_store=make_store(),
        validator=validator,
    )
    candidates = planner.propose(
        query="Find the selected project",
        history=[],
        observation=session.observation(),
    )
    assert candidates[0][0].tool == "RETRIEVE"
    assert "final JSON action array" in client.prompt
    assert '"reflection"' not in client.prompt
    assert "Return a memory-tool policy JSON" not in client.prompt
    assert "candidates" not in client.prompt


def test_chat_planner_does_not_inject_fallback_for_visible_pool() -> None:
    client = CapturingClient()
    validator = InteractiveActionValidator()
    planner = ChatInteractivePlanner(client, validator)
    session = ExecutorSession(
        query="Find the selected project",
        memory_store=make_store(),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
    )
    session.execute_chunk(
        [{"tool": "RETRIEVE", "method": "bm25", "top_k": 2}]
    )
    candidates = planner.propose(
        query="Find the selected project",
        history=session.history,
        observation=session.observation(),
        candidate_count=2,
    )
    assert len(candidates) == 1
    assert [action.tool for action in candidates[0]] == ["RETRIEVE"]
    signature = json.dumps(
        [action.to_dict() for action in candidates[0]],
        sort_keys=True,
        ensure_ascii=False,
    )
    assert planner.last_candidate_sources[signature] == "planner"


def test_chat_planner_softly_prompts_visual_feedback_without_injection() -> None:
    client = CapturingInspectRawClient()
    validator = InteractiveActionValidator(allow_inspect_raw=True)
    planner = ChatInteractivePlanner(client, validator)
    session = ExecutorSession(
        query="Which image matches the question?",
        memory_store=make_store(),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
    )
    session.execute_chunk(
        [{"tool": "RETRIEVE", "method": "bm25", "top_k": 2}]
    )
    candidates = planner.propose(
        query="Which image matches the question?",
        history=session.history,
        observation=session.observation(),
        candidate_count=1,
        privileged_feedback={
            "continue_required": True,
            "failure_diagnostic": {
                "failure_type": "uninspected_visual_evidence",
                "evidence_gap": "Visual memories are present.",
                "recommended_change": "Use INSPECT_RAW on candidates.",
            },
        },
    )
    assert candidates[0][0].tool == "INSPECT_RAW"
    assert "final JSON object" in client.prompt
    assert "expected_gain" in client.prompt
    assert "visual/image gap" in client.prompt
    assert "Use INSPECT_RAW on candidates." in client.prompt
    signature = json.dumps(
        [action.to_dict() for action in candidates[0]],
        sort_keys=True,
        ensure_ascii=False,
    )
    assert planner.last_candidate_sources[signature] == "planner"
    assert planner.last_candidate_rationales[signature]["next_tool"] == (
        "INSPECT_RAW"
    )
    assert planner.last_candidate_rationales[signature]["expected_gain"] == (
        "raw visual details"
    )


class NeverCalledClient:
    def complete(self, messages, max_tokens=512, temperature=0.0):
        raise AssertionError("empty evidence should be rejected locally")


def test_gold_verifier_never_accepts_empty_evidence() -> None:
    verifier = ChatGoldEvidenceVerifier(NeverCalledClient())
    result = verifier.evaluate(
        "Was a timer mentioned?",
        "Not mentioned.",
        [],
    )
    assert not result.answerable
    assert result.completeness == 0.0


class LowGroundingClient:
    def complete(self, messages, max_tokens=512, temperature=0.0):
        return (
            '{"answerable":true,"relevance":0.2,"completeness":0.3,'
            '"redundancy":0.0,"reason":"weak"}'
        )


def test_gold_verifier_applies_grounding_thresholds() -> None:
    verifier = ChatGoldEvidenceVerifier(LowGroundingClient())
    result = verifier.evaluate(
        "Which project?",
        "banana",
        [EvidenceItem("D1:1", {"content": "unrelated"})],
    )
    assert not result.answerable


class EvidenceEchoAnswer:
    def answer(self, query, evidence, question_image=None):
        return " ".join(
            str(value)
            for item in evidence
            for value in item.fields.values()
        )


class ContainsBananaJudge:
    def evaluate(self, query, prediction, gold_answer):
        correct = "banana" in prediction.lower()
        return correct, float(correct), "contains banana"


class MissingTextDiagnosticJudge:
    def __init__(self):
        self.called = False

    def evaluate(self, query, prediction, gold_answer):
        return False, 0.0, "The required textual fact is missing."

    def diagnose_failure(
        self,
        query,
        prediction,
        evidence,
    ):
        self.called = True
        return {
            "failure_type": "missing_text_evidence",
            "evidence_gap": "The retrieved text does not state the subject.",
            "recommended_change": (
                "Rewrite the text query and retrieve a different pool."
            ),
        }


class GoldAwareAssessAnswer:
    def __init__(self, answerable):
        self.answerable = answerable
        self.calls = 0

    def answer(self, query, evidence, question_image=None):
        raise AssertionError("assess_evidence should replace answer+judge")

    def assess_evidence(self, query, gold_answer, evidence, question_image=None):
        self.calls += 1
        if self.answerable:
            return {
                "answerable": True,
                "score": 1.0,
                "predicted_answer": gold_answer,
                "reason": "evidence supports gold",
            }
        return {
            "answerable": False,
            "score": 0.1,
            "predicted_answer": "wrong",
            "failure_type": "missing_text_evidence",
            "evidence_gap": "The evidence does not contain the needed fact.",
            "recommended_change": "Retrieve a more focused text memory.",
            "reason": "missing support",
        }


class ExplodingJudge:
    def evaluate(self, query, prediction, gold_answer):
        raise AssertionError("judge should not be called for assess_evidence")


def test_strict_answer_validator_prefers_answer_model_assessment() -> None:
    answer_model = GoldAwareAssessAnswer(answerable=False)
    validator = StrictAnswerValidator(answer_model, ExplodingJudge())
    result = validator.evaluate(
        "Which project?",
        "banana",
        [EvidenceItem("D1:1", {"content": "Alice discussed apples."})],
    )
    assert answer_model.calls == 1
    assert not result.correct
    assert result.diagnostic["failure_type"] == "missing_text_evidence"
    feedback = result.failure_feedback(
        [EvidenceItem("D1:1", {"content": "Alice discussed apples."})]
    )
    assert feedback["recommended_change"] == "Retrieve a more focused text memory."


class CapturingJudgeClient:
    def __init__(self):
        self.prompt = ""

    def complete(self, messages, max_tokens=512, temperature=0.0):
        self.prompt = "\n".join(str(message["content"]) for message in messages)
        return (
            '{"failure_type":"missing_text_evidence",'
            '"evidence_gap":"missing support",'
            '"recommended_change":"retrieve more context"}'
        )


def test_answer_judge_failure_diagnosis_prompt_is_gold_free() -> None:
    client = CapturingJudgeClient()
    judge = ChatAnswerJudge(client)
    diagnostic = judge.diagnose_failure(
        "What names were considered?",
        "Lumi",
        [EvidenceItem("D1:1", {"content": "Lumi was approved."})],
    )
    assert diagnostic["failure_type"] == "missing_text_evidence"
    assert "SECRET_GOLD" not in client.prompt
    assert "Gold answer" not in client.prompt


def test_strict_answer_validator_uses_judge_diagnosis_over_visual_presence() -> None:
    judge = MissingTextDiagnosticJudge()
    validator = StrictAnswerValidator(
        EvidenceEchoAnswer(),
        judge,
    )
    result = validator.evaluate(
        "What subject is Lena majoring in?",
        "Life Sciences",
        [
            EvidenceItem(
                "D1:1",
                {
                    "content": "Lena discussed animal behavior.",
                    "modality": "image",
                    "raw_pointer": "/tmp/incidental.jpg",
                },
            )
        ],
    )
    assert judge.called
    feedback = result.failure_feedback(
        [
            EvidenceItem(
                "D1:1",
                {
                    "content": "Lena discussed animal behavior.",
                    "modality": "image",
                    "raw_pointer": "/tmp/incidental.jpg",
                },
            )
        ],
        can_inspect_raw=True,
    )
    assert feedback["failure_type"] == "missing_text_evidence"
    assert "judge_reason" not in feedback
    assert "INSPECT_RAW" not in feedback["recommended_change"]


def test_strict_answer_validator_requires_exact_gold_image_ids() -> None:
    validator = StrictAnswerValidator(
        EvidenceEchoAnswer(),
        ContainsBananaJudge(),
        min_score=0.9,
    )
    result = validator.evaluate(
        "Which images?",
        "D1:IMG_001, D1:IMG_002",
        [
            EvidenceItem(
                "D1:1",
                {"content": "banana D1:IMG_001 D1:IMG_003"},
            )
        ],
    )
    assert not result.correct
    assert result.image_ids_match is False


class AlwaysSufficientVerifier:
    def __init__(self):
        self.calls = 0

    def evaluate(self, query, gold_answer, evidence):
        self.calls += 1
        return VerificationResult(
            answerable=bool(evidence),
            relevance=1.0,
            completeness=1.0,
            redundancy=0.0,
        )


class CorrectingPlanner:
    def __init__(self):
        self.calls = 0
        self.last_raw_response = ""
        self.last_candidate_sources = {}
        self.feedback = []

    def propose(
        self,
        query,
        history,
        observation,
        candidate_count=1,
        privileged_feedback=None,
    ):
        self.calls += 1
        self.feedback.append(privileged_feedback)
        term = "apples" if not history else "banana"
        actions = [
            ToolAction(
                "RETRIEVE",
                {"method": "bm25", "top_k": 1, "query": term},
            ),
            ToolAction("READ", {"fields": ["content"]}),
        ]
        signature = json.dumps(
            [action.to_dict() for action in actions],
            sort_keys=True,
        )
        self.last_candidate_sources = {signature: "planner"}
        return [actions]


def test_teacher_search_continues_when_strict_answer_validation_fails() -> None:
    answer_validator = StrictAnswerValidator(
        EvidenceEchoAnswer(),
        ContainsBananaJudge(),
    )
    planner = CorrectingPlanner()
    result = InteractiveTeacherSearch(
        planner=planner,
        verifier=AlwaysSufficientVerifier(),
        validator=InteractiveActionValidator(),
        retriever=TurnAwareHybridRetriever(context_window=0),
        answer_validator=answer_validator,
        max_rounds=2,
        beam_size=1,
        candidates_per_node=1,
    ).search(
        query="Which project did Bob select?",
        gold_answer="banana",
        memory_store=make_store(),
    )
    assert result.answer_validation is not None
    assert result.answer_validation.correct
    assert result.verifier_calls == 0
    assert result.answer_validator_calls == 2
    assert result.failure_diagnostics
    assert planner.feedback[0] is None
    failure = planner.feedback[1]["failure_diagnostic"]
    assert failure["failure_type"] == "answer_mismatch"
    assert "Alice discussed apples." in failure["predicted_answer"]
    assert "recommended_change" in failure
    assert any(
        action.arguments.get("query") == "banana"
        for action in result.actions
        if action.tool == "RETRIEVE"
    )


def test_support_grounded_sft_filter_uses_available_annotations() -> None:
    verification = VerificationResult(True, 1.0, 1.0, 0.0)
    assert _select_sft_trajectory(
        "support-grounded",
        verification,
        {
            "evidence_clue_recall_any": True,
            "gold_image_recall_any": None,
        },
    )
    assert not _select_sft_trajectory(
        "support-grounded",
        verification,
        {
            "evidence_clue_recall_any": True,
            "gold_image_recall_all": False,
        },
    )
    assert not _select_sft_trajectory(
        "answer-correct",
        verification,
        {
            "evidence_clue_recall_any": True,
            "gold_image_recall_all": None,
        },
        answer_correct=False,
    )
    assert _select_sft_trajectory(
        "answer-correct",
        verification,
        {
            "evidence_clue_recall_any": True,
            "gold_image_recall_all": None,
        },
        answer_correct=True,
    )
    assert not _select_sft_trajectory(
        "support-grounded",
        verification,
        {
            "evidence_clue_recall_any": False,
            "gold_image_recall_any": None,
        },
    )
    assert _select_sft_trajectory(
        "support-grounded",
        verification,
        {
            "evidence_clue_recall_any": None,
            "gold_image_recall_any": None,
        },
    )


class BrokenPlanner:
    def __init__(self) -> None:
        self.calls = 0
        self.last_raw_response = ""

    def propose(self, *args, **kwargs):
        self.calls += 1
        raise ValueError("invalid model output")


def test_teacher_search_uses_online_fallback_instead_of_empty_stop() -> None:
    result = InteractiveTeacherSearch(
        planner=BrokenPlanner(),
        verifier=FakeGoldVerifier(),
        validator=InteractiveActionValidator(),
        retriever=TurnAwareHybridRetriever(context_window=0),
        max_rounds=1,
        beam_size=1,
        candidates_per_node=1,
    ).search(
        query="Which project did Bob select?",
        gold_answer="banana",
        memory_store=make_store(),
    )
    assert result.verification.answerable
    assert any(action.tool == "READ" for action in result.actions)
    assert result.decisions[0].action_source == "controller_fallback"


def test_online_distillation_buffer_deduplicates_examples(tmp_path) -> None:
    buffer = OnlineDistillationBuffer()
    example = SFTExample(
        "one",
        "input",
        "target",
        metadata={
            "opd": {
                "student_prompt_template": "simple_tools_v1",
                "teacher_input": "teacher prompt",
            }
        },
    )
    assert buffer.add(example, round_index=0)
    assert not buffer.add(example, round_index=1)
    assert len(buffer) == 1
    output = tmp_path / "buffer.jsonl"
    buffer.write_jsonl(output)
    row = json.loads(output.read_text().strip())
    assert row["buffer"]["seen_count"] == 2
    assert row["buffer"]["last_round"] == 1
    assert row["metadata"]["opd"]["teacher_input"] == "teacher prompt"


def test_opd_dataset_row_keeps_teacher_prompt_in_metadata_only() -> None:
    example = SFTExample(
        "sample:online:0:0",
        "student visible prompt",
        '[{"tool":"STOP"}]',
        metadata={
            "opd": {
                "teacher_input": "teacher compact prompt with feedback",
                "student_prompt_template": "simple_tools_v1",
            }
        },
    )
    correction = OnlineCorrection(
        sample_id="sample",
        state_index=0,
        student_actions=[ToolAction("STOP")],
        teacher_actions=[ToolAction("STOP")],
        teacher_answer_validation=AnswerValidationResult(True, 1.0, ""),
        example=example,
    )
    result = OnlineSampleResult(
        sample_id="sample",
        student_actions=[ToolAction("STOP")],
        student_evidence_sufficiency=AnswerValidationResult(True, 1.0, ""),
        student_answer_validation=AnswerValidationResult(False, 0.0, ""),
        corrections=[correction],
        student_planner_calls=1,
    )
    row = correction_to_dataset_row(result, 0, "scenario", 0)
    assert row["input"] == "student visible prompt"
    assert "teacher compact" not in row["input"]
    assert row["metadata"]["opd"]["teacher_input"].startswith("teacher compact")
    assert row["metadata"]["dataset"]["student_answer_correct_no_gold"] is False
    assert row["metadata"]["dataset"]["teacher_actions"] == [{"tool": "STOP"}]


def test_opd_dataset_split_is_stable() -> None:
    first = split_name("scenario:sample:state", 0.2, 7)
    second = split_name("scenario:sample:state", 0.2, 7)
    assert first == second
    assert split_name("anything", 0.0, 7) == "train"
    assert split_name("anything", 1.0, 7) == "val"


def test_online_opd_reload_command_template() -> None:
    rendered = render_command(
        "reload --round {round} --model {checkpoint} --data {data}",
        round_index=2,
        checkpoint=Path("/tmp/ckpt"),
        data=Path("/tmp/train.jsonl"),
        run_dir=Path("/tmp/run"),
        collection_dir=Path("/tmp/collection"),
    )
    assert rendered == "reload --round 2 --model /tmp/ckpt --data /tmp/train.jsonl"


def test_planner_repaired_and_system_stop_are_trainable_path() -> None:
    validator = InteractiveActionValidator()
    session = ExecutorSession(
        query="Which project did Bob select?",
        memory_store=make_store(),
        validator=validator,
    )
    observation = session.observation()
    repaired = InteractiveDecision(
        observation=observation,
        observation_after=observation,
        history=[],
        actions=[
            ToolAction("RETRIEVE", {"method": "hybrid", "top_k": 5}),
            ToolAction("READ", {"fields": ["summary"]}),
        ],
        privileged_feedback={},
        verification_after=VerificationResult(True, 1.0, 1.0, 0.0),
        action_source="planner_repaired",
    )
    stop = InteractiveDecision(
        observation=observation,
        observation_after=observation,
        history=[],
        actions=[ToolAction("STOP")],
        privileged_feedback={},
        verification_after=VerificationResult(True, 1.0, 1.0, 0.0),
        action_source="answer_stop",
    )
    fallback = InteractiveDecision(
        observation=observation,
        observation_after=observation,
        history=[],
        actions=[ToolAction("READ", {"fields": ["summary"]})],
        privileged_feedback={},
        verification_after=VerificationResult(True, 1.0, 1.0, 0.0),
        action_source="controller_fallback",
    )
    budget_stop = InteractiveDecision(
        observation=observation,
        observation_after=observation,
        history=[],
        actions=[ToolAction("STOP")],
        privileged_feedback={},
        verification_after=VerificationResult(False, 0.0, 0.0, 0.0),
        action_source="budget_stop",
    )
    assert _is_trainable_teacher_path([repaired, stop])
    assert not _is_trainable_teacher_path([fallback, stop])
    assert not _is_trainable_teacher_path([repaired, budget_stop])


class OnlineStudentPlanner:
    def __init__(self):
        self.calls = 0

    def propose(
        self,
        query,
        history,
        observation,
        candidate_count=1,
        privileged_feedback=None,
    ):
        self.calls += 1
        if history:
            return [[ToolAction("STOP")]]
        return [
            [
                ToolAction(
                    "RETRIEVE",
                    {"method": "bm25", "top_k": 1, "query": "apples"},
                ),
                ToolAction("READ", {"fields": ["content"]}),
            ]
        ]


class WrongNoGoldAnswer:
    def answer(self, query, evidence, question_image=None):
        return "apples"

    def assess_evidence(self, query, gold_answer, evidence, question_image=None):
        return {
            "answerable": True,
            "score": 1.0,
            "predicted_answer": gold_answer,
            "reason": "gold-aware evidence check passed",
        }


def test_online_self_distiller_labels_student_visited_states() -> None:
    validator = InteractiveActionValidator()
    answer_validator = StrictAnswerValidator(
        EvidenceEchoAnswer(),
        ContainsBananaJudge(),
    )
    teacher_search = InteractiveTeacherSearch(
        planner=CorrectingPlanner(),
        verifier=AlwaysSufficientVerifier(),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
        answer_validator=answer_validator,
        max_rounds=2,
        beam_size=1,
        candidates_per_node=1,
    )
    buffer = OnlineDistillationBuffer()
    distiller = OnlineSelfDistiller(
        student_planner=OnlineStudentPlanner(),
        teacher_search=teacher_search,
        answer_validator=answer_validator,
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
        max_student_rounds=2,
        buffer=buffer,
    )
    result = distiller.collect_sample(
        OPDSample(
            sample_id="sample",
            query="Which project did Bob select?",
            gold_answer="banana",
            memory_store=make_store(),
        ),
        round_index=3,
    )
    assert len(result.corrections) == 5
    assert len(result.teacher_attempts) == 2
    assert result.teacher_attempts[0]["answer_validation"]["correct"]
    assert len(distiller.buffer) == 3
    assert len(buffer) == 3
    assert any(
        correction.teacher_action_source == "answer_stop"
        and all(action.tool == "STOP" for action in correction.teacher_actions)
        for correction in result.corrections
    )
    second = next(
        correction
        for correction in result.corrections
        if any(
            action.arguments.get("query") == "banana"
            for action in correction.teacher_actions
            if action.tool == "RETRIEVE"
        )
    )
    assert any(
        action.arguments.get("query") == "banana"
        for action in second.teacher_actions
        if action.tool == "RETRIEVE"
    )
    assert "Alice discussed apples" in second.example.input
    assert second.example.round_index == 3
    row = second.example.to_dict()
    row_text = json.dumps(row, ensure_ascii=False)
    assert "metadata" not in row
    assert "teacher_reflection" not in row_text
    assert "teacher_answer_score" not in row_text
    assert "trainable_teacher_source" not in row_text
    assert "banana" not in row["input"]


def test_online_self_distiller_keeps_repaired_inspect_raw_path() -> None:
    class InspectRawTeacher:
        def __init__(self):
            self.calls = 0

        def search(
            self,
            query,
            gold_answer,
            memory_store,
            question_image=None,
            initial_session=None,
        ):
            self.calls += 1
            session = initial_session or ExecutorSession(
                query=query,
                memory_store=memory_store,
                validator=validator,
            )
            observation = session.observation()
            session.execute_chunk(
                [ToolAction("RETRIEVE", {"method": "vision", "top_k": 1})]
            )
            retrieved_observation = session.observation()
            retrieve_decision = InteractiveDecision(
                observation=observation,
                observation_after=retrieved_observation,
                history=[],
                actions=[
                    ToolAction(
                        "RETRIEVE",
                        {"method": "vision", "top_k": 1},
                    )
                ],
                privileged_feedback={},
                verification_after=VerificationResult(
                    False, 0.2, 1.0, 0.0
                ),
                action_source="planner",
            )
            session.execute_chunk(
                [
                    ToolAction(
                        "INSPECT_RAW",
                        {
                            "target": "current_pool",
                            "instruction": (
                                "answer_query_related_visual_details"
                            ),
                        },
                    )
                ]
            )
            inspected_observation = session.observation()
            inspect_decision = InteractiveDecision(
                observation=retrieved_observation,
                observation_after=inspected_observation,
                history=list(retrieve_decision.actions),
                actions=[
                    ToolAction(
                        "INSPECT_RAW",
                        {
                            "target": "current_pool",
                            "instruction": (
                                "answer_query_related_visual_details"
                            ),
                        },
                    )
                ],
                privileged_feedback={},
                verification_after=VerificationResult(
                    True, 1.0, 1.0, 1.0
                ),
                action_source="planner_repaired",
            )
            stop_decision = InteractiveDecision(
                observation=inspected_observation,
                observation_after=inspected_observation,
                history=list(retrieve_decision.actions)
                + list(inspect_decision.actions),
                actions=[ToolAction("STOP")],
                privileged_feedback={},
                verification_after=VerificationResult(
                    True, 1.0, 1.0, 1.0
                ),
                action_source="answer_stop",
            )
            return type(
                "TeacherResult",
                (),
                {
                    "actions": retrieve_decision.actions
                    + inspect_decision.actions
                    + stop_decision.actions,
                    "decisions": [
                        retrieve_decision,
                        inspect_decision,
                        stop_decision,
                    ],
                    "verification": stop_decision.verification_after,
                    "answer_validation": AnswerValidationResult(
                        True, 1.0, "ok", "banana"
                    ),
                    "failure_diagnostics": [],
                    "planner_calls": 1,
                    "verifier_calls": 0,
                    "answer_validator_calls": 1,
                },
            )()

    validator = InteractiveActionValidator(allow_inspect_raw=True)
    buffer = OnlineDistillationBuffer()
    distiller = OnlineSelfDistiller(
        student_planner=OnlineStudentPlanner(),
        teacher_search=InspectRawTeacher(),
        answer_validator=StrictAnswerValidator(
            EvidenceEchoAnswer(),
            ContainsBananaJudge(),
        ),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
        max_student_rounds=1,
        buffer=buffer,
    )
    result = distiller.collect_sample(
        OPDSample(
            sample_id="sample",
            query="Which project did Bob select?",
            gold_answer="banana",
            memory_store=make_store(),
        ),
        round_index=4,
    )
    assert result.teacher_attempts[0]["selected_path_trainable"]
    assert len(result.corrections) == 3
    assert any(
        action.tool == "INSPECT_RAW"
        for correction in result.corrections
        for action in correction.teacher_actions
    )
    stop_corrections = [
        correction
        for correction in result.corrections
        if any(action.tool == "STOP" for action in correction.teacher_actions)
    ]
    assert len(stop_corrections) == 1
    assert stop_corrections[0].teacher_action_source == "answer_stop"
    assert any(
        correction.teacher_verification.get("answerable")
        for correction in result.corrections
    )


def test_online_student_metrics_split_gold_aware_and_no_gold() -> None:
    validator = InteractiveActionValidator()
    answer_model = WrongNoGoldAnswer()
    answer_validator = StrictAnswerValidator(
        answer_model,
        ContainsBananaJudge(),
    )
    teacher_search = InteractiveTeacherSearch(
        planner=CorrectingPlanner(),
        verifier=AlwaysSufficientVerifier(),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
        answer_validator=answer_validator,
        max_rounds=2,
        beam_size=1,
        candidates_per_node=1,
    )
    distiller = OnlineSelfDistiller(
        student_planner=OnlineStudentPlanner(),
        teacher_search=teacher_search,
        answer_validator=answer_validator,
        answer_model=answer_model,
        answer_judge=ContainsBananaJudge(),
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
        max_student_rounds=1,
    )
    result = distiller.collect_sample(
        OPDSample(
            sample_id="sample",
            query="Which project did Bob select?",
            gold_answer="banana",
            memory_store=make_store(),
        )
    )
    assert result.student_evidence_sufficiency.correct
    assert not result.student_answer_validation.correct
    row = result.to_dict()
    assert row["student_evidence_sufficiency"]["correct"]
    assert not row["student_answer_validation"]["correct"]
