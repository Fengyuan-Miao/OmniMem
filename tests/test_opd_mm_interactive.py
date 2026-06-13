from __future__ import annotations

import pytest

from opd_mm_baseline.interactive import (
    ChatGoldEvidenceVerifier,
    ChatInteractivePlanner,
    ExecutorSession,
    InteractiveActionValidator,
    InteractivePolicyRunner,
    InteractiveTeacherSearch,
    InteractiveValidationError,
    VerificationResult,
)
from opd_mm_baseline.memgallery_interactive_pipeline import (
    _select_sft_trajectory,
)
from opd_mm_baseline.models import EvidenceItem, MemoryRecord, ToolAction
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
    assert "banana" in str(after.evidence_previews).lower()


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


def test_teacher_search_selects_answerable_branch_without_sft_privilege() -> None:
    planner = FakePlanner()
    verifier = FakeGoldVerifier()
    validator = InteractiveActionValidator()
    result = InteractiveTeacherSearch(
        planner=planner,
        verifier=verifier,
        validator=validator,
        retriever=TurnAwareHybridRetriever(context_window=0),
        max_rounds=2,
        beam_size=2,
        candidates_per_node=2,
    ).search(
        query="Which project did Bob select?",
        gold_answer="SECRET_GOLD banana",
        memory_store=make_store(),
    )
    assert result.verification.answerable
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
    def __init__(self) -> None:
        self.prompt = ""

    def complete(self, messages, max_tokens=512, temperature=0.0):
        self.prompt = messages[0]["content"]
        return (
            '{"candidates":[[{"tool":"RETRIEVE","method":"bm25",'
            '"top_k":2,"query":"banana"}]]}'
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


def test_chat_planner_reserves_evidence_branch_for_visible_pool() -> None:
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
    assert any(
        action.tool == "READ"
        for candidate in candidates
        for action in candidate
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
