"""Online self-distillation over student-visited interactive states."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

from .interactive import (
    AnswerValidationResult,
    ExecutorSession,
    InteractiveActionValidator,
    InteractiveTeacherSearch,
    StrictAnswerValidator,
    fallback_action_chunks,
)
from .models import OPDSample, SFTExample, ToolAction
from .retrieval import TurnAwareHybridRetriever


TRAINABLE_TEACHER_ACTION_SOURCES = {"planner"}
SYSTEM_STOP_ACTION_SOURCES = {"verifier_stop"}


def _is_trainable_teacher_path(decisions: List[Any]) -> bool:
    """Only distill trajectories whose non-stop actions came from the model."""
    if not decisions:
        return False
    for decision in decisions:
        if decision.action_source in TRAINABLE_TEACHER_ACTION_SOURCES:
            continue
        if (
            decision.action_source in SYSTEM_STOP_ACTION_SOURCES
            and all(action.tool == "STOP" for action in decision.actions)
        ):
            continue
        return False
    return True


class InteractivePlanner(Protocol):
    calls: int

    def propose(
        self,
        query: str,
        history: List[ToolAction],
        observation: Any,
        candidate_count: int = 1,
        privileged_feedback: Optional[Dict[str, Any]] = None,
    ) -> List[List[ToolAction]]:
        ...


@dataclass
class OnlineCorrection:
    sample_id: str
    state_index: int
    student_actions: List[ToolAction]
    teacher_actions: List[ToolAction]
    teacher_answer_validation: AnswerValidationResult
    example: SFTExample

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "state_index": self.state_index,
            "student_actions": [
                action.to_dict() for action in self.student_actions
            ],
            "teacher_actions": [
                action.to_dict() for action in self.teacher_actions
            ],
            "teacher_answer_validation": (
                self.teacher_answer_validation.to_dict()
            ),
            "example": self.example.to_dict(),
        }


@dataclass
class OnlineSampleResult:
    sample_id: str
    student_actions: List[ToolAction]
    student_answer_validation: AnswerValidationResult
    corrections: List[OnlineCorrection]
    student_planner_calls: int
    teacher_attempts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "student_actions": [
                action.to_dict() for action in self.student_actions
            ],
            "student_answer_validation": (
                self.student_answer_validation.to_dict()
            ),
            "corrections": [
                correction.to_dict() for correction in self.corrections
            ],
            "student_planner_calls": self.student_planner_calls,
            "teacher_attempts": self.teacher_attempts,
        }


@dataclass
class BufferedExample:
    example: SFTExample
    first_round: int
    last_round: int
    seen_count: int = 1


class OnlineDistillationBuffer:
    """Deduplicated replay buffer for strictly validated state corrections."""

    def __init__(self, max_examples: Optional[int] = None):
        self.max_examples = (
            max(1, int(max_examples))
            if max_examples is not None
            else None
        )
        self._items: Dict[str, BufferedExample] = {}

    @staticmethod
    def _key(example: SFTExample) -> str:
        value = f"{example.input}\n<target>\n{example.target}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def add(self, example: SFTExample, round_index: int) -> bool:
        key = self._key(example)
        current = self._items.get(key)
        if current is not None:
            current.last_round = round_index
            current.seen_count += 1
            return False
        self._items[key] = BufferedExample(
            example=example,
            first_round=round_index,
            last_round=round_index,
        )
        if (
            self.max_examples is not None
            and len(self._items) > self.max_examples
        ):
            oldest = min(
                self._items,
                key=lambda item_key: (
                    self._items[item_key].last_round,
                    self._items[item_key].first_round,
                ),
            )
            del self._items[oldest]
        return True

    def examples(self) -> List[SFTExample]:
        return [
            item.example
            for item in sorted(
                self._items.values(),
                key=lambda value: (
                    value.first_round,
                    value.example.sample_id,
                ),
            )
        ]

    def __len__(self) -> int:
        return len(self._items)

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for item in sorted(
                self._items.values(),
                key=lambda value: (
                    value.first_round,
                    value.example.sample_id,
                ),
            ):
                row = item.example.to_dict()
                row["buffer"] = {
                    "first_round": item.first_round,
                    "last_round": item.last_round,
                    "seen_count": item.seen_count,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


StudentUpdater = Callable[
    [InteractivePlanner, List[SFTExample], int],
    InteractivePlanner,
]


class OnlineSelfDistiller:
    """DAgger-style self-distillation on states visited by the student."""

    def __init__(
        self,
        student_planner: InteractivePlanner,
        teacher_search: InteractiveTeacherSearch,
        answer_validator: StrictAnswerValidator,
        validator: InteractiveActionValidator,
        retriever: Optional[TurnAwareHybridRetriever] = None,
        max_student_rounds: int = 3,
        max_student_actions: int = 9,
        buffer: Optional[OnlineDistillationBuffer] = None,
        raw_inspector: Optional[Any] = None,
        max_raw_inspections: int = 3,
    ):
        self.student_planner = student_planner
        self.teacher_search = teacher_search
        self.answer_validator = answer_validator
        self.validator = validator
        self.retriever = retriever or TurnAwareHybridRetriever()
        self.max_student_rounds = max(1, int(max_student_rounds))
        self.max_student_actions = max(1, int(max_student_actions))
        self.raw_inspector = raw_inspector
        self.max_raw_inspections = max(0, int(max_raw_inspections))
        self.buffer = (
            buffer if buffer is not None else OnlineDistillationBuffer()
        )

    def collect_sample(
        self,
        sample: OPDSample,
        round_index: int = 0,
    ) -> OnlineSampleResult:
        question_image = sample.metadata.get("question_image")
        session = ExecutorSession(
            query=sample.query,
            memory_store=sample.memory_store,
            validator=self.validator,
            retriever=self.retriever,
            question_image=question_image,
            raw_inspector=self.raw_inspector,
            max_raw_inspections=self.max_raw_inspections,
        )
        corrections: List[OnlineCorrection] = []
        teacher_attempts: List[Dict[str, Any]] = []
        calls_before = getattr(self.student_planner, "calls", 0)

        for state_index in range(self.max_student_rounds):
            if session.stopped or len(session.history) >= self.max_student_actions:
                break
            observation = session.observation()
            history = list(session.history)

            teacher_result = self.teacher_search.search(
                query=sample.query,
                gold_answer=sample.gold_answer,
                memory_store=sample.memory_store,
                question_image=question_image,
                initial_session=session,
            )
            decision_sources = [
                decision.action_source for decision in teacher_result.decisions
            ]
            decision_reflections = [
                decision.teacher_reflection
                for decision in teacher_result.decisions
            ]
            trainable_path = _is_trainable_teacher_path(
                teacher_result.decisions
            )
            teacher_attempts.append(
                {
                    "state_index": state_index,
                    "selected_actions": [
                        action.to_dict() for action in teacher_result.actions
                    ],
                    "selected_action_sources": decision_sources,
                    "selected_reflections": decision_reflections,
                    "selected_first_action_source": (
                        decision_sources[0] if decision_sources else None
                    ),
                    "selected_first_action_trainable": (
                        bool(decision_sources)
                        and decision_sources[0]
                        in TRAINABLE_TEACHER_ACTION_SOURCES
                    ),
                    "selected_path_trainable": trainable_path,
                    "verification": teacher_result.verification.to_dict(),
                    "answer_validation": (
                        teacher_result.answer_validation.to_dict()
                        if teacher_result.answer_validation is not None
                        else None
                    ),
                    "failure_diagnostics": (
                        teacher_result.failure_diagnostics
                    ),
                    "planner_calls": teacher_result.planner_calls,
                    "verifier_calls": teacher_result.verifier_calls,
                    "answer_validator_calls": (
                        teacher_result.answer_validator_calls
                    ),
                }
            )

            try:
                student_chunks = self.student_planner.propose(
                    query=sample.query,
                    history=session.history,
                    observation=observation,
                    candidate_count=1,
                    privileged_feedback=None,
                )
                student_actions = student_chunks[0]
            except Exception:
                fallback = fallback_action_chunks(
                    observation,
                    self.validator,
                    candidate_count=1,
                )
                student_actions = (
                    fallback[0] if fallback else [ToolAction("STOP")]
                )

            teacher_validation = teacher_result.answer_validation
            if (
                teacher_validation is not None
                and teacher_validation.correct
                and teacher_result.decisions
            ):
                decision = teacher_result.decisions[0]
                is_trainable_source = (
                    decision.action_source in TRAINABLE_TEACHER_ACTION_SOURCES
                    and trainable_path
                )
                example = decision.sft_example(
                    sample.sample_id,
                    state_index,
                    sample.query,
                    self.validator.schema_text(),
                )
                example.sample_id = (
                    f"{sample.sample_id}:online:{round_index}:{state_index}"
                )
                example.round_index = round_index
                example.metadata.update(
                    {
                        "online_state_index": state_index,
                        "student_actions": [
                            action.to_dict() for action in student_actions
                        ],
                        "teacher_answer_score": teacher_validation.score,
                        "teacher_action_source": decision.action_source,
                        "trainable_teacher_source": is_trainable_source,
                        "teacher_reflection": decision.teacher_reflection,
                    }
                )
                correction = OnlineCorrection(
                    sample_id=sample.sample_id,
                    state_index=state_index,
                    student_actions=list(student_actions),
                    teacher_actions=list(decision.actions),
                    teacher_answer_validation=teacher_validation,
                    example=example,
                )
                if is_trainable_source:
                    corrections.append(correction)
                    self.buffer.add(example, round_index)

            if len(session.history) + len(student_actions) > self.max_student_actions:
                break
            session.execute_chunk(student_actions)

        if not session.stopped:
            session.execute_chunk([ToolAction("STOP")])
        student_validation = self.answer_validator.evaluate(
            sample.query,
            sample.gold_answer,
            session.evidence,
            question_image=question_image,
        )
        return OnlineSampleResult(
            sample_id=sample.sample_id,
            student_actions=list(session.history),
            student_answer_validation=student_validation,
            corrections=corrections,
            student_planner_calls=(
                getattr(self.student_planner, "calls", 0) - calls_before
            ),
            teacher_attempts=teacher_attempts,
        )

    def collect_round(
        self,
        samples: Iterable[OPDSample],
        round_index: int = 0,
    ) -> List[OnlineSampleResult]:
        return [
            self.collect_sample(sample, round_index=round_index)
            for sample in samples
        ]

    def run_rounds(
        self,
        samples: Iterable[OPDSample],
        num_rounds: int,
        student_updater: Optional[StudentUpdater] = None,
    ) -> List[List[OnlineSampleResult]]:
        materialized = list(samples)
        all_rounds = []
        for round_index in range(max(1, int(num_rounds))):
            results = self.collect_round(
                materialized,
                round_index=round_index,
            )
            all_rounds.append(results)
            if student_updater is not None:
                self.student_planner = student_updater(
                    self.student_planner,
                    self.buffer.examples(),
                    round_index,
                )
        return all_rounds
