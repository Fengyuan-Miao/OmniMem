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
    verification_from_answer_validation,
    fallback_action_chunks,
)
from .models import OPDSample, SFTExample, ToolAction
from .retrieval import TurnAwareHybridRetriever


TRAINABLE_TEACHER_ACTION_SOURCES = {"planner", "planner_repaired"}
SYSTEM_STOP_ACTION_SOURCES = {"answer_stop", "verifier_stop"}


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


def _is_pure_stop_decision(decision: Any) -> bool:
    return bool(decision.actions) and all(
        action.tool == "STOP" for action in decision.actions
    )


def _is_distillable_teacher_decision(decision: Any) -> bool:
    if decision.action_source in TRAINABLE_TEACHER_ACTION_SOURCES:
        return True
    return (
        decision.action_source in SYSTEM_STOP_ACTION_SOURCES
        and _is_pure_stop_decision(decision)
    )


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


class AnswerGenerator(Protocol):
    def answer(
        self,
        query: str,
        evidence: List[Any],
        question_image: Optional[str] = None,
    ) -> str:
        ...


class AnswerJudge(Protocol):
    def evaluate(
        self,
        query: str,
        prediction: str,
        gold_answer: str,
    ) -> tuple[bool, float, str]:
        ...


@dataclass
class OnlineCorrection:
    sample_id: str
    state_index: int
    student_actions: List[ToolAction]
    teacher_actions: List[ToolAction]
    teacher_answer_validation: AnswerValidationResult
    example: SFTExample
    student_raw_response: str = ""
    teacher_action_source: str = ""
    teacher_verification: Optional[Dict[str, Any]] = None
    trigger_verification: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "state_index": self.state_index,
            "student_actions": [
                action.to_dict() for action in self.student_actions
            ],
            "student_raw_response": self.student_raw_response,
            "teacher_actions": [
                action.to_dict() for action in self.teacher_actions
            ],
            "teacher_answer_validation": (
                self.teacher_answer_validation.to_dict(
                    include_reason=False
                )
            ),
            "teacher_action_source": self.teacher_action_source,
            "teacher_verification": self.teacher_verification or {},
            "trigger_verification": self.trigger_verification or {},
            "example": self.example.to_dict(),
        }


@dataclass
class OnlineSampleResult:
    sample_id: str
    student_actions: List[ToolAction]
    student_evidence_sufficiency: AnswerValidationResult
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
            "student_evidence_sufficiency": (
                self.student_evidence_sufficiency.to_dict(
                    include_reason=False
                )
            ),
            "student_answer_validation": (
                self.student_answer_validation.to_dict(
                    include_reason=False
                )
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
                row = item.example.to_dict(include_metadata=True)
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
        answer_model: Optional[AnswerGenerator] = None,
        answer_judge: Optional[AnswerJudge] = None,
        retriever: Optional[TurnAwareHybridRetriever] = None,
        max_student_rounds: int = 3,
        max_student_actions: int = 9,
        buffer: Optional[OnlineDistillationBuffer] = None,
        raw_inspector: Optional[Any] = None,
        max_raw_inspections: int = 3,
        teacher_trigger: str = "failure",
        stop_when_student_evidence_sufficient: bool = False,
    ):
        self.student_planner = student_planner
        self.teacher_search = teacher_search
        self.answer_validator = answer_validator
        self.answer_model = answer_model
        self.answer_judge = answer_judge
        self.validator = validator
        self.retriever = retriever or TurnAwareHybridRetriever()
        self.max_student_rounds = max(1, int(max_student_rounds))
        self.max_student_actions = max(1, int(max_student_actions))
        self.raw_inspector = raw_inspector
        self.max_raw_inspections = max(0, int(max_raw_inspections))
        self.stop_when_student_evidence_sufficient = bool(
            stop_when_student_evidence_sufficient
        )
        if teacher_trigger not in {"failure", "always"}:
            raise ValueError(f"invalid teacher_trigger: {teacher_trigger}")
        self.teacher_trigger = teacher_trigger
        self.buffer = (
            buffer if buffer is not None else OnlineDistillationBuffer()
        )

    def _append_teacher_corrections(
        self,
        *,
        sample: OPDSample,
        question_image: Optional[str],
        session: ExecutorSession,
        state_index: int,
        round_index: int,
        student_actions: List[ToolAction],
        student_raw_response: str,
        student_evidence_after: AnswerValidationResult,
        corrections: List[OnlineCorrection],
        teacher_attempts: List[Dict[str, Any]],
    ) -> None:
        trigger_verification = verification_from_answer_validation(
            student_evidence_after,
            session.evidence,
            can_inspect_raw=(
                self.validator.allow_inspect_raw
                and self.raw_inspector is not None
            ),
        )
        trigger_feedback = trigger_verification.planner_feedback()
        teacher_result = self.teacher_search.search(
            query=sample.query,
            gold_answer=sample.gold_answer,
            memory_store=sample.memory_store,
            question_image=question_image,
            initial_session=session,
            initial_verification=trigger_verification,
        )
        decision_sources = [
            decision.action_source for decision in teacher_result.decisions
        ]
        decision_reflections = [
            dict(decision.planner_rationale)
            for decision in teacher_result.decisions
        ]
        trainable_path = _is_trainable_teacher_path(
            teacher_result.decisions
        )
        selected_evidence = getattr(
            getattr(teacher_result, "execution", None),
            "evidence",
            [],
        )
        teacher_attempts.append(
            {
                "state_index": state_index,
                "trigger": self.teacher_trigger,
                "student_actions_before_correction": [
                    action.to_dict() for action in student_actions
                ],
                "selected_actions": [
                    action.to_dict() for action in teacher_result.actions
                ],
                "selected_action_count": len(teacher_result.actions),
                "selected_evidence_count": len(selected_evidence),
                "selected_action_sources": decision_sources,
                "selected_reflections": decision_reflections,
                "selected_first_action_source": (
                    decision_sources[0] if decision_sources else None
                ),
                "selected_first_action_trainable": (
                    bool(decision_sources)
                    and decision_sources[0] in TRAINABLE_TEACHER_ACTION_SOURCES
                ),
                "selected_path_trainable": trainable_path,
                "verification": teacher_result.verification.to_dict(),
                "trigger_verification": trigger_verification.to_dict(),
                "trigger_feedback": trigger_feedback,
                "answer_validation": (
                    teacher_result.answer_validation.to_dict(
                        include_reason=False
                    )
                    if teacher_result.answer_validation is not None
                    else None
                ),
                "failure_diagnostics": teacher_result.failure_diagnostics,
                "planner_calls": teacher_result.planner_calls,
                "verifier_calls": teacher_result.verifier_calls,
                "answer_validator_calls": (
                    teacher_result.answer_validator_calls
                ),
            }
        )

        teacher_validation = teacher_result.answer_validation
        if (
            teacher_validation is None
            or not teacher_validation.correct
            or not teacher_result.decisions
            or not trainable_path
        ):
            return
        for decision_index, decision in enumerate(teacher_result.decisions):
            if not _is_distillable_teacher_decision(decision):
                continue
            example = decision.sft_example(
                sample.sample_id,
                state_index,
                sample.query,
                self.validator.schema_text(),
                allow_inspect_raw=self.validator.allow_inspect_raw,
            )
            example.sample_id = (
                f"{sample.sample_id}:online:{round_index}:"
                f"{state_index}:{decision_index}"
            )
            example.round_index = round_index
            example.metadata.update(
                {
                    "online_state_index": state_index,
                    "teacher_decision_index": decision_index,
                    "teacher_trigger": self.teacher_trigger,
                    "trigger_feedback": trigger_feedback,
                }
            )
            correction = OnlineCorrection(
                sample_id=sample.sample_id,
                state_index=state_index,
                student_actions=list(student_actions),
                student_raw_response=student_raw_response,
                teacher_actions=list(decision.actions),
                teacher_answer_validation=teacher_validation,
                example=example,
                teacher_action_source=decision.action_source,
                teacher_verification=decision.verification_after.to_dict(),
                trigger_verification=trigger_verification.to_dict(),
            )
            corrections.append(correction)
            self.buffer.add(example, round_index)

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
            try:
                student_actions = self.validator.repair(student_actions)
            except Exception:
                fallback = fallback_action_chunks(
                    observation,
                    self.validator,
                    candidate_count=1,
                )
                student_actions = (
                    fallback[0] if fallback else [ToolAction("STOP")]
                )
            student_raw_response = str(
                getattr(self.student_planner, "last_raw_response", "") or ""
            ).strip()

            if len(session.history) + len(student_actions) > self.max_student_actions:
                break
            correction_session = session
            if all(action.tool == "STOP" for action in student_actions):
                student_evidence_after = self.answer_validator.evaluate(
                    sample.query,
                    sample.gold_answer,
                    session.evidence,
                    question_image=question_image,
                )
                should_correct = (
                    self.teacher_trigger == "always"
                    or not student_evidence_after.correct
                )
                if should_correct:
                    self._append_teacher_corrections(
                        sample=sample,
                        question_image=question_image,
                        session=correction_session,
                        state_index=state_index,
                        round_index=round_index,
                        student_actions=student_actions,
                        student_raw_response=student_raw_response,
                        student_evidence_after=student_evidence_after,
                        corrections=corrections,
                        teacher_attempts=teacher_attempts,
                    )
                session.execute_chunk(student_actions)
                break

            session.execute_chunk(student_actions)
            student_evidence_after = self.answer_validator.evaluate(
                sample.query,
                sample.gold_answer,
                session.evidence,
                question_image=question_image,
            )
            if (
                self.stop_when_student_evidence_sufficient
                and student_evidence_after.correct
            ):
                session.execute_chunk([ToolAction("STOP")])
                break
            should_correct = (
                self.teacher_trigger == "always"
                or not student_evidence_after.correct
            )
            if should_correct:
                self._append_teacher_corrections(
                    sample=sample,
                    question_image=question_image,
                    session=session,
                    state_index=state_index,
                    round_index=round_index,
                    student_actions=student_actions,
                    student_raw_response=student_raw_response,
                    student_evidence_after=student_evidence_after,
                    corrections=corrections,
                    teacher_attempts=teacher_attempts,
                )

        if not session.stopped:
            session.execute_chunk([ToolAction("STOP")])
        student_sufficiency = self.answer_validator.evaluate(
            sample.query,
            sample.gold_answer,
            session.evidence,
            question_image=question_image,
        )
        student_validation = self._evaluate_student_answer_no_gold(
            sample,
            session,
            question_image=question_image,
        )
        return OnlineSampleResult(
            sample_id=sample.sample_id,
            student_actions=list(session.history),
            student_evidence_sufficiency=student_sufficiency,
            student_answer_validation=student_validation,
            corrections=corrections,
            student_planner_calls=(
                getattr(self.student_planner, "calls", 0) - calls_before
            ),
            teacher_attempts=teacher_attempts,
        )

    def _evaluate_student_answer_no_gold(
        self,
        sample: OPDSample,
        session: ExecutorSession,
        question_image: Optional[str] = None,
    ) -> AnswerValidationResult:
        if not session.evidence:
            return AnswerValidationResult(
                correct=False,
                score=0.0,
                prediction="",
                reason="No retrieved evidence was provided.",
            )
        if self.answer_model is None or self.answer_judge is None:
            return self.answer_validator.evaluate(
                sample.query,
                sample.gold_answer,
                session.evidence,
                question_image=question_image,
            )
        try:
            prediction = self.answer_model.answer(
                sample.query,
                session.evidence,
                question_image=question_image,
            )
            correct, score, reason = self.answer_judge.evaluate(
                sample.query,
                prediction,
                sample.gold_answer,
            )
            score = max(0.0, min(1.0, float(score)))
            return AnswerValidationResult(
                correct=bool(correct),
                score=score,
                prediction=prediction,
                reason=str(reason or ""),
            )
        except Exception as exc:
            return AnswerValidationResult(
                correct=False,
                score=0.0,
                prediction="",
                error=str(exc),
                reason="No-gold answer validation failed.",
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
