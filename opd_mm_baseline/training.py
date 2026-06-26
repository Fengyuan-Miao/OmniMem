"""On-policy rollout, hindsight correction, and SFT data generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

from .clients import build_student_prompt
from .executor import ToolExecutor
from .models import (
    ExecutionResult,
    OPDRollout,
    OPDSample,
    PolicyOutput,
    SFTExample,
    ToolAction,
)


class StudentPolicy(Protocol):
    def generate_trace(self, query: str) -> PolicyOutput:
        ...


class TeacherPolicy(Protocol):
    def correct(
        self,
        query: str,
        gold_answer: str,
        student_policy: PolicyOutput,
        student_answer: str,
        correct: bool,
        execution: Optional[Any] = None,
        privileged_context: Optional[dict[str, Any]] = None,
    ) -> PolicyOutput:
        ...


class AnswerModel(Protocol):
    def answer(
        self,
        query: str,
        evidence: list[Any],
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


class OnPolicyDistiller:
    def __init__(
        self,
        student: StudentPolicy,
        teacher: TeacherPolicy,
        executor: ToolExecutor,
        answer_model: AnswerModel,
        judge: AnswerJudge,
        teacher_feedback_rounds: int = 0,
        teacher_evidence_budget: int = 20,
    ):
        self.student = student
        self.teacher = teacher
        self.executor = executor
        self.answer_model = answer_model
        self.judge = judge
        self.teacher_feedback_rounds = max(0, int(teacher_feedback_rounds))
        self.teacher_evidence_budget = max(1, int(teacher_evidence_budget))

    @staticmethod
    def _execute_policy(
        executor: ToolExecutor,
        policy: PolicyOutput,
        sample: OPDSample,
    ) -> Optional[ExecutionResult]:
        try:
            return executor.run(
                trace=policy.actions,
                query=sample.query,
                memory_store=sample.memory_store,
                question_image=sample.metadata.get("question_image"),
            )
        except Exception:
            return None

    @staticmethod
    def _support_hits(
        execution: Optional[ExecutionResult],
        support_turn_ids: Iterable[str],
    ) -> List[str]:
        if execution is None:
            return []
        targets = set(support_turn_ids)
        hits = set()
        for item in execution.evidence:
            for turn_id in targets:
                if (
                    item.memory_id == turn_id
                    or item.memory_id.startswith(turn_id + ":")
                ):
                    hits.add(turn_id)
        return sorted(hits)

    @staticmethod
    def _support_record_hit_count(
        execution: Optional[ExecutionResult],
        sample: OPDSample,
        support_turn_ids: Iterable[str],
    ) -> tuple[int, int]:
        targets = set(support_turn_ids)
        target_memory_ids = {
            item.memory.memory_id
            for item in sample.memory_store.initial_pool()
            if item.memory.turn_id in targets
        }
        evidence_memory_ids = (
            {item.memory_id for item in execution.evidence}
            if execution is not None
            else set()
        )
        return (
            len(target_memory_ids & evidence_memory_ids),
            len(target_memory_ids),
        )

    @staticmethod
    def _oracle_policy(sample: OPDSample) -> Optional[PolicyOutput]:
        context = sample.metadata.get("teacher_privileged_context") or {}
        advice = context.get("verified_action_advice") or {}
        recommended = advice.get("recommended") or {}
        method = recommended.get("method")
        top_k = recommended.get("minimum_top_k")
        if method not in {"bm25", "dense", "hybrid"}:
            return None
        if not isinstance(top_k, int) or top_k <= 0:
            return None
        actions = [
            ToolAction("RETRIEVE", {"method": method, "top_k": top_k}),
            ToolAction(
                "READ",
                {
                    "fields": [
                        "summary",
                        "content",
                        "timestamp",
                        "turn_id",
                        "author",
                        "modality",
                        "source_type",
                    ]
                },
            ),
            ToolAction("STOP"),
        ]
        return PolicyOutput(
            actions=actions,
            raw_response=json.dumps(
                [action.to_dict() for action in actions],
                ensure_ascii=False,
            ),
        )

    def _select_teacher_policy(
        self,
        sample: OPDSample,
        model_policy: PolicyOutput,
        student_policy: PolicyOutput,
        student_answer: str,
        student_correct: bool,
        student_execution: ExecutionResult,
    ) -> tuple[
        PolicyOutput,
        Optional[ExecutionResult],
        List[Dict[str, Any]],
        str,
    ]:
        support_turn_ids = list(
            sample.metadata.get("gold_clue_turn_ids") or []
        )
        candidates: List[tuple[str, PolicyOutput]] = [
            ("llm_teacher", model_policy)
        ]
        privilege_mode = getattr(self.teacher, "privilege_mode", "")
        if privilege_mode == "oracle-feedback":
            previous_policy = model_policy
            for attempt_index in range(1, self.teacher_feedback_rounds + 1):
                previous_execution = self._execute_policy(
                    self.executor,
                    previous_policy,
                    sample,
                )
                support_hits = self._support_hits(
                    previous_execution,
                    support_turn_ids,
                )
                record_hits, record_count = self._support_record_hit_count(
                    previous_execution,
                    sample,
                    support_turn_ids,
                )
                evidence_count = (
                    len(previous_execution.evidence)
                    if previous_execution is not None
                    else 0
                )
                if (
                    record_count
                    and record_hits == record_count
                    and evidence_count <= self.teacher_evidence_budget
                ):
                    break
                revise = getattr(self.teacher, "revise", None)
                if revise is None:
                    break
                replay_feedback = {
                    "support_turns_covered": len(support_hits),
                    "support_turn_count": len(support_turn_ids),
                    "support_records_covered": record_hits,
                    "support_record_count": record_count,
                    "evidence_count": evidence_count,
                    "evidence_budget": self.teacher_evidence_budget,
                    "execution_error": (
                        previous_execution.error
                        if previous_execution is not None
                        else "replay_failed"
                    ),
                }
                revised_policy = revise(
                    query=sample.query,
                    gold_answer=sample.gold_answer,
                    student_policy=student_policy,
                    student_answer=student_answer,
                    correct=student_correct,
                    previous_policy=previous_policy,
                    replay_feedback=replay_feedback,
                    attempt_index=attempt_index,
                    execution=student_execution,
                    privileged_context=sample.metadata.get(
                        "teacher_privileged_context"
                    ),
                )
                candidates.append(
                    (f"llm_teacher_feedback_{attempt_index}", revised_policy)
                )
                previous_policy = revised_policy
        else:
            oracle_policy = self._oracle_policy(sample)
            if oracle_policy is not None:
                candidates.append(("oracle_action_advisor", oracle_policy))

        evaluated = []
        for source, policy in candidates:
            execution = self._execute_policy(self.executor, policy, sample)
            hits = self._support_hits(execution, support_turn_ids)
            record_hit_count, support_record_count = (
                self._support_record_hit_count(
                    execution,
                    sample,
                    support_turn_ids,
                )
            )
            evaluated.append(
                {
                    "source": source,
                    "policy": policy,
                    "execution": execution,
                    "support_hits": hits,
                    "support_hit_count": len(hits),
                    "support_record_hit_count": record_hit_count,
                    "support_record_count": support_record_count,
                    "evidence_count": (
                        len(execution.evidence) if execution is not None else 0
                    ),
                    "execution_error": (
                        execution.error if execution is not None else "replay_failed"
                    ),
                }
            )

        if support_turn_ids:
            selected = max(
                evaluated,
                key=lambda item: (
                    item["support_record_hit_count"],
                    item["support_hit_count"],
                    not bool(item["execution_error"]),
                    -item["evidence_count"],
                    item["source"] == "llm_teacher",
                ),
            )
        else:
            selected = evaluated[0]

        diagnostics = [
            {
                "source": item["source"],
                "support_hits": item["support_hits"],
                "support_hit_count": item["support_hit_count"],
                "support_turn_count": len(support_turn_ids),
                "support_record_hit_count": item["support_record_hit_count"],
                "support_record_count": item["support_record_count"],
                "evidence_count": item["evidence_count"],
                "execution_error": item["execution_error"],
                "actions": [
                    action.to_dict() for action in item["policy"].actions
                ],
                "selected": item is selected,
            }
            for item in evaluated
        ]
        return (
            selected["policy"],
            selected["execution"],
            diagnostics,
            selected["source"],
        )

    def rollout(self, sample: OPDSample, round_index: int = 0) -> OPDRollout:
        student_policy = self.student.generate_trace(sample.query)
        execution = self.executor.run(
            trace=student_policy.actions,
            query=sample.query,
            memory_store=sample.memory_store,
            question_image=sample.metadata.get("question_image"),
        )
        question_image = sample.metadata.get("question_image")
        try:
            student_answer = self.answer_model.answer(
                query=sample.query,
                evidence=execution.evidence,
                question_image=question_image,
            )
        except Exception as exc:
            student_answer = ""
            answer_error = str(exc)
        else:
            answer_error = ""
        try:
            correct, score, reason = self.judge.evaluate(
                sample.query,
                student_answer,
                sample.gold_answer,
            )
        except Exception as exc:
            correct, score, reason = False, 0.0, f"judge_error: {exc}"
        teacher_model_policy = self.teacher.correct(
            query=sample.query,
            gold_answer=sample.gold_answer,
            student_policy=student_policy,
            student_answer=student_answer,
            correct=correct,
            execution=execution,
            privileged_context=sample.metadata.get("teacher_privileged_context"),
        )
        (
            teacher_policy,
            teacher_execution,
            teacher_candidate_diagnostics,
            teacher_selection_source,
        ) = self._select_teacher_policy(
            sample,
            teacher_model_policy,
            student_policy,
            student_answer,
            correct,
            execution,
        )
        validator = getattr(self.student, "validator", None)
        tool_schema = validator.schema_text() if validator is not None else None
        target = json.dumps(
            [action.to_dict() for action in teacher_policy.actions],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        sft_example = SFTExample(
            sample_id=sample.sample_id,
            input=build_student_prompt(sample.query, tool_schema),
            target=target,
            round_index=round_index,
            metadata={
                "student_correct": correct,
                "student_score": score,
                "student_policy_error": student_policy.error,
                "teacher_policy_error": teacher_policy.error,
                "teacher_execution_error": (
                    teacher_execution.error if teacher_execution is not None else ""
                ),
                "teacher_selection_source": teacher_selection_source,
                "teacher_candidate_diagnostics": teacher_candidate_diagnostics,
            },
        )
        metadata = dict(sample.metadata)
        metadata["teacher_selection_source"] = teacher_selection_source
        if answer_error:
            metadata["answer_error"] = answer_error
        return OPDRollout(
            sample_id=sample.sample_id,
            query=sample.query,
            gold_answer=sample.gold_answer,
            student_policy=student_policy,
            execution=execution,
            student_answer=student_answer,
            correct=correct,
            score=score,
            evaluation_reason=reason,
            teacher_policy=teacher_policy,
            teacher_execution=teacher_execution,
            sft_example=sft_example,
            metadata=metadata,
            teacher_model_policy=teacher_model_policy,
            teacher_candidate_diagnostics=teacher_candidate_diagnostics,
        )

    def run_round(
        self,
        samples: Iterable[OPDSample],
        round_index: int = 0,
    ) -> List[OPDRollout]:
        return [
            self.rollout(sample, round_index=round_index)
            for sample in samples
        ]

    def run_rounds(
        self,
        samples: Iterable[OPDSample],
        num_rounds: int = 3,
        student_updater: Optional[
            Callable[[StudentPolicy, List[SFTExample], int], StudentPolicy]
        ] = None,
    ) -> List[OPDRollout]:
        materialized = list(samples)
        all_rollouts: List[OPDRollout] = []
        accumulated: List[SFTExample] = []
        for round_index in range(max(1, int(num_rounds))):
            rollouts = self.run_round(materialized, round_index=round_index)
            all_rollouts.extend(rollouts)
            accumulated.extend(rollout.sft_example for rollout in rollouts)
            if student_updater is not None:
                self.student = student_updater(
                    self.student,
                    list(accumulated),
                    round_index,
                )
        return all_rollouts


def write_rollouts(path: str | Path, rollouts: Iterable[OPDRollout]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(json.dumps(rollout.to_dict(), ensure_ascii=False) + "\n")


def write_sft_examples(path: str | Path, examples: Iterable[SFTExample]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
