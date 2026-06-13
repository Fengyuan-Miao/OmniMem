"""Executor for validated OPD-MM tool trajectories."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Protocol

from .models import (
    EvidenceItem,
    ExecutionResult,
    ExecutionStep,
    PoolItem,
    ToolAction,
)
from .retrieval import HiddenMemoryStore, HybridRetriever
from .schema import TrajectoryValidator


class RawInspector(Protocol):
    def inspect(
        self,
        image_path: str,
        query: str,
        question_image: Optional[str] = None,
        text_context: Optional[str] = None,
    ) -> str:
        ...


class ToolExecutor:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        raw_inspector: Optional[RawInspector] = None,
        validator: Optional[TrajectoryValidator] = None,
        max_raw_inspections: int = 3,
    ):
        self.retriever = retriever or HybridRetriever()
        self.raw_inspector = raw_inspector
        self.validator = validator or TrajectoryValidator()
        self.max_raw_inspections = max(0, int(max_raw_inspections))

    def run(
        self,
        trace: List[ToolAction] | List[dict[str, Any]],
        query: str,
        memory_store: HiddenMemoryStore,
        question_image: Optional[str] = None,
    ) -> ExecutionResult:
        actions = self.validator.validate(trace)
        pool = memory_store.initial_pool()
        evidence: List[EvidenceItem] = []
        steps: List[ExecutionStep] = []
        stopped = False
        raw_calls = 0
        error = ""

        for index, action in enumerate(actions):
            before = len(pool)
            evidence_before = len(evidence)
            step_error = ""
            try:
                if action.tool == "FILTER":
                    pool = self._filter(pool, **action.arguments)
                elif action.tool == "SORT":
                    pool = self._sort(pool, **action.arguments)
                elif action.tool == "TOPK":
                    pool = self._topk_turns(pool, action.arguments["k"])
                elif action.tool == "RETRIEVE":
                    pool = self.retriever.retrieve(
                        pool,
                        query=query,
                        store=memory_store,
                        method=action.arguments.get("method", "hybrid"),
                        top_k=action.arguments.get("top_k", 5),
                        question_image=question_image,
                    )
                elif action.tool == "READ":
                    evidence.extend(self._read(pool, action.arguments["fields"]))
                elif action.tool == "INSPECT_RAW":
                    remaining = max(0, self.max_raw_inspections - raw_calls)
                    inspected = self._inspect_raw(
                        pool,
                        query,
                        remaining,
                        question_image=question_image,
                    )
                    raw_calls += len(inspected)
                    evidence.extend(inspected)
                elif action.tool == "STOP":
                    stopped = True
            except Exception as exc:
                step_error = str(exc)
                error = f"action {index} {action.tool}: {exc}"
            steps.append(
                ExecutionStep(
                    index=index,
                    action=action,
                    pool_before=before,
                    pool_after=len(pool),
                    evidence_added=len(evidence) - evidence_before,
                    error=step_error,
                )
            )
            if stopped or step_error:
                break

        return ExecutionResult(
            evidence=evidence,
            steps=steps,
            final_pool_size=len(pool),
            final_memory_ids=[item.memory.memory_id for item in pool],
            stopped=stopped,
            error=error,
            raw_inspection_calls=raw_calls,
        )

    @staticmethod
    def _topk_turns(pool: List[PoolItem], k: int) -> List[PoolItem]:
        selected_turns = []
        selected = []
        for item in pool:
            turn_id = item.memory.turn_id
            if turn_id not in selected_turns:
                if len(selected_turns) >= k:
                    continue
                selected_turns.append(turn_id)
            selected.append(item)
        return selected

    @staticmethod
    def _filter(
        pool: List[PoolItem],
        field: str,
        op: str,
        value: Any,
    ) -> List[PoolItem]:
        target = str(value).lower()

        def keep(item: PoolItem) -> bool:
            current_value = item.memory.field_value(field)
            current = str(current_value or "").lower()
            if op == "eq":
                return current == target
            if op == "neq":
                return current != target
            if op == "contains":
                return target in current
            if op == "before":
                return current < target
            if op == "after":
                return current > target
            return False

        return [item for item in pool if keep(item)]

    @classmethod
    def _sort(
        cls,
        pool: List[PoolItem],
        field: str,
        order: str,
    ) -> List[PoolItem]:
        reverse = order == "desc"
        if field == "score":
            key = lambda item: item.score
        elif field == "turn_id":
            key = lambda item: cls._natural_key(item.memory.turn_id)
        else:
            key = lambda item: str(item.memory.field_value(field) or "")
        return sorted(pool, key=key, reverse=reverse)

    @staticmethod
    def _natural_key(value: str) -> tuple[Any, ...]:
        return tuple(
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value or ""))
        )

    @staticmethod
    def _read(pool: List[PoolItem], fields: List[str]) -> List[EvidenceItem]:
        evidence = []
        for item in pool:
            values = {
                field: item.memory.field_value(field)
                for field in fields
            }
            if item.score:
                values["retrieval_score"] = item.score
            evidence.append(
                EvidenceItem(
                    memory_id=item.memory.memory_id,
                    fields=values,
                    source="READ",
                )
            )
        return evidence

    def _inspect_raw(
        self,
        pool: List[PoolItem],
        query: str,
        limit: int,
        question_image: Optional[str] = None,
    ) -> List[EvidenceItem]:
        if self.raw_inspector is None:
            return []
        evidence = []
        text_by_turn = self._text_context_by_turn(pool)
        for item in pool:
            if len(evidence) >= limit:
                break
            pointer = item.memory.raw_pointer
            if not pointer:
                continue
            context = text_by_turn.get(item.memory.turn_id, "")
            observation = self.raw_inspector.inspect(
                pointer,
                query,
                question_image=question_image,
                text_context=context,
            )
            evidence.append(
                EvidenceItem(
                    memory_id=item.memory.memory_id,
                    fields={
                        "visual_observation": observation,
                        "linked_text_context": context,
                        "image_label": (
                            f"turn={item.memory.turn_id}; "
                            f"context={context[:220]}"
                        ),
                        "timestamp": item.memory.timestamp,
                        "turn_id": item.memory.turn_id,
                        "raw_pointer": pointer,
                    },
                    source="INSPECT_RAW",
                )
            )
        return evidence

    @staticmethod
    def _text_context_by_turn(pool: List[PoolItem]) -> dict[str, str]:
        contexts: dict[str, List[str]] = {}
        for item in pool:
            memory = item.memory
            if memory.raw_pointer:
                continue
            text = " ".join(
                value
                for value in [memory.summary, memory.content, memory.ocr]
                if value
            ).strip()
            if not text:
                continue
            contexts.setdefault(memory.turn_id, []).append(text)
        return {
            turn_id: " ".join(values)[:1200]
            for turn_id, values in contexts.items()
        }
