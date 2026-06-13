"""Data models for the OPD-MM baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    turn_id: str
    timestamp: str
    author: str
    modality: str
    source_type: str
    summary: str = ""
    content: str = ""
    ocr: str = ""
    raw_pointer: Optional[str] = None
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        return " ".join(
            value
            for value in [self.summary, self.content, self.ocr]
            if value
        )

    def field_value(self, field_name: str) -> Any:
        if hasattr(self, field_name):
            return getattr(self, field_name)
        return self.metadata.get(field_name)

    def to_dict(self, include_internal_id: bool = True) -> Dict[str, Any]:
        data = {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "author": self.author,
            "modality": self.modality,
            "source_type": self.source_type,
            "summary": self.summary,
            "content": self.content or None,
            "ocr": self.ocr or None,
            "raw_pointer": self.raw_pointer,
            "status": self.status,
            "metadata": self.metadata,
        }
        if include_internal_id:
            data["memory_id"] = self.memory_id
        return data


@dataclass
class ToolAction:
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ToolAction":
        return cls(
            tool=str(value.get("tool") or "").upper(),
            arguments={key: item for key, item in value.items() if key != "tool"},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, **self.arguments}


@dataclass
class PolicyOutput:
    actions: List[ToolAction]
    raw_response: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "raw_response": self.raw_response,
            "error": self.error,
        }


@dataclass
class PoolItem:
    memory: MemoryRecord
    score: float = 0.0


@dataclass
class EvidenceItem:
    memory_id: str
    fields: Dict[str, Any] = field(default_factory=dict)
    source: str = "READ"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "source": self.source,
            **self.fields,
        }


@dataclass
class ExecutionStep:
    index: int
    action: ToolAction
    pool_before: int
    pool_after: int
    evidence_added: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action.to_dict(),
            "pool_before": self.pool_before,
            "pool_after": self.pool_after,
            "evidence_added": self.evidence_added,
            "error": self.error,
        }


@dataclass
class ExecutionResult:
    evidence: List[EvidenceItem]
    steps: List[ExecutionStep]
    final_pool_size: int
    final_memory_ids: List[str]
    stopped: bool
    error: str = ""
    raw_inspection_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "steps": [step.to_dict() for step in self.steps],
            "final_pool_size": self.final_pool_size,
            "final_memory_ids": self.final_memory_ids,
            "stopped": self.stopped,
            "error": self.error,
            "raw_inspection_calls": self.raw_inspection_calls,
        }


@dataclass
class OPDSample:
    sample_id: str
    query: str
    gold_answer: str
    memory_store: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SFTExample:
    sample_id: str
    input: str
    target: str
    round_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "input": self.input,
            "target": self.target,
            "round_index": self.round_index,
            "metadata": self.metadata,
        }


@dataclass
class OPDRollout:
    sample_id: str
    query: str
    gold_answer: str
    student_policy: PolicyOutput
    execution: ExecutionResult
    student_answer: str
    correct: bool
    score: float
    teacher_policy: PolicyOutput
    teacher_execution: Optional[ExecutionResult]
    sft_example: SFTExample
    evaluation_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    teacher_model_policy: Optional[PolicyOutput] = None
    teacher_candidate_diagnostics: List[Dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "gold_answer": self.gold_answer,
            "student_policy": self.student_policy.to_dict(),
            "execution": self.execution.to_dict(),
            "student_answer": self.student_answer,
            "correct": self.correct,
            "score": self.score,
            "evaluation_reason": self.evaluation_reason,
            "teacher_policy": self.teacher_policy.to_dict(),
            "teacher_execution": (
                self.teacher_execution.to_dict()
                if self.teacher_execution is not None
                else None
            ),
            "teacher_model_policy": (
                self.teacher_model_policy.to_dict()
                if self.teacher_model_policy is not None
                else None
            ),
            "teacher_candidate_diagnostics": self.teacher_candidate_diagnostics,
            "sft_example": self.sft_example.to_dict(),
            "metadata": self.metadata,
        }
