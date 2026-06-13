"""Interactive chunked policy search for OPD-MM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .clients import OpenAICompatibleClient, extract_json_object
from .executor import ToolExecutor
from .models import (
    EvidenceItem,
    ExecutionResult,
    ExecutionStep,
    PoolItem,
    SFTExample,
    ToolAction,
)
from .retrieval import HiddenMemoryStore, TurnAwareHybridRetriever
from .schema import (
    FILTER_FIELDS,
    FILTER_OPS,
    INSPECT_INSTRUCTIONS,
    INSPECT_TARGETS,
    MEMORY_ID_PATTERN,
    READ_FIELDS,
    RETRIEVAL_METHODS,
    SORT_FIELDS,
    SORT_ORDERS,
)


POOL_MUTATING_TOOLS = {
    "FILTER",
    "SORT",
    "TOPK",
    "RETRIEVE",
    "EXPAND_NEIGHBORS",
}
INTERACTIVE_TOOLS = POOL_MUTATING_TOOLS | {"READ", "INSPECT_RAW", "STOP"}
RETRIEVE_SCOPES = {"all", "current"}
PUBLIC_IMAGE_ID_PATTERN = re.compile(r"\bD\d+:IMG_\d+\b")
DEFAULT_READ_FIELDS = [
    "summary",
    "content",
    "ocr",
    "timestamp",
    "turn_id",
    "author",
    "modality",
    "source_type",
    "raw_pointer",
]


def _evidence_mentions_visual_content(item: EvidenceItem) -> bool:
    if item.source == "INSPECT_RAW":
        return True
    fields = item.fields
    if fields.get("raw_pointer") or fields.get("visual_observation"):
        return True
    modality_values = fields.get("modality", [])
    source_values = fields.get("source_type", [])
    values = (
        modality_values
        if isinstance(modality_values, list)
        else [modality_values]
    ) + (
        source_values if isinstance(source_values, list) else [source_values]
    )
    return any(
        token in str(value).lower()
        for value in values
        for token in ("image", "visual", "photo", "uploaded")
    )


def build_interactive_schema(allow_inspect_raw: bool = False) -> str:
    lines = [
        "Allowed executable tools:",
        "RETRIEVE(method=bm25|dense|hybrid, top_k=positive integer,",
        "         query=optional rewritten retrieval query, scope=all|current)",
        "FILTER(field=modality|author|source_type|timestamp|status,",
        "       op=eq|neq|before|after|contains, value=...)",
        "SORT(field=timestamp|turn_id|score, order=asc|desc)",
        "TOPK(k=positive integer)",
        "EXPAND_NEIGHBORS(window=1|2|3)",
        "READ(fields=[summary|content|ocr|timestamp|turn_id|author|modality|source_type|raw_pointer])",
    ]
    if allow_inspect_raw:
        lines.extend(
            [
                "INSPECT_RAW(target=current_pool,",
                "            instruction=answer_query_related_visual_details)",
            ]
        )
    lines.extend(
        [
            "STOP()",
            "",
            "Return one short JSON array action chunk. RETRIEVE may rewrite the",
            "query using only the user request and observed evidence. Never emit",
            "memory IDs or file paths. Pool-changing actions must occur before READ.",
        ]
    )
    return "\n".join(lines)


class InteractiveValidationError(ValueError):
    pass


class InteractiveActionValidator:
    def __init__(
        self,
        max_chunk_actions: int = 3,
        max_top_k: int = 50,
        max_query_chars: int = 300,
        allow_inspect_raw: bool = False,
    ):
        self.max_chunk_actions = max(1, int(max_chunk_actions))
        self.max_top_k = max(1, int(max_top_k))
        self.max_query_chars = max(20, int(max_query_chars))
        self.allow_inspect_raw = bool(allow_inspect_raw)

    def schema_text(self) -> str:
        return build_interactive_schema(self.allow_inspect_raw)

    def validate(
        self,
        values: Iterable[Dict[str, Any] | ToolAction],
    ) -> List[ToolAction]:
        actions = [
            value if isinstance(value, ToolAction) else ToolAction.from_dict(value)
            for value in values
        ]
        if not actions:
            raise InteractiveValidationError("action chunk is empty")
        if len(actions) > self.max_chunk_actions:
            raise InteractiveValidationError(
                f"chunk has {len(actions)} actions; maximum is "
                f"{self.max_chunk_actions}"
            )
        seen_read = False
        for index, action in enumerate(actions):
            self._validate_action(action, index)
            if action.tool == "STOP" and index != len(actions) - 1:
                raise InteractiveValidationError("STOP must be final in a chunk")
            if seen_read and action.tool in POOL_MUTATING_TOOLS:
                raise InteractiveValidationError(
                    f"action {index}: {action.tool} after READ is a dead action"
                )
            if action.tool in {"READ", "INSPECT_RAW"}:
                seen_read = True
        return actions

    def repair(
        self,
        values: Iterable[Dict[str, Any] | ToolAction],
    ) -> List[ToolAction]:
        """Remove harmless schema drift without inventing semantic actions."""
        repaired: List[ToolAction] = []
        for value in values:
            action = (
                value
                if isinstance(value, ToolAction)
                else ToolAction.from_dict(value)
            )
            args = action.arguments
            if action.tool == "RETRIEVE":
                top_k = args.get("top_k", 5)
                if not isinstance(top_k, int) or isinstance(top_k, bool):
                    top_k = 5
                repaired_args: Dict[str, Any] = {
                    "method": (
                        args.get("method")
                        if args.get("method") in RETRIEVAL_METHODS
                        else "hybrid"
                    ),
                    "top_k": min(max(1, top_k), self.max_top_k),
                }
                query = args.get("query")
                if isinstance(query, str) and query.strip():
                    repaired_args["query"] = query.strip()[
                        : self.max_query_chars
                    ]
                if args.get("scope") in RETRIEVE_SCOPES:
                    repaired_args["scope"] = args["scope"]
                repaired.append(ToolAction("RETRIEVE", repaired_args))
            elif action.tool == "READ":
                fields = args.get("fields")
                if not isinstance(fields, list):
                    fields = DEFAULT_READ_FIELDS
                valid_fields = [
                    field
                    for field in fields
                    if isinstance(field, str) and field in READ_FIELDS
                ]
                repaired.append(
                    ToolAction(
                        "READ",
                        {"fields": valid_fields or list(DEFAULT_READ_FIELDS)},
                    )
                )
            elif action.tool == "FILTER":
                repaired.append(
                    ToolAction(
                        "FILTER",
                        {
                            key: args[key]
                            for key in ("field", "op", "value")
                            if key in args
                        },
                    )
                )
            elif action.tool == "SORT":
                repaired.append(
                    ToolAction(
                        "SORT",
                        {
                            key: args[key]
                            for key in ("field", "order")
                            if key in args
                        },
                    )
                )
            elif action.tool == "TOPK":
                repaired.append(
                    ToolAction("TOPK", {"k": args.get("k")})
                )
            elif action.tool == "EXPAND_NEIGHBORS":
                repaired.append(
                    ToolAction(
                        "EXPAND_NEIGHBORS",
                        {"window": args.get("window", 1)},
                    )
                )
            elif action.tool == "INSPECT_RAW":
                repaired.append(
                    ToolAction(
                        "INSPECT_RAW",
                        {
                            key: args[key]
                            for key in ("target", "instruction")
                            if key in args
                        },
                    )
                )
            elif action.tool == "STOP":
                repaired.append(ToolAction("STOP"))
            else:
                repaired.append(action)
        return self.validate(repaired)

    def _validate_action(self, action: ToolAction, index: int) -> None:
        if action.tool not in INTERACTIVE_TOOLS:
            raise InteractiveValidationError(
                f"action {index}: unsupported tool {action.tool!r}"
            )
        if action.tool == "INSPECT_RAW" and not self.allow_inspect_raw:
            raise InteractiveValidationError("INSPECT_RAW is unavailable")
        for value in action.arguments.values():
            if MEMORY_ID_PATTERN.search(json.dumps(value, ensure_ascii=False)):
                raise InteractiveValidationError(
                    f"action {index}: memory IDs are not allowed"
                )
        validator = getattr(self, f"_validate_{action.tool.lower()}")
        validator(action.arguments, index)

    @staticmethod
    def _keys(
        args: Dict[str, Any],
        required: set[str],
        optional: set[str],
        index: int,
    ) -> None:
        missing = required - set(args)
        unknown = set(args) - required - optional
        if missing:
            raise InteractiveValidationError(
                f"action {index}: missing arguments {sorted(missing)}"
            )
        if unknown:
            raise InteractiveValidationError(
                f"action {index}: unknown arguments {sorted(unknown)}"
            )

    def _positive_int(self, value: Any, index: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise InteractiveValidationError(
                f"action {index}: {name} must be an integer"
            )
        if value <= 0 or value > self.max_top_k:
            raise InteractiveValidationError(
                f"action {index}: {name} must be between 1 and {self.max_top_k}"
            )

    def _validate_retrieve(self, args: Dict[str, Any], index: int) -> None:
        self._keys(args, set(), {"method", "top_k", "query", "scope"}, index)
        if args.get("method", "hybrid") not in RETRIEVAL_METHODS:
            raise InteractiveValidationError(
                f"action {index}: invalid RETRIEVE method"
            )
        self._positive_int(args.get("top_k", 5), index, "top_k")
        if args.get("scope", "all") not in RETRIEVE_SCOPES:
            raise InteractiveValidationError(
                f"action {index}: invalid RETRIEVE scope"
            )
        rewritten = args.get("query")
        if rewritten is not None:
            if not isinstance(rewritten, str) or not rewritten.strip():
                raise InteractiveValidationError(
                    f"action {index}: query must be non-empty text"
                )
            if len(rewritten) > self.max_query_chars:
                raise InteractiveValidationError(
                    f"action {index}: rewritten query is too long"
                )

    def _validate_filter(self, args: Dict[str, Any], index: int) -> None:
        self._keys(args, {"field", "op", "value"}, set(), index)
        if args["field"] not in FILTER_FIELDS or args["op"] not in FILTER_OPS:
            raise InteractiveValidationError(
                f"action {index}: invalid FILTER arguments"
            )
        if not isinstance(args["value"], (str, int, float, bool)):
            raise InteractiveValidationError(
                f"action {index}: invalid FILTER value"
            )

    def _validate_sort(self, args: Dict[str, Any], index: int) -> None:
        self._keys(args, {"field", "order"}, set(), index)
        if args["field"] not in SORT_FIELDS or args["order"] not in SORT_ORDERS:
            raise InteractiveValidationError(
                f"action {index}: invalid SORT arguments"
            )

    def _validate_topk(self, args: Dict[str, Any], index: int) -> None:
        self._keys(args, {"k"}, set(), index)
        self._positive_int(args["k"], index, "k")

    def _validate_expand_neighbors(
        self,
        args: Dict[str, Any],
        index: int,
    ) -> None:
        self._keys(args, {"window"}, set(), index)
        window = args["window"]
        if not isinstance(window, int) or isinstance(window, bool) or not 1 <= window <= 3:
            raise InteractiveValidationError(
                f"action {index}: window must be 1, 2, or 3"
            )

    def _validate_read(self, args: Dict[str, Any], index: int) -> None:
        self._keys(args, {"fields"}, set(), index)
        fields = args["fields"]
        if not isinstance(fields, list) or not fields:
            raise InteractiveValidationError(
                f"action {index}: READ fields must be a non-empty list"
            )
        unknown = set(fields) - READ_FIELDS
        if unknown:
            raise InteractiveValidationError(
                f"action {index}: invalid READ fields {sorted(unknown)}"
            )

    def _validate_inspect_raw(self, args: Dict[str, Any], index: int) -> None:
        self._keys(args, set(), {"target", "instruction"}, index)
        if args.get("target", "current_pool") not in INSPECT_TARGETS:
            raise InteractiveValidationError(
                f"action {index}: invalid INSPECT_RAW target"
            )
        if (
            args.get("instruction", "answer_query_related_visual_details")
            not in INSPECT_INSTRUCTIONS
        ):
            raise InteractiveValidationError(
                f"action {index}: invalid INSPECT_RAW instruction"
            )

    def _validate_stop(self, args: Dict[str, Any], index: int) -> None:
        self._keys(args, set(), set(), index)


def _clip(value: Any, limit: int = 320) -> Any:
    if isinstance(value, str):
        value = re.sub(r"\bD\d+:IMG_\d+\b", "[image-id]", value)
        return value[:limit]
    if isinstance(value, list):
        return [_clip(item, limit) for item in value[:8]]
    if isinstance(value, dict):
        return {
            str(key): _clip(item, limit)
            for key, item in list(value.items())[:12]
            if key not in {"raw_pointer", "memory_id", "relative_path"}
        }
    return value


@dataclass
class InteractiveObservation:
    pool_record_count: int
    pool_turn_count: int
    score_min: float
    score_max: float
    candidate_previews: List[Dict[str, Any]]
    evidence_count: int
    new_evidence_count: int
    evidence_previews: List[Dict[str, Any]]
    stopped: bool
    last_error: str = ""
    has_question_image: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_record_count": self.pool_record_count,
            "pool_turn_count": self.pool_turn_count,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "candidate_previews": self.candidate_previews,
            "evidence_count": self.evidence_count,
            "new_evidence_count": self.new_evidence_count,
            "evidence_previews": self.evidence_previews,
            "stopped": self.stopped,
            "last_error": self.last_error,
            "has_question_image": self.has_question_image,
        }


class ExecutorSession:
    """Persistent executor state that accepts one short action chunk at a time."""

    def __init__(
        self,
        query: str,
        memory_store: HiddenMemoryStore,
        validator: InteractiveActionValidator,
        retriever: Optional[TurnAwareHybridRetriever] = None,
        raw_inspector: Optional[Any] = None,
        question_image: Optional[str] = None,
        max_raw_inspections: int = 3,
        preview_turns: int = 5,
        preview_evidence: int = 8,
    ):
        self.query = query
        self.memory_store = memory_store
        self.validator = validator
        self.retriever = retriever or TurnAwareHybridRetriever()
        self.raw_inspector = raw_inspector
        self.question_image = question_image
        self.max_raw_inspections = max(0, int(max_raw_inspections))
        self.preview_turns = max(1, int(preview_turns))
        self.preview_evidence = max(1, int(preview_evidence))
        self.pool = memory_store.initial_pool()
        self.evidence: List[EvidenceItem] = []
        self.steps: List[ExecutionStep] = []
        self.history: List[ToolAction] = []
        self.stopped = False
        self.error = ""
        self.raw_inspection_calls = 0
        self.pool_observable = False

    def clone(self) -> "ExecutorSession":
        cloned = ExecutorSession(
            query=self.query,
            memory_store=self.memory_store,
            validator=self.validator,
            retriever=self.retriever,
            raw_inspector=self.raw_inspector,
            question_image=self.question_image,
            max_raw_inspections=self.max_raw_inspections,
            preview_turns=self.preview_turns,
            preview_evidence=self.preview_evidence,
        )
        cloned.pool = list(self.pool)
        cloned.evidence = list(self.evidence)
        cloned.steps = list(self.steps)
        cloned.history = list(self.history)
        cloned.stopped = self.stopped
        cloned.error = self.error
        cloned.raw_inspection_calls = self.raw_inspection_calls
        cloned.pool_observable = self.pool_observable
        return cloned

    def execute_chunk(
        self,
        values: Iterable[Dict[str, Any] | ToolAction],
    ) -> InteractiveObservation:
        if self.stopped:
            raise InteractiveValidationError("session is already stopped")
        actions = self.validator.validate(values)
        evidence_before_chunk = len(self.evidence)
        helper = ToolExecutor(
            retriever=self.retriever,
            raw_inspector=self.raw_inspector,
            max_raw_inspections=self.max_raw_inspections,
        )
        for action in actions:
            before = len(self.pool)
            evidence_before = len(self.evidence)
            step_error = ""
            try:
                if action.tool == "FILTER":
                    self.pool = helper._filter(self.pool, **action.arguments)
                    self.pool_observable = True
                elif action.tool == "SORT":
                    self.pool = helper._sort(self.pool, **action.arguments)
                    self.pool_observable = True
                elif action.tool == "TOPK":
                    self.pool = helper._topk_turns(
                        self.pool,
                        action.arguments["k"],
                    )
                    self.pool_observable = True
                elif action.tool == "RETRIEVE":
                    source = (
                        self.memory_store.initial_pool()
                        if action.arguments.get("scope", "all") == "all"
                        else self.pool
                    )
                    retrieval_query = str(
                        action.arguments.get("query") or self.query
                    )
                    self.pool = self.retriever.retrieve(
                        source,
                        query=retrieval_query,
                        store=self.memory_store,
                        method=action.arguments.get("method", "hybrid"),
                        top_k=action.arguments.get("top_k", 5),
                        question_image=self.question_image,
                    )
                    self.pool_observable = True
                elif action.tool == "EXPAND_NEIGHBORS":
                    if not self.pool_observable:
                        raise InteractiveValidationError(
                            "EXPAND_NEIGHBORS requires a selected candidate pool"
                        )
                    self.pool = self._expand_neighbors(
                        action.arguments["window"]
                    )
                elif action.tool == "READ":
                    self._append_evidence(
                        helper._read(self.pool, action.arguments["fields"])
                    )
                elif action.tool == "INSPECT_RAW":
                    remaining = max(
                        0,
                        self.max_raw_inspections - self.raw_inspection_calls,
                    )
                    inspected = helper._inspect_raw(
                        self.pool,
                        self.query,
                        remaining,
                        question_image=self.question_image,
                    )
                    self.raw_inspection_calls += len(inspected)
                    self._append_evidence(inspected)
                elif action.tool == "STOP":
                    self.stopped = True
            except Exception as exc:
                step_error = str(exc)
                self.error = f"{action.tool}: {exc}"
            self.history.append(action)
            self.steps.append(
                ExecutionStep(
                    index=len(self.steps),
                    action=action,
                    pool_before=before,
                    pool_after=len(self.pool),
                    evidence_added=len(self.evidence) - evidence_before,
                    error=step_error,
                )
            )
            if self.stopped or step_error:
                break
        return self.observation(
            new_evidence_count=len(self.evidence) - evidence_before_chunk
        )

    def _append_evidence(self, values: List[EvidenceItem]) -> None:
        existing = {
            (item.memory_id, item.source, json.dumps(item.fields, sort_keys=True, default=str))
            for item in self.evidence
        }
        for item in values:
            signature = (
                item.memory_id,
                item.source,
                json.dumps(item.fields, sort_keys=True, default=str),
            )
            if signature not in existing:
                self.evidence.append(item)
                existing.add(signature)

    def _expand_neighbors(self, window: int) -> List[PoolItem]:
        selected = {
            (
                str(item.memory.metadata.get("session_id") or ""),
                item.memory.metadata.get("turn_index"),
            )
            for item in self.pool
        }
        expanded_keys = set(selected)
        for session_id, turn_index in selected:
            if not session_id or not isinstance(turn_index, int):
                continue
            for distance in range(1, window + 1):
                expanded_keys.add((session_id, turn_index - distance))
                expanded_keys.add((session_id, turn_index + distance))
        score_by_turn = {
            item.memory.turn_id: item.score for item in self.pool
        }
        expanded = []
        for item in self.memory_store.initial_pool():
            key = (
                str(item.memory.metadata.get("session_id") or ""),
                item.memory.metadata.get("turn_index"),
            )
            if key in expanded_keys:
                expanded.append(
                    PoolItem(
                        item.memory,
                        score_by_turn.get(item.memory.turn_id, 0.0),
                    )
                )
        expanded.sort(
            key=lambda item: (
                str(item.memory.metadata.get("session_id") or ""),
                int(item.memory.metadata.get("turn_index") or 0),
                item.memory.timestamp,
            )
        )
        return expanded

    def observation(self, new_evidence_count: int = 0) -> InteractiveObservation:
        scores = [float(item.score) for item in self.pool]
        previews = []
        seen_turns = set()
        visible_pool = self.pool if self.pool_observable else []
        for item in visible_pool:
            turn_id = item.memory.turn_id
            if turn_id in seen_turns:
                continue
            seen_turns.add(turn_id)
            records = [
                candidate.memory
                for candidate in self.pool
                if candidate.memory.turn_id == turn_id
            ]
            previews.append(
                {
                    "rank": len(previews) + 1,
                    "score": round(float(item.score), 4),
                    "timestamp": item.memory.timestamp,
                    "modalities": sorted(
                        {record.modality for record in records}
                    ),
                    "source_types": sorted(
                        {record.source_type for record in records}
                    ),
                    "summary": _clip(
                        " ".join(
                            record.summary for record in records if record.summary
                        ),
                        420,
                    ),
                }
            )
            if len(previews) >= self.preview_turns:
                break
        evidence_previews = [
            {
                "source": item.source,
                "fields": _clip(item.fields, 420),
            }
            for item in self.evidence[-self.preview_evidence :]
        ]
        return InteractiveObservation(
            pool_record_count=len(self.pool),
            pool_turn_count=len({item.memory.turn_id for item in self.pool}),
            score_min=round(min(scores), 4) if scores else 0.0,
            score_max=round(max(scores), 4) if scores else 0.0,
            candidate_previews=previews,
            evidence_count=len(self.evidence),
            new_evidence_count=new_evidence_count,
            evidence_previews=evidence_previews,
            stopped=self.stopped,
            last_error=self.error,
            has_question_image=bool(self.question_image),
        )

    def result(self) -> ExecutionResult:
        return ExecutionResult(
            evidence=list(self.evidence),
            steps=list(self.steps),
            final_pool_size=len(self.pool),
            final_memory_ids=[item.memory.memory_id for item in self.pool],
            stopped=self.stopped,
            error=self.error,
            raw_inspection_calls=self.raw_inspection_calls,
        )


@dataclass
class VerificationResult:
    answerable: bool
    relevance: float
    completeness: float
    redundancy: float
    reason: str = ""
    error: str = ""
    diagnostic: Dict[str, Any] = field(default_factory=dict)

    def planner_feedback(self) -> Dict[str, Any]:
        def band(value: float) -> str:
            if value >= 0.8:
                return "high"
            if value >= 0.45:
                return "medium"
            return "low"

        feedback = {
            "answerable": self.answerable,
            "relevance": band(self.relevance),
            "completeness": (
                "sufficient" if self.answerable else band(self.completeness)
            ),
            "redundancy": band(self.redundancy),
            "continue_required": not self.answerable,
        }
        if self.diagnostic:
            feedback["failure_diagnostic"] = self.diagnostic
        return feedback

    def to_dict(self, include_reason: bool = True) -> Dict[str, Any]:
        data = {
            "answerable": self.answerable,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "redundancy": self.redundancy,
            "error": self.error,
            "diagnostic": self.diagnostic,
        }
        if include_reason:
            data["reason"] = self.reason
        return data


class EvidenceVerifier(Protocol):
    def evaluate(
        self,
        query: str,
        gold_answer: str,
        evidence: List[EvidenceItem],
    ) -> VerificationResult:
        ...


@dataclass
class AnswerValidationResult:
    correct: bool
    score: float
    prediction: str
    reason: str = ""
    error: str = ""
    image_ids_match: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct": self.correct,
            "score": self.score,
            "prediction": self.prediction,
            "reason": self.reason,
            "error": self.error,
            "image_ids_match": self.image_ids_match,
        }

    def failure_feedback(
        self,
        evidence: List[EvidenceItem],
        can_inspect_raw: bool = False,
    ) -> Dict[str, Any]:
        """Return training-only diagnostics without exposing the gold answer."""
        if self.correct:
            return {}

        has_raw_inspection = any(
            item.source == "INSPECT_RAW" for item in evidence
        )
        has_visual_evidence = any(
            _evidence_mentions_visual_content(item) for item in evidence
        )

        if not evidence:
            failure_type = "no_evidence"
            evidence_gap = "No evidence has been read for the answer model."
            recommended_change = (
                "Retrieve a focused candidate pool, then READ relevant fields."
            )
        elif self.error:
            failure_type = "answer_validation_error"
            evidence_gap = "The answer validation call failed."
            recommended_change = (
                "Continue with a different evidence path and validate again."
            )
        elif self.image_ids_match is False:
            failure_type = "image_id_mismatch"
            evidence_gap = (
                "The predicted public image IDs do not match the required set."
            )
            recommended_change = (
                "Refine retrieval to isolate the relevant visual memories and "
                "inspect the raw candidates before answering."
                if can_inspect_raw
                else "Refine retrieval to isolate the relevant visual memories."
            )
        elif has_visual_evidence and not has_raw_inspection:
            failure_type = "uninspected_visual_evidence"
            evidence_gap = (
                "Visual memories are present, but the answer model could not "
                "derive a correct answer from the currently read representation."
            )
            recommended_change = (
                "Use INSPECT_RAW on the strongest visual candidates, then "
                "validate the answer again."
                if can_inspect_raw
                else "Reformulate retrieval to surface more discriminative "
                "visual captions or surrounding context."
            )
        elif has_raw_inspection:
            failure_type = "answer_mismatch_after_raw_inspection"
            evidence_gap = (
                "The inspected visual evidence still does not support a correct "
                "answer."
            )
            recommended_change = (
                "Change the retrieval query or candidate pool instead of "
                "re-inspecting the same evidence."
            )
        else:
            failure_type = "answer_mismatch"
            evidence_gap = (
                "The answer model could not derive the required answer from the "
                "current evidence."
            )
            recommended_change = (
                "Reformulate retrieval, change retrieval method, or expand "
                "neighboring turns before reading again."
            )

        diagnostic = {
            "failure_type": failure_type,
            "predicted_answer": _clip(self.prediction, 300),
            "judge_reason": _clip(self.reason or self.error, 500),
            "answer_score": round(float(self.score), 4),
            "image_ids_match": self.image_ids_match,
            "evidence_gap": evidence_gap,
            "recommended_change": recommended_change,
        }
        return {
            key: value
            for key, value in diagnostic.items()
            if value not in {"", None}
        }


class AnswerGenerator(Protocol):
    def answer(
        self,
        query: str,
        evidence: List[EvidenceItem],
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


class StrictAnswerValidator:
    """Validate a trajectory through the actual answer model and judge."""

    def __init__(
        self,
        answer_model: AnswerGenerator,
        judge: AnswerJudge,
        min_score: float = 0.9,
    ):
        self.answer_model = answer_model
        self.judge = judge
        self.min_score = max(0.0, min(1.0, float(min_score)))
        self.calls = 0

    def evaluate(
        self,
        query: str,
        gold_answer: str,
        evidence: List[EvidenceItem],
        question_image: Optional[str] = None,
    ) -> AnswerValidationResult:
        self.calls += 1
        if not evidence:
            return AnswerValidationResult(
                correct=False,
                score=0.0,
                prediction="",
                reason="No retrieved evidence was provided.",
            )
        try:
            prediction = self.answer_model.answer(
                query,
                evidence,
                question_image=question_image,
            )
            judge_correct, score, reason = self.judge.evaluate(
                query,
                prediction,
                gold_answer,
            )
            score = max(0.0, min(1.0, float(score)))
            gold_ids = set(PUBLIC_IMAGE_ID_PATTERN.findall(gold_answer))
            predicted_ids = set(
                PUBLIC_IMAGE_ID_PATTERN.findall(prediction)
            )
            image_ids_match = (
                predicted_ids == gold_ids if gold_ids else None
            )
            correct = (
                bool(judge_correct)
                and score >= self.min_score
                and image_ids_match is not False
            )
            return AnswerValidationResult(
                correct=correct,
                score=score,
                prediction=prediction,
                reason=str(reason or ""),
                image_ids_match=image_ids_match,
            )
        except Exception as exc:
            return AnswerValidationResult(
                correct=False,
                score=0.0,
                prediction="",
                error=str(exc),
                reason="Answer validation failed.",
            )


class ChatGoldEvidenceVerifier:
    """Training-only verifier. Its reason never enters student SFT input."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        max_tokens: int = 192,
    ):
        self.client = client
        self.max_tokens = max_tokens
        self.calls = 0

    def evaluate(
        self,
        query: str,
        gold_answer: str,
        evidence: List[EvidenceItem],
    ) -> VerificationResult:
        self.calls += 1
        if not evidence:
            return VerificationResult(
                answerable=False,
                relevance=0.0,
                completeness=0.0,
                redundancy=0.0,
                reason="No retrieved evidence was provided.",
            )
        public_evidence = [
            {"source": item.source, "fields": _clip(item.fields, 700)}
            for item in evidence
        ]
        prompt = f"""You are a training-only evidence verifier.

Given the user query, gold answer, and currently retrieved evidence, judge
whether the evidence alone is sufficient for an answer model to derive the
gold answer. Do not require exact wording. Penalize missing comparison sides,
missing list entities, unsupported temporal claims, and excessive irrelevant
evidence. Empty evidence is never sufficient, including when the gold answer
is "Not mentioned." For absence claims, require relevant retrieved evidence
that lets an answer model assess the requested fact rather than inferring
absence from an empty result.

Return only JSON:
{{
  "answerable": true,
  "relevance": 0.0,
  "completeness": 0.0,
  "redundancy": 0.0,
  "reason": "short diagnostic"
}}

User query:
{query}

Gold answer:
{gold_answer}

Retrieved evidence:
{json.dumps(public_evidence, ensure_ascii=False)}
"""
        raw = ""
        try:
            raw = self.client.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            value = extract_json_object(raw)
            relevance = max(
                0.0,
                min(1.0, float(value.get("relevance", 0.0))),
            )
            completeness = max(
                0.0,
                min(1.0, float(value.get("completeness", 0.0))),
            )
            answerable = (
                bool(value.get("answerable"))
                and relevance >= 0.5
                and completeness >= 0.5
            )
            return VerificationResult(
                answerable=answerable,
                relevance=relevance,
                completeness=completeness,
                redundancy=max(
                    0.0,
                    min(1.0, float(value.get("redundancy", 0.0))),
                ),
                reason=str(value.get("reason") or ""),
            )
        except Exception as exc:
            return VerificationResult(
                answerable=False,
                relevance=0.0,
                completeness=0.0,
                redundancy=1.0,
                error=str(exc),
                reason=raw[:500],
            )


def build_online_policy_prompt(
    query: str,
    history: List[ToolAction],
    observation: InteractiveObservation,
    schema: str,
) -> str:
    return f"""You are an interactive multimodal memory retrieval policy.
Choose the next short executable action chunk using only online-available
information. Do not assume the answer or hidden memory locations.

{schema}

User query:
{query}

Executed action history:
{json.dumps([action.to_dict() for action in history], ensure_ascii=False)}

Current executor observation:
{json.dumps(observation.to_dict(), ensure_ascii=False)}
"""


def fallback_action_chunks(
    observation: InteractiveObservation,
    validator: InteractiveActionValidator,
    candidate_count: int = 2,
) -> List[List[ToolAction]]:
    """General recovery actions derived only from online executor state."""
    candidates: List[List[ToolAction]] = []
    if not observation.candidate_previews:
        candidates.append(
            [
                ToolAction(
                    "RETRIEVE",
                    {"method": "hybrid", "top_k": 5, "scope": "all"},
                ),
                ToolAction("READ", {"fields": list(DEFAULT_READ_FIELDS)}),
            ]
        )
    elif observation.evidence_count == 0:
        candidates.append(
            [ToolAction("READ", {"fields": list(DEFAULT_READ_FIELDS)})]
        )
        candidates.append(
            [
                ToolAction("EXPAND_NEIGHBORS", {"window": 1}),
                ToolAction("READ", {"fields": list(DEFAULT_READ_FIELDS)}),
            ]
        )
        if validator.allow_inspect_raw:
            candidates.append(
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
    else:
        candidates.append(
            [
                ToolAction("EXPAND_NEIGHBORS", {"window": 1}),
                ToolAction("READ", {"fields": list(DEFAULT_READ_FIELDS)}),
            ]
        )
        candidates.append(
            [
                ToolAction(
                    "RETRIEVE",
                    {"method": "hybrid", "top_k": 10, "scope": "all"},
                ),
                ToolAction("READ", {"fields": list(DEFAULT_READ_FIELDS)}),
            ]
        )
        if validator.allow_inspect_raw:
            candidates.append(
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
    validated = []
    for candidate in candidates:
        try:
            validated.append(validator.validate(candidate))
        except InteractiveValidationError:
            continue
    return validated[: max(1, candidate_count)]


class ChatInteractivePlanner:
    """Gold-free planner used by both teacher search and the future student."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        validator: InteractiveActionValidator,
        max_tokens: int = 768,
    ):
        self.client = client
        self.validator = validator
        self.max_tokens = max_tokens
        self.calls = 0
        self.last_raw_response = ""
        self.last_candidate_sources: Dict[str, str] = {}

    def propose(
        self,
        query: str,
        history: List[ToolAction],
        observation: InteractiveObservation,
        candidate_count: int = 3,
        privileged_feedback: Optional[Dict[str, Any]] = None,
    ) -> List[List[ToolAction]]:
        self.calls += 1
        online_prompt = build_online_policy_prompt(
            query,
            history,
            observation,
            self.validator.schema_text(),
        )
        feedback = (
            "\nTraining-only verifier feedback for search (never included in "
            "student input):\n"
            + json.dumps(privileged_feedback, ensure_ascii=False)
            if privileged_feedback
            else ""
        )
        prompt = f"""{online_prompt}
{feedback}

Propose {max(1, candidate_count)} materially different candidate chunks.
Prefer changing retrieval query, method, scope, neighbor expansion, or action
ordering before merely increasing top_k. STOP only when observed evidence is
sufficient and at least one READ or INSPECT_RAW has produced evidence.
When training-only failure_diagnostic is present, directly address its
evidence_gap and recommended_change. Do not repeat the failed action pattern.
RETRIEVE changes the candidate pool but does not add answer evidence. After a
useful candidate pool is visible, use READ with exactly the allowed fields.
Do not add arguments such as limit, reason, filter, pool_record_id, or IDs.
Pool-changing actions must precede READ.

Return only JSON:
{{"candidates": [[{{"tool": "RETRIEVE", ...}}, ...], ...]}}
"""
        raw = self.client.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        self.last_raw_response = raw
        value = extract_json_object(raw)
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("planner response has no candidates list")
        validated = []
        signatures = set()
        self.last_candidate_sources = {}
        for candidate in candidates:
            if not isinstance(candidate, list):
                continue
            try:
                actions = self.validator.repair(candidate)
            except Exception:
                continue
            signature = json.dumps(
                [action.to_dict() for action in actions],
                sort_keys=True,
                ensure_ascii=False,
            )
            if signature not in signatures:
                validated.append(actions)
                signatures.add(signature)
                raw_signature = json.dumps(
                    [
                        ToolAction.from_dict(value).to_dict()
                        for value in candidate
                        if isinstance(value, dict)
                    ],
                    sort_keys=True,
                    ensure_ascii=False,
                )
                self.last_candidate_sources[signature] = (
                    "planner"
                    if raw_signature == signature
                    else "planner_repaired"
                )
        fallback_chunks = fallback_action_chunks(
            observation,
            self.validator,
            candidate_count,
        )
        for actions in fallback_chunks:
            signature = json.dumps(
                [action.to_dict() for action in actions],
                sort_keys=True,
                ensure_ascii=False,
            )
            if signature not in signatures:
                validated.append(actions)
                signatures.add(signature)
                self.last_candidate_sources[signature] = "controller_fallback"
        if not validated:
            raise ValueError("planner produced no valid action chunks")
        selected = validated[: max(1, candidate_count)]
        if (
            observation.candidate_previews
            and not any(
                action.tool in {"READ", "INSPECT_RAW"}
                for actions in selected
                for action in actions
            )
        ):
            evidence_chunk = next(
                (
                    actions
                    for actions in fallback_chunks
                    if any(
                        action.tool in {"READ", "INSPECT_RAW"}
                        for action in actions
                    )
                ),
                None,
            )
            if evidence_chunk is not None:
                if len(selected) >= max(1, candidate_count):
                    selected[-1] = evidence_chunk
                else:
                    selected.append(evidence_chunk)
        return selected


@dataclass
class InteractiveDecision:
    observation: InteractiveObservation
    observation_after: InteractiveObservation
    history: List[ToolAction]
    actions: List[ToolAction]
    privileged_feedback: Dict[str, Any]
    verification_after: VerificationResult
    planner_raw_response: str = ""
    action_source: str = "planner"

    def sft_example(
        self,
        sample_id: str,
        step_index: int,
        query: str,
        schema: str,
    ) -> SFTExample:
        return SFTExample(
            sample_id=f"{sample_id}:step:{step_index}",
            input=build_online_policy_prompt(
                query,
                self.history,
                self.observation,
                schema,
            ),
            target=json.dumps(
                [action.to_dict() for action in self.actions],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            round_index=step_index,
            metadata={
                "answerable_after": self.verification_after.answerable,
                "relevance_after": self.verification_after.relevance,
                "completeness_after": self.verification_after.completeness,
                "evidence_count_after": (
                    self.observation_after.evidence_count
                ),
                "privileged_feedback_used_for_teacher_search": bool(
                    self.privileged_feedback
                ),
                "action_source": self.action_source,
            },
        )


@dataclass
class SearchNode:
    session: ExecutorSession
    decisions: List[InteractiveDecision] = field(default_factory=list)
    verification: VerificationResult = field(
        default_factory=lambda: VerificationResult(False, 0.0, 0.0, 0.0)
    )
    answer_validation: Optional[AnswerValidationResult] = None

    def score(self) -> tuple[Any, ...]:
        raw_evidence = sum(
            1 for item in self.session.evidence if item.source == "INSPECT_RAW"
        )
        no_op_reads = self._redundant_read_count()
        return (
            int(
                self.answer_validation is not None
                and self.answer_validation.correct
            ),
            int(self.verification.answerable),
            self.verification.completeness,
            self.verification.relevance,
            -self.verification.redundancy,
            min(raw_evidence, 3),
            -no_op_reads,
            -len(self.session.history),
        )

    def _redundant_read_count(self) -> int:
        reads = []
        pool_fingerprint = ""
        count = 0
        for step in self.session.steps:
            action = step.action
            if action.tool in POOL_MUTATING_TOOLS:
                pool_fingerprint = f"{step.index}:{step.pool_after}"
            if action.tool != "READ":
                continue
            signature = (
                pool_fingerprint,
                tuple(action.arguments.get("fields", [])),
                step.evidence_added,
            )
            if signature in reads:
                count += 1
            reads.append(signature)
        return count


@dataclass
class InteractiveSearchResult:
    actions: List[ToolAction]
    execution: ExecutionResult
    verification: VerificationResult
    decisions: List[InteractiveDecision]
    candidates_evaluated: int
    planner_calls: int
    verifier_calls: int
    answer_validation: Optional[AnswerValidationResult] = None
    answer_validator_calls: int = 0
    failure_diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    def sft_examples(
        self,
        sample_id: str,
        query: str,
        schema: str,
    ) -> List[SFTExample]:
        return [
            decision.sft_example(sample_id, index, query, schema)
            for index, decision in enumerate(self.decisions)
        ]


@dataclass
class InteractivePolicyResult:
    actions: List[ToolAction]
    execution: ExecutionResult
    planner_calls: int
    chunks: int
    planner_raw_responses: List[str] = field(default_factory=list)


class InteractivePolicyRunner:
    """Run the same observation-to-next-chunk interface without gold feedback."""

    def __init__(
        self,
        planner: ChatInteractivePlanner,
        validator: InteractiveActionValidator,
        retriever: Optional[TurnAwareHybridRetriever] = None,
        raw_inspector: Optional[Any] = None,
        max_rounds: int = 3,
        max_actions: int = 9,
        max_raw_inspections: int = 3,
    ):
        self.planner = planner
        self.validator = validator
        self.retriever = retriever or TurnAwareHybridRetriever()
        self.raw_inspector = raw_inspector
        self.max_rounds = max(1, int(max_rounds))
        self.max_actions = max(1, int(max_actions))
        self.max_raw_inspections = max(0, int(max_raw_inspections))

    def run(
        self,
        query: str,
        memory_store: HiddenMemoryStore,
        question_image: Optional[str] = None,
    ) -> InteractivePolicyResult:
        session = ExecutorSession(
            query=query,
            memory_store=memory_store,
            validator=self.validator,
            retriever=self.retriever,
            raw_inspector=self.raw_inspector,
            question_image=question_image,
            max_raw_inspections=self.max_raw_inspections,
        )
        calls_before = getattr(self.planner, "calls", 0)
        chunks = 0
        raw_responses: List[str] = []
        for _round_index in range(self.max_rounds):
            if session.stopped or len(session.history) >= self.max_actions:
                break
            try:
                proposed = self.planner.propose(
                    query=query,
                    history=session.history,
                    observation=session.observation(),
                    candidate_count=1,
                    privileged_feedback=None,
                )
                raw_responses.append(
                    str(getattr(self.planner, "last_raw_response", ""))
                )
                chunk = proposed[0]
            except Exception:
                fallback = fallback_action_chunks(
                    session.observation(),
                    self.validator,
                    candidate_count=1,
                )
                chunk = fallback[0] if fallback else [ToolAction("STOP")]
            if len(session.history) + len(chunk) > self.max_actions:
                break
            session.execute_chunk(chunk)
            chunks += 1
        if not session.stopped:
            session.execute_chunk([ToolAction("STOP")])
            chunks += 1
        return InteractivePolicyResult(
            actions=list(session.history),
            execution=session.result(),
            planner_calls=getattr(self.planner, "calls", 0) - calls_before,
            chunks=chunks,
            planner_raw_responses=raw_responses,
        )


class InteractiveTeacherSearch:
    def __init__(
        self,
        planner: ChatInteractivePlanner,
        verifier: EvidenceVerifier,
        validator: InteractiveActionValidator,
        retriever: Optional[TurnAwareHybridRetriever] = None,
        raw_inspector: Optional[Any] = None,
        max_rounds: int = 3,
        beam_size: int = 2,
        candidates_per_node: int = 3,
        max_actions: int = 9,
        max_evidence: int = 40,
        max_raw_inspections: int = 3,
        answer_validator: Optional[StrictAnswerValidator] = None,
    ):
        self.planner = planner
        self.verifier = verifier
        self.validator = validator
        self.retriever = retriever or TurnAwareHybridRetriever()
        self.raw_inspector = raw_inspector
        self.max_rounds = max(1, int(max_rounds))
        self.beam_size = max(1, int(beam_size))
        self.candidates_per_node = max(1, int(candidates_per_node))
        self.max_actions = max(1, int(max_actions))
        self.max_evidence = max(1, int(max_evidence))
        self.max_raw_inspections = max(0, int(max_raw_inspections))
        self.answer_validator = answer_validator

    def search(
        self,
        query: str,
        gold_answer: str,
        memory_store: HiddenMemoryStore,
        question_image: Optional[str] = None,
        initial_session: Optional[ExecutorSession] = None,
    ) -> InteractiveSearchResult:
        initial = (
            initial_session.clone()
            if initial_session is not None
            else ExecutorSession(
                query=query,
                memory_store=memory_store,
                validator=self.validator,
                retriever=self.retriever,
                raw_inspector=self.raw_inspector,
                question_image=question_image,
                max_raw_inspections=self.max_raw_inspections,
            )
        )
        if initial.query != query or initial.memory_store is not memory_store:
            raise ValueError(
                "initial_session must belong to the current query and store"
            )
        beam = [SearchNode(session=initial)]
        finished: List[SearchNode] = []
        candidates_evaluated = 0
        failure_diagnostics: List[Dict[str, Any]] = []
        planner_calls_before = getattr(self.planner, "calls", 0)
        verifier_calls_before = getattr(self.verifier, "calls", 0)
        answer_validator_calls_before = getattr(
            self.answer_validator,
            "calls",
            0,
        )

        for _round_index in range(self.max_rounds):
            children: List[SearchNode] = []
            for node in beam:
                if node.session.stopped:
                    finished.append(node)
                    continue
                observation = node.session.observation()
                feedback = (
                    node.verification.planner_feedback()
                    if node.decisions
                    else None
                )
                try:
                    chunks = self.planner.propose(
                        query=query,
                        history=node.session.history,
                        observation=observation,
                        candidate_count=self.candidates_per_node,
                        privileged_feedback=feedback,
                    )
                    planner_raw = str(
                        getattr(self.planner, "last_raw_response", "")
                    )
                    candidate_sources = dict(
                        getattr(
                            self.planner,
                            "last_candidate_sources",
                            {},
                        )
                    )
                except Exception:
                    chunks = fallback_action_chunks(
                        observation,
                        self.validator,
                        self.candidates_per_node,
                    )
                    if not chunks:
                        chunks = [[ToolAction("STOP")]]
                    planner_raw = ""
                    candidate_sources = {}
                for chunk in chunks:
                    if len(node.session.history) + len(chunk) > self.max_actions:
                        continue
                    child_session = node.session.clone()
                    try:
                        observation_after = child_session.execute_chunk(chunk)
                    except Exception:
                        continue
                    verification = self.verifier.evaluate(
                        query,
                        gold_answer,
                        child_session.evidence,
                    )
                    answer_validation = None
                    if (
                        verification.answerable
                        and self.answer_validator is not None
                    ):
                        answer_validation = self.answer_validator.evaluate(
                            query,
                            gold_answer,
                            child_session.evidence,
                            question_image=question_image,
                        )
                        if not answer_validation.correct:
                            diagnostic = answer_validation.failure_feedback(
                                child_session.evidence,
                                can_inspect_raw=(
                                    self.validator.allow_inspect_raw
                                    and self.raw_inspector is not None
                                ),
                            )
                            verification = VerificationResult(
                                answerable=False,
                                relevance=verification.relevance,
                                completeness=min(
                                    verification.completeness,
                                    0.49,
                                ),
                                redundancy=verification.redundancy,
                                reason=(
                                    "Strict answer validation failed: "
                                    + (
                                        answer_validation.reason
                                        or answer_validation.error
                                    )
                                ),
                                error=answer_validation.error,
                                diagnostic=diagnostic,
                            )
                            failure_diagnostics.append(
                                {
                                    "round_index": _round_index,
                                    "history": [
                                        action.to_dict()
                                        for action in node.session.history
                                    ],
                                    "candidate_actions": [
                                        action.to_dict() for action in chunk
                                    ],
                                    "answer_validation": (
                                        answer_validation.to_dict()
                                    ),
                                    "teacher_feedback": (
                                        verification.planner_feedback()
                                    ),
                                    "evidence_count": len(
                                        child_session.evidence
                                    ),
                                    "raw_inspection_calls": (
                                        child_session.raw_inspection_calls
                                    ),
                                }
                            )
                    candidates_evaluated += 1
                    decision = InteractiveDecision(
                        observation=observation,
                        observation_after=observation_after,
                        history=list(node.session.history),
                        actions=list(chunk),
                        privileged_feedback=feedback or {},
                        verification_after=verification,
                        planner_raw_response=planner_raw,
                        action_source=candidate_sources.get(
                            json.dumps(
                                [
                                    action.to_dict()
                                    for action in chunk
                                ],
                                sort_keys=True,
                                ensure_ascii=False,
                            ),
                            "controller_fallback",
                        ),
                    )
                    child = SearchNode(
                        session=child_session,
                        decisions=[*node.decisions, decision],
                        verification=verification,
                        answer_validation=answer_validation,
                    )
                    if child_session.stopped:
                        finished.append(child)
                    elif verification.answerable:
                        stop_session = child_session.clone()
                        stop_observation = stop_session.observation()
                        stop_history = list(stop_session.history)
                        stop_observation_after = stop_session.execute_chunk(
                            [ToolAction("STOP")]
                        )
                        finished.append(
                            SearchNode(
                                session=stop_session,
                                decisions=[
                                    *child.decisions,
                                    InteractiveDecision(
                                        observation=stop_observation,
                                        observation_after=stop_observation_after,
                                        history=stop_history,
                                        actions=[ToolAction("STOP")],
                                        privileged_feedback=(
                                            verification.planner_feedback()
                                        ),
                                        verification_after=verification,
                                        action_source="verifier_stop",
                                    ),
                                ],
                                verification=verification,
                                answer_validation=answer_validation,
                            )
                        )
                    elif len(child_session.evidence) <= self.max_evidence:
                        children.append(child)
            if not children:
                break
            deduped: Dict[str, SearchNode] = {}
            for child in children:
                signature = json.dumps(
                    [action.to_dict() for action in child.session.history],
                    sort_keys=True,
                    ensure_ascii=False,
                )
                previous = deduped.get(signature)
                if previous is None or child.score() > previous.score():
                    deduped[signature] = child
            beam = sorted(
                deduped.values(),
                key=lambda node: node.score(),
                reverse=True,
            )[: self.beam_size]

        pool = [*finished, *beam]
        if not pool:
            pool = [SearchNode(session=initial)]
        selected = max(pool, key=lambda node: node.score())
        if not selected.session.stopped:
            observation_before_stop = selected.session.observation()
            history_before_stop = list(selected.session.history)
            observation_after_stop = selected.session.execute_chunk(
                [ToolAction("STOP")]
            )
            selected.decisions.append(
                InteractiveDecision(
                    observation=observation_before_stop,
                    observation_after=observation_after_stop,
                    history=history_before_stop,
                    actions=[ToolAction("STOP")],
                    privileged_feedback=selected.verification.planner_feedback(),
                    verification_after=selected.verification,
                    action_source="budget_stop",
                )
            )
        return InteractiveSearchResult(
            actions=list(selected.session.history),
            execution=selected.session.result(),
            verification=selected.verification,
            decisions=selected.decisions,
            candidates_evaluated=candidates_evaluated,
            planner_calls=getattr(self.planner, "calls", 0)
            - planner_calls_before,
            verifier_calls=getattr(self.verifier, "calls", 0)
            - verifier_calls_before,
            answer_validation=selected.answer_validation,
            answer_validator_calls=(
                getattr(self.answer_validator, "calls", 0)
                - answer_validator_calls_before
            ),
            failure_diagnostics=failure_diagnostics,
        )
