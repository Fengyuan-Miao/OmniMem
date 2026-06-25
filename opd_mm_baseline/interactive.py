"""Interactive chunked policy search for OPD-MM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .clients import (
    OpenAICompatibleClient,
    extract_json_array,
    extract_json_object,
)
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
    "session_date",
    "turn_id",
    "author",
    "modality",
    "source_type",
    "raw_pointer",
]


def raw_inspect_chunk() -> List[ToolAction]:
    return [
        ToolAction(
            "INSPECT_RAW",
            {
                "target": "current_pool",
                "instruction": "answer_query_related_visual_details",
            },
        )
    ]


def _actions_signature(actions: List[ToolAction]) -> str:
    return json.dumps(
        [action.to_dict() for action in actions],
        sort_keys=True,
        ensure_ascii=False,
    )


def _clip_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_interactive_schema(
    allow_inspect_raw: bool = False,
    max_chunk_actions: int = 3,
) -> str:
    atomic = max(1, int(max_chunk_actions)) == 1
    lines = [
        "Allowed executable tools:",
        "RETRIEVE(method=bm25|dense|vision|hybrid, top_k=positive integer,",
        "         query=optional rewritten retrieval query, scope=all|current)",
        "FILTER(field=modality|author|source_type|timestamp|status,",
        "       op=eq|neq|before|after|contains, value=...)",
        "SORT(field=timestamp|turn_id|score, order=asc|desc)",
        "TOPK(k=positive integer)",
        "EXPAND_NEIGHBORS(window=1|2|3)",
        "READ(fields=[summary|content|ocr|timestamp|session_date|turn_id|author|modality|source_type|raw_pointer])",
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
            (
                "The executable action list must contain exactly one executable action."
                if atomic
                else "The executable action list should be one short chunk."
            ),
            "RETRIEVE may rewrite the",
            "query using only the user request and observed evidence. Never emit",
            "memory IDs or file paths.",
        ]
    )
    if not atomic:
        lines.append("Pool-changing actions must occur before READ.")
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
        return build_interactive_schema(
            self.allow_inspect_raw,
            max_chunk_actions=self.max_chunk_actions,
        )

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
        if self.max_chunk_actions == 1:
            for value in values:
                repaired = self._repair_action(value)
                try:
                    return self.validate([repaired])
                except InteractiveValidationError:
                    continue
            return self.validate([])

        repaired: List[ToolAction] = []
        for value in values:
            repaired.append(self._repair_action(value))
        return self.validate(repaired)

    def _repair_action(
        self,
        value: Dict[str, Any] | ToolAction,
    ) -> ToolAction:
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
            return ToolAction("RETRIEVE", repaired_args)
        if action.tool == "READ":
            fields = args.get("fields")
            if not isinstance(fields, list):
                fields = DEFAULT_READ_FIELDS
            valid_fields = [
                field
                for field in fields
                if isinstance(field, str) and field in READ_FIELDS
            ]
            return ToolAction(
                "READ",
                {"fields": valid_fields or list(DEFAULT_READ_FIELDS)},
            )
        if action.tool == "FILTER":
            return ToolAction(
                "FILTER",
                {
                    key: args[key]
                    for key in ("field", "op", "value")
                    if key in args
                },
            )
        if action.tool == "SORT":
            return ToolAction(
                "SORT",
                {
                    key: args[key]
                    for key in ("field", "order")
                    if key in args
                },
            )
        if action.tool == "TOPK":
            return ToolAction("TOPK", {"k": args.get("k")})
        if action.tool == "EXPAND_NEIGHBORS":
            return ToolAction(
                "EXPAND_NEIGHBORS",
                {"window": args.get("window", 1)},
            )
        if action.tool == "INSPECT_RAW":
            return ToolAction(
                "INSPECT_RAW",
                {
                    key: args[key]
                    for key in ("target", "instruction")
                    if key in args
                },
            )
        if action.tool == "STOP":
            return ToolAction("STOP")
        return action

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


def _compact_prompt_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Keep planner observations short enough for 4k-context local models."""
    compact: Dict[str, Any] = {}
    preferred = [
        "content",
        "summary",
        "visual_observation",
        "linked_text_context",
        "ocr",
        "image_label",
        "timestamp",
        "session_date",
        "turn_id",
        "author",
        "modality",
        "source_type",
    ]
    text_limits = {
        "content": 220,
        "summary": 180,
        "visual_observation": 220,
        "linked_text_context": 180,
        "ocr": 120,
        "image_label": 80,
    }
    for key in preferred:
        if key not in fields:
            continue
        value = fields[key]
        limit = text_limits.get(key, 80)
        if isinstance(value, list):
            compact[key] = [_clip(item, limit) for item in value[:3]]
        else:
            compact[key] = _clip(value, limit)
    return {
        key: value
        for key, value in compact.items()
        if value is not None and value != "" and value != []
    }


@dataclass
class InteractiveObservation:
    pool_record_count: int
    pool_turn_count: int
    score_min: float
    score_max: float
    candidate_previews: List[Dict[str, Any]]
    evidence_count: int
    new_evidence_count: int
    last_retrieval_signature: Dict[str, Any]
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
            "last_retrieval_signature": self.last_retrieval_signature,
            "evidence_previews": self.evidence_previews,
            "stopped": self.stopped,
            "last_error": self.last_error,
            "has_question_image": self.has_question_image,
        }

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "pool_record_count": self.pool_record_count,
            "pool_turn_count": self.pool_turn_count,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "candidate_previews": [
                {
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "timestamp": item.get("timestamp"),
                    "modalities": item.get("modalities"),
                    "source_types": item.get("source_types"),
                    "summary": _clip(item.get("summary", ""), 180),
                }
                for item in self.candidate_previews[:3]
            ],
            "evidence_count": self.evidence_count,
            "new_evidence_count": self.new_evidence_count,
            "last_retrieval_signature": self.last_retrieval_signature,
            "evidence_previews": [
                {
                    "source": item.get("source"),
                    "fields": _compact_prompt_fields(
                        item.get("fields", {})
                        if isinstance(item.get("fields"), dict)
                        else {}
                    ),
                }
                for item in self.evidence_previews[-4:]
            ],
            "stopped": self.stopped,
            "last_error": _clip(self.last_error, 180),
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
        self.last_chunk_new_evidence_count = 0
        self.last_retrieval_signature: Dict[str, Any] = {}

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
        cloned.last_chunk_new_evidence_count = (
            self.last_chunk_new_evidence_count
        )
        cloned.last_retrieval_signature = dict(
            self.last_retrieval_signature
        )
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
                    self.last_retrieval_signature = {
                        "method": action.arguments.get("method", "hybrid"),
                        "top_k": action.arguments.get("top_k", 5),
                        "query": retrieval_query,
                        "scope": action.arguments.get("scope", "all"),
                    }
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
        self.last_chunk_new_evidence_count = (
            len(self.evidence) - evidence_before_chunk
        )
        return self.observation()

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

    def observation(self) -> InteractiveObservation:
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
            new_evidence_count=self.last_chunk_new_evidence_count,
            last_retrieval_signature=dict(self.last_retrieval_signature),
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
    diagnostic: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_reason: bool = True) -> Dict[str, Any]:
        data = {
            "correct": self.correct,
            "score": self.score,
            "prediction": self.prediction,
            "error": self.error,
            "image_ids_match": self.image_ids_match,
            "diagnostic": self.diagnostic,
        }
        if include_reason:
            data["reason"] = self.reason
        return data

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
        if not evidence:
            failure_type = "no_evidence"
            evidence_gap = "No evidence has been read for the answer model."
            recommended_change = (
                "Retrieve a focused candidate pool, then READ relevant fields."
            )
            action_hints: Dict[str, Any] = {
                "recommended_tool": "READ_OR_RETRIEVE",
                "needs_text_evidence": True,
                "needs_visual_evidence": False,
                "avoid_action": "STOP",
            }
        elif self.error:
            failure_type = "answer_validation_error"
            evidence_gap = "The answer validation call failed."
            recommended_change = (
                "Continue with a different evidence path and validate again."
            )
            action_hints = {
                "recommended_tool": "RETRIEVE",
                "recommended_retrieval_method": "hybrid",
                "avoid_action": "STOP",
            }
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
            action_hints = {
                "recommended_tool": (
                    "INSPECT_RAW" if can_inspect_raw else "RETRIEVE"
                ),
                "recommended_retrieval_method": "vision",
                "needs_visual_evidence": True,
                "avoid_action": "STOP",
            }
        elif self.diagnostic:
            failure_type = self.diagnostic.get(
                "failure_type",
                "answer_mismatch",
            )
            evidence_gap = self.diagnostic.get(
                "evidence_gap",
                "The current evidence does not support the required answer.",
            )
            recommended_change = self.diagnostic.get(
                "recommended_change",
                "Change the retrieval query or candidate pool, then read the "
                "new evidence.",
            )
            action_hints = {
                key: value
                for key, value in self.diagnostic.items()
                if key
                in {
                    "recommended_tool",
                    "recommended_retrieval_method",
                    "recommended_query_focus",
                    "needs_text_evidence",
                    "needs_visual_evidence",
                    "needs_neighbor_context",
                    "avoid_action",
                }
            }
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
            action_hints = {
                "recommended_tool": "RETRIEVE",
                "recommended_retrieval_method": "hybrid",
                "avoid_action": "INSPECT_RAW",
            }
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
            action_hints = {
                "recommended_tool": "RETRIEVE",
                "recommended_retrieval_method": "hybrid",
                "needs_text_evidence": True,
                "avoid_action": "STOP",
            }

        diagnostic = {
            "failure_type": failure_type,
            "predicted_answer": _clip(self.prediction, 300),
            "answer_score": round(float(self.score), 4),
            "image_ids_match": self.image_ids_match,
            "evidence_gap": evidence_gap,
            "recommended_change": recommended_change,
            **action_hints,
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
    """Validate a trajectory through the answer model.

    If the answer model exposes ``assess_evidence``, use it as the strict
    training-time assessor: it sees the gold answer and current evidence, then
    reports whether the evidence is sufficient and what is missing. Otherwise
    fall back to answer generation followed by an external judge.
    """

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
        assessment_error = ""
        try:
            assess_evidence = getattr(
                self.answer_model,
                "assess_evidence",
                None,
            )
            if callable(assess_evidence):
                try:
                    data = assess_evidence(
                        query,
                        gold_answer,
                        evidence,
                        question_image=question_image,
                    )
                except Exception as exc:
                    assessment_error = str(exc)
                else:
                    answerable = bool(data.get("answerable"))
                    try:
                        score = float(
                            data.get(
                                "score",
                                1.0 if answerable else 0.0,
                            )
                        )
                    except (TypeError, ValueError):
                        score = 1.0 if answerable else 0.0
                    score = max(0.0, min(1.0, score))
                    prediction = str(
                        data.get("predicted_answer")
                        or data.get("answer")
                        or ""
                    )
                    reason = str(data.get("reason") or "")
                    gold_ids = set(
                        PUBLIC_IMAGE_ID_PATTERN.findall(gold_answer)
                    )
                    predicted_ids = set(
                        PUBLIC_IMAGE_ID_PATTERN.findall(prediction)
                    )
                    image_ids_match = (
                        predicted_ids == gold_ids if gold_ids else None
                    )
                    correct = (
                        answerable
                        and score >= self.min_score
                        and image_ids_match is not False
                    )
                    diagnostic = {}
                    for key in (
                        "failure_type",
                        "evidence_gap",
                        "recommended_change",
                        "recommended_tool",
                        "recommended_retrieval_method",
                        "recommended_query_focus",
                        "needs_text_evidence",
                        "needs_visual_evidence",
                        "needs_neighbor_context",
                        "avoid_action",
                    ):
                        value = data.get(key)
                        if value is not None and value != "" and value != []:
                            diagnostic[key] = value
                    return AnswerValidationResult(
                        correct=correct,
                        score=score,
                        prediction=prediction,
                        reason=reason,
                        image_ids_match=image_ids_match,
                        diagnostic=diagnostic,
                    )

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
            diagnostic: Dict[str, str] = {}
            diagnose_failure = getattr(
                self.judge,
                "diagnose_failure",
                None,
            )
            if not correct and callable(diagnose_failure):
                try:
                    diagnostic = diagnose_failure(
                        query,
                        prediction,
                        evidence,
                    )
                except Exception:
                    diagnostic = {}
            if not correct and not diagnostic and assessment_error:
                diagnostic = {
                    "failure_type": "assessment_parse_error",
                    "evidence_gap": (
                        "The structured evidence assessor failed to return "
                        "valid JSON, and the fallback answer path did not "
                        "identify a specific evidence gap."
                    ),
                    "recommended_change": (
                        "Try a different evidence path, then validate again."
                    ),
                    "recommended_tool": "RETRIEVE",
                    "recommended_retrieval_method": "hybrid",
                    "avoid_action": "STOP",
                }
            return AnswerValidationResult(
                correct=correct,
                score=score,
                prediction=prediction,
                reason=str(reason or "") or (
                    "Used answer+judge fallback after assessment JSON error."
                    if assessment_error
                    else ""
                ),
                image_ids_match=image_ids_match,
                diagnostic=diagnostic,
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
{json.dumps(observation.to_prompt_dict(), ensure_ascii=False)}
"""


def build_simple_student_policy_prompt(
    query: str,
    history: List[ToolAction],
    observation: InteractiveObservation,
    schema: str,
) -> str:
    action_count = (
        "exactly 1 tool action"
        if "exactly one executable action" in schema
        else "1-3 tool actions"
    )
    action_shape = (
        "[{\"tool\":\"READ\",\"fields\":[\"summary\"]}]"
        if "exactly one executable action" in schema
        else "[{\"tool\":\"RETRIEVE\",\"method\":\"hybrid\",\"top_k\":5,\"scope\":\"all\"}]"
    )
    return f"""You are a memory-tool policy.
Choose the next executable action.
Use only the available tools and the current observation.
Do not answer the user query. Do not assume hidden facts.
Return only a JSON array of {action_count}.
Do not mention hidden labels, hidden support, private signals, memory IDs, or
file paths.

Available tools:
{schema}

User query:
{query}

Executed actions:
{json.dumps([action.to_dict() for action in history], ensure_ascii=False)}

Current observation:
{json.dumps(observation.to_prompt_dict(), ensure_ascii=False)}

Final JSON shape:
{action_shape}
"""


def build_compact_interactive_schema(allow_inspect_raw: bool = False) -> str:
    tools = [
        (
            "RETRIEVE(method=bm25|dense|vision|hybrid, top_k, query?, scope?): "
            "search memory and replace/extend the candidate pool; it does NOT "
            "add answer evidence"
        ),
        (
            "READ(fields): read text and metadata from the current candidate "
            "pool into answer evidence"
        ),
        (
            "EXPAND_NEIGHBORS(window): add nearby turns around current "
            "candidates to recover surrounding context; it does NOT read them"
        ),
        (
            "FILTER(field, op, value): narrow the current candidate pool by "
            "metadata; it does NOT add evidence"
        ),
        (
            "SORT(field, order): reorder the current candidate pool; it does "
            "NOT add evidence"
        ),
        (
            "TOPK(k): keep only the first k current candidates; it does NOT "
            "add evidence"
        ),
    ]
    if allow_inspect_raw:
        tools.append(
            "INSPECT_RAW(current_pool): inspect raw images in the current pool "
            "and add query-relevant visual observations as evidence"
        )
    tools.append(
        "STOP(): finish only when the accumulated evidence is sufficient to "
        "answer; STOP does not retrieve or read anything"
    )
    return "\n".join(f"- {tool}" for tool in tools)


def _compact_feedback(feedback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not feedback:
        return {}
    compact = {
        key: feedback[key]
        for key in (
            "answerable",
            "relevance",
            "completeness",
            "continue_required",
        )
        if key in feedback
    }
    diagnostic = feedback.get("failure_diagnostic")
    if isinstance(diagnostic, dict):
        compact["failure_diagnostic"] = {
            key: (
                diagnostic.get(key)
                if isinstance(diagnostic.get(key), bool)
                else _clip_text(diagnostic.get(key), 180)
            )
            for key in (
                "failure_type",
                "evidence_gap",
                "recommended_change",
                "recommended_tool",
                "recommended_retrieval_method",
                "recommended_query_focus",
                "needs_text_evidence",
                "needs_visual_evidence",
                "needs_neighbor_context",
                "avoid_action",
            )
            if diagnostic.get(key) not in {"", None}
        }
    return compact


def _planner_observation_state(
    observation: InteractiveObservation,
) -> Dict[str, Any]:
    prompt_observation = observation.to_prompt_dict()
    return {
        "pool": {
            "records": observation.pool_record_count,
            "turns": observation.pool_turn_count,
            "score_min": observation.score_min,
            "score_max": observation.score_max,
        },
        "candidates": [
            {
                "rank": item.get("rank"),
                "score": item.get("score"),
                "time": item.get("timestamp") or item.get("session_date"),
                "modalities": item.get("modalities"),
                "summary": _clip_text(item.get("summary"), 80),
            }
            for item in prompt_observation.get("candidate_previews", [])[:2]
        ],
        "evidence": {
            "count": observation.evidence_count,
            "new": observation.new_evidence_count,
            "items": [
                {
                    "source": item.get("source"),
                    "fields": _clip_text(item.get("fields"), 120),
                }
                for item in prompt_observation.get("evidence_previews", [])[-2:]
            ],
        },
        "last_retrieval": observation.last_retrieval_signature,
        "stopped": observation.stopped,
        "last_error": _clip_text(observation.last_error, 120),
        "has_question_image": observation.has_question_image,
    }


def build_compact_planner_prompt(
    query: str,
    history: List[ToolAction],
    observation: InteractiveObservation,
    allow_inspect_raw: bool,
    candidate_count: int,
    privileged_feedback: Optional[Dict[str, Any]] = None,
    max_actions_per_candidate: int = 3,
) -> str:
    payload = {
        "q": query,
        "history": [action.to_dict() for action in history],
        "obs": _planner_observation_state(observation),
    }
    feedback = _compact_feedback(privileged_feedback)
    if feedback:
        payload["fb"] = feedback
    atomic = max(1, int(max_actions_per_candidate)) == 1
    action_limit = (
        "exactly 1 action each"
        if atomic
        else "1-3 actions each"
    )
    retrieve_guidance = (
        "- RETRIEVE alone cannot support an answer. If you retrieve now, the "
        "next observation must be used to decide whether to READ, filter, "
        "inspect, or search again."
        if atomic
        else (
            "- RETRIEVE alone cannot support an answer. Usually follow a "
            "useful retrieval with READ in the same chunk. If the missing "
            "evidence is visual, use INSPECT_RAW; if the missing evidence "
            "may be in nearby dialogue turns, use EXPAND_NEIGHBORS then READ."
        )
    )
    final_shape = (
        '{"candidates":[{"diagnosis":"why this is the right next step",'
        '"next_tool":"RETRIEVE","expected_gain":"find a focused candidate '
        'pool","actions":[{"tool":"RETRIEVE","method":"vision","top_k":5,'
        '"scope":"all"}]}]}'
        if atomic
        else (
            '{"candidates":[{"diagnosis":"why this chunk addresses the '
            'current feedback","next_tool":"RETRIEVE","expected_gain":"find '
            'and read a focused candidate pool","actions":[{"tool":'
            '"RETRIEVE","method":"vision","top_k":5,"scope":"all"},'
            '{"tool":"READ","fields":["summary","content","ocr",'
            '"timestamp","session_date","turn_id","author","modality",'
            '"source_type","raw_pointer"]}]}]}'
        )
    )
    return f"""Return a memory-tool policy JSON.

State:
{json.dumps(payload, ensure_ascii=False)}

Tools:
{build_compact_interactive_schema(allow_inspect_raw)}

Constraints:
- {max(1, candidate_count)} candidates; {action_limit}.
- Pool-changing actions before READ/INSPECT_RAW.
- Analyze the query before choosing RETRIEVE.method.
- Use bm25 for exact names, IDs, dates, quoted phrases, or distinctive words.
- Use dense for semantic text memory when wording may differ.
- Use vision for SigLIP visual search over memory images. Prefer it when
  obs.has_question_image is true, or when the query asks what is visible,
  which image matches, visual similarity, object identity, color, layout, or
  fine-grained visual attributes.
- Use hybrid when both text/caption clues and visual evidence are useful, or
  when unsure which route should dominate.
{retrieve_guidance}
- If candidates exist but evidence is empty, read or inspect them instead of
  repeating the same retrieval.
- Do not STOP while relevant candidates remain unread or uninspected.
- No memory IDs or file paths.
- If fb says visual/image gap, use INSPECT_RAW.
- If fb says text/context/date/person relation is missing after a READ,
  consider EXPAND_NEIGHBORS, FILTER, SORT, or a different retrieval route
  instead of another similar RETRIEVE.

Answer-model feedback:
- fb is a privileged assessment of whether the current evidence can support an
  answer. Use it as the main diagnosis, but do not copy hidden answer content.
- If fb.continue_required is true, do not STOP. Choose the next action that
  most directly addresses fb.failure_diagnostic.evidence_gap.
- If fb.failure_diagnostic has recommended_tool, make at least one candidate
  follow that tool unless the observation makes it impossible.
- If recommended_tool is READ, read the current candidate pool instead of
  retrieving again. If it is INSPECT_RAW, inspect current visual candidates. If
  it is RETRIEVE, change the retrieval method, query focus, scope, or top_k so
  the attempt is not a repeat of last_retrieval. If it is EXPAND_NEIGHBORS,
  recover surrounding turns before reading. If it names STOP, stop only when
  fb says the evidence is answerable.
- Respect avoid_action as a warning about a repeated or unhelpful move.
- Treat repeated failure as a signal to switch tools. For example, after a
  failed semantic search try bm25/vision/hybrid with a focused query; after
  reading a plausible memory but missing chronology, expand neighbors or sort
  by time; after seeing image candidates but lacking visual confirmation,
  inspect raw images.

Planning guidance:
- evidence.new is the evidence added by the most recent action chunk.
  last_retrieval is the exact retrieval that actually ran.
- When evidence.new is 0, the last attempt made no progress. Repeating the
  same retrieval is likely to return the same pool, so reconsider the search
  rather than merely trying it again.
- Use the observed evidence gap to make the next attempt meaningfully
  different. Depending on the state, this may mean expressing the missing
  concept in a focused query, changing retrieval breadth or method, exploring
  neighboring turns, or inspecting relevant raw images.
- In chunk mode, prefer a compact repair plan over a single habitual action:
  retrieve+read for a new pool, expand+read for surrounding context,
  filter/sort+read for narrowing or temporal questions, inspect_raw for
  visual verification. Keep the chunk short and purposeful.
- Candidate metadata and executable actions must agree. If next_tool says a
  more targeted search or broader context is needed, the action parameters
  should visibly implement that change.
- Each candidate should include a short diagnosis explaining which feedback
  gap it addresses. Prefer candidates with different tools or retrieval
  methods when multiple plausible repairs exist.
- Prefer a deliberate step whose effect can be judged from the next
  observation.

Final JSON shape (format example only; choose the method from query analysis):
{final_shape}
"""


def _extract_candidates_object(raw: str) -> Dict[str, Any]:
    try:
        value = json.loads(str(raw or "").strip())
        if isinstance(value, dict) and isinstance(value.get("candidates"), list):
            return value
    except json.JSONDecodeError:
        pass

    text = str(raw or "")
    candidates: List[Dict[str, Any]] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if (
                        isinstance(value, dict)
                        and isinstance(value.get("candidates"), list)
                    ):
                        candidates.append(value)
                    break
    if candidates:
        return candidates[-1]
    if "[empty_content_with_reasoning]" in text:
        raise ValueError("planner returned reasoning without final candidates JSON")
    return extract_json_object(raw)


def fallback_action_chunks(
    observation: InteractiveObservation,
    validator: InteractiveActionValidator,
    candidate_count: int = 2,
) -> List[List[ToolAction]]:
    """General recovery actions derived only from online executor state."""
    candidates: List[List[ToolAction]] = []
    if validator.max_chunk_actions == 1:
        if not observation.candidate_previews:
            candidates.append(
                [
                    ToolAction(
                        "RETRIEVE",
                        {"method": "hybrid", "top_k": 5, "scope": "all"},
                    )
                ]
            )
        elif observation.evidence_count == 0:
            candidates.append(
                [ToolAction("READ", {"fields": list(DEFAULT_READ_FIELDS)})]
            )
        else:
            if validator.allow_inspect_raw:
                candidates.append(raw_inspect_chunk())
            candidates.append(
                [ToolAction("EXPAND_NEIGHBORS", {"window": 1})]
            )
            candidates.append([ToolAction("STOP")])
        validated = []
        for candidate in candidates:
            try:
                validated.append(validator.validate(candidate))
            except InteractiveValidationError:
                continue
        return validated[: max(1, candidate_count)]

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
            candidates.append(raw_inspect_chunk())
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
            candidates.append(raw_inspect_chunk())
    validated = []
    for candidate in candidates:
        try:
            validated.append(validator.validate(candidate))
        except InteractiveValidationError:
            continue
    return validated[: max(1, candidate_count)]


class ChatInteractivePlanner:
    """Planner used by student policy rollout and teacher search."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        validator: InteractiveActionValidator,
        max_tokens: int = 768,
        thinking_token_budget: Optional[int] = None,
        prompt_mode: str = "teacher_compact",
        enable_thinking: Optional[bool] = None,
    ):
        self.client = client
        self.validator = validator
        self.max_tokens = max_tokens
        self.thinking_token_budget = thinking_token_budget
        self.enable_thinking = enable_thinking
        if prompt_mode not in {"teacher_compact", "student_simple"}:
            raise ValueError(f"invalid planner prompt mode: {prompt_mode}")
        self.prompt_mode = prompt_mode
        self.calls = 0
        self.last_raw_response = ""
        self.last_candidate_sources: Dict[str, str] = {}
        self.last_candidate_rationales: Dict[str, Dict[str, str]] = {}

    def _parse_candidate(
        self,
        candidate: Any,
    ) -> tuple[Optional[List[ToolAction]], Dict[str, str]]:
        rationale: Dict[str, str] = {}
        action_values = candidate
        if isinstance(candidate, dict):
            action_values = candidate.get("actions")
            rationale = {
                "diagnosis": _clip_text(candidate.get("diagnosis"), 240),
                "next_tool": _clip_text(candidate.get("next_tool"), 80),
                "expected_gain": _clip_text(
                    candidate.get("expected_gain"),
                    240,
                ),
            }
            rationale = {
                key: value for key, value in rationale.items() if value
            }
        if not isinstance(action_values, list):
            return None, {}
        try:
            return self.validator.repair(action_values), rationale
        except Exception:
            return None, {}

    def _completion_kwargs(self) -> Dict[str, Any]:
        completion_kwargs: Dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        extra_body: Dict[str, Any] = {}
        if (
            self.thinking_token_budget is not None
            and self.thinking_token_budget > 0
            and self.enable_thinking is not False
        ):
            extra_body["thinking_token_budget"] = self.thinking_token_budget
        if self.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": bool(self.enable_thinking)
            }
            if self.enable_thinking is False:
                completion_kwargs["prefill_assistant"] = "</think>\n\n"
        if extra_body:
            completion_kwargs["extra_body"] = extra_body
        return completion_kwargs

    def _propose_student_simple(
        self,
        query: str,
        history: List[ToolAction],
        observation: InteractiveObservation,
    ) -> List[List[ToolAction]]:
        prompt = build_simple_student_policy_prompt(
            query=query,
            history=history,
            observation=observation,
            schema=self.validator.schema_text(),
        )
        raw = self.client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Think privately if enabled, then output only the "
                        "final JSON action array."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **self._completion_kwargs(),
        )
        self.last_raw_response = raw
        actions = self.validator.repair(extract_json_array(raw))
        signature = _actions_signature(actions)
        self.last_candidate_sources = {signature: "planner"}
        self.last_candidate_rationales = {}
        return [actions]

    def propose(
        self,
        query: str,
        history: List[ToolAction],
        observation: InteractiveObservation,
        candidate_count: int = 3,
        privileged_feedback: Optional[Dict[str, Any]] = None,
    ) -> List[List[ToolAction]]:
        self.calls += 1
        if self.prompt_mode == "student_simple":
            return self._propose_student_simple(query, history, observation)
        prompt = build_compact_planner_prompt(
            query=query,
            history=history,
            observation=observation,
            allow_inspect_raw=self.validator.allow_inspect_raw,
            candidate_count=candidate_count,
            privileged_feedback=privileged_feedback,
            max_actions_per_candidate=self.validator.max_chunk_actions,
        )
        raw = self.client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Think in at most two short sentences, then write "
                        "</think> and output only the final JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **self._completion_kwargs(),
        )
        self.last_raw_response = raw
        value = _extract_candidates_object(raw)
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("planner response has no candidates list")
        validated = []
        signatures = set()
        self.last_candidate_sources = {}
        self.last_candidate_rationales = {}
        for candidate in candidates:
            actions, rationale = self._parse_candidate(candidate)
            if actions is None:
                continue
            signature = _actions_signature(actions)
            if signature not in signatures:
                validated.append(actions)
                signatures.add(signature)
                if isinstance(candidate, dict):
                    raw_values = candidate.get("actions")
                else:
                    raw_values = candidate
                raw_signature = ""
                if isinstance(raw_values, list):
                    raw_signature = json.dumps(
                        [
                            ToolAction.from_dict(value).to_dict()
                            for value in raw_values
                            if isinstance(value, dict)
                        ],
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                source = (
                    "planner"
                    if raw_signature == signature
                    else "planner_repaired"
                )
                self.last_candidate_sources[signature] = source
                if rationale:
                    self.last_candidate_rationales[signature] = rationale
        if not validated:
            raise ValueError("planner produced no valid action chunks")
        return validated[: max(1, candidate_count)]


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
    planner_rationale: Dict[str, str] = field(default_factory=dict)

    def sft_example(
        self,
        sample_id: str,
        step_index: int,
        query: str,
        schema: str,
        allow_inspect_raw: bool = False,
    ) -> SFTExample:
        student_input = build_simple_student_policy_prompt(
            query,
            self.history,
            self.observation,
            schema,
        )
        teacher_input = build_compact_planner_prompt(
            query=query,
            history=self.history,
            observation=self.observation,
            allow_inspect_raw=allow_inspect_raw,
            candidate_count=1,
            privileged_feedback=self.privileged_feedback,
            max_actions_per_candidate=(
                1
                if "exactly one executable action" in schema
                else 3
            ),
        )
        return SFTExample(
            sample_id=f"{sample_id}:step:{step_index}",
            input=student_input,
            target=json.dumps(
                [action.to_dict() for action in self.actions],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            round_index=step_index,
            metadata={
                "evidence_count_after": (
                    self.observation_after.evidence_count
                ),
                "opd": {
                    "student_prompt_template": "simple_tools_v1",
                    "teacher_prompt_template": "compact_planner_v1",
                    "teacher_input": teacher_input,
                    "teacher_privileged_feedback_used": bool(
                        self.privileged_feedback
                    ),
                    "action_source": self.action_source,
                    "teacher_rationale": dict(self.planner_rationale),
                },
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

    def score(
        self,
        *,
        action_cost: float = 0.0,
        evidence_cost: float = 0.0,
    ) -> tuple[Any, ...]:
        raw_evidence = sum(
            1 for item in self.session.evidence if item.source == "INSPECT_RAW"
        )
        no_op_reads = self._redundant_read_count()
        answer_score = (
            float(self.answer_validation.score)
            if self.answer_validation is not None
            else 0.0
        )
        path_cost = (
            float(action_cost) * len(self.session.history)
            + float(evidence_cost) * len(self.session.evidence)
        )
        return (
            int(
                self.answer_validation is not None
                and self.answer_validation.correct
            ),
            int(self.verification.answerable),
            answer_score,
            -path_cost,
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
        allow_inspect_raw: bool = False,
    ) -> List[SFTExample]:
        return [
            decision.sft_example(
                sample_id,
                index,
                query,
                schema,
                allow_inspect_raw=allow_inspect_raw,
            )
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
        verifier: Optional[EvidenceVerifier],
        validator: InteractiveActionValidator,
        retriever: Optional[TurnAwareHybridRetriever] = None,
        raw_inspector: Optional[Any] = None,
        max_rounds: int = 5,
        beam_size: int = 2,
        candidates_per_node: int = 3,
        max_actions: int = 15,
        max_evidence: int = 40,
        max_raw_inspections: int = 3,
        answer_validator: Optional[StrictAnswerValidator] = None,
        trajectory_action_cost: float = 0.0,
        trajectory_evidence_cost: float = 0.0,
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
        self.trajectory_action_cost = max(0.0, float(trajectory_action_cost))
        self.trajectory_evidence_cost = max(0.0, float(trajectory_evidence_cost))

    @staticmethod
    def _evidence_redundancy(evidence: List[EvidenceItem]) -> float:
        if not evidence:
            return 0.0
        signatures = {
            (
                item.memory_id,
                item.source,
                json.dumps(item.fields, sort_keys=True, default=str),
            )
            for item in evidence
        }
        return max(0.0, 1.0 - (len(signatures) / len(evidence)))

    def _validate_candidate_evidence(
        self,
        query: str,
        gold_answer: str,
        evidence: List[EvidenceItem],
        question_image: Optional[str],
    ) -> tuple[VerificationResult, Optional[AnswerValidationResult]]:
        redundancy = self._evidence_redundancy(evidence)
        if self.answer_validator is None:
            return (
                VerificationResult(
                    answerable=bool(evidence),
                    relevance=1.0 if evidence else 0.0,
                    completeness=1.0 if evidence else 0.0,
                    redundancy=redundancy,
                    reason=(
                        "No answer validator configured; non-empty evidence "
                        "is treated as tentatively answerable."
                        if evidence
                        else "No retrieved evidence was provided."
                    ),
                ),
                None,
            )

        answer_validation = self.answer_validator.evaluate(
            query,
            gold_answer,
            evidence,
            question_image=question_image,
        )
        if answer_validation.correct:
            return (
                VerificationResult(
                    answerable=True,
                    relevance=1.0,
                    completeness=1.0,
                    redundancy=redundancy,
                    reason=answer_validation.reason,
                ),
                answer_validation,
            )

        diagnostic = answer_validation.failure_feedback(
            evidence,
            can_inspect_raw=(
                self.validator.allow_inspect_raw
                and self.raw_inspector is not None
            ),
        )
        partial_score = max(0.0, min(0.49, float(answer_validation.score)))
        return (
            VerificationResult(
                answerable=False,
                relevance=partial_score,
                completeness=partial_score,
                redundancy=redundancy,
                reason=(
                    "Answer-model evidence assessment failed: "
                    + (answer_validation.reason or answer_validation.error)
                ),
                error=answer_validation.error,
                diagnostic=diagnostic,
            ),
            answer_validation,
        )

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
                    candidate_rationales = dict(
                        getattr(
                            self.planner,
                            "last_candidate_rationales",
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
                    candidate_rationales = {}
                for chunk in chunks:
                    if len(node.session.history) + len(chunk) > self.max_actions:
                        continue
                    child_session = node.session.clone()
                    try:
                        observation_after = child_session.execute_chunk(chunk)
                    except Exception:
                        continue
                    verification, answer_validation = (
                        self._validate_candidate_evidence(
                            query,
                            gold_answer,
                            child_session.evidence,
                            question_image,
                        )
                    )
                    if (
                        answer_validation is not None
                        and not answer_validation.correct
                    ):
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
                                    answer_validation.to_dict(
                                        include_reason=False
                                    )
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
                            _actions_signature(chunk),
                            "controller_fallback",
                        ),
                        planner_rationale=candidate_rationales.get(
                            _actions_signature(chunk),
                            {},
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
                                        action_source="answer_stop",
                                        planner_rationale={
                                            "diagnosis": (
                                                "answer model judged current "
                                                "evidence sufficient"
                                            ),
                                            "next_tool": "STOP",
                                        },
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
                if previous is None or child.score(
                    action_cost=self.trajectory_action_cost,
                    evidence_cost=self.trajectory_evidence_cost,
                ) > previous.score(
                    action_cost=self.trajectory_action_cost,
                    evidence_cost=self.trajectory_evidence_cost,
                ):
                    deduped[signature] = child
            beam = sorted(
                deduped.values(),
                key=lambda node: node.score(
                    action_cost=self.trajectory_action_cost,
                    evidence_cost=self.trajectory_evidence_cost,
                ),
                reverse=True,
            )[: self.beam_size]

        pool = [*finished, *beam]
        if not pool:
            pool = [SearchNode(session=initial)]
        selected = max(
            pool,
            key=lambda node: node.score(
                action_cost=self.trajectory_action_cost,
                evidence_cost=self.trajectory_evidence_cost,
            ),
        )
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
                    planner_rationale={
                        "diagnosis": "search budget ended before a better action",
                        "next_tool": "STOP",
                    },
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
            verifier_calls=0,
            answer_validation=selected.answer_validation,
            answer_validator_calls=(
                getattr(self.answer_validator, "calls", 0)
                - answer_validator_calls_before
            ),
            failure_diagnostics=failure_diagnostics,
        )
