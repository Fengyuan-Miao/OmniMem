"""Strict executable action schema for OPD-MM policies."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from .models import ToolAction


ALLOWED_TOOLS = {
    "FILTER",
    "SORT",
    "TOPK",
    "RETRIEVE",
    "READ",
    "INSPECT_RAW",
    "STOP",
}
FILTER_FIELDS = {"modality", "author", "source_type", "timestamp", "status"}
FILTER_OPS = {"eq", "neq", "before", "after", "contains"}
SORT_FIELDS = {"timestamp", "turn_id", "score"}
SORT_ORDERS = {"asc", "desc"}
RETRIEVAL_METHODS = {"bm25", "dense", "hybrid"}
READ_FIELDS = {
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
}
INSPECT_TARGETS = {"current_pool"}
INSPECT_INSTRUCTIONS = {"answer_query_related_visual_details"}
FORBIDDEN_ARGUMENT_KEYS = {
    "memory_id",
    "memory_ids",
    "candidate_id",
    "candidate_ids",
    "query",
    "search_query",
}
MEMORY_ID_PATTERN = re.compile(r"\b(?:m|memory|mau)[-_]?\d+\b", re.IGNORECASE)

TOOL_SCHEMA_TEXT = """Allowed executable tools:
FILTER(field=modality|author|source_type|timestamp|status,
       op=eq|neq|before|after|contains, value=...)
SORT(field=timestamp|turn_id|score, order=asc|desc)
TOPK(k=positive integer)
RETRIEVE(method=bm25|dense|hybrid, top_k=positive integer)
READ(fields=[summary|content|ocr|timestamp|session_date|turn_id|author|modality|source_type|raw_pointer])
STOP()

Return only a JSON array of tool calls. Do not emit memory IDs. RETRIEVE always
uses the original user query and therefore accepts no query argument."""


def build_tool_schema(allow_inspect_raw: bool = True) -> str:
    lines = TOOL_SCHEMA_TEXT.splitlines()
    if allow_inspect_raw:
        stop_index = lines.index("STOP()")
        lines[stop_index:stop_index] = [
            "INSPECT_RAW(target=current_pool,",
            "            instruction=answer_query_related_visual_details)",
        ]
    return "\n".join(lines)


class TrajectoryValidationError(ValueError):
    pass


class TrajectoryValidator:
    def __init__(
        self,
        max_actions: int = 8,
        max_top_k: int = 50,
        allow_inspect_raw: bool = True,
    ):
        self.max_actions = max(1, int(max_actions))
        self.max_top_k = max(1, int(max_top_k))
        self.allow_inspect_raw = bool(allow_inspect_raw)

    def schema_text(self) -> str:
        return build_tool_schema(self.allow_inspect_raw)

    def validate(self, values: Iterable[Dict[str, Any] | ToolAction]) -> List[ToolAction]:
        actions = [
            value if isinstance(value, ToolAction) else ToolAction.from_dict(value)
            for value in values
        ]
        if not actions:
            raise TrajectoryValidationError("trajectory is empty")
        if len(actions) > self.max_actions:
            raise TrajectoryValidationError(
                f"trajectory has {len(actions)} actions; maximum is {self.max_actions}"
            )

        validated = []
        for index, action in enumerate(actions):
            self._validate_action(action, index)
            if action.tool == "STOP" and index != len(actions) - 1:
                raise TrajectoryValidationError("STOP must be the final action")
            validated.append(action)
        if validated[-1].tool != "STOP":
            if len(validated) >= self.max_actions:
                raise TrajectoryValidationError("trajectory must end with STOP")
            validated.append(ToolAction("STOP"))
        return validated

    def _validate_action(self, action: ToolAction, index: int) -> None:
        if action.tool not in ALLOWED_TOOLS:
            raise TrajectoryValidationError(
                f"action {index}: unsupported tool {action.tool!r}"
            )
        if action.tool == "INSPECT_RAW" and not self.allow_inspect_raw:
            raise TrajectoryValidationError(
                f"action {index}: INSPECT_RAW is unavailable in this run"
            )
        forbidden = FORBIDDEN_ARGUMENT_KEYS & set(action.arguments)
        if forbidden:
            raise TrajectoryValidationError(
                f"action {index}: forbidden arguments {sorted(forbidden)}"
            )
        for value in action.arguments.values():
            if MEMORY_ID_PATTERN.search(json.dumps(value, ensure_ascii=False)):
                raise TrajectoryValidationError(
                    f"action {index}: memory IDs are not allowed"
                )

        validator = getattr(self, f"_validate_{action.tool.lower()}")
        validator(action.arguments, index)

    @staticmethod
    def _require_exact_keys(
        arguments: Dict[str, Any],
        required: set[str],
        optional: set[str],
        index: int,
    ) -> None:
        missing = required - set(arguments)
        unknown = set(arguments) - required - optional
        if missing:
            raise TrajectoryValidationError(
                f"action {index}: missing arguments {sorted(missing)}"
            )
        if unknown:
            raise TrajectoryValidationError(
                f"action {index}: unknown arguments {sorted(unknown)}"
            )

    def _validate_filter(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, {"field", "op", "value"}, set(), index)
        if args["field"] not in FILTER_FIELDS:
            raise TrajectoryValidationError(f"action {index}: invalid FILTER field")
        if args["op"] not in FILTER_OPS:
            raise TrajectoryValidationError(f"action {index}: invalid FILTER op")
        if not isinstance(args["value"], (str, int, float, bool)):
            raise TrajectoryValidationError(f"action {index}: invalid FILTER value")

    def _validate_sort(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, {"field", "order"}, set(), index)
        if args["field"] not in SORT_FIELDS or args["order"] not in SORT_ORDERS:
            raise TrajectoryValidationError(f"action {index}: invalid SORT arguments")

    def _validate_topk(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, {"k"}, set(), index)
        self._validate_k(args["k"], index, "k")

    def _validate_retrieve(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, set(), {"method", "top_k"}, index)
        if args.get("method", "hybrid") not in RETRIEVAL_METHODS:
            raise TrajectoryValidationError(f"action {index}: invalid RETRIEVE method")
        self._validate_k(args.get("top_k", 5), index, "top_k")

    def _validate_read(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, {"fields"}, set(), index)
        fields = args["fields"]
        if not isinstance(fields, list) or not fields:
            raise TrajectoryValidationError(f"action {index}: READ fields must be a list")
        unknown = set(fields) - READ_FIELDS
        if unknown:
            raise TrajectoryValidationError(
                f"action {index}: invalid READ fields {sorted(unknown)}"
            )

    def _validate_inspect_raw(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, set(), {"target", "instruction"}, index)
        if args.get("target", "current_pool") not in INSPECT_TARGETS:
            raise TrajectoryValidationError(f"action {index}: invalid INSPECT_RAW target")
        if (
            args.get("instruction", "answer_query_related_visual_details")
            not in INSPECT_INSTRUCTIONS
        ):
            raise TrajectoryValidationError(
                f"action {index}: invalid INSPECT_RAW instruction"
            )

    def _validate_stop(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, set(), set(), index)

    def _validate_k(self, value: Any, index: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TrajectoryValidationError(f"action {index}: {name} must be an integer")
        if value <= 0 or value > self.max_top_k:
            raise TrajectoryValidationError(
                f"action {index}: {name} must be between 1 and {self.max_top_k}"
            )
