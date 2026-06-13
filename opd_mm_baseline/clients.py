"""LLM-backed student, teacher, answer, inspection, and judge clients."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import EvidenceItem, ExecutionResult, PolicyOutput, ToolAction
from .schema import TrajectoryValidator


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(
        r'("tool"\s*:\s*"STOP")\s*:\s*\{\s*\}',
        r"\1",
        cleaned,
    )
    try:
        value = json.loads(cleaned)
        if isinstance(value, list):
            return value
    except json.JSONDecodeError:
        pass
    start = cleaned.find("[")
    if start < 0:
        raise ValueError("no JSON array found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                value = json.loads(cleaned[start : index + 1])
                if not isinstance(value, list):
                    raise ValueError("policy output is not a JSON array")
                return value
    raise ValueError("unterminated JSON array")


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(cleaned[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("response is not a JSON object")
                return value
    raise ValueError("unterminated JSON object")


def image_data_url(path: str | Path) -> str:
    source = Path(path)
    mime = mimetypes.guess_type(source.name)[0] or "image/jpeg"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        timeout: int = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'ollama'}",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"].get("content") or "").strip()


def build_student_prompt(
    query: str,
    tool_schema: Optional[str] = None,
) -> str:
    schema = tool_schema or TrajectoryValidator().schema_text()
    return f"""You are a multimodal memory retrieval planner.
Generate a short sequence of executable tool calls for the user query.

You cannot see the memory store, memory index, candidate memories, or memory IDs.
Do not invent answer words or a new search query. RETRIEVE automatically uses
the original user query. Use only the allowed schema.

{schema}

User query:
{query}
"""


class ChatStudentPolicy:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        validator: Optional[TrajectoryValidator] = None,
        max_tokens: int = 512,
    ):
        self.client = client
        self.validator = validator or TrajectoryValidator()
        self.max_tokens = max_tokens

    def generate_trace(self, query: str) -> PolicyOutput:
        raw = ""
        try:
            raw = self.client.complete(
                [
                    {
                        "role": "user",
                        "content": build_student_prompt(
                            query,
                            self.validator.schema_text(),
                        ),
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            values = extract_json_array(raw)
            actions = self.validator.validate(values)
            return PolicyOutput(actions=actions, raw_response=raw)
        except Exception as exc:
            return PolicyOutput(
                actions=[ToolAction("STOP")],
                raw_response=raw,
                error=str(exc),
            )


class ChatHindsightTeacher:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        validator: Optional[TrajectoryValidator] = None,
        max_tokens: int = 512,
        privilege_mode: str = "diagnostic",
    ):
        self.client = client
        self.validator = validator or TrajectoryValidator()
        self.max_tokens = max_tokens
        if privilege_mode not in {
            "minimal",
            "diagnostic",
            "oracle-feedback",
            "oracle-profile",
        }:
            raise ValueError(f"invalid teacher privilege mode: {privilege_mode}")
        self.privilege_mode = privilege_mode

    def correct(
        self,
        query: str,
        gold_answer: str,
        student_policy: PolicyOutput,
        student_answer: str,
        correct: bool,
        execution: Optional[ExecutionResult] = None,
        privileged_context: Optional[Dict[str, Any]] = None,
    ) -> PolicyOutput:
        return self._generate_correction(
            query=query,
            gold_answer=gold_answer,
            student_policy=student_policy,
            student_answer=student_answer,
            correct=correct,
            execution=execution,
            privileged_context=privileged_context,
        )

    def revise(
        self,
        query: str,
        gold_answer: str,
        student_policy: PolicyOutput,
        student_answer: str,
        correct: bool,
        previous_policy: PolicyOutput,
        replay_feedback: Dict[str, Any],
        attempt_index: int,
        execution: Optional[ExecutionResult] = None,
        privileged_context: Optional[Dict[str, Any]] = None,
    ) -> PolicyOutput:
        return self._generate_correction(
            query=query,
            gold_answer=gold_answer,
            student_policy=student_policy,
            student_answer=student_answer,
            correct=correct,
            execution=execution,
            privileged_context=privileged_context,
            previous_policy=previous_policy,
            replay_feedback=replay_feedback,
            attempt_index=attempt_index,
        )

    def _generate_correction(
        self,
        query: str,
        gold_answer: str,
        student_policy: PolicyOutput,
        student_answer: str,
        correct: bool,
        execution: Optional[ExecutionResult],
        privileged_context: Optional[Dict[str, Any]],
        previous_policy: Optional[PolicyOutput] = None,
        replay_feedback: Optional[Dict[str, Any]] = None,
        attempt_index: int = 0,
    ) -> PolicyOutput:
        student_trace = json.dumps(
            [action.to_dict() for action in student_policy.actions],
            ensure_ascii=False,
        )
        privilege = self._privileged_section(execution, privileged_context)
        revision = ""
        if previous_policy is not None and replay_feedback is not None:
            previous_trace = json.dumps(
                [action.to_dict() for action in previous_policy.actions],
                ensure_ascii=False,
            )
            revision = f"""
This is correction attempt {attempt_index}. The previous teacher candidate was:
{previous_trace}

Hidden replay feedback for that candidate:
{json.dumps(replay_feedback, ensure_ascii=False)}

Revise the candidate rather than repeating it. If support coverage is
incomplete, first reconsider retrieval method, useful metadata filters, and
tool ordering. Increase top_k only when those structural changes are
insufficient. If coverage is complete but evidence is broad, try a materially
more selective trajectory while preserving coverage.
"""
        oracle_rules = ""
        if self.privilege_mode == "oracle-profile":
            oracle_rules = """- When an oracle support-rank profile is present, choose a retrieval method and
  top_k that can actually reach the support. Do not copy support metadata into
  FILTER values.
- When verified_action_advice is present, follow its minimum_top_k requirement,
  READ the retrieved pool, and do not apply a smaller TOPK afterward.
"""
        prompt = f"""You are a hindsight teacher for multimodal memory retrieval.

You can see only the user query, gold answer, student tool trajectory, student
answer, correctness, and the explicitly provided training diagnostics below.
You cannot see the full memory store, full index, unobserved candidates, raw
file paths, or internal memory IDs.

Return a corrected abstract tool trajectory. Do not put gold-answer words into
a new query. RETRIEVE uses the original user query and accepts no query field.
Prefer a short, general, executable trajectory.

General correction principles:
- If retrieval evidence missed the answer, change retrieval/filter ordering or
  increase top_k; merely shortening the same failed plan is not a correction.
- For relevance questions, preserve RETRIEVE ranking; do not SORT by time after
  RETRIEVE unless temporal order is explicitly required.
- For recency/first/last questions, FILTER and SORT the hidden pool before TOPK.
- Always READ useful fields before STOP.
- Use INSPECT_RAW only when it is present in the allowed schema.
{oracle_rules}
{revision}

{self.validator.schema_text()}

User query:
{query}

Gold answer:
{gold_answer}

Student trajectory:
{student_trace}

Student answer:
{student_answer}

Student answer correct:
{str(bool(correct)).lower()}

{privilege}
"""
        raw = ""
        try:
            raw = self.client.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            actions = self.validator.validate(extract_json_array(raw))
            return PolicyOutput(actions=actions, raw_response=raw)
        except Exception as exc:
            return PolicyOutput(
                actions=list(student_policy.actions),
                raw_response=raw,
                error=str(exc),
            )

    def _privileged_section(
        self,
        execution: Optional[ExecutionResult],
        privileged_context: Optional[Dict[str, Any]],
    ) -> str:
        if self.privilege_mode == "minimal":
            return "Training diagnostics: unavailable."
        diagnostics: Dict[str, Any] = {
            "runtime_capabilities": {
                "inspect_raw_available": self.validator.allow_inspect_raw,
            }
        }
        if execution is not None:
            diagnostics["execution_steps"] = [
                {
                    "tool": step.action.tool,
                    "arguments": step.action.arguments,
                    "pool_before": step.pool_before,
                    "pool_after": step.pool_after,
                    "evidence_added": step.evidence_added,
                    "error": step.error,
                }
                for step in execution.steps
            ]
            diagnostics["observed_evidence"] = [
                {
                    "source": item.source,
                    "fields": {
                        key: self._sanitize_diagnostic_value(value)
                        for key, value in item.fields.items()
                        if key not in {"raw_pointer", "memory_id"}
                    },
                }
                for item in execution.evidence
            ]
            diagnostics["execution_error"] = execution.error
        if self.privilege_mode == "oracle-profile" and privileged_context:
            diagnostics["gold_support_abstract_profile"] = privileged_context
        elif self.privilege_mode == "oracle-feedback" and privileged_context:
            allowed = {
                "support_count",
                "modalities",
                "authors",
                "source_types",
                "has_raw_media",
            }
            diagnostics["gold_support_abstract_profile"] = {
                key: value
                for key, value in privileged_context.items()
                if key in allowed
            }
        return (
            "Training-only diagnostics (not available to the student):\n"
            + json.dumps(diagnostics, ensure_ascii=False)
        )

    @classmethod
    def _sanitize_diagnostic_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = re.sub(r"\bD\d+:IMG_\d+\b", "[public-image-id]", value)
            return value[:2000]
        if isinstance(value, list):
            return [cls._sanitize_diagnostic_value(item) for item in value[:20]]
        if isinstance(value, dict):
            return {
                str(key): cls._sanitize_diagnostic_value(item)
                for key, item in list(value.items())[:20]
            }
        return value


class PassthroughTeacher:
    def correct(
        self,
        query: str,
        gold_answer: str,
        student_policy: PolicyOutput,
        student_answer: str,
        correct: bool,
        execution: Optional[ExecutionResult] = None,
        privileged_context: Optional[Dict[str, Any]] = None,
    ) -> PolicyOutput:
        return PolicyOutput(actions=list(student_policy.actions))


class ChatRawInspector:
    def __init__(self, client: OpenAICompatibleClient, max_tokens: int = 160):
        self.client = client
        self.max_tokens = max_tokens

    def inspect(self, image_path: str, query: str) -> str:
        prompt = (
            "Inspect this memory image. Report only visible details relevant to "
            f"the query, without guessing.\nQuery: {query}"
        )
        return self.client.complete(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url(image_path)},
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )


class ChatAnswerModel:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        max_tokens: int = 128,
        max_images: int = 3,
    ):
        self.client = client
        self.max_tokens = max_tokens
        self.max_images = max(0, int(max_images))

    def answer(
        self,
        query: str,
        evidence: List[EvidenceItem],
        question_image: Optional[str] = None,
    ) -> str:
        public_evidence = []
        image_paths: List[str] = []
        for item in evidence:
            fields = dict(item.fields)
            pointer = fields.pop("raw_pointer", None)
            public_evidence.append({"source": item.source, **fields})
            if pointer and Path(str(pointer)).is_file() and pointer not in image_paths:
                image_paths.append(str(pointer))
        prompt = f"""Answer the user query using only the retrieved memory evidence.
If the evidence is insufficient, answer "Not mentioned." Be concise. Do not
mention internal memory IDs or file paths.

User query:
{query}

Retrieved evidence:
{json.dumps(public_evidence, ensure_ascii=False)}
"""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if question_image and Path(question_image).is_file():
            content.extend(
                [
                    {"type": "text", "text": "Attached image from the question:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(question_image)},
                    },
                ]
            )
        for pointer in image_paths[: self.max_images]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(pointer)},
                }
            )
        messages: List[Dict[str, Any]]
        if len(content) == 1:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": content}]
        return self.client.complete(
            messages,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )


class ChatAnswerJudge:
    def __init__(self, client: OpenAICompatibleClient, max_tokens: int = 192):
        self.client = client
        self.max_tokens = max_tokens

    def evaluate(
        self,
        query: str,
        prediction: str,
        gold_answer: str,
    ) -> tuple[bool, float, str]:
        prompt = f"""Judge whether the prediction correctly answers the question
relative to the gold answer. Semantic equivalence is sufficient. Respect
yes/no polarity and require all core entities for list answers.

Return only JSON:
{{"correct": true, "score": 1.0, "reason": "short reason"}}

Question: {query}
Gold answer: {gold_answer}
Prediction: {prediction}
"""
        raw = self.client.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        data = extract_json_object(raw)
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        return bool(data.get("correct")), score, str(data.get("reason") or "")


class HeuristicAnswerJudge:
    @staticmethod
    def _normalize(value: str) -> str:
        value = str(value or "").lower()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def evaluate(
        self,
        query: str,
        prediction: str,
        gold_answer: str,
    ) -> tuple[bool, float, str]:
        prediction_norm = self._normalize(prediction)
        gold_norm = self._normalize(gold_answer)
        exact = prediction_norm == gold_norm
        contains = bool(gold_norm) and gold_norm in prediction_norm
        correct = exact or contains
        return correct, float(correct), "exact_or_contains"
