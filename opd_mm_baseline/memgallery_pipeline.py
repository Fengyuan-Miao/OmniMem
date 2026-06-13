"""Mem-Gallery runner for the query-only OPD-MM baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dual_encoder_memory import MiniLMTextEncoder, SigLIPVisionEncoder
from omnimem.config import (
    PROJECT_ROOT,
    default_memgallery_dir,
    default_minilm_model,
    default_siglip_model,
    require_memgallery_dir,
)

from .clients import (
    ChatAnswerJudge,
    ChatAnswerModel,
    ChatHindsightTeacher,
    ChatRawInspector,
    ChatStudentPolicy,
    HeuristicAnswerJudge,
    OpenAICompatibleClient,
    PassthroughTeacher,
)
from .executor import ToolExecutor
from .memgallery import (
    IMAGE_ID_PATTERN,
    build_scenario_store,
    iter_scenario_paths,
    scenario_samples,
)
from .models import MemoryRecord, OPDRollout
from .retrieval import TurnAwareHybridRetriever
from .schema import TrajectoryValidator
from .training import OnPolicyDistiller


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "memgallery_opd_mm"


class EmptyAnswerModel:
    def answer(
        self,
        query: str,
        evidence: list[Any],
        question_image: Optional[str] = None,
    ) -> str:
        return ""


class AlwaysIncorrectJudge:
    def evaluate(
        self,
        query: str,
        prediction: str,
        gold_answer: str,
    ) -> tuple[bool, float, str]:
        return False, 0.0, "teacher_recall_only"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class ProgressBar:
    def __init__(self, total: int, label: str, enabled: bool = True):
        self.total = max(0, int(total))
        self.label = label
        self.enabled = enabled and self.total > 0
        self.current = 0

    def update(self, current: int, message: str = "") -> None:
        if not self.enabled:
            return
        self.current = min(max(0, current), self.total)
        fraction = self.current / self.total
        filled = round(24 * fraction)
        suffix = f" {message}" if message else ""
        sys.stderr.write(
            f"\r[{self.label}] [{'#' * filled}{'-' * (24 - filled)}] "
            f"{self.current}/{self.total} {fraction * 100:5.1f}%{suffix}"
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def parse_scenario_names(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item for item in re.split(r"[,;\s]+", value) if item]


def resolve_scenarios(args: argparse.Namespace) -> List[Path]:
    if args.all_scenarios:
        paths = list(iter_scenario_paths(args.data_dir))
    else:
        names = parse_scenario_names(args.scenarios)
        if not names:
            names = [args.scenario or "Academic_Animal_Pet_Research_Life"]
        paths = list(iter_scenario_paths(args.data_dir, names))
    paths = [path for path in paths if path.is_file()]
    if args.max_scenarios is not None:
        paths = paths[: max(0, args.max_scenarios)]
    return paths


def make_components(args: argparse.Namespace) -> Dict[str, Any]:
    validator = TrajectoryValidator(
        max_actions=args.max_actions,
        max_top_k=args.max_top_k,
        allow_inspect_raw=args.raw_inspection,
    )
    student_client = OpenAICompatibleClient(
        args.student_base_url,
        args.student_model,
        args.student_api_key,
    )
    student = ChatStudentPolicy(
        student_client,
        validator=validator,
        max_tokens=args.policy_max_tokens,
    )
    if args.mode == "collect-sft" and args.teacher_mode == "llm":
        teacher = ChatHindsightTeacher(
            OpenAICompatibleClient(
                args.teacher_base_url,
                args.teacher_model,
                args.teacher_api_key,
            ),
            validator=validator,
            max_tokens=args.policy_max_tokens,
            privilege_mode=args.teacher_privilege,
        )
    else:
        teacher = PassthroughTeacher()

    if args.teacher_recall_only:
        answer_model = EmptyAnswerModel()
        raw_inspector = None
    else:
        answer_client = OpenAICompatibleClient(
            args.answer_base_url,
            args.answer_model,
            args.answer_api_key,
        )
        answer_model = ChatAnswerModel(
            answer_client,
            max_tokens=args.answer_max_tokens,
            max_images=args.answer_max_images,
        )
        raw_inspector = (
            ChatRawInspector(answer_client, max_tokens=args.inspect_max_tokens)
            if args.raw_inspection
            else None
        )
    executor = ToolExecutor(
        retriever=TurnAwareHybridRetriever(args.hybrid_alpha),
        raw_inspector=raw_inspector,
        validator=validator,
        max_raw_inspections=args.max_raw_inspections,
    )
    if args.teacher_recall_only:
        judge = AlwaysIncorrectJudge()
    elif args.judge_mode == "llm":
        judge = ChatAnswerJudge(
            OpenAICompatibleClient(
                args.judge_base_url,
                args.judge_model,
                args.judge_api_key,
            ),
            max_tokens=args.judge_max_tokens,
        )
    else:
        judge = HeuristicAnswerJudge()
    return {
        "student": student,
        "teacher": teacher,
        "answer_model": answer_model,
        "executor": executor,
        "judge": judge,
    }


def evidence_image_ids(rollout: OPDRollout) -> List[str]:
    value = json.dumps(
        [item.to_dict() for item in rollout.execution.evidence],
        ensure_ascii=False,
    )
    return list(dict.fromkeys(IMAGE_ID_PATTERN.findall(value)))


def rollout_row(rollout: OPDRollout) -> Dict[str, Any]:
    row = rollout.to_dict()
    found_ids = evidence_image_ids(rollout)
    gold_ids = list(rollout.metadata.get("gold_image_ids") or [])
    row["evidence_image_ids"] = found_ids
    row["gold_image_recall_any"] = (
        bool(set(found_ids) & set(gold_ids)) if gold_ids else None
    )
    evidence_memory_ids = [
        item.memory_id for item in rollout.execution.evidence
    ]
    clue_turn_ids = list(rollout.metadata.get("gold_clue_turn_ids") or [])
    row["evidence_clue_recall_any"] = (
        any(
            memory_id == clue
            or memory_id.startswith(clue + ":")
            for memory_id in evidence_memory_ids
            for clue in clue_turn_ids
        )
        if clue_turn_ids
        else None
    )
    teacher_evidence = (
        rollout.teacher_execution.evidence
        if rollout.teacher_execution is not None
        else []
    )
    teacher_memory_ids = [item.memory_id for item in teacher_evidence]
    row["teacher_evidence_clue_recall_any"] = (
        any(
            memory_id == clue
            or memory_id.startswith(clue + ":")
            for memory_id in teacher_memory_ids
            for clue in clue_turn_ids
        )
        if clue_turn_ids
        else None
    )
    row["teacher_evidence_clue_recall_all"] = (
        all(
            any(
                memory_id == clue
                or memory_id.startswith(clue + ":")
                for memory_id in teacher_memory_ids
            )
            for clue in clue_turn_ids
        )
        if clue_turn_ids
        else None
    )
    selected_teacher_candidate = next(
        (
            item
            for item in rollout.teacher_candidate_diagnostics
            if item.get("selected")
        ),
        None,
    )
    support_record_count = int(
        (selected_teacher_candidate or {}).get("support_record_count") or 0
    )
    row["teacher_support_record_recall"] = (
        int(
            (selected_teacher_candidate or {}).get(
                "support_record_hit_count"
            )
            or 0
        )
        / support_record_count
        if support_record_count
        else None
    )
    teacher_evidence_json = json.dumps(
        [item.to_dict() for item in teacher_evidence],
        ensure_ascii=False,
    )
    teacher_image_ids = list(
        dict.fromkeys(IMAGE_ID_PATTERN.findall(teacher_evidence_json))
    )
    row["teacher_evidence_image_ids"] = teacher_image_ids
    row["teacher_gold_image_recall_any"] = (
        bool(set(teacher_image_ids) & set(gold_ids)) if gold_ids else None
    )
    row["retrieval_pool_scans"] = sum(
        step.pool_before
        for step in rollout.execution.steps
        if step.action.tool == "RETRIEVE"
    )
    row["tool_count"] = len(rollout.student_policy.actions)
    row["student_tools"] = [
        action.tool for action in rollout.student_policy.actions
    ]
    return row


def select_sft_example(row: Dict[str, Any], mode: str) -> bool:
    teacher_policy_error = bool((row.get("teacher_policy") or {}).get("error"))
    teacher_execution_error = bool(
        (row.get("teacher_execution") or {}).get("error")
    )
    if mode == "all":
        return True
    if teacher_policy_error or teacher_execution_error:
        return False
    if mode == "valid":
        return True
    support_result = row.get("teacher_evidence_clue_recall_any")
    return bool(support_result) if support_result is not None else bool(row.get("correct"))


def summarize(rows: List[Dict[str, Any]], memory_count: int, elapsed: float) -> Dict[str, Any]:
    def average(values: Iterable[float]) -> float:
        items = list(values)
        return sum(items) / len(items) if items else 0.0

    gold_image_rows = [
        row for row in rows if row.get("gold_image_recall_any") is not None
    ]
    clue_rows = [
        row for row in rows if row.get("evidence_clue_recall_any") is not None
    ]
    teacher_clue_rows = [
        row
        for row in rows
        if row.get("teacher_evidence_clue_recall_any") is not None
    ]
    teacher_all_clue_rows = [
        row
        for row in rows
        if row.get("teacher_evidence_clue_recall_all") is not None
    ]
    teacher_record_rows = [
        row
        for row in rows
        if row.get("teacher_support_record_recall") is not None
    ]
    by_point: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_point[str((row.get("metadata") or {}).get("point") or "unknown")].append(row)
    tool_counts = Counter(
        tool
        for row in rows
        for tool in row.get("student_tools") or []
    )
    teacher_shapes = Counter(
        ">".join(
            action.get("tool", "")
            for action in (row.get("teacher_policy") or {}).get("actions") or []
        )
        for row in rows
    )
    teacher_candidate_shapes = Counter(
        ">".join(
            action.get("tool", "")
            for action in candidate.get("actions") or []
        )
        for row in rows
        for candidate in row.get("teacher_candidate_diagnostics") or []
    )
    teacher_retrieve_top_ks = [
        int(action.get("top_k", 5))
        for row in rows
        for action in (row.get("teacher_policy") or {}).get("actions") or []
        if action.get("tool") == "RETRIEVE"
    ]
    exact_advice_matches = []
    for row in rows:
        advice = (
            ((row.get("metadata") or {}).get("teacher_privileged_context") or {})
            .get("verified_action_advice", {})
            .get("recommended")
        )
        retrieves = [
            action
            for action in (row.get("teacher_policy") or {}).get("actions") or []
            if action.get("tool") == "RETRIEVE"
        ]
        if advice and retrieves:
            exact_advice_matches.append(
                advice.get("method") == retrieves[0].get("method")
                and advice.get("minimum_top_k")
                == retrieves[0].get("top_k", 5)
            )

    def percentile(values: List[int], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int(max(0.0, min(1.0, fraction)) * (len(ordered) - 1))
        return float(ordered[index])

    return {
        "num_results": len(rows),
        "memory_count": memory_count,
        "primary_metric": "answer_accuracy",
        "answer_accuracy": average(float(row["correct"]) for row in rows),
        "answer_score": average(float(row["score"]) for row in rows),
        "evidence_image_recall_any": average(
            float(row["gold_image_recall_any"]) for row in gold_image_rows
        ),
        "gold_image_questions": len(gold_image_rows),
        "evidence_recall_at_k": average(
            float(row["evidence_clue_recall_any"]) for row in clue_rows
        ),
        "clue_labeled_questions": len(clue_rows),
        "teacher_evidence_recall_at_k": average(
            float(row["teacher_evidence_clue_recall_any"])
            for row in teacher_clue_rows
        ),
        "teacher_evidence_recall_all": average(
            float(row["teacher_evidence_clue_recall_all"])
            for row in teacher_all_clue_rows
        ),
        "teacher_support_record_recall": average(
            float(row["teacher_support_record_recall"])
            for row in teacher_record_rows
        ),
        "teacher_evidence_recall_delta": (
            average(
                float(row["teacher_evidence_clue_recall_any"])
                for row in teacher_clue_rows
            )
            - average(float(row["evidence_clue_recall_any"]) for row in clue_rows)
        ),
        "teacher_image_recall_any": average(
            float(row["teacher_gold_image_recall_any"])
            for row in gold_image_rows
        ),
        "student_policy_error_rate": average(
            float(bool((row.get("student_policy") or {}).get("error")))
            for row in rows
        ),
        "teacher_policy_error_rate": average(
            float(bool((row.get("teacher_policy") or {}).get("error")))
            for row in rows
        ),
        "teacher_execution_error_rate": average(
            float(bool((row.get("teacher_execution") or {}).get("error")))
            for row in rows
        ),
        "sft_examples_selected": sum(bool(row.get("sft_selected")) for row in rows),
        "sft_selection_rate": average(
            float(bool(row.get("sft_selected"))) for row in rows
        ),
        "execution_error_rate": average(
            float(bool((row.get("execution") or {}).get("error")))
            for row in rows
        ),
        "raw_image_invocation_rate": average(
            float((row.get("execution") or {}).get("raw_inspection_calls", 0) > 0)
            for row in rows
        ),
        "avg_raw_image_calls": average(
            float((row.get("execution") or {}).get("raw_inspection_calls", 0))
            for row in rows
        ),
        "avg_tool_steps": average(float(row.get("tool_count", 0)) for row in rows),
        "avg_evidence_items": average(
            float(len((row.get("execution") or {}).get("evidence") or []))
            for row in rows
        ),
        "avg_teacher_evidence_items": average(
            float(
                len(
                    (row.get("teacher_execution") or {}).get("evidence")
                    or []
                )
            )
            for row in rows
        ),
        "avg_retrieval_pool_scans": average(
            float(row.get("retrieval_pool_scans", 0)) for row in rows
        ),
        "tool_usage": dict(sorted(tool_counts.items())),
        "teacher_selection_sources": dict(
            sorted(
                Counter(
                    (row.get("metadata") or {}).get(
                        "teacher_selection_source",
                        "unknown",
                    )
                    for row in rows
                ).items()
            )
        ),
        "teacher_trajectory_shapes": dict(
            sorted(teacher_shapes.items(), key=lambda item: (-item[1], item[0]))
        ),
        "teacher_candidate_trajectory_shapes": dict(
            sorted(
                teacher_candidate_shapes.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "avg_teacher_candidate_count": average(
            float(len(row.get("teacher_candidate_diagnostics") or []))
            for row in rows
        ),
        "avg_teacher_retrieve_top_k": average(
            float(value) for value in teacher_retrieve_top_ks
        ),
        "p90_teacher_retrieve_top_k": percentile(
            teacher_retrieve_top_ks,
            0.9,
        ),
        "max_teacher_retrieve_top_k": (
            max(teacher_retrieve_top_ks) if teacher_retrieve_top_ks else 0
        ),
        "teacher_exact_oracle_advice_follow_rate": (
            average(float(value) for value in exact_advice_matches)
            if exact_advice_matches
            else None
        ),
        "elapsed_seconds": elapsed,
        "by_point": {
            point: {
                "count": len(items),
                "answer_accuracy": average(float(item["correct"]) for item in items),
                "answer_score": average(float(item["score"]) for item in items),
            }
            for point, items in sorted(by_point.items())
        },
    }


def write_memory_manifest(path: Path, records: List[MemoryRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def run_scenario(
    path: Path,
    args: argparse.Namespace,
    components: Dict[str, Any],
    dense_encoder: Optional[Any],
    vision_encoder: Optional[Any],
) -> Path:
    started = time.time()
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario = path.stem
    run_dir = args.output_dir / scenario / f"{now_stamp()}_opd_mm"
    run_dir.mkdir(parents=True, exist_ok=True)
    store, records = build_scenario_store(
        data,
        data_dir=args.data_dir,
        dense_encoder=dense_encoder,
        vision_encoder=vision_encoder,
        max_sessions=args.max_sessions,
        max_turns=args.max_turns,
    )
    samples = scenario_samples(
        data,
        store=store,
        data_dir=args.data_dir,
        scenario=scenario,
        max_questions=args.max_questions,
        include_oracle_profile=args.teacher_privilege == "oracle-profile",
    )
    distiller = OnPolicyDistiller(
        student=components["student"],
        teacher=components["teacher"],
        executor=components["executor"],
        answer_model=components["answer_model"],
        judge=components["judge"],
        teacher_feedback_rounds=args.teacher_feedback_rounds,
        teacher_evidence_budget=args.teacher_evidence_budget,
    )
    predictions_path = run_dir / "predictions.jsonl"
    sft_path = run_dir / "sft_data.jsonl"
    rows = []
    progress = ProgressBar(len(samples), f"{scenario} OPD", not args.no_progress)
    progress.update(0)
    with predictions_path.open("w", encoding="utf-8") as prediction_handle, sft_path.open(
        "w",
        encoding="utf-8",
    ) as sft_handle:
        for index, sample in enumerate(samples, start=1):
            rollout = distiller.rollout(sample)
            row = rollout_row(rollout)
            row["sft_selected"] = (
                args.mode == "collect-sft"
                and select_sft_example(row, args.sft_quality_filter)
            )
            rows.append(row)
            prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            prediction_handle.flush()
            if row["sft_selected"]:
                sft_handle.write(
                    json.dumps(rollout.sft_example.to_dict(), ensure_ascii=False)
                    + "\n"
                )
                sft_handle.flush()
            progress.update(
                index,
                message=f"point={sample.metadata.get('point')} correct={int(rollout.correct)}",
            )
    progress.close()
    write_memory_manifest(run_dir / "hidden_memory_manifest.jsonl", records)
    metrics = summarize(rows, memory_count=len(store), elapsed=time.time() - started)
    if args.teacher_recall_only:
        metrics["primary_metric"] = "teacher_evidence_recall_at_k"
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = vars(args) | {"scenario": scenario}
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[INFO] Saved OPD-MM run: {run_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return run_dir


def run_scenarios(args: argparse.Namespace) -> List[Path]:
    paths = resolve_scenarios(args)
    if not paths:
        raise FileNotFoundError("no Mem-Gallery scenario files matched")
    dense_encoder = (
        MiniLMTextEncoder(args.dense_model, device=args.dense_device)
        if args.dense_mode == "minilm"
        else None
    )
    vision_encoder = (
        SigLIPVisionEncoder(args.vision_model, device=args.vision_device)
        if args.vision_mode == "siglip"
        else None
    )
    components = make_components(args)
    outputs = []
    for index, path in enumerate(paths, start=1):
        print(f"[INFO] Scenario {index}/{len(paths)}: {path.stem}")
        outputs.append(
            run_scenario(
                path,
                args=args,
                components=components,
                dense_encoder=dense_encoder,
                vision_encoder=vision_encoder,
            )
        )
    return outputs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run query-only OPD-MM tool planning on Mem-Gallery."
    )
    parser.add_argument("--data-dir", type=Path, default=default_memgallery_dir())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--mode", choices=["evaluate", "collect-sft"], default="collect-sft")
    parser.add_argument(
        "--teacher-recall-only",
        action="store_true",
        help=(
            "Skip answer generation and answer judging while retaining student "
            "and teacher policy calls, for fast teacher retrieval evaluation."
        ),
    )
    parser.add_argument(
        "--sft-quality-filter",
        choices=["all", "valid", "support-verified"],
        default="valid",
        help=(
            "Filter corrected trajectories before writing SFT data. "
            "support-verified requires teacher replay to hit annotated support."
        ),
    )
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument(
        "--max-top-k",
        type=int,
        default=512,
        help=(
            "Maximum retrieval pool size. The larger default lets the "
            "training-only oracle advisor replay support beyond rank 50."
        ),
    )
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--dense-mode", choices=["minilm", "off"], default="minilm")
    parser.add_argument("--dense-model", default=default_minilm_model())
    parser.add_argument("--dense-device", default="cpu")
    parser.add_argument(
        "--vision-mode",
        choices=["siglip", "off"],
        default="siglip",
    )
    parser.add_argument("--vision-model", default=default_siglip_model())
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--student-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--student-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--student-api-key", default="ollama")
    parser.add_argument("--teacher-mode", choices=["llm", "off"], default="llm")
    parser.add_argument(
        "--teacher-privilege",
        choices=[
            "minimal",
            "diagnostic",
            "oracle-feedback",
            "oracle-profile",
        ],
        default="diagnostic",
    )
    parser.add_argument(
        "--teacher-feedback-rounds",
        type=int,
        default=2,
        help=(
            "Maximum hidden replay feedback revisions in oracle-feedback mode."
        ),
    )
    parser.add_argument(
        "--teacher-evidence-budget",
        type=int,
        default=20,
        help=(
            "Request a lower-cost revision when support is covered but the "
            "teacher reads more evidence items than this budget."
        ),
    )
    parser.add_argument("--teacher-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--teacher-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--teacher-api-key", default="ollama")
    parser.add_argument("--policy-max-tokens", type=int, default=512)
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--answer-model", default="qwen3-vl-8b-instruct-ctx8k:latest")
    parser.add_argument("--answer-api-key", default="ollama")
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--answer-max-images", type=int, default=3)
    parser.add_argument("--raw-inspection", action="store_true")
    parser.add_argument("--max-raw-inspections", type=int, default=3)
    parser.add_argument("--inspect-max-tokens", type=int, default=160)
    parser.add_argument("--judge-mode", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--judge-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--judge-api-key", default="ollama")
    parser.add_argument("--judge-max-tokens", type=int, default=192)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = require_memgallery_dir(args.data_dir)
    args.output_dir = args.output_dir.expanduser().resolve()
    run_scenarios(args)


if __name__ == "__main__":
    main()
