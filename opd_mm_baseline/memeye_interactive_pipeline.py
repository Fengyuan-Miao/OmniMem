"""MemEye runner for interactive next-action OPD-MM policies."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dual_encoder_memory import MiniLMTextEncoder, SigLIPVisionEncoder
from omnimem.config import (
    PROJECT_ROOT,
    default_minilm_model,
    default_siglip_model,
)

from .interactive import VerificationResult
from .memeye import (
    DEFAULT_OPEN_TASKS,
    IMAGE_ID_PATTERN,
    build_scenario_store,
    iter_task_paths,
    require_memeye_data_dir,
    scenario_samples,
)
from .memgallery_interactive_pipeline import (
    _evidence_redundancy,
    _safe_judge,
    _verification_from_answer_validation,
    load_excluded_sample_ids,
    make_components,
    summarize,
)
from .memgallery_pipeline import ProgressBar, now_stamp, write_memory_manifest
from .models import EvidenceItem, ExecutionResult, MemoryRecord, OPDSample


DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "benchmark" / "MemEye" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "memeye_opd_interactive"
READ_FIELDS = [
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


def parse_task_names(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def resolve_tasks(args: argparse.Namespace) -> List[Path]:
    if args.all_tasks:
        names = DEFAULT_OPEN_TASKS
    else:
        names = parse_task_names(args.tasks)
        if not names:
            names = [args.task or DEFAULT_OPEN_TASKS[0]]
    paths = iter_task_paths(args.data_dir, names)
    if args.max_tasks is not None:
        paths = paths[: max(0, args.max_tasks)]
    return paths


def _evidence_image_ids(evidence: List[EvidenceItem]) -> List[str]:
    serialized = json.dumps(
        [item.to_dict() for item in evidence],
        ensure_ascii=False,
    )
    return list(dict.fromkeys(IMAGE_ID_PATTERN.findall(serialized)))


def _support_metrics(
    execution: ExecutionResult,
    clue_turn_ids: List[str],
    gold_image_ids: List[str],
) -> Dict[str, Any]:
    evidence_memory_ids = [item.memory_id for item in execution.evidence]
    hit_turns = [
        clue
        for clue in clue_turn_ids
        if any(
            memory_id == clue or memory_id.startswith(clue + ":")
            for memory_id in evidence_memory_ids
        )
    ]
    found_images = _evidence_image_ids(execution.evidence)
    return {
        "evidence_image_ids": found_images,
        "gold_image_recall_any": (
            bool(set(found_images) & set(gold_image_ids))
            if gold_image_ids
            else None
        ),
        "gold_image_recall_all": (
            set(gold_image_ids).issubset(found_images)
            if gold_image_ids
            else None
        ),
        "evidence_clue_recall_any": (
            bool(hit_turns) if clue_turn_ids else None
        ),
        "evidence_clue_recall_all": (
            len(hit_turns) == len(clue_turn_ids)
            if clue_turn_ids
            else None
        ),
        "support_turn_recall": (
            len(hit_turns) / len(clue_turn_ids)
            if clue_turn_ids
            else None
        ),
    }


def _select_round_robin(
    task_samples: List[tuple[str, List[OPDSample]]],
    limit: Optional[int],
) -> List[OPDSample]:
    if limit is None:
        return [sample for _task, samples in task_samples for sample in samples]
    selected: List[OPDSample] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for _task, samples in task_samples:
            if offset < len(samples):
                selected.append(samples[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def _task_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task") or "unknown")].append(row)
    return {
        task: {
            "count": len(items),
            "answer_accuracy": sum(float(item["correct"]) for item in items)
            / len(items),
            "avg_actions": sum(len(item.get("actions") or []) for item in items)
            / len(items),
            "avg_planner_calls": sum(float(item["planner_calls"]) for item in items)
            / len(items),
        }
        for task, items in sorted(by_task.items())
    }


def run_memeye(args: argparse.Namespace) -> Path:
    started = time.time()
    args.data_dir = require_memeye_data_dir(args.data_dir)
    args.output_dir = args.output_dir.expanduser().resolve()
    run_dir = args.output_dir / f"{now_stamp()}_opd_interactive_memeye"
    run_dir.mkdir(parents=True, exist_ok=True)

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
    excluded_sample_ids = load_excluded_sample_ids(args.exclude_sample_ids_file)

    task_paths = resolve_tasks(args)
    if not task_paths:
        raise FileNotFoundError("no MemEye task files matched")

    task_samples: List[tuple[str, List[OPDSample]]] = []
    all_records: List[MemoryRecord] = []
    memory_count = 0
    for path in task_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        task = path.stem
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
            scenario=task,
            max_questions=args.max_questions_per_task,
        )
        if excluded_sample_ids:
            samples = [
                sample
                for sample in samples
                if sample.sample_id not in excluded_sample_ids
            ]
        for sample in samples:
            sample.metadata["task"] = task
        task_samples.append((task, samples))
        all_records.extend(records)
        memory_count += len(store)

    selected_samples = _select_round_robin(task_samples, args.max_total_questions)
    if args.max_total_questions is not None:
        selected_samples = selected_samples[: max(0, args.max_total_questions)]

    rows: List[Dict[str, Any]] = []
    progress = ProgressBar(
        len(selected_samples),
        "MemEye interactive",
        not args.no_progress,
    )
    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as prediction_handle:
        for index, sample in enumerate(selected_samples, start=1):
            question_image = sample.metadata.get("question_image")
            policy = components["policy_runner"].run(
                query=sample.query,
                memory_store=sample.memory_store,
                question_image=question_image,
            )
            execution = policy.execution
            strict_validation = (
                components["answer_validator"].evaluate(
                    sample.query,
                    sample.gold_answer,
                    execution.evidence,
                    question_image=question_image,
                )
                if components["answer_validator"] is not None
                else None
            )
            verification = (
                _verification_from_answer_validation(
                    strict_validation,
                    execution.evidence,
                    can_inspect_raw=(
                        args.raw_inspection
                        and components["raw_inspector"] is not None
                    ),
                )
                if strict_validation is not None
                else VerificationResult(
                    answerable=bool(execution.evidence),
                    relevance=1.0 if execution.evidence else 0.0,
                    completeness=1.0 if execution.evidence else 0.0,
                    redundancy=_evidence_redundancy(execution.evidence),
                )
            )
            if strict_validation is not None:
                prediction = strict_validation.prediction
                correct = strict_validation.correct
                score = strict_validation.score
                reason = strict_validation.reason
                judge_error = strict_validation.error
            else:
                prediction = components["answer_model"].answer(
                    sample.query,
                    execution.evidence,
                    question_image=question_image,
                )
                correct, score, reason, judge_error = _safe_judge(
                    components["judge"],
                    sample.query,
                    prediction,
                    sample.gold_answer,
                )
            support = _support_metrics(
                execution,
                list(sample.metadata.get("gold_clue_turn_ids") or []),
                list(sample.metadata.get("gold_image_ids") or []),
            )
            row = {
                "sample_id": sample.sample_id,
                "task": sample.metadata.get("task"),
                "query": sample.query,
                "gold_answer": sample.gold_answer,
                "point": sample.metadata.get("point"),
                "raw_point": sample.metadata.get("raw_point"),
                "question_id": sample.metadata.get("question_id"),
                "question_image": question_image,
                "actions": [action.to_dict() for action in policy.actions],
                "execution": execution.to_dict(),
                "verification": verification.to_dict(),
                "teacher_decisions": [
                    {"planner_raw_response": raw}
                    for raw in policy.planner_raw_responses
                ],
                "prediction": prediction,
                "correct": correct,
                "score": score,
                "judge_reason": reason,
                "judge_error": judge_error,
                "planner_calls": policy.planner_calls,
                "verifier_calls": 0,
                "answer_validator_calls": 1 if strict_validation is not None else 0,
                "strict_answer_validation": (
                    strict_validation.to_dict()
                    if strict_validation is not None
                    else None
                ),
                "candidates_evaluated": 0,
                "chunk_count": len(policy.planner_raw_responses),
                "sft_selected": False,
                **support,
            }
            rows.append(row)
            prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            prediction_handle.flush()
            progress.update(
                index,
                message=(
                    f"task={sample.metadata.get('task')} "
                    f"point={sample.metadata.get('point')}"
                ),
            )
    progress.close()

    write_memory_manifest(run_dir / "hidden_memory_manifest.jsonl", all_records)
    metrics = summarize(
        rows,
        memory_count=memory_count,
        elapsed=time.time() - started,
        mode=args.mode,
        teacher_recall_only=args.teacher_recall_only,
    )
    metrics["dataset"] = "MemEye"
    metrics["task_count"] = len(task_paths)
    metrics["sample_strategy"] = args.sample_strategy
    metrics["by_task"] = _task_metrics(rows)
    metrics["point_distribution"] = dict(
        sorted(Counter(row.get("point") for row in rows).items())
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = vars(args) | {
        "task_files": [str(path) for path in task_paths],
        "selected_sample_count": len(selected_samples),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[INFO] Saved MemEye interactive OPD-MM run: {run_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return run_dir


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interactive next-action OPD-MM on MemEye."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-questions-per-task", type=int, default=None)
    parser.add_argument("--max-total-questions", type=int, default=None)
    parser.add_argument(
        "--sample-strategy",
        choices=["round_robin"],
        default="round_robin",
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate"],
        default="evaluate",
    )
    parser.add_argument("--teacher-recall-only", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=9)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--candidates-per-node", type=int, default=3)
    parser.add_argument("--max-chunk-actions", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=9)
    parser.add_argument("--max-top-k", type=int, default=50)
    parser.add_argument("--max-evidence", type=int, default=40)
    parser.add_argument("--trajectory-action-cost", type=float, default=0.08)
    parser.add_argument("--trajectory-evidence-cost", type=float, default=0.01)
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
    parser.add_argument("--vision-device", default="cpu")
    parser.add_argument("--planner-base-url", default="http://127.0.0.1:11440/v1")
    parser.add_argument("--planner-model", default="qwen3-vl-4b-opd-lowaccum-step29")
    parser.add_argument("--planner-api-key", default="ollama")
    parser.add_argument(
        "--planner-service",
        choices=["auto", "local", "api"],
        default="auto",
    )
    parser.add_argument("--planner-max-tokens", type=int, default=768)
    parser.add_argument("--planner-thinking-token-budget", type=int, default=512)
    parser.add_argument(
        "--planner-prompt-mode",
        choices=["teacher_compact", "student_simple"],
        default="student_simple",
    )
    parser.add_argument(
        "--planner-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--exclude-sample-ids-file", type=Path, default=None)
    parser.add_argument("--verifier-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--verifier-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--verifier-api-key", default="ollama")
    parser.add_argument("--verifier-max-tokens", type=int, default=192)
    parser.add_argument(
        "--teacher-validation",
        choices=["answer", "evidence"],
        default="answer",
    )
    parser.add_argument("--teacher-min-answer-score", type=float, default=0.9)
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--answer-model", default="qwen3-vl-8b-instruct-ctx8k:latest")
    parser.add_argument("--answer-api-key", default="ollama")
    parser.add_argument(
        "--answer-service",
        choices=["auto", "local", "api"],
        default="auto",
    )
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--answer-max-images", type=int, default=3)
    parser.add_argument("--raw-inspection", action="store_true")
    parser.add_argument("--max-raw-inspections", type=int, default=3)
    parser.add_argument("--inspect-max-tokens", type=int, default=160)
    parser.add_argument("--judge-mode", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--judge-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--judge-api-key", default="ollama")
    parser.add_argument(
        "--judge-service",
        choices=["auto", "local", "api"],
        default="auto",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=192)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_memeye(args)


if __name__ == "__main__":
    main()
