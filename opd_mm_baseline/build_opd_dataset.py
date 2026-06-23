"""Export offline OPD/SDFT inspection datasets from Mem-Gallery rollouts.

This is not the full online OPD loop. It is useful for debugging, cold-start
SFT, and inspecting teacher labels. Use ``opd_online_train.py`` when the student
must be updated between rounds before collecting the next on-policy batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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

from .memgallery import build_scenario_store, scenario_samples
from .memgallery_online_pipeline import make_components
from .memgallery_pipeline import ProgressBar, now_stamp, resolve_scenarios
from .online import OnlineDistillationBuffer, OnlineSampleResult, OnlineSelfDistiller


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "opd_dataset"


def stable_split_bucket(sample_id: str, seed: int = 13) -> float:
    value = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(value[:12], 16) / float(16**12)


def split_name(sample_id: str, val_ratio: float, seed: int = 13) -> str:
    ratio = max(0.0, min(1.0, float(val_ratio)))
    return "val" if stable_split_bucket(sample_id, seed) < ratio else "train"


def _row_key(row: Dict[str, Any]) -> str:
    value = f"{row.get('input', '')}\n<target>\n{row.get('target', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _quality_passes(
    result: OnlineSampleResult,
    quality_filter: str,
) -> tuple[bool, str]:
    if quality_filter == "teacher-correct":
        return True, "teacher_correct"
    if quality_filter == "student-answer-failure":
        if result.student_answer_validation.correct:
            return False, "student_answer_already_correct"
        return True, "student_answer_failed"
    if quality_filter == "student-sufficiency-failure":
        if result.student_evidence_sufficiency.correct:
            return False, "student_evidence_already_sufficient"
        return True, "student_evidence_insufficient"
    raise ValueError(f"invalid quality filter: {quality_filter}")


def correction_to_dataset_row(
    result: OnlineSampleResult,
    correction_index: int,
    scenario: str,
    round_index: int,
) -> Dict[str, Any]:
    correction = result.corrections[correction_index]
    row = correction.example.to_dict(include_metadata=True)
    metadata = row.setdefault("metadata", {})
    metadata["dataset"] = {
        "source": "memgallery_online_opd",
        "scenario": scenario,
        "source_sample_id": result.sample_id,
        "correction_index": correction_index,
        "online_state_index": correction.state_index,
        "round_index": round_index,
        "student_answer_correct_no_gold": bool(
            result.student_answer_validation.correct
        ),
        "student_answer_score_no_gold": float(
            result.student_answer_validation.score
        ),
        "student_evidence_sufficiency_gold_aware": bool(
            result.student_evidence_sufficiency.correct
        ),
        "student_evidence_sufficiency_score_gold_aware": float(
            result.student_evidence_sufficiency.score
        ),
        "teacher_answer_score": float(
            correction.teacher_answer_validation.score
        ),
        "student_actions": [
            action.to_dict() for action in correction.student_actions
        ],
        "teacher_actions": [
            action.to_dict() for action in correction.teacher_actions
        ],
    }
    return row


def rejected_result_row(
    result: OnlineSampleResult,
    scenario: str,
    round_index: int,
    reason: str,
    correction_index: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "source": "memgallery_online_opd",
        "scenario": scenario,
        "sample_id": result.sample_id,
        "round_index": round_index,
        "correction_index": correction_index,
        "reason": reason,
        "student_answer_validation": result.student_answer_validation.to_dict(
            include_reason=False
        ),
        "student_evidence_sufficiency": (
            result.student_evidence_sufficiency.to_dict(include_reason=False)
        ),
        "correction_count": len(result.corrections),
        "teacher_attempts": result.teacher_attempts,
    }


def _write_jsonl(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _make_distiller(args: argparse.Namespace, components: Dict[str, Any]) -> OnlineSelfDistiller:
    return OnlineSelfDistiller(
        student_planner=components["student_planner"],
        teacher_search=components["teacher_search"],
        answer_validator=components["answer_validator"],
        answer_model=components["answer_model"],
        answer_judge=components["judge"],
        validator=components["validator"],
        retriever=components["retriever"],
        max_student_rounds=args.student_max_rounds,
        max_student_actions=args.max_actions,
        buffer=OnlineDistillationBuffer(args.max_buffer_examples),
        raw_inspector=components["raw_inspector"],
        max_raw_inspections=args.max_raw_inspections,
        teacher_trigger=args.teacher_trigger,
    )


def build_dataset(args: argparse.Namespace) -> Path:
    started = time.time()
    args.data_dir = require_memgallery_dir(args.data_dir)
    args.output_dir = args.output_dir.expanduser().resolve()
    run_dir = args.output_dir / f"{now_stamp()}_opd_dataset"
    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

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

    train_path = run_dir / "train.jsonl"
    val_path = run_dir / "val.jsonl"
    accepted_path = run_dir / "accepted.jsonl"
    rejected_path = run_dir / "rejected.jsonl"
    config_path = run_dir / "config.json"
    manifest_path = run_dir / "manifest.json"

    config_path.write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    counts = {
        "scenarios": 0,
        "samples": 0,
        "rollouts": 0,
        "accepted": 0,
        "train": 0,
        "val": 0,
        "rejected": 0,
        "duplicate": 0,
    }
    seen_rows: set[str] = set()
    scenario_summaries: List[Dict[str, Any]] = []

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open(
        "w", encoding="utf-8"
    ) as val_handle, accepted_path.open(
        "w", encoding="utf-8"
    ) as accepted_handle, rejected_path.open(
        "w", encoding="utf-8"
    ) as rejected_handle:
        for path in paths:
            scenario = path.stem
            scenario_started = time.time()
            data = json.loads(path.read_text(encoding="utf-8"))
            store, _records = build_scenario_store(
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
                include_oracle_profile=False,
            )
            for sample in samples:
                sample.metadata.pop("teacher_privileged_context", None)

            counts["scenarios"] += 1
            counts["samples"] += len(samples)
            scenario_dir = rollouts_dir / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            distiller = _make_distiller(args, components)
            scenario_counts = {
                "scenario": scenario,
                "samples": len(samples),
                "accepted": 0,
                "train": 0,
                "val": 0,
                "rejected": 0,
                "duplicate": 0,
            }

            for round_index in range(args.distill_rounds):
                rollout_path = scenario_dir / f"round_{round_index:02d}.jsonl"
                progress = ProgressBar(
                    len(samples),
                    f"{scenario} opd-data r{round_index}",
                    not args.no_progress,
                )
                with rollout_path.open("w", encoding="utf-8") as rollout_handle:
                    for sample_index, sample in enumerate(samples, start=1):
                        result = distiller.collect_sample(
                            sample,
                            round_index=round_index,
                        )
                        counts["rollouts"] += 1
                        _write_jsonl(rollout_handle, result.to_dict())

                        if not result.corrections:
                            rejected = rejected_result_row(
                                result,
                                scenario,
                                round_index,
                                "no_teacher_validated_correction",
                            )
                            _write_jsonl(rejected_handle, rejected)
                            counts["rejected"] += 1
                            scenario_counts["rejected"] += 1

                        for correction_index, _correction in enumerate(
                            result.corrections
                        ):
                            keep, reason = _quality_passes(
                                result,
                                args.quality_filter,
                            )
                            if not keep:
                                rejected = rejected_result_row(
                                    result,
                                    scenario,
                                    round_index,
                                    reason,
                                    correction_index=correction_index,
                                )
                                _write_jsonl(rejected_handle, rejected)
                                counts["rejected"] += 1
                                scenario_counts["rejected"] += 1
                                continue

                            row = correction_to_dataset_row(
                                result,
                                correction_index,
                                scenario,
                                round_index,
                            )
                            key = _row_key(row)
                            if key in seen_rows:
                                rejected = rejected_result_row(
                                    result,
                                    scenario,
                                    round_index,
                                    "duplicate_input_target",
                                    correction_index=correction_index,
                                )
                                _write_jsonl(rejected_handle, rejected)
                                counts["duplicate"] += 1
                                scenario_counts["duplicate"] += 1
                                continue
                            seen_rows.add(key)

                            source_sample_id = row["metadata"]["dataset"][
                                "source_sample_id"
                            ]
                            row_split = split_name(
                                source_sample_id,
                                args.val_ratio,
                                args.split_seed,
                            )
                            row.setdefault("metadata", {})["dataset"].update(
                                {
                                    "split": row_split,
                                    "quality_filter": args.quality_filter,
                                    "quality_reason": reason,
                                    "dedupe_key": key,
                                }
                            )
                            _write_jsonl(accepted_handle, row)
                            if row_split == "val":
                                _write_jsonl(val_handle, row)
                                counts["val"] += 1
                                scenario_counts["val"] += 1
                            else:
                                _write_jsonl(train_handle, row)
                                counts["train"] += 1
                                scenario_counts["train"] += 1
                            counts["accepted"] += 1
                            scenario_counts["accepted"] += 1

                        progress.update(
                            sample_index,
                            message=(
                                f"accepted={scenario_counts['accepted']} "
                                f"rejected={scenario_counts['rejected']}"
                            ),
                        )
                progress.close()

            scenario_counts["elapsed_seconds"] = time.time() - scenario_started
            scenario_summaries.append(scenario_counts)

    manifest = {
        "created_at": now_stamp(),
        "elapsed_seconds": time.time() - started,
        "paths": {
            "train": str(train_path),
            "val": str(val_path),
            "accepted": str(accepted_path),
            "rejected": str(rejected_path),
            "rollouts": str(rollouts_dir),
        },
        "counts": counts,
        "scenarios": scenario_summaries,
        "split": {
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
        },
        "quality_filter": args.quality_filter,
        "student_prompt_mode": args.student_prompt_mode,
        "teacher_prompt_mode": args.teacher_prompt_mode,
        "note": (
            "Offline export for inspection/cold-start data. train/val inputs "
            "are student-visible prompts; teacher prompts are stored only "
            "under metadata.opd.teacher_input. For true OPD, run "
            "opd_online_train.py so the student is updated before the next "
            "collection round."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Saved OPD dataset: {run_dir}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return run_dir


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=default_memgallery_dir())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--distill-rounds", type=int, default=1)
    parser.add_argument("--student-max-rounds", type=int, default=9)
    parser.add_argument("--teacher-max-rounds", type=int, default=9)
    parser.add_argument("--teacher-max-actions", type=int, default=9)
    parser.add_argument("--teacher-beam-size", type=int, default=2)
    parser.add_argument("--teacher-candidates", type=int, default=3)
    parser.add_argument("--max-chunk-actions", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=9)
    parser.add_argument("--max-top-k", type=int, default=50)
    parser.add_argument("--max-evidence", type=int, default=40)
    parser.add_argument("--max-buffer-examples", type=int, default=None)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--dense-mode", choices=["minilm", "off"], default="minilm")
    parser.add_argument("--dense-model", default=default_minilm_model())
    parser.add_argument("--dense-device", default="cpu")
    parser.add_argument("--vision-mode", choices=["siglip", "off"], default="siglip")
    parser.add_argument("--vision-model", default=default_siglip_model())
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument(
        "--student-backend",
        choices=["openai", "hf-qwen-vl"],
        default="openai",
    )
    parser.add_argument("--student-base-url", default="http://127.0.0.1:11438/v1")
    parser.add_argument("--student-model", default="qwen3-vl-4b-thinking-vllm")
    parser.add_argument("--student-api-key", default="ollama")
    parser.add_argument("--student-device", default="cuda:1")
    parser.add_argument("--student-dtype", default="auto")
    parser.add_argument(
        "--teacher-backend",
        choices=["openai", "hf-qwen-vl"],
        default="openai",
    )
    parser.add_argument("--teacher-base-url", default="http://127.0.0.1:11438/v1")
    parser.add_argument("--teacher-model", default="qwen3-vl-4b-thinking-vllm")
    parser.add_argument("--teacher-api-key", default="ollama")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--teacher-dtype", default="auto")
    parser.add_argument("--planner-max-tokens", type=int, default=768)
    parser.add_argument("--planner-thinking-token-budget", type=int, default=256)
    parser.add_argument(
        "--student-planner-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--teacher-planner-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--student-prompt-mode",
        choices=["student_simple", "teacher_compact"],
        default="student_simple",
    )
    parser.add_argument(
        "--teacher-prompt-mode",
        choices=["teacher_compact", "student_simple"],
        default="teacher_compact",
    )
    parser.add_argument(
        "--answer-backend",
        choices=["openai", "hf-qwen-vl"],
        default="openai",
    )
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--answer-model", default="qwen3-vl-8b-instruct-ctx8k:latest")
    parser.add_argument("--answer-api-key", default="ollama")
    parser.add_argument("--answer-device", default="cuda:1")
    parser.add_argument("--answer-dtype", default="auto")
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--answer-max-images", type=int, default=3)
    parser.add_argument(
        "--raw-inspection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-raw-inspections", type=int, default=3)
    parser.add_argument("--inspect-max-tokens", type=int, default=160)
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--judge-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--judge-api-key", default="ollama")
    parser.add_argument("--judge-max-tokens", type=int, default=192)
    parser.add_argument("--min-answer-score", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument(
        "--quality-filter",
        choices=[
            "teacher-correct",
            "student-answer-failure",
            "student-sufficiency-failure",
        ],
        default="teacher-correct",
    )
    parser.add_argument(
        "--teacher-trigger",
        choices=["failure", "always"],
        default="failure",
        help=(
            "failure runs teacher correction only after the current student "
            "state fails evidence validation; always preserves the older "
            "label-every-visited-state behavior."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OPD/SDFT datasets from Mem-Gallery online rollouts."
    )
    add_common_args(parser)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    build_dataset(args)


if __name__ == "__main__":
    main()
