"""Online self-distillation runner for interactive OPD-MM on Mem-Gallery."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    ChatRawInspector,
    HFQwenVLClient,
    OpenAICompatibleClient,
)
from .interactive import (
    ChatInteractivePlanner,
    InteractiveActionValidator,
    InteractiveTeacherSearch,
    StrictAnswerValidator,
)
from .memgallery import build_scenario_store, scenario_samples
from .memgallery_pipeline import (
    ProgressBar,
    now_stamp,
    resolve_scenarios,
)
from .online import OnlineDistillationBuffer, OnlineSelfDistiller
from .retrieval import TurnAwareHybridRetriever


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "memgallery_opd_online"


def _make_chat_client(
    *,
    backend: str,
    base_url: str,
    model: str,
    api_key: str,
    service_mode: str = "auto",
    device: str,
    dtype: str,
    hf_cache: Dict[tuple[str, str, str], HFQwenVLClient],
) -> Any:
    if backend == "hf-qwen-vl":
        key = (model, device, dtype)
        if key not in hf_cache:
            hf_cache[key] = HFQwenVLClient(model, device=device, dtype=dtype)
        return hf_cache[key]
    return OpenAICompatibleClient(
        base_url,
        model,
        api_key,
        service_mode=service_mode,
    )


def make_components(args: argparse.Namespace) -> Dict[str, Any]:
    validator = InteractiveActionValidator(
        max_chunk_actions=args.max_chunk_actions,
        max_top_k=args.max_top_k,
        allow_inspect_raw=args.raw_inspection,
    )
    hf_cache: Dict[tuple[str, str, str], HFQwenVLClient] = {}
    student_planner = ChatInteractivePlanner(
        _make_chat_client(
            backend=args.student_backend,
            base_url=args.student_base_url,
            model=args.student_model,
            api_key=args.student_api_key,
            service_mode=getattr(args, "student_service", "auto"),
            device=args.student_device,
            dtype=args.student_dtype,
            hf_cache=hf_cache,
        ),
        validator=validator,
        max_tokens=args.planner_max_tokens,
        thinking_token_budget=args.planner_thinking_token_budget,
        prompt_mode=args.student_prompt_mode,
        enable_thinking=getattr(args, "student_planner_enable_thinking", None),
    )
    teacher_planner = ChatInteractivePlanner(
        _make_chat_client(
            backend=args.teacher_backend,
            base_url=args.teacher_base_url,
            model=args.teacher_model,
            api_key=args.teacher_api_key,
            service_mode=getattr(args, "teacher_service", "auto"),
            device=args.teacher_device,
            dtype=args.teacher_dtype,
            hf_cache=hf_cache,
        ),
        validator=validator,
        max_tokens=args.planner_max_tokens,
        thinking_token_budget=args.planner_thinking_token_budget,
        prompt_mode=args.teacher_prompt_mode,
        enable_thinking=getattr(args, "teacher_planner_enable_thinking", None),
    )
    if args.answer_backend == "hf-qwen-vl":
        answer_client = HFQwenVLClient(
            args.answer_model,
            device=args.answer_device,
            dtype=args.answer_dtype,
        )
    else:
        answer_client = OpenAICompatibleClient(
            args.answer_base_url,
            args.answer_model,
            args.answer_api_key,
            service_mode=getattr(args, "answer_service", "auto"),
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
    judge = ChatAnswerJudge(
        OpenAICompatibleClient(
            args.judge_base_url,
            args.judge_model,
            args.judge_api_key,
            service_mode=getattr(args, "judge_service", "auto"),
        ),
        max_tokens=args.judge_max_tokens,
    )
    answer_validator = StrictAnswerValidator(
        answer_model,
        judge,
        min_score=args.min_answer_score,
    )
    retriever = TurnAwareHybridRetriever(args.hybrid_alpha)
    teacher_search = InteractiveTeacherSearch(
        planner=teacher_planner,
        verifier=None,
        validator=validator,
        retriever=retriever,
        max_rounds=args.teacher_max_rounds,
        beam_size=args.teacher_beam_size,
        candidates_per_node=args.teacher_candidates,
        max_actions=args.teacher_max_actions,
        max_evidence=args.max_evidence,
        raw_inspector=raw_inspector,
        max_raw_inspections=args.max_raw_inspections,
        answer_validator=answer_validator,
        trajectory_action_cost=args.trajectory_action_cost,
        trajectory_evidence_cost=args.trajectory_evidence_cost,
    )
    return {
        "validator": validator,
        "student_planner": student_planner,
        "teacher_search": teacher_search,
        "answer_model": answer_model,
        "judge": judge,
        "answer_validator": answer_validator,
        "retriever": retriever,
        "raw_inspector": raw_inspector,
    }


def _round_metrics(results: List[Any], buffer_size: int) -> Dict[str, Any]:
    count = len(results)
    corrections = [
        correction
        for result in results
        for correction in result.corrections
    ]
    feedback_events = [
        diagnostic
        for result in results
        for attempt in result.teacher_attempts
        for diagnostic in attempt.get("failure_diagnostics", [])
    ]
    student_answer_failures = [
        result
        for result in results
        if not result.student_answer_validation.correct
    ]
    student_sufficiency_failures = [
        result
        for result in results
        if not result.student_evidence_sufficiency.correct
    ]
    recovered_failures = [
        result for result in student_answer_failures if result.corrections
    ]
    feedback_assisted_recoveries = sum(
        any(
            attempt.get("state_index")
            in {correction.state_index for correction in result.corrections}
            and attempt.get("failure_diagnostics")
            for attempt in result.teacher_attempts
        )
        for result in recovered_failures
    )
    corrections_from_failures = sum(
        len(result.corrections) for result in student_answer_failures
    )
    return {
        "samples": count,
        "primary_metric": "student_answer_accuracy_no_gold",
        "student_answer_accuracy_no_gold": (
            sum(result.student_answer_validation.correct for result in results)
            / count
            if count
            else 0.0
        ),
        "student_answer_score_no_gold": (
            sum(result.student_answer_validation.score for result in results)
            / count
            if count
            else 0.0
        ),
        "student_evidence_sufficiency_gold_aware": (
            sum(result.student_evidence_sufficiency.correct for result in results)
            / count
            if count
            else 0.0
        ),
        "student_evidence_sufficiency_score_gold_aware": (
            sum(result.student_evidence_sufficiency.score for result in results)
            / count
            if count
            else 0.0
        ),
        "student_strict_accuracy": (
            sum(result.student_evidence_sufficiency.correct for result in results)
            / count
            if count
            else 0.0
        ),
        "student_answer_score": (
            sum(result.student_evidence_sufficiency.score for result in results)
            / count
            if count
            else 0.0
        ),
        "correction_states": len(corrections),
        "corrected_sample_rate": (
            sum(bool(result.corrections) for result in results) / count
            if count
            else 0.0
        ),
        "avg_corrections_per_sample": (
            len(corrections) / count if count else 0.0
        ),
        "teacher_feedback_events": len(feedback_events),
        "student_failure_count": len(student_answer_failures),
        "student_answer_failure_count_no_gold": len(student_answer_failures),
        "student_evidence_sufficiency_failure_count": (
            len(student_sufficiency_failures)
        ),
        "teacher_recovered_student_failures": len(recovered_failures),
        "teacher_failure_recovery_rate": (
            len(recovered_failures) / len(student_answer_failures)
            if student_answer_failures
            else 0.0
        ),
        "buffer_true_correction_fraction": (
            corrections_from_failures / len(corrections)
            if corrections
            else 0.0
        ),
        "teacher_recovered_after_feedback_rate": (
            feedback_assisted_recoveries / len(student_answer_failures)
            if student_answer_failures
            else 0.0
        ),
        "buffer_size": buffer_size,
    }


def _run_update_command(
    template: str,
    data_path: Path,
    output_dir: Path,
    round_index: int,
) -> None:
    rendered = template.format(
        data=str(data_path),
        output_dir=str(output_dir),
        round=round_index,
    )
    subprocess.run(shlex.split(rendered), check=True)


def run_scenario(
    path: Path,
    args: argparse.Namespace,
    components: Dict[str, Any],
    dense_encoder: Optional[Any],
    vision_encoder: Optional[Any],
) -> Path:
    started = time.time()
    scenario = path.stem
    run_dir = (
        args.output_dir
        / scenario
        / f"{now_stamp()}_online_self_distill"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
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

    buffer = OnlineDistillationBuffer(args.max_buffer_examples)
    distiller = OnlineSelfDistiller(
        student_planner=components["student_planner"],
        teacher_search=components["teacher_search"],
        answer_validator=components["answer_validator"],
        answer_model=components["answer_model"],
        answer_judge=components["judge"],
        validator=components["validator"],
        retriever=components["retriever"],
        max_student_rounds=args.student_max_rounds,
        max_student_actions=args.max_actions,
        buffer=buffer,
        raw_inspector=components["raw_inspector"],
        max_raw_inspections=args.max_raw_inspections,
        teacher_trigger=args.teacher_trigger,
        stop_when_student_evidence_sufficient=(
            args.stop_when_student_evidence_sufficient
        ),
    )
    round_summaries = []
    for round_index in range(args.distill_rounds):
        round_dir = run_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = round_dir / "online_rollouts.jsonl"
        buffer_path = run_dir / "online_sft_buffer.jsonl"
        partial_metrics_path = round_dir / "partial_metrics.json"
        progress = ProgressBar(
            len(samples),
            f"{scenario} online r{round_index}",
            not args.no_progress,
        )
        results = []
        with rollout_path.open("w", encoding="utf-8") as rollout_handle:
            for sample_index, sample in enumerate(samples, start=1):
                result = distiller.collect_sample(
                    sample,
                    round_index=round_index,
                )
                results.append(result)
                rollout_handle.write(
                    json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
                )
                rollout_handle.flush()
                buffer.write_jsonl(buffer_path)
                partial_metrics = _round_metrics(results, len(buffer))
                partial_metrics["round"] = round_index
                partial_metrics["completed_samples"] = sample_index
                partial_metrics["total_samples"] = len(samples)
                partial_metrics["partial"] = sample_index < len(samples)
                partial_metrics_path.write_text(
                    json.dumps(
                        partial_metrics,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                progress.update(
                    sample_index,
                    message=(
                        f"student={int(result.student_answer_validation.correct)} "
                        f"suff={int(result.student_evidence_sufficiency.correct)} "
                        f"labels={len(result.corrections)}"
                    ),
                )
        progress.close()
        buffer.write_jsonl(buffer_path)
        metrics = _round_metrics(results, len(buffer))
        metrics["round"] = round_index
        metrics["completed_samples"] = len(results)
        metrics["total_samples"] = len(samples)
        metrics["partial"] = False
        (round_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        partial_metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        round_summaries.append(metrics)
        if args.student_update_command:
            update_dir = run_dir / f"student_round_{round_index:02d}"
            update_dir.mkdir(parents=True, exist_ok=True)
            _run_update_command(
                args.student_update_command,
                buffer_path,
                update_dir,
                round_index,
            )

    summary = {
        "scenario": scenario,
        "rounds": round_summaries,
        "final_buffer_size": len(buffer),
        "elapsed_seconds": time.time() - started,
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "student_prompt_mode": args.student_prompt_mode,
        "teacher_prompt_mode": args.teacher_prompt_mode,
        "self_distillation": args.student_model == args.teacher_model,
        "teacher_assessment": (
            "answer model receives gold answer and retrieved evidence during "
            "training-time validation; separate evidence verifier is disabled"
        ),
        "student_primary_metric": "student_answer_accuracy_no_gold",
        "student_secondary_metric": "student_evidence_sufficiency_gold_aware",
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = vars(args) | {
        "teacher_assessment": summary["teacher_assessment"],
        "deprecated_verifier_args": "retained for old commands but not used",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[INFO] Saved online self-distillation run: {run_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
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
    return [
        run_scenario(
            path,
            args,
            components,
            dense_encoder,
            vision_encoder,
        )
        for path in paths
    ]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online self-distillation for interactive OPD-MM."
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
    parser.add_argument("--distill-rounds", type=int, default=3)
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
    parser.add_argument("--student-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--student-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--student-api-key", default="ollama")
    parser.add_argument(
        "--student-service",
        choices=["auto", "local", "api"],
        default="auto",
    )
    parser.add_argument("--student-device", default="cuda:1")
    parser.add_argument("--student-dtype", default="auto")
    parser.add_argument(
        "--teacher-backend",
        choices=["openai", "hf-qwen-vl"],
        default="openai",
    )
    parser.add_argument("--teacher-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--teacher-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--teacher-api-key", default="ollama")
    parser.add_argument(
        "--teacher-service",
        choices=["auto", "local", "api"],
        default="auto",
    )
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
    parser.add_argument("--verifier-base-url", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--verifier-model", default="gemma3-12b-it-q4km-judge:latest")
    parser.add_argument("--verifier-api-key", default="ollama")
    parser.add_argument("--verifier-max-tokens", type=int, default=192)
    parser.add_argument(
        "--answer-backend",
        choices=["openai", "hf-qwen-vl"],
        default="openai",
    )
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--answer-model", default="qwen3-vl-8b-instruct-ctx8k:latest")
    parser.add_argument("--answer-api-key", default="ollama")
    parser.add_argument(
        "--answer-service",
        choices=["auto", "local", "api"],
        default="auto",
    )
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
    parser.add_argument(
        "--judge-service",
        choices=["auto", "local", "api"],
        default="auto",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=192)
    parser.add_argument("--min-answer-score", type=float, default=0.9)
    parser.add_argument(
        "--teacher-trigger",
        choices=["failure", "always"],
        default="failure",
        help=(
            "failure runs teacher correction only after the current student "
            "state fails evidence validation; always labels every visited "
            "student state."
        ),
    )
    parser.add_argument(
        "--student-update-command",
        default=None,
        help=(
            "Optional command run after each round. Available placeholders: "
            "{data}, {output_dir}, and {round}. The command must update or "
            "restart the configured student endpoint before returning."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    args.data_dir = require_memgallery_dir(args.data_dir)
    args.output_dir = args.output_dir.expanduser().resolve()
    run_scenarios(args)


if __name__ == "__main__":
    main()
