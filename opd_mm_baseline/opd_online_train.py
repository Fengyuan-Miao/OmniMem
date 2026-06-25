"""Online OPD/SDFT loop for Mem-Gallery.

This is the true online loop: each round collects trajectories from the current
student, trains on the teacher-corrected states from that round, then uses the
updated student for the next round. The standalone dataset builder remains
useful for inspection and cold-start data, but this runner is the OPD path.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnimem.config import PROJECT_ROOT

try:
    from .build_opd_dataset import add_common_args, build_dataset
    from .memgallery_pipeline import now_stamp
    from .opd_train import train_opd
    from .opd_stream_train import run_streaming_opd
except ImportError:
    from opd_mm_baseline.build_opd_dataset import add_common_args, build_dataset
    from opd_mm_baseline.memgallery_pipeline import now_stamp
    from opd_mm_baseline.opd_train import train_opd
    from opd_mm_baseline.opd_stream_train import run_streaming_opd


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "opd_online_train"
DEFAULT_TRAIN_MODEL = "/home/miaofy/models/Qwen3-VL-4B-Thinking"


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_manifest(path: Path) -> Dict[str, Any]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def render_command(
    template: str,
    *,
    round_index: int,
    checkpoint: Path,
    data: Path,
    run_dir: Path,
    collection_dir: Path,
) -> str:
    return template.format(
        round=round_index,
        checkpoint=str(checkpoint),
        model=str(checkpoint),
        data=str(data),
        run_dir=str(run_dir),
        collection_dir=str(collection_dir),
    )


def run_reload_command(
    template: Optional[str],
    *,
    round_index: int,
    checkpoint: Path,
    data: Path,
    run_dir: Path,
    collection_dir: Path,
) -> None:
    if not template:
        return
    command = render_command(
        template,
        round_index=round_index,
        checkpoint=checkpoint,
        data=data,
        run_dir=run_dir,
        collection_dir=collection_dir,
    )
    subprocess.run(shlex.split(command), check=True)


def _bool_flag(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def train_with_accelerate(
    args: argparse.Namespace,
    *,
    model_name_or_path: str,
    train_path: Path,
    checkpoint_dir: Path,
    round_index: int,
) -> None:
    script_path = Path(__file__).resolve().with_name("opd_train.py")
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(args.accelerate_num_processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16" if args.bf16 else "fp16" if args.fp16 else "no",
        "--gpu_ids",
        str(args.gpu_devices),
    ]
    if args.accelerate_config_file:
        command.extend(["--config_file", str(args.accelerate_config_file)])
    command.extend(
        [
            str(script_path),
            "--model",
            str(model_name_or_path),
            "--data",
            str(train_path),
            "--output-dir",
            str(checkpoint_dir),
            "--epochs",
            str(args.train_epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--batch-size",
            str(args.train_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--max-length",
            str(args.train_max_length),
            "--distill-kl-weight",
            str(args.distill_kl_weight),
            "--distill-temperature",
            str(args.distill_temperature),
            "--gpu-devices",
            str(args.gpu_devices),
            "--report-to",
            str(args.report_to),
            "--wandb-project",
            str(args.wandb_project),
            "--wandb-run-name",
            args.wandb_run_name
            or f"opd-online-{checkpoint_dir.parent.parent.name}-r{round_index:02d}",
            "--wandb-group",
            args.wandb_group or checkpoint_dir.parent.parent.name,
            "--wandb-mode",
            str(args.wandb_mode),
            "--logging-steps",
            str(args.logging_steps),
        ]
    )
    if args.wandb_entity:
        command.extend(["--wandb-entity", str(args.wandb_entity)])
    command.extend(_bool_flag("bf16", args.bf16))
    if args.fp16:
        command.append("--fp16")
    command.extend(
        _bool_flag("gradient-checkpointing", args.gradient_checkpointing)
    )
    log_path = checkpoint_dir / "accelerate_train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_devices)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{pythonpath}"
        if pythonpath
        else str(PROJECT_ROOT)
    )
    env["PYTHONUNBUFFERED"] = "1"
    if args.report_to == "wandb" and args.wandb_mode != "disabled":
        env.setdefault("WANDB_PROJECT", str(args.wandb_project))
        env.setdefault("WANDB_MODE", str(args.wandb_mode))
        if args.wandb_entity:
            env.setdefault("WANDB_ENTITY", str(args.wandb_entity))
        proxy = str(args.proxy_url or "").strip()
        if proxy:
            env["HTTP_PROXY"] = proxy
            env["HTTPS_PROXY"] = proxy
            env["http_proxy"] = proxy
            env["https_proxy"] = proxy
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(
            json.dumps(
                {
                    "round": round_index,
                    "command": command,
                    "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                    "distill_kl_weight": args.distill_kl_weight,
                    "distill_temperature": args.distill_temperature,
                    "report_to": args.report_to,
                    "wandb_project": args.wandb_project,
                    "wandb_mode": args.wandb_mode,
                    "proxy_url": args.proxy_url,
                    "train_path": str(train_path),
                    "model": str(model_name_or_path),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        log_handle.flush()
        subprocess.run(command, check=True, env=env, stdout=log_handle, stderr=log_handle)


def collection_args(
    args: argparse.Namespace,
    run_dir: Path,
    round_index: int,
) -> argparse.Namespace:
    value = copy.copy(args)
    value.output_dir = run_dir / "collections" / f"round_{round_index:02d}"
    value.distill_rounds = 1
    return value


def run_online_opd(args: argparse.Namespace) -> Path:
    started = time.time()
    run_dir = args.output_dir.expanduser().resolve() / f"{now_stamp()}_opd_online"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    rounds_path = run_dir / "rounds.jsonl"
    config_path.write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    current_train_model = str(args.train_model)
    round_summaries = []
    for round_index in range(max(1, int(args.opd_rounds))):
        collect_args = collection_args(args, run_dir, round_index)
        collection_dir = build_dataset(collect_args)
        manifest = read_manifest(collection_dir)
        train_path = Path(manifest["paths"]["train"])
        train_rows = count_jsonl(train_path)

        checkpoint_dir = run_dir / "checkpoints" / f"round_{round_index:02d}"
        trained = False
        skipped_reason = ""
        if train_rows <= 0:
            skipped_reason = "empty_train_split"
        elif args.dry_run:
            skipped_reason = "dry_run"
        else:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if args.train_launcher == "accelerate":
                train_with_accelerate(
                    args,
                    model_name_or_path=current_train_model,
                    train_path=train_path,
                    checkpoint_dir=checkpoint_dir,
                    round_index=round_index,
                )
            else:
                train_opd(
                    model_name_or_path=current_train_model,
                    data_path=train_path,
                    output_dir=checkpoint_dir,
                    epochs=args.train_epochs,
                    learning_rate=args.learning_rate,
                    batch_size=args.train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_length=args.train_max_length,
                    bf16=args.bf16,
                    fp16=args.fp16,
                    gradient_checkpointing=args.gradient_checkpointing,
                    distill_kl_weight=args.distill_kl_weight,
                    distill_temperature=args.distill_temperature,
                    gpu_devices=args.gpu_devices,
                    report_to=args.report_to,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_run_name=(
                        args.wandb_run_name
                        or f"opd-online-{run_dir.name}-r{round_index:02d}"
                    ),
                    wandb_group=args.wandb_group or run_dir.name,
                    wandb_mode=args.wandb_mode,
                    logging_steps=args.logging_steps,
                )
            trained = True
            current_train_model = str(checkpoint_dir)
            if args.student_backend == "hf-qwen-vl":
                args.student_model = str(checkpoint_dir)
            else:
                if args.student_reload_command:
                    run_reload_command(
                        args.student_reload_command,
                        round_index=round_index,
                        checkpoint=checkpoint_dir,
                        data=train_path,
                        run_dir=run_dir,
                        collection_dir=collection_dir,
                    )
                elif round_index + 1 < args.opd_rounds and not args.allow_static_student:
                    raise RuntimeError(
                        "OpenAI/vLLM student backend needs --student-reload-command "
                        "between OPD rounds, otherwise later trajectories are not "
                        "sampled from the updated student. Use "
                        "--allow-static-student only for debugging."
                    )

        round_summary = {
            "round": round_index,
            "collection_dir": str(collection_dir),
            "train_path": str(train_path),
            "train_rows": train_rows,
            "manifest_counts": manifest.get("counts", {}),
            "checkpoint_dir": str(checkpoint_dir) if trained else "",
            "trained": trained,
            "skipped_reason": skipped_reason,
            "student_model_for_next_round": args.student_model,
            "train_model_for_next_round": current_train_model,
        }
        round_summaries.append(round_summary)
        with rounds_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(round_summary, ensure_ascii=False) + "\n")
        summary = {
            "run_dir": str(run_dir),
            "rounds": round_summaries,
            "elapsed_seconds": time.time() - started,
            "online_opd": True,
            "note": (
                "Each round collects trajectories from the current student, "
                "then trains before the next collection round."
            ),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[INFO] Saved online OPD run: {run_dir}")
    print(summary_path.read_text(encoding="utf-8"))
    return run_dir


def launch_streaming_with_accelerate(args: argparse.Namespace) -> None:
    script_path = Path(__file__).resolve()
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(args.accelerate_num_processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16" if args.bf16 else "fp16",
        "--gpu_ids",
        str(args.gpu_devices),
    ]
    if args.accelerate_config_file:
        command.extend(["--config_file", str(args.accelerate_config_file)])
    command.extend([str(script_path), *sys.argv[1:], "--stream-worker"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_devices)
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{pythonpath}"
        if pythonpath
        else str(PROJECT_ROOT)
    )
    proxy = str(args.proxy_url or "").strip()
    if proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[key] = proxy
    subprocess.run(command, check=True, env=env)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run true online OPD/SDFT over Mem-Gallery."
    )
    add_common_args(parser)
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--online-mode",
        choices=["streaming", "round-batch"],
        default="streaming",
        help=(
            "streaming performs rollout -> logits distillation -> update in "
            "one live loop. round-batch preserves the old dataset-then-train "
            "baseline."
        ),
    )
    parser.add_argument(
        "--stream-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--opd-rounds", type=int, default=3)
    parser.add_argument("--train-model", default=DEFAULT_TRAIN_MODEL)
    parser.add_argument("--train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--train-max-length", type=int, default=2048)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--activation-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Offload autograd-saved activations to CPU during the trainable "
            "student forward pass."
        ),
    )
    parser.add_argument("--distill-kl-weight", type=float, default=0.2)
    parser.add_argument(
        "--distill-nll-weight",
        type=float,
        default=1.0,
        help=(
            "Hard-label weight for validated teacher next-action tokens in "
            "streaming OPD."
        ),
    )
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument(
        "--train-launcher",
        choices=["accelerate", "inprocess"],
        default="accelerate",
    )
    parser.add_argument("--accelerate-num-processes", type=int, default=4)
    parser.add_argument("--accelerate-config-file", type=Path, default=None)
    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=[0, 2],
        default=0,
        help=(
            "Use DeepSpeed ZeRO-2 to shard gradients and optimizer state. "
            "Stage 0 keeps the existing DDP path."
        ),
    )
    parser.add_argument(
        "--zero-offload-optimizer",
        choices=["none", "cpu"],
        default="none",
        help=(
            "Optional ZeRO-2 optimizer-state offload. GPU is faster; CPU saves "
            "more VRAM."
        ),
    )
    parser.add_argument(
        "--report-to",
        choices=["none", "wandb"],
        default="wandb",
    )
    parser.add_argument("--wandb-project", default="omnimem-opd")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--proxy-url", default="http://127.0.0.1:7896")
    parser.add_argument(
        "--student-reload-command",
        default=None,
        help=(
            "Command run after each successful train step for OpenAI/vLLM "
            "student backends. Placeholders: {round}, {checkpoint}, {model}, "
            "{data}, {run_dir}, {collection_dir}."
        ),
    )
    parser.add_argument(
        "--allow-static-student",
        action="store_true",
        help="Debug only: allow multiple rounds without reloading the student.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect online data but skip the training/update step.",
    )
    parser.add_argument("--online-samples-per-rank", type=int, default=1)
    parser.add_argument(
        "--training-mode",
        choices=["full", "lora"],
        default="full",
        help="Train full language-policy parameters by default; LoRA is an ablation.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["paged_adamw8bit", "adamw8bit", "adamw"],
        default="paged_adamw8bit",
    )
    parser.add_argument(
        "--freeze-vision-tower",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze Qwen3-VL's vision tower because planner prompts contain "
            "text observations only."
        ),
    )
    parser.add_argument(
        "--freeze-token-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze the tied input embedding/lm_head table while fully "
            "training all language transformer blocks."
        ),
    )
    parser.add_argument("--student-rollout-max-tokens", type=int, default=192)
    parser.add_argument("--student-rollout-temperature", type=float, default=0.7)
    parser.add_argument("--student-rollout-top-p", type=float, default=0.9)
    parser.add_argument(
        "--student-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable private thinking during online student rollouts. The "
            "thinking tokens are not supervised unless --supervise-thinking "
            "is set."
        ),
    )
    parser.add_argument(
        "--supervise-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablation only: include thinking tokens in the supervised prompt. "
            "Default false keeps loss on the final action JSON array."
        ),
    )
    parser.add_argument("--max-prompt-length", type=int, default=1792)
    parser.add_argument("--prompt-head-tokens", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--state0-keep-ratio", type=float, default=0.35)
    parser.add_argument("--positive-state-repeat", type=int, default=2)
    parser.add_argument(
        "--trajectory-action-cost",
        type=float,
        default=0.08,
        help=(
            "Cost used when selecting teacher targets and ranking teacher "
            "search paths. Larger values prefer shorter successful paths."
        ),
    )
    parser.add_argument(
        "--trajectory-evidence-cost",
        type=float,
        default=0.01,
        help=(
            "Cost used when selecting teacher targets and ranking teacher "
            "search paths. Larger values prefer less bloated evidence sets."
        ),
    )
    parser.add_argument(
        "--normalize-trajectory-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize total supervised weight per sample so long successful "
            "trajectories do not dominate state-level training."
        ),
    )
    parser.add_argument(
        "--stop-when-student-evidence-sufficient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During online rollout collection only, stop the student path once "
            "the answer validator judges current evidence sufficient."
        ),
    )
    parser.add_argument("--distill-top-k", type=int, default=32)
    parser.add_argument(
        "--distill-add-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--teacher-ema-decay", type=float, default=0.99)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.online_mode == "round-batch":
        run_online_opd(args)
        return
    if args.dry_run:
        raise ValueError("--dry-run is only supported by --online-mode round-batch")
    launched_by_accelerate = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.stream_worker or launched_by_accelerate:
        run_streaming_opd(args)
    else:
        launch_streaming_with_accelerate(args)


if __name__ == "__main__":
    main()