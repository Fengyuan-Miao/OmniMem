"""SDFT-style OPD training for interactive memory-tool policies.

The online collector writes student-visible prompts plus teacher action chunks.
This trainer fits the student on those on-policy states. When teacher prompts
are available in metadata, it can also add a teacher-conditioned KL term: the
same model sees the richer teacher prompt under no_grad, while the trainable
student sees only the simple tool prompt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_opd_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("input") and row.get("target"):
                rows.append(row)
    return rows


def _teacher_input(row: Dict[str, Any]) -> Optional[str]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    opd = metadata.get("opd")
    if not isinstance(opd, dict):
        return None
    value = opd.get("teacher_input")
    return str(value) if value else None


def _load_policy_model(
    model_name_or_path: str,
    *,
    dtype: Any,
    device_map: str = "auto",
) -> Any:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        Qwen3VLForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "device_map": device_map,
    }
    if getattr(config, "model_type", "") == "qwen3_vl":
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **kwargs,
    )


def train_opd(
    model_name_or_path: str,
    data_path: str | Path,
    output_dir: str | Path,
    epochs: float = 1.0,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    bf16: bool = True,
    fp16: bool = False,
    gradient_checkpointing: bool = True,
    distill_kl_weight: float = 0.0,
    distill_temperature: float = 1.0,
    gpu_devices: str = "0,1,2,3",
    report_to: str = "none",
    wandb_project: str = "omnimem-opd",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_mode: str = "online",
    logging_steps: int = 5,
) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_devices)
    if report_to == "wandb" and wandb_mode != "disabled":
        os.environ.update(
            {
                "WANDB__DISABLE_STATS": "true",
                "WANDB__DISABLE_MACHINE_INFO": "true",
                "WANDB__DISABLE_META": "true",
                "WANDB__SAVE_REQUIREMENTS": "false",
                "WANDB_DISABLE_CODE": "true",
                "WANDB_DISABLE_GIT": "true",
                "WANDB_CONSOLE": "off",
            }
        )
        os.environ.setdefault("WANDB_PROJECT", wandb_project)
        os.environ.setdefault("WANDB_MODE", wandb_mode)
        if wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", wandb_entity)
        if wandb_group:
            os.environ.setdefault("WANDB_RUN_GROUP", wandb_group)
        if wandb_run_name:
            os.environ.setdefault("WANDB_NAME", wandb_run_name)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    rows = load_opd_rows(data_path)
    if not rows:
        raise ValueError(f"no OPD rows found in {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode_pair(prompt: str, target: str) -> Dict[str, List[int]]:
        prompt_text = str(prompt).rstrip() + "\n"
        target_text = str(target).strip() + tokenizer.eos_token
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, max_length - len(prompt_ids)),
        )["input_ids"]
        input_ids = (prompt_ids + target_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    class OPDDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            row = rows[index]
            student = encode_pair(row["input"], row["target"])
            result = {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in student.items()
            }
            teacher_prompt = _teacher_input(row)
            if teacher_prompt:
                teacher = encode_pair(teacher_prompt, row["target"])
            else:
                teacher = student
            result.update(
                {
                    f"teacher_{key}": torch.tensor(value, dtype=torch.long)
                    for key, value in teacher.items()
                }
            )
            return result

    class OPDCollator:
        def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
            result: Dict[str, torch.Tensor] = {}
            keys = [
                "input_ids",
                "attention_mask",
                "labels",
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_labels",
            ]
            for key in keys:
                if key.endswith("input_ids"):
                    pad_value = tokenizer.pad_token_id
                elif key.endswith("attention_mask"):
                    pad_value = 0
                else:
                    pad_value = -100
                max_size = max(feature[key].size(0) for feature in features)
                padded = []
                for feature in features:
                    tensor = feature[key]
                    pad = torch.full(
                        (max_size - tensor.size(0),),
                        pad_value,
                        dtype=tensor.dtype,
                    )
                    padded.append(torch.cat([tensor, pad]))
                result[key] = torch.stack(padded)
            return result

    class OPDTrainer(Trainer):
        @staticmethod
        def _target_prediction_logits(
            logits: torch.Tensor,
            labels: torch.Tensor,
        ) -> List[torch.Tensor]:
            """Return logits that predict target tokens, one tensor per item."""
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            result = []
            for item_logits, item_labels in zip(shifted_logits, shifted_labels):
                mask = item_labels.ne(-100)
                result.append(item_logits[mask])
            return result

        def compute_loss(
            self,
            model: Any,
            inputs: Dict[str, torch.Tensor],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            teacher_inputs = {
                key.removeprefix("teacher_"): inputs.pop(key)
                for key in list(inputs)
                if key.startswith("teacher_")
            }
            outputs = model(**inputs)
            loss = outputs.loss
            if distill_kl_weight > 0.0:
                with torch.no_grad():
                    teacher_outputs = model(**teacher_inputs)
                student_items = self._target_prediction_logits(
                    outputs.logits,
                    inputs["labels"],
                )
                teacher_items = self._target_prediction_logits(
                    teacher_outputs.logits,
                    teacher_inputs["labels"],
                )
                aligned_student = []
                aligned_teacher = []
                for student_logits, teacher_logits in zip(
                    student_items,
                    teacher_items,
                ):
                    length = min(
                        student_logits.size(0),
                        teacher_logits.size(0),
                    )
                    if length <= 0:
                        continue
                    aligned_student.append(student_logits[:length])
                    aligned_teacher.append(teacher_logits[:length])
                if aligned_student:
                    student_logits = torch.cat(aligned_student, dim=0)
                    teacher_logits = torch.cat(aligned_teacher, dim=0)
                    temperature = max(float(distill_temperature), 1e-6)
                    student_log_probs = F.log_softmax(
                        student_logits / temperature,
                        dim=-1,
                    )
                    teacher_probs = F.softmax(
                        teacher_logits / temperature,
                        dim=-1,
                    )
                    kl_loss = F.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction="batchmean",
                    ) * (temperature**2)
                    loss = loss + float(distill_kl_weight) * kl_loss
            return (loss, outputs) if return_outputs else loss

    model_dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else "auto"
    model = _load_policy_model(
        model_name_or_path,
        dtype=model_dtype,
        device_map="auto",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="epoch",
        remove_unused_columns=False,
        bf16=bf16,
        fp16=fp16,
        report_to=(
            ["wandb"]
            if report_to == "wandb" and wandb_mode != "disabled"
            else []
        ),
        run_name=wandb_run_name,
        logging_steps=max(1, int(logging_steps)),
    )
    trainer = OPDTrainer(
        model=model,
        args=args,
        train_dataset=OPDDataset(),
        data_collator=OPDCollator(),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an OPD-MM student policy on online teacher corrections."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--distill-kl-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument(
        "--report-to",
        choices=["none", "wandb"],
        default="none",
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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    train_opd(
        model_name_or_path=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        distill_kl_weight=args.distill_kl_weight,
        distill_temperature=args.distill_temperature,
        gpu_devices=args.gpu_devices,
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()
