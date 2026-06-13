"""Generic causal-LM SFT for corrected OPD-MM trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_sft_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("input") and row.get("target"):
                    rows.append(row)
    return rows


def train_sft(
    model_name_or_path: str,
    data_path: str | Path,
    output_dir: str | Path,
    epochs: float = 2.0,
    learning_rate: float = 2e-5,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    bf16: bool = False,
    fp16: bool = False,
) -> None:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    rows = load_sft_rows(data_path)
    if not rows:
        raise ValueError(f"no SFT examples found in {data_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class TrajectoryDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            row = rows[index]
            prompt = str(row["input"]).rstrip() + "\n"
            target = str(row["target"]).strip() + tokenizer.eos_token
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            target_ids = tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, max_length - len(prompt_ids)),
            )["input_ids"]
            input_ids = (prompt_ids + target_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
            attention_mask = [1] * len(input_ids)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    class PaddingCollator:
        def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
            max_size = max(feature["input_ids"].size(0) for feature in features)
            result = {}
            for key, pad_value in [
                ("input_ids", tokenizer.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]:
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

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        bf16=bf16,
        fp16=fp16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=TrajectoryDataset(),
        data_collator=PaddingCollator(),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT an OPD-MM student policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    train_sft(
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
    )


if __name__ == "__main__":
    main()
