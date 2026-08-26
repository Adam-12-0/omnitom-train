#!/usr/bin/env python3
"""LoRA SFT scaffold for OmniToM Stage 1 chat-format datasets."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
from pathlib import Path
from typing import Any, Callable


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
EXPECTED_ROLES = ["system", "user", "assistant"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.bf16 and args.fp16:
        parser.error("--bf16 and --fp16 are mutually exclusive")
    for name in ("max_train_examples", "max_val_examples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path.name}: {exc.msg}") from exc
            validate_example(record, path.name, line_number)
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No examples found in {path.name}")
    return records


def validate_example(record: Any, filename: str, line_number: int) -> None:
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        raise ValueError(f"{filename}:{line_number}: expected an object with a messages array")
    messages = record["messages"]
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != EXPECTED_ROLES or len(messages) != 3:
        raise ValueError(f"{filename}:{line_number}: messages must have system, user, assistant roles")
    if any(not isinstance(message.get("content"), str) or not message["content"] for message in messages):
        raise ValueError(f"{filename}:{line_number}: every message requires non-empty string content")
    if not messages[2]["content"].startswith("Actor | Belief | Order"):
        raise ValueError(f"{filename}:{line_number}: assistant target has the wrong header")


def fallback_chat_format(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"<|{message['role']}|>\n{message['content']}" for message in messages)


def tokenizer_formatter(tokenizer: Any) -> Callable[[list[dict[str, str]]], str]:
    if getattr(tokenizer, "chat_template", None):
        return lambda messages: tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return fallback_chat_format


def local_tokenizer_for_dry_run(model_name_or_path: str) -> Any | None:
    """Load only an already-cached tokenizer; never contact a model hub."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except (OSError, ValueError):
        return None


def run_dry_run(args: argparse.Namespace, train: list[dict[str, Any]], val: list[dict[str, Any]]) -> None:
    tokenizer = local_tokenizer_for_dry_run(args.model_name_or_path)
    formatter = tokenizer_formatter(tokenizer) if tokenizer is not None else fallback_chat_format
    sample = (train + val)[:32]
    formatted = [formatter(record["messages"]) for record in sample]
    if tokenizer is not None:
        lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in formatted]
        measurement = "tokenizer_exact_local_cache"
    else:
        lengths = [len(text.split()) for text in formatted]
        measurement = "whitespace_estimate_no_tokenizer_download"

    summary = {
        "dry_run": True,
        "model_loaded": False,
        "train_examples": len(train),
        "validation_examples": len(val),
        "formatted_first_example": bool(formatted),
        "first_example_characters": len(formatted[0]) if formatted else 0,
        "length_measurement": measurement,
        "examples_measured": len(lengths),
        "minimum_token_length": min(lengths, default=0),
        "average_token_length": round(statistics.mean(lengths), 2) if lengths else 0.0,
        "maximum_token_length": max(lengths, default=0),
        "over_max_seq_length": sum(length > args.max_seq_length for length in lengths),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def require_training_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are unavailable. Install mutually compatible versions of "
            "transformers, datasets, peft, and torch in an allocated GPU-node environment "
            "before running without --dry-run."
        ) from exc
    try:
        from trl import SFTTrainer
        try:
            from trl import SFTConfig
        except ImportError:
            SFTConfig = None
    except ImportError as exc:
        raise SystemExit(
            "TRL is unavailable. Install a TRL version compatible with the selected "
            "transformers/datasets/peft stack before running training."
        ) from exc
    return (
        torch,
        transformers,
        Dataset,
        LoraConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        set_seed,
        SFTTrainer,
        SFTConfig,
    )


def supported_kwargs(callable_object: Any, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    return {key: value for key, value in values.items() if key in parameters}


def main() -> None:
    args = parse_args()
    train_records = load_jsonl(args.train_file, args.max_train_examples)
    val_records = load_jsonl(args.val_file, args.max_val_examples)

    if args.dry_run:
        run_dry_run(args, train_records, val_records)
        return

    (
        torch,
        transformers,
        Dataset,
        LoraConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        set_seed,
        SFTTrainer,
        SFTConfig,
    ) = require_training_dependencies()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatter = tokenizer_formatter(tokenizer)

    def add_text(record: dict[str, Any]) -> dict[str, str]:
        return {"text": formatter(record["messages"])}

    train_dataset = Dataset.from_list(train_records).map(add_text)
    val_dataset = Dataset.from_list(val_records).map(add_text)

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    if dtype is not None:
        transformers_major = int(transformers.__version__.split(".", maxsplit=1)[0])
        model_kwargs["dtype" if transformers_major >= 5 else "torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    common_config = {
        "output_dir": str(args.output_dir.expanduser()),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
        "dataset_text_field": "text",
        "max_length": args.max_seq_length,
        "max_seq_length": args.max_seq_length,
        "packing": False,
    }
    if SFTConfig is not None:
        training_config = SFTConfig(**supported_kwargs(SFTConfig, common_config))
    else:
        training_config = TrainingArguments(**supported_kwargs(TrainingArguments, common_config))

    trainer_values = {
        "model": model,
        "args": training_config,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "peft_config": lora_config,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
    }
    trainer = SFTTrainer(**supported_kwargs(SFTTrainer, trainer_values))
    trainer.train()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
