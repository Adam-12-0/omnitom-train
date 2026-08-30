#!/usr/bin/env python3
"""LoRA SFT for OmniToM Stage 1 belief extraction."""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import time
import inspect
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from peft import LoraConfig

BASE_MODEL = "Qwen/Qwen3-14B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ROOT = Path(__file__).resolve().parents[1]


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def read_jsonl(path: Path, smoke: bool) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[:4] if smoke else rows


def render(tokenizer, example: dict) -> str:
    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/omnitom_stage1_sft")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/checkpoints/omnitom_stage1_sft_qwen3_14b")
    ap.add_argument("--model-id", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--assistant-only-loss", action="store_true",
                    help="Mask system and user tokens; optimize assistant completions only.")
    args = ap.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.data_dir / "train.jsonl", args.smoke)
    valid_rows = read_jsonl(args.data_dir / "validation.jsonl", args.smoke)
    if not train_rows:
        raise RuntimeError("No training examples found; run convert_omnitom_stage1.py first")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    # Keep the role structure so SFTTrainer can construct an assistant-only
    # loss mask from the model chat template when requested.
    train_text = [{"messages": x["messages"]} for x in train_rows]
    valid_text = [{"messages": x["messages"]} for x in valid_rows]
    try:
        from datasets import Dataset
        train_ds, valid_ds = Dataset.from_list(train_text), Dataset.from_list(valid_text)
    except ImportError as exc:
        raise RuntimeError("datasets is required for SFT; install sft/requirements.txt") from exc
    try:
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError("TRL is required for SFT; install sft/requirements.txt") from exc
    common = dict(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, optim="adamw_torch", bf16=True,
        gradient_checkpointing=True, logging_steps=1,
        eval_steps=10, save_strategy="steps", save_steps=10, save_total_limit=2,
        seed=args.seed, report_to="none", remove_unused_columns=False,
        max_steps=args.max_steps,
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    common["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "steps"
    try:
        from trl import SFTConfig
        common["assistant_only_loss"] = args.assistant_only_loss
        training_args = SFTConfig(**common, max_length=args.max_seq_length)
        has_sft_config = True
    except (ImportError, TypeError):
        training_args = TrainingArguments(**common)
        has_sft_config = False
    trainer_kwargs = dict(
        model=model, train_dataset=train_ds, eval_dataset=valid_ds,
        peft_config=peft_config, args=training_args,
    )
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    trainer = SFTTrainer(**trainer_kwargs)
    start = time.time()
    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metrics = dict(result.metrics)
    if valid_ds:
        metrics.update({f"validation_{k}": v for k, v in trainer.evaluate().items()})
    provenance = {
        "base_model_id": args.model_id, "omnitom_commit": git_head(ROOT / "omnitom-benchmark-review"),
        "defender_commit": git_head(ROOT / "AIDoubleAgentDefenders"), "seed": args.seed,
        "learning_rate": args.learning_rate, "epochs": args.epochs, "optimizer": "AdamW",
        "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "target_modules": TARGET_MODULES,
        "bf16": True, "gradient_checkpointing": True, "max_sequence_length": args.max_seq_length,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "assistant_only_loss": args.assistant_only_loss,
        "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-set"), "gpu_type": os.environ.get("GPU_TYPE", "unknown"),
        "memory_note": "bf16 configuration preserved; normal partition 15.77 GiB GPU OOMed, so jobs use highgpu NVIDIA H100 80GB HBM3. No training hyperparameters changed.",
        "host": socket.gethostname(), "runtime_seconds": round(time.time() - start, 2),
        "final_adapter_path": str(args.output_dir.resolve()), "metrics": metrics,
    }
    (args.output_dir / "training_provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, default=str))


if __name__ == "__main__":
    main()
