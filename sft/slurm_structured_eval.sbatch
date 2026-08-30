#!/usr/bin/env bash
# Held-out Stage-1 structured evaluation for base or one LoRA adapter.
# Required: STRUCTURED_LABEL. Optional: ADAPTER_PATH (unset = base).
#SBATCH --job-name=omnitom-structured-eval
#SBATCH --account=ai
#SBATCH --partition=highgpu,normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --exclude=evc101
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/structured_eval-%j.out
#SBATCH --error=logs/structured_eval-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
: "${STRUCTURED_LABEL:?}"
source "$ROOT/.venv/bin/activate"
OUT="$ROOT/results/structured_eval/${STRUCTURED_LABEL}.json"
ARGS=(--data "$ROOT/data/omnitom_stage1_sft_stratified/validation.jsonl" --output "$OUT")
if [[ -n "${ADAPTER_PATH:-}" ]]; then ARGS+=(--adapter "$ADAPTER_PATH"); fi
python -u "$ROOT/sft/evaluate_stage1_structured.py" "${ARGS[@]}"
