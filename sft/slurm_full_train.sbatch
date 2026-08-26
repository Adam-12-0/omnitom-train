#!/usr/bin/env bash
#SBATCH --job-name=omnitom-sft-qwen3-14b
#SBATCH --account=ai
#SBATCH --partition=highgpu
#SBATCH --output=logs/omnitom_sft_%j.out
#SBATCH --error=logs/omnitom_sft_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export GPU_TYPE="${GPU_TYPE:-NVIDIA H100 80GB HBM3 (nvidia_h100_80gb_hbm3)}"
if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH" ]]; then source "$VENV_PATH"; elif [[ -f "$ROOT/.venv/bin/activate" ]]; then source "$ROOT/.venv/bin/activate"; fi
python "$ROOT/sft/convert_omnitom_stage1.py" --output-dir "$ROOT/data/omnitom_stage1_sft"
python "$ROOT/sft/train_omnitom_stage1.py" \
  --data-dir "$ROOT/data/omnitom_stage1_sft" \
  --output-dir "$ROOT/results/checkpoints/omnitom_stage1_sft_qwen3_14b" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
