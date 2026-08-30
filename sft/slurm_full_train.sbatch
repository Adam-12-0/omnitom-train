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
if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH" ]]; then
  source "$VENV_PATH"
else
  source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
  conda activate /home/ad906660/.conda/envs/tom
fi
echo "python=$(command -v python) env=${CONDA_DEFAULT_ENV:-venv}"
DATA_DIR="${SFT_DATA_DIR:-$ROOT/data/omnitom_stage1_sft_stratified}"
OUTPUT_DIR="${SFT_OUTPUT_DIR:-$ROOT/results/checkpoints/omnitom_stage1_sft_qwen3_14b_stratified}"
if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
  python "$ROOT/sft/convert_omnitom_stage1.py" --output-dir "$DATA_DIR"
fi
python "$ROOT/sft/train_omnitom_stage1.py" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
