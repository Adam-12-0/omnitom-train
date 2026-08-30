#!/usr/bin/env bash
# Double-Agent JSON-format control: no OmniToM labels or stories.
#SBATCH --job-name=da-format-control-sft
#SBATCH --account=ai
#SBATCH --partition=highgpu,normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --exclude=evc101
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/da_format_control_%j.out
#SBATCH --error=logs/da_format_control_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
source "$ROOT/.venv/bin/activate"
DATA="$ROOT/data/double_agent_format_control"
python "$ROOT/sft/convert_double_agent_format_control.py" --output-dir "$DATA"
python "$ROOT/sft/train_omnitom_stage1.py" --data-dir "$DATA" \
  --output-dir "$ROOT/results/checkpoints/double_agent_json_format_control_qwen3_14b" \
  --assistant-only-loss --per-device-batch-size 1 --gradient-accumulation-steps 16
