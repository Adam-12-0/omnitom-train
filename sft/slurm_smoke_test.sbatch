#!/usr/bin/env bash
#SBATCH --job-name=omnitom-sft-smoke
#SBATCH --account=ai
#SBATCH --partition=highgpu
#SBATCH --output=logs/omnitom_sft_smoke_%j.out
#SBATCH --error=logs/omnitom_sft_smoke_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export GPU_TYPE="${GPU_TYPE:-NVIDIA H100 80GB HBM3 (nvidia_h100_80gb_hbm3)}"
if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH" ]]; then source "$VENV_PATH"; elif [[ -f "$ROOT/.venv/bin/activate" ]]; then source "$ROOT/.venv/bin/activate"; fi
bash "$ROOT/sft/smoke_test.sh"
