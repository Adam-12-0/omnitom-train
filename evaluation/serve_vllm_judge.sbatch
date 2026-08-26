#!/bin/bash
#SBATCH --job-name=omni-judge
#SBATCH --account=ai
#SBATCH --partition=highgpu,normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_judge-%j.out
#SBATCH --error=logs/eval_judge-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate /home/ad906660/.conda/envs/tom
mkdir -p logs

MODEL_ID="mistralai/Mistral-Large-Instruct-2407"
PORT="${JUDGE_PORT:-8002}"
export HF_HOME="${HF_HOME:-/home/ad906660/.cache/huggingface}"
[[ -f .hf_token.env ]] && source .hf_token.env
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is required for Mistral-Large-Instruct-2407" >&2; exit 2; }

echo "started=$(date -Is) host=$(hostname) job=${SLURM_JOB_ID}"
echo "model=${MODEL_ID} port=${PORT} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$MODEL_ID" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
