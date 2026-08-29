#!/bin/bash
# Full open-weight GRPO reproduction for one reward configuration.
# Submit one job per GRPO_CONFIG/GRPO_LABEL pair after service jobs are ready.
# For an OmniToM-SFT-initialized ablation, pass MODEL_ENGINE and
# MODEL_CHECKPOINTS_DIR explicitly; the default trains from base Qwen3-14B.
#SBATCH --job-name=ada-grpo-full
#SBATCH --account=ai
#SBATCH --partition=highgpu,normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --exclude=evc101
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/grpo_full-%j.out
#SBATCH --error=logs/grpo_full-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
: "${GRPO_CONFIG:?Set GRPO_CONFIG to an open_repro_traj_*.yaml config}"
: "${GRPO_LABEL:?Set GRPO_LABEL to a unique result/checkpoint label}"
# Resolve config paths before changing into the embedded project directory.
if [[ "$GRPO_CONFIG" != /* ]]; then
  GRPO_CONFIG="$ROOT/$GRPO_CONFIG"
fi
cd "$ROOT/AIDoubleAgentDefenders"

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate /home/ad906660/.conda/envs/tom
mkdir -p "$ROOT/logs" "$ROOT/results/grpo/$GRPO_LABEL" "$ROOT/results/checkpoints/$GRPO_LABEL"

echo "GPU diagnostic: host=$(hostname) job=${SLURM_JOB_ID:-unknown}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable on this allocation" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free --format=csv,noheader
nvidia-smi

export RESULTS_DIR="$ROOT/results/grpo/$GRPO_LABEL"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-unused}"
export WANDB_MODE="offline"
export WANDB_PROJECT="omnitom-open-grpo"
export SHELLS_LAUNCHER_LOG_NAME="ada_grpo_${GRPO_LABEL}_${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
# A 20k-token reservation exhausts a single H100 during the training
# log-probability backward pass.  The actual ToM-SB replies are short; cap the
# generated completion to keep the GRPO reproduction runnable on one H100.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${ATTACKER_ENDPOINT:-}" || -z "${JUDGE_ENDPOINT:-}" ]]; then
  : "${ATTACKER_JOB_ID:?Set service endpoint variables or service job IDs}"
  : "${JUDGE_JOB_ID:?Set service endpoint variables or service job IDs}"
  for attempt in $(seq 1 720); do
    attacker_state="$(squeue -h -j "$ATTACKER_JOB_ID" -o '%T' 2>/dev/null || true)"
    judge_state="$(squeue -h -j "$JUDGE_JOB_ID" -o '%T' 2>/dev/null || true)"
    if [[ "$attacker_state" == RUNNING && "$judge_state" == RUNNING ]]; then
      export ATTACKER_ENDPOINT="http://$(squeue -h -j "$ATTACKER_JOB_ID" -o '%N'):8001/v1"
      export JUDGE_ENDPOINT="http://$(squeue -h -j "$JUDGE_JOB_ID" -o '%N'):8002/v1"
      break
    fi
    if [[ "$attacker_state" == FAILED* || "$attacker_state" == CANCELLED* || "$judge_state" == FAILED* || "$judge_state" == CANCELLED* ]]; then
      echo "A serving job failed: attacker=$attacker_state judge=$judge_state" >&2
      exit 1
    fi
    sleep 5
  done
fi
: "${ATTACKER_ENDPOINT:?Attacker endpoint was not resolved}"
: "${JUDGE_ENDPOINT:?Judge endpoint was not resolved}"

for attempt in $(seq 1 720); do
  attacker_http="$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' "${ATTACKER_ENDPOINT%/v1}/health" || true)"
  judge_http="$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' "${JUDGE_ENDPOINT%/v1}/health" || true)"
  if [[ "$attacker_http" == 200 && "$judge_http" == 200 ]]; then break; fi
  if (( attempt % 12 == 0 )); then echo "waiting for vLLM health: attacker=$attacker_http judge=$judge_http"; fi
  sleep 5
done
if [[ "$attacker_http" != 200 || "$judge_http" != 200 ]]; then
  echo "vLLM health timeout: attacker=$attacker_http judge=$judge_http" >&2
  exit 1
fi

MODEL_ENGINE="${MODEL_ENGINE:-Qwen3-14B}"
MODEL_CHECKPOINTS_DIR="${MODEL_CHECKPOINTS_DIR:-}"
GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-1024}"
MODEL_ARGS=(--engine "$MODEL_ENGINE" --model_savepath "$ROOT/results/checkpoints/$GRPO_LABEL")
if [[ -n "$MODEL_CHECKPOINTS_DIR" ]]; then MODEL_ARGS+=(--checkpoints_dir "$MODEL_CHECKPOINTS_DIR"); fi

python -u -m main_scripts.main_training_script \
  --config "$GRPO_CONFIG" \
  --results_dir "$RESULTS_DIR" \
  "${MODEL_ARGS[@]}" \
  --attacker_endpoint "$ATTACKER_ENDPOINT" \
  --judge_endpoint "$JUDGE_ENDPOINT" \
  --attacker_model meta-llama/Llama-3.3-70B-Instruct \
  --judge_model mistralai/Mistral-Large-Instruct-2407 \
  --attacker_prompts_dir utils/sequence_of_prompt_ablations/orig_v4_2x \
  --gradient_accumulation_steps 16 \
  --max_iterations 15 --num_generations 2 --max_completion_length "$GRPO_MAX_COMPLETION_LENGTH" \
  --learning_rate 1e-5 --lr_scheduler linear --loss_type dr_grpo \
  --torch_seed 42 --seed 42 \
  --load_in_4bit \
  --gradient_checkpointing \
  --save_trajectory_outputs "$RESULTS_DIR/trajectories.jsonl"
