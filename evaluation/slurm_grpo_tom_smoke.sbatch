#!/bin/bash
#SBATCH --job-name=ada-tom-smoke
#SBATCH --account=ai
#SBATCH --partition=highgpu,normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --exclude=evc101
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/ada_tom_smoke-%j.out
#SBATCH --error=logs/ada_tom_smoke-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$ROOT/AIDoubleAgentDefenders"
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate /home/ad906660/.conda/envs/tom
mkdir -p "$ROOT/logs" "$ROOT/results/checkpoints/ada_tom_only_qwen3_14b_open"
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
    sleep 5
  done
fi
: "${ATTACKER_ENDPOINT:?Attacker endpoint was not resolved}"
: "${JUDGE_ENDPOINT:?Judge endpoint was not resolved}"
export RESULTS_DIR="$ROOT/results/grpo_smoke"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-unused}"
export WANDB_MODE="offline"
export SHELLS_LAUNCHER_LOG_NAME="ada_tom_smoke_${SLURM_JOB_ID}"
export WANDB_PROJECT="omnitom-open-eval"

# vLLM reports a RUNNING Slurm state while it loads model weights.  Do not
# start the training smoke until both APIs answer health checks.
for attempt in $(seq 1 720); do
  attacker_http="$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' "${ATTACKER_ENDPOINT%/v1}/health" || true)"
  judge_http="$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' "${JUDGE_ENDPOINT%/v1}/health" || true)"
  if [[ "$attacker_http" == 200 && "$judge_http" == 200 ]]; then
    break
  fi
  sleep 5
done
if [[ "$attacker_http" != 200 || "$judge_http" != 200 ]]; then
  echo "vLLM health timeout: attacker=$attacker_http judge=$judge_http" >&2
  exit 1
fi

python -u -m main_scripts.main_training_script \
  --config config_launchers/configs/open_repro_traj_PToM.yaml \
  --results_dir "$RESULTS_DIR" \
  --attacker_endpoint "$ATTACKER_ENDPOINT" \
  --judge_endpoint "$JUDGE_ENDPOINT" \
  --testing --eval_limit 2 --epochs 1 --max_iterations 2 --num_generations 2 \
  --max_completion_length 256 --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 --torch_seed 42 --seed 42 \
  --model_savepath "$ROOT/results/checkpoints/ada_tom_only_qwen3_14b_open_smoke" \
  --attacker_prompts_dir utils/sequence_of_prompt_ablations/orig_v4_2x \
  --save_trajectory_outputs "$ROOT/results/grpo_smoke/trajectories.jsonl"
