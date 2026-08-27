#!/bin/bash
# Full common-stack ToM-SB evaluation for one defender. Submit once per
# EVAL_LABEL/EVAL_CONFIG pair after the attacker and judge services are ready.
#SBATCH --job-name=omni-eval-full
#SBATCH --account=ai
#SBATCH --partition=highgpu,normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --exclude=evc101
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/eval_full-%j.out
#SBATCH --error=logs/eval_full-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
: "${EVAL_CONFIG:?Set EVAL_CONFIG to a defender evaluation YAML}"
: "${EVAL_LABEL:?Set EVAL_LABEL to a stable defender result label}"
cd "$ROOT/AIDoubleAgentDefenders"

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate /home/ad906660/.conda/envs/tom
mkdir -p "$ROOT/logs" "$ROOT/results/evaluations/full/$EVAL_LABEL"

export RESULTS_DIR="$ROOT/results/evaluations/full/$EVAL_LABEL"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-unused}"
export WANDB_MODE="offline"
export SHELLS_LAUNCHER_LOG_NAME="omnitom_eval_full_${EVAL_LABEL}_${SLURM_JOB_ID}"
export WANDB_PROJECT="omnitom-open-eval"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${ATTACKER_ENDPOINT:-}" || -z "${JUDGE_ENDPOINT:-}" ]]; then
  : "${ATTACKER_JOB_ID:?Set service endpoint variables or service job IDs}"
  : "${JUDGE_JOB_ID:?Set service endpoint variables or service job IDs}"
  for attempt in $(seq 1 120); do
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
echo "attacker_endpoint=$ATTACKER_ENDPOINT judge_endpoint=$JUDGE_ENDPOINT"

# A Slurm allocation can be RUNNING while vLLM is still loading multi-GB
# quantized weights.  Wait for both actual HTTP APIs so an evaluation never
# begins against an unopened port.
for attempt in $(seq 1 720); do
  attacker_http="$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' "${ATTACKER_ENDPOINT%/v1}/health" || true)"
  judge_http="$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' "${JUDGE_ENDPOINT%/v1}/health" || true)"
  if [[ "$attacker_http" == 200 && "$judge_http" == 200 ]]; then
    break
  fi
  if (( attempt % 12 == 0 )); then
    echo "waiting for vLLM health: attacker=$attacker_http judge=$judge_http"
  fi
  sleep 5
done
if [[ "$attacker_http" != 200 || "$judge_http" != 200 ]]; then
  echo "vLLM health timeout: attacker=$attacker_http judge=$judge_http" >&2
  exit 1
fi

# Uses all 75 held-out scenarios and the repository's two orig_v4 attacker
# prompt variants (150 trajectories per defender). These flags force the same
# evaluation/reward protocol for every defender condition.
python -u -m main_scripts.main_training_script \
  --config "$EVAL_CONFIG" \
  --results_dir "$RESULTS_DIR" \
  --eval_only --do_run_first_eval 1 --n_evals 1 \
  --eval_batch_size 1 --eval_num_workers 1 \
  --max_iterations 15 --max_completion_length 20000 \
  --temperature 0.7 \
  --attacker_endpoint "$ATTACKER_ENDPOINT" \
  --judge_endpoint "$JUDGE_ENDPOINT" \
  --attacker_model meta-llama/Llama-3.3-70B-Instruct \
  --judge_model mistralai/Mistral-Large-Instruct-2407 \
  --attacker_prompts_dir utils/sequence_of_prompt_ablations/orig_v4_2x \
  --reward_functions fooling_successful \
  --eval_reward_functions backward_ToM_belief \
  --eval_trajectory_level_rewards prior_knowledge_ToM \
  --torch_seed 42 --seed 42 --loss_type dr_grpo \
  --save_trajectory_outputs "$RESULTS_DIR/trajectories.jsonl"
