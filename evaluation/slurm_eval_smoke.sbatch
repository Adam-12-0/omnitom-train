#!/bin/bash
#SBATCH --job-name=omni-eval-smoke
#SBATCH --account=ai
#SBATCH --partition=highgpu,normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
# evc101 has no usable NVIDIA device in the Codex allocation; avoid it.
#SBATCH --exclude=evc101
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_smoke-%j.out
#SBATCH --error=logs/eval_smoke-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$ROOT/AIDoubleAgentDefenders"
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate /home/ad906660/.conda/envs/tom
mkdir -p logs results/evaluations/smoke

export RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/evaluations/smoke}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-unused}"
export WANDB_MODE="offline"
export SHELLS_LAUNCHER_LOG_NAME="omnitom_eval_smoke_${SLURM_JOB_ID}"
export WANDB_PROJECT="omnitom-open-eval"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${ATTACKER_ENDPOINT:-}" || -z "${JUDGE_ENDPOINT:-}" ]]; then
  : "${ATTACKER_JOB_ID:?Set ATTACKER_ENDPOINT/JUDGE_ENDPOINT or the corresponding service job IDs}"
  : "${JUDGE_JOB_ID:?Set ATTACKER_ENDPOINT/JUDGE_ENDPOINT or the corresponding service job IDs}"
  for attempt in $(seq 1 120); do
    attacker_state="$(squeue -h -j "$ATTACKER_JOB_ID" -o '%T' 2>/dev/null || true)"
    judge_state="$(squeue -h -j "$JUDGE_JOB_ID" -o '%T' 2>/dev/null || true)"
    if [[ "$attacker_state" == RUNNING && "$judge_state" == RUNNING ]]; then
      attacker_node="$(squeue -h -j "$ATTACKER_JOB_ID" -o '%N')"
      judge_node="$(squeue -h -j "$JUDGE_JOB_ID" -o '%N')"
      export ATTACKER_ENDPOINT="http://${attacker_node}:8001/v1"
      export JUDGE_ENDPOINT="http://${judge_node}:8002/v1"
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

COMMON=(
  --eval_only --do_run_first_eval 1 --testing --eval_limit 2
  --eval_batch_size 1 --eval_num_workers 1
  --max_iterations 3 --max_completion_length 512
  --attacker_endpoint "$ATTACKER_ENDPOINT"
  --judge_endpoint "$JUDGE_ENDPOINT"
  --attacker_model meta-llama/Llama-3.3-70B-Instruct
  --judge_model mistralai/Mistral-Large-Instruct-2407
  --attacker_prompts_dir utils/sequence_of_prompt_ablations/orig_v4_2x
  # Force identical evaluation rewards for Base and OmniToM; the adapter YAML
  # otherwise contributes its training reward and makes fooling incomparable.
  --reward_functions fooling_successful
  --eval_reward_functions backward_ToM_belief
  --eval_trajectory_level_rewards prior_knowledge_ToM
  --torch_seed 42 --seed 42 --loss_type dr_grpo
)

run_one() {
  local name="$1" config="$2"
  echo "START ${name} $(date -Is)"
  python -u -m main_scripts.main_training_script --config "$config" \
    --results_dir "$RESULTS_DIR/$name" "${COMMON[@]}" \
    --save_trajectory_outputs "$RESULTS_DIR/$name/trajectories.jsonl"
  echo "DONE ${name} $(date -Is)"
}

run_one base_qwen3_14b config_launchers/configs/eval_base_qwen3_14b.yaml
run_one omnitom_stage1_sft config_launchers/configs/eval_omnitom_stage1_sft.yaml
echo "smoke_completed=$(date -Is)"
