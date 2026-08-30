#!/bin/bash
# Paper-aligned Online SFT reproduction with the shared open attacker/judge.
# One trajectory per training scenario; SFT updates only after successful fooling.
#SBATCH --job-name=ada-online-sft
#SBATCH --account=ai
#SBATCH --partition=highgpu,normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --exclude=evc101
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/online_sft-%j.out
#SBATCH --error=logs/online_sft-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate /home/ad906660/.conda/envs/tom
cd "$ROOT/AIDoubleAgentDefenders"
export RESULTS_DIR="$ROOT/results/checkpoints/double_agent_online_sft_qwen3_14b"
export WANDB_MODE=offline WANDB_PROJECT=omnitom-online-sft
export SHELLS_LAUNCHER_LOG_NAME="online_sft_${SLURM_JOB_ID}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-unused}"
export TOKENIZERS_PARALLELISM=false
: "${ATTACKER_JOB_ID:?}" "${JUDGE_JOB_ID:?}"
for attempt in $(seq 1 720); do
  ATTACKER_ENDPOINT="http://$(squeue -h -j "$ATTACKER_JOB_ID" -o '%N'):8001/v1"
  JUDGE_ENDPOINT="http://$(squeue -h -j "$JUDGE_JOB_ID" -o '%N'):8002/v1"
  a=$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' "${ATTACKER_ENDPOINT%/v1}/health" || true)
  j=$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' "${JUDGE_ENDPOINT%/v1}/health" || true)
  [[ "$a" == 200 && "$j" == 200 ]] && break
  sleep 5
done
[[ "$a" == 200 && "$j" == 200 ]]
python -u -m main_scripts.main_training_script \
  --config "$ROOT/AIDoubleAgentDefenders/config_launchers/configs/open_repro_online_sft.yaml" \
  --results_dir "$ROOT/results/checkpoints" \
  --model_savepath "$RESULTS_DIR" \
  --attacker_endpoint "$ATTACKER_ENDPOINT" --judge_endpoint "$JUDGE_ENDPOINT" \
  --load_in_4bit --gradient_checkpointing --torch_seed 42 --seed 42 \
  --attacker_prompts_dir utils/sequence_of_prompt_ablations/orig_v4_2x \
  --save_trajectory_outputs "$RESULTS_DIR/trajectories.jsonl"
