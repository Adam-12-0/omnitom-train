#!/bin/bash
#SBATCH --job-name=ada-tom-smoke
#SBATCH --account=ai
#SBATCH --partition=highgpu,normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/ada_tom_smoke-%j.out
#SBATCH --error=logs/ada_tom_smoke-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate /home/ad906660/.conda/envs/tom
mkdir -p logs results/checkpoints/ada_tom_only_qwen3_14b_open
: "${ATTACKER_ENDPOINT:?Set the running attacker vLLM endpoint}"
: "${JUDGE_ENDPOINT:?Set the running judge vLLM endpoint}"
export RESULTS_DIR="$PWD/results/grpo_smoke"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-unused}"
export WANDB_MODE="offline"

python -u AIDoubleAgentDefenders/main_scripts/main_training_script.py \
  --config AIDoubleAgentDefenders/config_launchers/configs/open_repro_traj_PToM.yaml \
  --results_dir "$RESULTS_DIR" \
  --attacker_endpoint "$ATTACKER_ENDPOINT" \
  --judge_endpoint "$JUDGE_ENDPOINT" \
  --testing --eval_limit 2 --epochs 1 --max_iterations 2 --num_generations 2 \
  --max_completion_length 256 --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 --torch_seed 42 --seed 42 \
  --model_savepath "$PWD/results/checkpoints/ada_tom_only_qwen3_14b_open_smoke" \
  --attacker_prompts_dir AIDoubleAgentDefenders/utils/sequence_of_prompt_ablations/orig_v4_2x \
  --save_trajectory_outputs "$PWD/results/grpo_smoke/trajectories.jsonl"
