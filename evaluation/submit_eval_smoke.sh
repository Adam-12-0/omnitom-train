#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

attacker_jid="$(sbatch --parsable evaluation/serve_vllm_attacker.sbatch)"
judge_jid="$(sbatch --parsable evaluation/serve_vllm_judge.sbatch)"
echo "attacker_job=${attacker_jid}"
echo "judge_job=${judge_jid}"

smoke_jid="$(sbatch --parsable --export=ALL,ATTACKER_JOB_ID="$attacker_jid",JUDGE_JOB_ID="$judge_jid" evaluation/slurm_eval_smoke.sbatch)"
echo "smoke_job=${smoke_jid}"
