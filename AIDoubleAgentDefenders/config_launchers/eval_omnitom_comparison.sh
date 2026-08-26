#!/usr/bin/env bash
set -euo pipefail
# Run all five defenders with the repository's unchanged evaluation launcher.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python shells_launcher.py -s config_launchers/launch_main_eval_ov4eval_multiattackeranalysis.sh \
  -l "omnitom_comparison_$(date +%Y%m%d_%H%M%S).txt" -a "${CUDA_VISIBLE_DEVICES:-0}" 42 \
  config_launchers/configs/eval_base_qwen3_14b.yaml \
  config_launchers/configs/eval_omnitom_stage1_sft.yaml \
  config_launchers/configs/eval_double_agent_tom_only.yaml \
  config_launchers/configs/eval_double_agent_fooling_only.yaml \
  config_launchers/configs/eval_double_agent_tom_and_fooling.yaml
