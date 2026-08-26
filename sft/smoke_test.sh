#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python "$ROOT/sft/convert_omnitom_stage1.py" --smoke --output-dir "$ROOT/data/omnitom_stage1_sft_smoke"
python "$ROOT/sft/train_omnitom_stage1.py" --smoke --max-steps 2 \
  --data-dir "$ROOT/data/omnitom_stage1_sft_smoke" \
  --output-dir "$ROOT/results/checkpoints/omnitom_stage1_sft_qwen3_14b_smoke"
