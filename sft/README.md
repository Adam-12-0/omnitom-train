# OmniToM Stage 1 SFT

This pipeline converts `omnitom-benchmark-review/benchmark_story_belief_labels.jsonl` using the released Stage 1 extraction prompt and trains only the `Actor | Belief | Order` target table. Stage 2 labels, an LLM judge, and GRPO are not used.

## Reproduction

```bash
python sft/convert_omnitom_stage1.py
sbatch sft/slurm_smoke_test.sbatch
sbatch sft/slurm_full_train.sbatch
```

The validated run used Slurm job `775964` for smoke testing and `775965` for full training. The final adapter is at `results/checkpoints/omnitom_stage1_sft_qwen3_14b`.

The default normal partition exposed a 15.77 GiB GPU and OOMed while placing the requested bf16 Qwen3-14B model. The jobs therefore request `highgpu` with an NVIDIA H100 80GB HBM3. This preserves all requested training hyperparameters; the memory-driven resource change is recorded in `training_provenance.json`.

Evaluation comparison configs and the unchanged Double Agent evaluation launcher are under `AIDoubleAgentDefenders/config_launchers/`.
