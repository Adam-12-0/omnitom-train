# OmniToM Train

Reproducible supervised fine-tuning experiment testing whether explicit belief modeling from OmniToM Stage 1 improves the Qwen3-14B defender in the AI Double Agent Defenders environment.

## Research question

Does supervised training on structured `(Actor, Belief, Order)` extraction improve Qwen3-14B as a Double Agent defender, and how does it compare with the original Double Agent ToM-only, fooling-only, and ToM+fooling GRPO specializations?

## What is included

- `sft/`: data conversion, LoRA SFT training, smoke-test and full Slurm jobs, dependency list, and reproduction notes.
- `AIDoubleAgentDefenders/`: the evaluation/training code and aligned comparison configurations.
- `omnitom-benchmark-review/`: the public OmniToM prompt builders and benchmark code.
- `results/checkpoints/omnitom_stage1_sft_qwen3_14b/`: the final LoRA adapter, tokenizer metadata, and provenance.
- `logs/`: selected smoke/full training logs.

The private OmniToM story/annotation JSONL is intentionally excluded from this repository. Place it at `omnitom-benchmark-review/benchmark_story_belief_labels.jsonl` locally before reproducing the conversion. Credentials, virtual environments, caches, optimizer states, and intermediate checkpoints are also excluded.

## Final run

The completed run used:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-14B` |
| Task | OmniToM Stage 1 extraction only |
| Target | `Actor \| Belief \| Order` table |
| Training | LoRA supervised fine-tuning with TRL `SFTTrainer` |
| Learning rate | `2e-5` |
| Epochs | `3` |
| Optimizer | AdamW |
| LoRA | rank `32`, alpha `64`, dropout `0.0` |
| Modules | q/k/v/o projections and gate/up/down projections |
| Precision | bf16 |
| Context | 8192 tokens |
| Batch | 1 per device × 16 accumulation = effective 16 |
| Seed | 42 |
| GPU | NVIDIA H100 80GB HBM3 |
| Slurm account | `ai` |
| Full Slurm job | `775995` |
| Validation loss | `0.359478` |

See `results/checkpoints/omnitom_stage1_sft_qwen3_14b/training_provenance.json` for commit hashes, runtime, exact paths, and metrics.

## Reproduction

Install the environment:

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -r sft/requirements.txt
```

With the private benchmark file present, run the smoke test first:

```bash
sbatch sft/slurm_smoke_test.sbatch
```

Then submit the full run:

```bash
sbatch sft/slurm_full_train.sbatch
```

Both jobs request `--account=ai`, the `highgpu` partition, and one H100-class GPU. The normal partition exposed a 15.77 GiB GPU and could not hold the requested bf16 14B model; the experiment therefore changed only the resource request, not the training hyperparameters.

## Evaluation comparison

`AIDoubleAgentDefenders/config_launchers/eval_omnitom_comparison.sh` invokes the existing evaluation launcher over the same ToM-SB scenarios, attacker prompts, defender prompts, ToM judge logic, fooling logic, trajectory rewards, stepwise ToM logic, and post-processing for:

1. Base Qwen3-14B
2. OmniToM Stage 1 SFT Qwen3-14B
3. Double Agent ToM-only
4. Double Agent fooling-only
5. Double Agent ToM+fooling

The preferred open evaluation models are configured as attacker `meta-llama/Llama-3.3-70B-Instruct` and judge `mistralai/Mistral-Large-Instruct-2407`. Existing Double Agent checkpoint paths may need to be set to the local locations of those trained adapters.

No Double Agent GRPO training is performed by the OmniToM SFT pipeline.
