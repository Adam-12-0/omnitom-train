# OmniToM Double Agent Reproduction Log

Inspection date: 2026-07-07

## Repository revisions

- `AIDoubleAgentDefenders`: `0f1ad2d336d9986eed736a7aa458275beb1535fc`
- `omnitom-benchmark-review`: `2a4da07b55165bf4ec1a199287c0c0ac1dc90009`

## Provenance

- `AIDoubleAgentDefenders`: `https://github.com/The-Inscrutable-X/AIDoubleAgentDefenders.git`
- `omnitom-benchmark-review`: `https://github.com/omnitom01/omnitom-benchmark-review.git`

No package installation, model download, training, model serving, or Slurm submission was performed during this inspection.

## Phase 1 implementation findings

- No public trained ADA checkpoint reference or tracked adapter weights were found.
- `Qwen3-14B` resolves to `Qwen/Qwen3-14B`.
- YAML configuration accepts `judge_endpoint` and `attacker_endpoint`; explicit CLI arguments take precedence.
- `Trajectory.compute_precursors()` uses the attacker model/client for extraction-success judging. The separately constructed fooling and ToM reward functions use the configured judge.
- Training YAML fields are `learning_rate=1e-5`, `rank=32`, `alpha=32`, `max_iterations=15`, `num_generations=8`, and `lr_scheduler=linear`.
- The training CLI exposes no dtype or quantization option. The inspected main load uses automatic dtype and does not activate 4-bit or 8-bit loading.
- Installation remains deferred pending GPU-node CUDA/driver checks. Direct requirements are pinned, but transitive dependencies are not locked; vLLM/PyTorch are large GPU-oriented installs, and pinned vLLM `0.10.2` has published security issues.

## OmniToM Stage 1 SFT scaffold

- Converter defaults: story-level split, seed `42`, train fraction `0.9`, and Stage 1-only `Actor | Belief | Order` targets.
- Synthetic smoke fixture conversion used seed `42` and train fraction `0.67`: 3 usable stories, 2 train, 1 validation, 9 source belief rows, and 1 intentionally malformed row skipped.
- All requested output files and JSONL message/header invariants passed validation.
- Training dry-run used model ID `Qwen/Qwen3-14B`, loaded no model, downloaded no tokenizer, and found 0 of 3 synthetic examples over the 8192-token limit using the explicitly reported whitespace-estimate fallback.
- `OMNITOM_JSON` was unset; private benchmark conversion was deferred without assuming or recording a path.
- At scaffold verification time, `transformers`, `datasets`, `peft`, and `trl` were not installed. The model-free dry-run is designed to work in this state and reports estimated lengths when no local tokenizer is available.
