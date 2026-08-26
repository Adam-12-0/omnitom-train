# Codex Progress

## 2026-07-07 — Phase 1 findings

- `AIDoubleAgentDefenders` commit: `0f1ad2d336d9986eed736a7aa458275beb1535fc`
- `omnitom-benchmark-review` commit: `2a4da07b55165bf4ec1a199287c0c0ac1dc90009`
- Public trained ADA checkpoints: none found in the repository.
- Qwen3-14B model mapping: `Qwen/Qwen3-14B`.
- `judge_endpoint` and `attacker_endpoint`: accepted from YAML and CLI; explicit CLI values override YAML values.
- `compute_precursors`: uses the attacker model, tokenizer, and client for extraction-success judging, not the configured judge. Other reward paths use the configured judge.
- Training fields: `learning_rate` (default `5e-5`, training YAML `1e-5`), `rank` (LoRA rank; default `16`, YAML `32`), `alpha` (LoRA alpha; default `16`, YAML `32`), `max_iterations` (default/YAML `15`), `num_generations` (default/YAML `8`), and `lr_scheduler` (default `constant`, YAML `linear`).
- Dtype: no YAML/CLI field; the main path calls `load_model(..., manual_precision=False)`, which uses `torch_dtype="auto"`.
- Quantization: no active YAML/CLI setting. Internal `load_in_8bit` and `load_in_4bit` arguments default false, and the constructed BitsAndBytes configuration is not passed through the inspected main loading path.
- Dependency risks: GPU/CUDA compatibility is unverified; vLLM and PyTorch wheels are large; transitive dependencies are not locked; pinned vLLM `0.10.2` has published security issues; parser construction expects environment variables including `RESULTS_DIR` and `AZURE_OPENAI_ENDPOINT`; W&B initialization is unconditional; repository launchers expect a `.env`, which must not be used to persist secrets in this workflow.

## 2026-07-07 — OmniToM Stage 1 scaffold

OMNITOM_JSON is not set, real dataset conversion deferred.

- Created the Stage 1 converter, synthetic three-story fixture, and LoRA SFT training scaffold.
- Synthetic conversion passed: 3 input/usable stories, 2 train stories, 1 validation story, 9 source belief rows, 1 skipped malformed row, and all five requested output files produced.
- JSONL validation passed for train, validation, and sample files. Every example has system/user/assistant messages, valid JSON, and an assistant target beginning with `Actor | Belief | Order`.
- Tiny SFT dry-run passed with `Qwen/Qwen3-14B`. It formatted all 3 examples and measured lengths without loading a model or downloading a tokenizer; whitespace estimates ranged from 125 to 143 tokens, with 0 over the 8192 limit.
- No package installation, model download, training, vLLM serving, or Slurm submission was performed.
- Current environment package probe: `transformers`, `datasets`, `peft`, and `trl` are absent. This does not affect the dependency-free dry-run; actual training remains intentionally blocked pending an approved compatible GPU environment.
