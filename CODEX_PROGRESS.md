# Codex Progress

## 2026-08-30 — Job audit, evaluation repair, and results dashboard

- Scheduler audit: the stratified OmniToM SFT `787522` and Double-Agent JSON-format control SFT `787524` completed successfully and wrote complete Qwen3-14B LoRA adapters. The live OmniToM Double-Agent rollout `787527` is healthy; its log contains completed trajectories (at least 50 observed during audit). ToM-only and fooling-only four-GPU GRPO runs `787498`/`787499` are also actively processing training examples; joint `787500` is pending priority. Attacker and Mistral judge services `787496`/`787497` remain running.
- The initial structured-evaluation jobs `787523`, `787525`, and `787526` failed after model load because `apply_chat_template` returned a `BatchEncoding` that was passed directly to `generate`. Corrected `sft/evaluate_stage1_structured.py` to pass `input_ids` and the encoded attention fields. Reruns: base `787748` (running), OmniToM `787749` (pending priority), and format control `787750` (pending priority).
- The failed format-control Double-Agent evaluation `787528` exposed a second issue: its descriptive adapter directory was parsed as base model `double`. The model resolver now explicitly maps that schema-only adapter to Qwen3-14B, so it can be reevaluated after its structured result is available.
- Added and executed `results/analysis/experiment_dashboard.ipynb`; it validates and renders the completed Base-vs-OmniToM behavioral plots, then dynamically incorporates completed structured results without fabricating pending metrics. Its rendered figure is in `results/analysis/figures/`.

## 2026-08-30 — Core OmniToM fine-tuning comparison queued

- Added a held-out Stage-1 generation evaluator reporting exact-set match, proposition precision/recall/F1, actor accuracy conditional on matched belief/order, and order accuracy conditional on matched actor/belief.
- Corrected all new SFT/evaluation launchers to invoke this checkout's `.venv/bin/python` directly. Its activation script contains an obsolete absolute workspace path and silently activated base Python, which caused the initial SFT/structured-eval/control jobs `787515`, `787518`, and `787519` to fail before model loading; this was not a GPU or model-memory failure.
- Resubmitted the core jobs: stratified assistant-only OmniToM Stage-1 SFT `787522`, Base held-out structured evaluation `787523`, and the no-OmniToM Double-Agent JSON-format control SFT `787524`.
- Queued successful-completion dependent jobs: OmniToM structured evaluation `787525`, format-control structured evaluation `787526`, OmniToM Double-Agent evaluation `787527`, and format-control Double-Agent evaluation `787528`. The Double-Agent launcher now resolves its YAML path before changing into the embedded repository.
- The format control is deliberately trained only on a fixed Double-Agent-compatible JSON schema derived from the 300-row Double-Agent dataset; it contains no OmniToM stories, beliefs, actors, or ordering labels. This separates output-schema/behavioral drift from structured belief supervision.

## 2026-08-30 — OmniToM calibration and follow-up design

- Located the adjacent `../OmniToM` workspace containing the merged 126-table human-agreement protocol (9 levels × 14 stories), prior judge outputs for GPT-5, Gemini, Claude, Llama, and DeepSeek, and the reusable Mistral-Large Stage-1 evaluator.
- Submitted Mistral-Large-Instruct-2407 calibration job `787508` (12-hour GPU allocation, 126-table protocol) from the adjacent workspace. Results are not available yet.
- The requested held-out structured metrics, assistant-only masked SFT, format-only control, Double-Agent JSON control, independent extraction judging, behavioral analysis, and multi-seed comparisons are not yet submitted; they require new scripts/configs beyond the existing trajectory evaluator.
- Corrected the OmniToM Stage 1 split to deterministic randomization within each story category (805 train / 90 validation; no test split), preserving proportional category representation. The regenerated data and manifest are in `data/omnitom_stage1_sft_stratified`; retraining submitted as SFT job `787510` to a new checkpoint path, leaving the historical adapter unchanged.
- SFT job `787510` failed immediately because `.venv` lacked Transformers. The launcher now activates the established `tom` environment; retry `787512` targets `omnitom_stage1_sft_qwen3_14b_stratified_v2`.

## 2026-08-28 — Full open-weight evaluations completed; GRPO reproductions launched

- Base Qwen3-14B full evaluation `780916` completed successfully: 150 trajectories, 62.67% fooling, 20.00% all-prompts fooling, 0.2067 prior-knowledge ToM trajectory mean.
- OmniToM Stage 1 SFT full evaluation `780338` completed successfully: 150 trajectories, 44.00% fooling, 15.00% all-prompts fooling, 0.2333 prior-knowledge ToM trajectory mean.
- ToM-only GRPO smoke `780249` completed successfully, trained and saved `results/checkpoints/ada_tom_only_qwen3_14b_open_smoke`; it exercised 2 examples, 2 generations, and 2 turns and reached checkpoint saving.
- Added `evaluation/slurm_grpo_full.sbatch`, which waits for live attacker/judge health endpoints, excludes `evc101`, and runs the original 15-turn/8-generation open-weight recipe with bounded judge calls.
- Full GRPO reproductions submitted from base Qwen3-14B: ToM-only `784989` (running), fooling-only `784990` (pending priority), and joint ToM+fooling `784991` (pending priority). All use the same Llama-3.3-70B attacker and Mistral-Large judge services (`780244`/`780245`).
- Paired Base-vs-OmniToM comparison over the 150 matched scenario/prompt trajectories: Base fooling `62.67%` versus OmniToM-SFT `44.00%` (difference `-18.67` percentage points; 20,000-resample paired bootstrap 95% CI approximately `[-30.0, -7.3]`; paired sign-randomization p approximately `0.0023`). OmniToM had higher prior-knowledge ToM trajectory mean (`0.2333` vs `0.2067`) and higher attacker extraction rate (`42.67%` vs `27.33%`), so structured belief supervision improved those signals but did not improve deception/fooling in this evaluation.

## 2026-08-28 — GRPO OOM mitigation and reruns

- Audit found the GRPO defender was not actually quantized: the loader constructed a BitsAndBytes config but did not pass it to `from_pretrained`, and the launcher supplied no 4-bit flag. Fixed the loader, added `--load_in_4bit`, and invoke `prepare_model_for_kbit_training` for quantized LoRA training.
- Added per-job `nvidia-smi` diagnostics (GPU name, compute capability, memory, and full status) to the GRPO launcher. The launcher continues to exclude known-bad `evc101`; a diagnostic failure now exits before training.

- Initial full GRPO jobs `784989` (ToM-only) and `784990` (fooling-only) failed during the defender per-token log-probability pass with CUDA OOM on H100 80GB; joint job `784991` was canceled before reaching the same failure path.
- Updated `evaluation/slurm_grpo_full.sbatch` to set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and default `GRPO_MAX_COMPLETION_LENGTH=4096` (overrideable), reducing activation memory while retaining 15 turns, 8 generations, and the original learning-rate/reward recipes.
- Pre-quantization pending reruns `785206`–`785208` were canceled. Corrected 4-bit reruns submitted: ToM-only `785220`, fooling-only `785221`, and joint ToM+fooling `785222`, using the existing healthy services; each now records `nvidia-smi` diagnostics before training.
- Jobs `785220`–`785222` reached healthy H100 PCIe nodes but failed immediately because relative `GRPO_CONFIG` paths were resolved after `cd AIDoubleAgentDefenders`. The launcher now canonicalizes the config path against `SLURM_SUBMIT_DIR`; no GPU fault was observed.
- Resubmitted after the path fix: ToM-only `787177`, fooling-only `787178`, and joint ToM+fooling `787179` (all use actual 4-bit defender loading and preflight `nvidia-smi`).
- Jobs `787177`–`787179` confirmed 4-bit `Linear4bit` loading and healthy H100s, then hit activation OOM in Qwen attention/log-prob computation. Added non-reentrant gradient checkpointing, disabled KV cache for k-bit training, and reduced the default completion cap to 2048 for the next retry.
- Resubmitted memory-safe retries: ToM-only `787189`, fooling-only `787190`, and joint ToM+fooling `787191`.
- Retries `787189`–`787191` still OOM'd during attention with 8-generation batches despite 4-bit weights, checkpointing, and disabled KV cache. Reduced the launcher to 2 generations/update (gradient accumulation 16) and a 1024 completion cap for the next retry.
- Retries `787292`–`787294` still OOM'd at 77–79 GiB on healthy H100s even with 2 generations, checkpointing, and KV-cache disabled; individual multi-turn sequences reached 2.4–2.7k tokens. Switched the launcher to 2 GPUs/job so `device_map=auto` can shard the defender and activation workload.
- Fooling-only `787309` subsequently OOM'd on GPU 1 despite two GPUs. Added explicit `device_map=balanced` placement and resubmitted it as `787322`; ToM-only `787308` and joint `787310` remain under observation.
- ToM-only `787308` later OOM'd on GPU 1 at ~78.4 GiB while joint `787310` and fooling-only `787322` continued. Escalated the launcher to four GPUs with balanced placement for the ToM-only retry.
- `787310` and `787322` ultimately OOM'd on GPU 1; four-GPU ToM-only `787334` instead reached the training/evaluation stage but failed when services `780244/780245` timed out at three days, producing connection-refused errors. Relaunched services `787496`/`787497` and resubmitted four-GPU GRPO jobs: ToM-only `787498`, fooling-only `787499`, joint `787500`.

## 2026-08-27 — Full common-stack evaluation launched

- Live services: attacker `780216` on `evc104:8001` (`meta-llama/Llama-3.3-70B-Instruct`) and judge `780217` on `evc28:8002` (`mistralai/Mistral-Large-Instruct-2407`). Both request a single H100 80GB-compatible `gpu80`, `account=ai`, `partition=highgpu,normal`, `12:00:00`, and load in vLLM BitsAndBytes 4-bit mode.
- Submitted full controlled evaluation jobs: Base Qwen3-14B `780238`, and OmniToM Stage 1 SFT Qwen3-14B `780239`. Each requests one `gpu80` under `ai` for `12:00:00`, uses the same 75 held-out ToM-SB scenarios and two original attacker prompt variants (150 trajectories/defender), and waits for HTTP health checks from both serving endpoints before rolling out.
- Submitted ToM-only GRPO reproduction smoke `780240` under `ai` (one `gpu80`, `06:00:00`). It will wait for the endpoints and then test only two examples with two generations and two turns; all three full ADA GRPO jobs remain gated on this smoke test.
- Adapter provenance: the primary comparison remains Base vs OmniToM-SFT vs ADA ToM-only trained from the *base* Qwen3-14B. An OmniToM-SFT-to-GRPO run is a separate sixth ablation, not a replacement for the paper-aligned ADA conditions.

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

## 2026-08-26 — Open-weight evaluation preparation

- Thorough local repository search found no usable trained ADA checkpoints for ToM-only, fooling-only, or ToM+fooling. The repository README documents how to load a checkpoint, but does not provide checkpoint URLs or embedded weights. Upstream web search found no official checkpoint release/Hugging Face adapter reference.
- The original configs are `train_traj_PToM.yaml`, `train_traj_fool.yaml`, and `train_traj_fool_and_PToM.yaml`. Their recorded recipe is: `Qwen3-14B`; `TrajectorywiseGRPO`; `epochs=1`; `learning_rate=1e-5`; `lr_scheduler=linear`; `gradient_accumulation_steps=4` from the launcher; `num_generations=8`; `max_iterations=15`; `max_completion_length=20000`; `loss_type=dr_grpo`; LoRA `rank=32`, `alpha=32`, `target_modules=all-linear`; warmup ratio `0.0`; launcher torch seed `42`; prompt set `orig_v4_2x`. The configs do not specify temperature, top_p, dtype, quantization, or an explicit training seed. Their judge default is Gemini, so open reproduction configs replace only attacker/judge models with the requested Llama/Mistral stack.
- Added separate vLLM Slurm service scripts for `meta-llama/Llama-3.3-70B-Instruct` and `mistralai/Mistral-Large-Instruct-2407`, each requesting `account=ai`, `partition=highgpu,normal`, `constraint=gpu80`, and one GPU.
- Added an evaluation smoke job for exactly two evaluation scenarios and the base/OmniToM defenders. It retains existing prompts, rollout, reward, parsing, and aggregation code; smoke-only overrides are `max_iterations=3`, `max_completion_length=512`, `eval_batch_size=1`, and `eval_limit=2`.
- Added `eval_limit` to the Double Agent entrypoint and fixed the eval YAMLs to provide the parser-required `loss_type=dr_grpo`.
- Added open-weight GRPO reproduction configs and a ToM-only GRPO smoke job, but no GRPO job has been submitted pending successful open-weight evaluation smoke results and explicit review of the proposed expensive allocation.
- The `tom` environment now has the missing evaluation packages `peft`, `datasets`, and `wandb`; it already contained `vllm=0.11.2`, `torch=2.6.0`, and `transformers=4.57.6`.
- Proposed GPU requests: one `gpu80` GPU per vLLM service and one `gpu80` GPU for the defender smoke/evaluation job; full five-defender evaluation can be serialized on one GPU or split into separate one-GPU jobs after the smoke test. No full evaluation or GRPO reproduction has been launched yet.
- Smoke attempt `777633` reached an H100 `gpu80` node under `ai` but failed before Python startup because the job scripts used a nonexistent `tom/bin/activate` path. The fix is to source cluster `conda.sh` and `conda activate /home/ad906660/.conda/envs/tom`; the service jobs `777626` and `777627` remain queued with Slurm reason `Priority` and scheduled start estimates.
- Submitted attacker service `777626` and judge service `777627`; both request `account=ai`, `partition=highgpu,normal`, `constraint=gpu80`, and one GPU. Initial smoke `777636` was cancelled while waiting for priority-queued services and replaced by smoke `777637` with `after:777626:777627`, so it will not occupy a GPU before both services start.
- Per request, jobs `777626`, `777627`, and `777637` were cancelled. Replacement jobs are attacker `777639`, judge `777640`, and dependent smoke `777641`; all use a 12-hour time limit and no explicit `--mem` directive. They remain under account `ai` with `partition=highgpu,normal` (Slurm canonicalizes this to `normal,highgpu`) and `constraint=gpu80`.
- Jobs `777639` and `777640` started on H100 80GB nodes but failed during vLLM model initialization with CUDA OOM: both 70B-class models require more than one GPU in bf16. Dependent smoke `777641` resolved the endpoints but failed because it invoked `main_scripts/main_training_script.py` from the experiment root rather than the nested `AIDoubleAgentDefenders` repository. The service scripts now request two `gpu80` GPUs and `tensor_parallel_size=2`; the smoke script now changes into the defender repository before invoking the unchanged evaluation code.
- The two-GPU jobs were cancelled and replaced with single-GPU 4-bit jobs per request. Both vLLM services now use BitsAndBytes quantization (`--quantization bitsandbytes`, `--load-format bitsandbytes`), `tensor_parallel_size=1`, and a 6-hour limit. They load `HF_TOKEN` from the exported environment first, then `.hf_token.env`, `$HOME/.config/huggingface/token`, or `$HOME/.cache/huggingface/token`, without printing the secret. Smoke and GRPO-smoke limits were also reduced to 6 hours.
- Verified the sibling `/home/ad906660/OmniToM/experiments` token pattern and confirmed `$HOME/.cache/huggingface/token` is present. Cancelled the prior pending workflow and submitted attacker `777735`, judge `777736`, and dependent smoke `777737`. Each requests one `gpu80` GPU, `account=ai`, `partition=highgpu,normal`, and `06:00:00`; Slurm's reported memory is its default allocation because no `--mem` directive is present.
- Jobs `777735` and `777736` are now running on H100 nodes and vLLM is loading both models successfully with BitsAndBytes 4-bit configuration. Smoke `777737` failed before evaluation with `ModuleNotFoundError: No module named 'utils'` because the script was invoked by file path; it is fixed to invoke `main_scripts.main_training_script` as a module and will be resubmitted against the running services.
- Status check: attacker `777735` successfully completed model loading and exposed the vLLM API on `evc24:8001`. Judge `777736` loaded all 51 shards but failed at KV-cache setup because 32768 context required 11.00 GiB while only 2.87 GiB remained; judge serving is reduced to `max_model_len=8192`. Smoke `777744` reached the evaluator but failed because the direct job lacked `SHELLS_LAUNCHER_LOG_NAME`; the smoke job now sets offline W&B metadata explicitly. No comparative defender metrics exist yet.
- Evaluation YAML checkpoint roots are now relative (`../results/checkpoints`) to avoid dependence on the mounted `omnitom_da_experiment` directory name. The parent filesystem is read-only, so the physical directory cannot be renamed in this environment; scripts derive their root from `SLURM_SUBMIT_DIR` where possible.
- Added the missing `Mistral-Large-Instruct-2407 -> mistralai` model mapping in `AIDoubleAgentDefenders/utils/model_utils.py`; Python compilation passed.
- Smoke job `778051` failed because of that missing mapping. The first corrected submission `778441` was cancelled because an `afterok` dependency incorrectly waited for the long-running serving jobs to finish. Corrected smoke job `778442` is now `RUNNING` on `evc101`, using attacker `777735` and judge `778050`; it resolved endpoints `evc24:8001` and `evc26:8002` and started the base-defender evaluation.
