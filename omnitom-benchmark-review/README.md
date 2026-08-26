# OmniToM Official Release

OmniToM is a benchmark for evaluating Theory of Mind in language models through explicit belief-structure modeling. Instead of scoring only endpoint answers, OmniToM exposes the intermediate belief structure that a model must construct in order to reason coherently about what different actors know, believe, infer, intend, or misunderstand.

This release contains the benchmark dataset, the prompt builders used for the final benchmark experiments, and a replication runner for Stage 1 belief extraction, Stage 2 belief labeling, GPT-5 semantic-judge evaluation, and metric aggregation.

<img src="assets/figure2_omnitom_pipeline.png" alt="OmniToM two-stage workflow" width="100%" />

## Important Notice

OmniToM is intended for evaluation and analysis. To reduce benchmark contamination, we strongly recommend using the released stories for benchmarking rather than training.

## Benchmark at a Glance

- `895` benchmark stories
- `22,343` labeled belief propositions
- `156,401` total schema labels (`22,343 x 7`)
- `7` retained ToMBench source categories

Category-level counts in the benchmark:

| Category | Stories | Beliefs | Avg. beliefs/story |
| --- | ---: | ---: | ---: |
| Ambiguous Story Task | 98 | 4,614 | 47.08 |
| False Belief Task | 97 | 2,168 | 22.35 |
| Faux-pas Recognition Test | 142 | 3,981 | 28.04 |
| Hinting Task Test | 100 | 2,317 | 23.17 |
| Persuasion Story Task | 97 | 1,204 | 12.41 |
| Scalar Implicature Test | 154 | 3,251 | 21.11 |
| Strange Story Task | 207 | 4,808 | 23.23 |
| **Total** | **895** | **22,343** | **24.96** |

Belief-order distribution:

- Order `0`: `32.57%`
- Order `1`: `57.12%`
- Order `2+`: `10.32%`

## Experiment Protocol

OmniToM evaluates two linked tasks:

1. **Stage 1: Belief Extraction**
   Input a story and extract `(Actor, Belief, Order)` tuples.
2. **Stage 2: Belief Labeling**
   Input a story and the benchmark belief table, then assign a seven-dimensional schema label vector to each belief.

The final benchmark experiments use fixed prompt structures. Stage 1 and Stage 2 are zero-shot; the semantic judge uses the three examples packaged in `prompt_evaluate.py`.

- Stage 1 uses the released extraction prompt in `prompts_extract.py`.
- Stage 2 uses the released labeling prompt in `prompts_label.py`.
- Stage 1 extraction is scored by the released semantic-judge prompt in `prompt_evaluate.py` with `--judge-fewshots 3`.
- OmniToM uses `gpt-5` as the semantic judge for Stage 1. GPT-5 is therefore omitted from Stage 1 model-generation results and reported for Stage 2 labeling only.

The evaluated model set reported in the paper is:

- Closed-source: `gemini-2.5-flash`, `gpt-5`
- Open-weight: `google/gemma-3-27b-it`, `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`, `mistralai/Mistral-Small-24B-Instruct-2501`, `mistralai/Mistral-Large-Instruct-2407`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`

The seven Stage 2 schema dimensions are:

- `Order`
- `Truth Status`
- `Knowledge Access`
- `Representation`
- `Content Type`
- `Mental Source`
- `Context`

<img src="assets/figure3_annotation_pipeline.png" alt="OmniToM annotation pipeline" width="100%" />

<img src="assets/figure4_schema_label_distribution.png" alt="OmniToM label distribution statistics" width="100%" />

## Included Files

- `benchmark_story_belief_labels.jsonl`: benchmark dataset, one JSON object per story
- `benchmark_prompting.py`: helpers for loading stories and reconstructing benchmark tables
- `prompts_extract.py`: Stage 1 extraction prompt builder
- `prompts_label.py`: Stage 2 labeling prompt builder
- `prompt_evaluate.py`: semantic-judge prompt builder with the three judge examples used by the final prompt
- `run_replication.py`: end-to-end experiment and evaluation runner
- `HF_DATASET_CARD.md`: dataset card for dataset-hosting platforms
- `LICENSE`: license for the OmniToM release package

## Dataset Format

Each line of `benchmark_story_belief_labels.jsonl` is one JSON object:

```json
{
  "story_id": 1,
  "story_category": "Ambiguous Story Task",
  "story": "Story text...",
  "beliefs": [
    {
      "actor": "world",
      "belief": "A minimal propositional statement.",
      "labels": {
        "order": "0",
        "truth_status": "True",
        "knowledge_access": "Public",
        "representation": "Explicit",
        "content_type": "Action/Event",
        "mental_source": "Narration",
        "context": "Neutral"
      }
    }
  ]
}
```

## Quick Prompt Usage

```python
from benchmark_prompting import load_story_record
from prompts_extract import build_extract_messages
from prompts_label import build_label_messages
from prompt_evaluate import build_evaluation_messages

record = load_story_record(1)
extract_system, extract_user = build_extract_messages(1)
label_system, label_user = build_label_messages(1)

predictions_csv = 'Actor,Belief\nworld,Example belief\n'
judge_system, judge_user = build_evaluation_messages(1, predictions_csv, fewshots=3)
```

## Installation

For the mock smoke test, Python `3.10+` and the standard library are sufficient.

For model runs:

```bash
pip install -r requirements.txt
```

For gated open-weight models, configure your local Hugging Face access in the normal way for your environment. For API models, set credentials through environment variables:

```bash
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
```

For an OpenAI-compatible endpoint, also set:

```bash
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint.example/v1"
```

## Running the Experiments

Smoke-test the full pipeline:

```bash
python run_replication.py \
  --backend mock \
  --output-dir runs/mock_smoke
```

Run Stage 1 extraction and Stage 2 labeling with an open-weight model:

```bash
python run_replication.py \
  --backend hf \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --load-in-4bit \
  --bnb-4bit-compute-dtype bf16 \
  --stages extract label \
  --output-dir runs/llama33_70b
```

Run a closed-source generator through the API backend:

```bash
python run_replication.py \
  --backend api \
  --api-provider google_genai \
  --model gemini-2.5-flash \
  --stages extract label \
  --output-dir runs/gemini25_flash
```

Run GPT-5 Stage 2 labeling:

```bash
python run_replication.py \
  --backend api \
  --api-provider openai \
  --model gpt-5 \
  --stages label \
  --output-dir runs/gpt5_stage2
```

After Stage 1 extraction exists for a model, run the GPT-5 semantic judge and metrics:

```bash
python run_replication.py \
  --backend api \
  --api-provider openai \
  --judge-model gpt-5 \
  --judge-fewshots 3 \
  --stages judge metrics \
  --output-dir runs/llama33_70b
```

Useful options:

- `--story-ids 1,2,5-7`
- `--max-stories 25`
- `--overwrite`
- `--no-story-context`
- `--max-new-tokens 6000`
- `--temperature 0`
- `--api-key-env MY_API_KEY_ENV`
- `--api-base-url-env MY_BASE_URL_ENV`

## Output Layout

```text
<output-dir>/
  manifest.json
  extraction/
    raw/
    csv/
  labeling/
    raw/
    csv/
  judge/
    raw/
    csv/
  metrics/
    stage1_story_metrics.csv
    stage1_overall.csv
    stage1_by_category.csv
    stage2_story_metrics.csv
    stage2_overall.csv
    stage2_by_category.csv
```

## Metrics

- Stage 1 belief extraction: per-story precision, recall, and F1 from semantic-judge `MatchCount` outputs, plus macro averages overall and by category
- Stage 2 belief labeling: per-dimension accuracy, overall macro labeling accuracy, and macro averages overall and by category

## License and Attribution

The OmniToM release package is distributed under the license in `LICENSE`.

Story text is derived from ToMBench:

- Repository: `zhchen18/ToMBench`
- URL: `https://github.com/zhchen18/ToMBench`
- Description: official repository for "ToMBench: Benchmarking Theory of Mind in Large Language Models" (ACL 2024)
- Upstream license: MIT

OmniToM reuses ToMBench-derived story text and adds new benchmark annotations, prompt builders, replication code, and release documentation. The upstream ToMBench repository states the following license and copyright notice:

```text
MIT License

Copyright (c) 2024 Zhuang Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
