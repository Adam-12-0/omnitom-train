---
pretty_name: OmniToM
language:
- en
license: mit
task_categories:
- text-generation
- text-classification
task_ids:
- text2text-generation
- multi-class-classification
size_categories:
- n<1K
tags:
- theory-of-mind
- benchmark
- social-reasoning
- belief-modeling
- llm-evaluation
- cognitive-reasoning
- mental-state-representation
---

# Dataset Card for OmniToM

## Dataset Details

### Dataset Description

OmniToM is a benchmark for evaluating Theory of Mind in language models through explicit belief-structure modeling. Instead of scoring only endpoint answers to social-reasoning questions, OmniToM exposes the intermediate belief structure that a model must build in order to reason coherently about what different actors know, believe, infer, intend, or misunderstand.

Each example is a short English story paired with:

- a set of actor-centered belief propositions
- a reserved `world` actor for narrator/world facts
- a seven-dimensional schema label vector for every belief

The benchmark supports two linked tasks:

1. **Belief Extraction**
   - Given a story, extract the relevant belief structure as `(Actor, Belief, Order)` tuples.
2. **Belief Labeling**
   - Given the story and belief tuples, label each belief along seven closed-set schema dimensions.

- **Language(s):** English
- **License:** MIT

## Uses

### Direct Use

OmniToM is designed for benchmark evaluation and diagnostic analysis. Suitable uses include:

- zero-shot belief extraction
- zero-shot belief labeling
- semantic-judge evaluation of extracted belief tables
- analysis of multi-actor and higher-order Theory-of-Mind reasoning
- process-sensitive evaluation beyond endpoint question answering

### Out-of-Scope Use

OmniToM should not be treated as:

- a direct measure of real-world social intelligence
- a measure of embodied, interactive, or multimodal social reasoning
- a safety certification benchmark for deployed systems
- a clinical, educational, or psychological assessment tool
- a complete coverage benchmark for all possible Theory-of-Mind phenomena

## Dataset Structure

### Data Instances

Each line in the release file is one JSON object:

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

### Data Fields

- `story_id`
  - Unique integer identifier for the story.
- `story_category`
  - One of seven retained benchmark categories.
- `story`
  - Raw story text used for both extraction and labeling tasks.
- `beliefs`
  - List of annotated belief propositions.
- `beliefs[].actor`
  - Belief holder. The reserved actor `world` denotes narrator/world facts.
- `beliefs[].belief`
  - Minimal propositional belief statement.
- `beliefs[].labels.order`
  - Recursive depth in `{0,1,2,3}`.
- `beliefs[].labels.truth_status`
  - `True`, `False`, or `Unknown`.
- `beliefs[].labels.knowledge_access`
  - `Private`, `Shared`, or `Public`.
- `beliefs[].labels.representation`
  - `Explicit` or `Implicit`.
- `beliefs[].labels.content_type`
  - One of: `Location`, `Contents/Physical State`, `Identity/Relation`, `Epistemic`, `Desire/Intention`, `Emotion`, `Trait/Value`, `Action/Event`.
- `beliefs[].labels.mental_source`
  - One of: `Narration`, `Perception`, `Memory`, `Testimony`, `Inference`, `Imagination`, `Unknown`.
- `beliefs[].labels.context`
  - `Neutral`, `Temporal`, `Deceptive`, or `Counterfactual`.

### Data Splits

This release contains one benchmark split:

- `train` / benchmark split: `895` stories

The split is named `train` in the Hugging Face dataset viewer for compatibility with the default dataset loading interface. It should be interpreted as the benchmark split, not as a recommended training set.

## Dataset Creation

### Curation Rationale

OmniToM was created to address a limitation in prior Theory-of-Mind benchmarks for language models: most evaluate endpoint question answering rather than whether a model constructs a coherent belief representation while reading the story.

OmniToM instead evaluates explicit belief-structure modeling. The benchmark is grounded in short stories from ToMBench and organizes reasoning around an ATOMS-grounded belief-level schema for fine-grained analysis of mental-state representations.

### Source Data

The benchmark sources its stories from ToMBench. From the original source corpus, OmniToM retains seven story categories whose stories provide sufficiently self-contained mental-state evidence for belief extraction from text alone:

- Ambiguous Story Task
- False Belief Task
- Faux-pas Recognition Test
- Hinting Task Test
- Persuasion Story Task
- Scalar Implicature Test
- Strange Story Task

### Annotation Process

The benchmark was built with a human-calibrated, LLM-assisted annotation pipeline:

- `1,383` source stories in the original corpus
- `916` stories retained after source filtering
- `895` stories released in the final benchmark
- `22,343` labeled belief propositions in the released benchmark
- `156,401` total schema labels in the released benchmark

The accompanying paper reports:

- Stage 1 expert-overlap validation after reconciliation: `83.72%`
- Stage 2 strict all-annotator exact-match label reliability: `92.23%`
- Human-human agreement on the semantic-alignment validation set: `88.86%`
- Human-judge agreement for the selected semantic judge: `72.03%`

### Who are the source data producers?

The story texts are sourced from ToMBench. The belief structures and schema labels are benchmark annotations produced through the OmniToM human-calibrated annotation pipeline described in the accompanying paper.

### Personal and Sensitive Information

The release consists of short benchmark stories and belief annotations. It is not designed to contain personal user data, private communications, direct identifiers, medical records, or real-world sensitive records. As with many story-based datasets, names, family roles, occupations, emotions, intentions, and social situations may appear in the source material, but the benchmark is intended for research evaluation rather than identification or profiling.

## Bias, Risks, and Limitations

OmniToM is story-based, text-only, English-language, and sourced from a specific upstream benchmark distribution. It therefore reflects the representational biases, writing conventions, scenario distribution, and coverage limitations of its source stories. Names, roles, social situations, and pragmatic conventions may also be unevenly distributed across categories.

Additional known limitations:

- the benchmark evaluates story-based Theory of Mind rather than interactive, embodied, or multimodal social reasoning
- the retained stories are short and self-contained, and do not stress-test long-horizon information tracking, dense temporal structure, or deeply nested mental states beyond the order-3 schema
- the released labels come from a human-calibrated LLM-assisted pipeline rather than fully manual annotation of every story
- the seven-dimensional schema is human-labeled and may retain interpretive subjectivity in socially ambiguous cases
- Stage 1 extraction evaluation in the paper relies on a human-calibrated semantic judge rather than full human adjudication across the full release
- the selected semantic judge reached `72.03%` agreement with human semantic-alignment decisions, so extraction `F1` should be interpreted as an approximate aggregate metric rather than an exact belief-level alignment score

### Recommendations

Users should interpret OmniToM as a diagnostic benchmark for explicit belief-structure modeling. Benchmark scores should not be treated as evidence of robust real-world interpersonal reasoning, embodied social competence, clinical or educational validity, or deployment safety.

## Citation

```bibtex
@misc{omnitom2026,
  title={OmniToM: Benchmarking Theory of Mind in LLMs via Explicit Belief Modeling},
  author={Anonymous Authors},
  year={2026},
  note={Anonymous review release}
}
```

## More Information

This anonymous release is intended to support peer review and benchmark inspection. Public release metadata and authorship details may be updated in a later non-anonymous version.
