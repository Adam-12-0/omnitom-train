#!/usr/bin/env python3
"""
Replication runner for the OmniToM benchmark release.

This script uses the released `benchmark_story_belief_labels.jsonl` file
together with the prompt builders in this folder to run:

1. Stage 1 belief extraction
2. Stage 2 belief labeling
3. LLM-as-a-judge evaluation for Stage 1
4. Metric aggregation for both stages

The default `mock` backend makes it easy to smoke-test the full pipeline
without model access. A lightweight Hugging Face backend is also included for
open-weight replication, and the API backend supports the GPT-5 judge setup
used for the OmniToM Stage 1 results.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_prompting import BENCHMARK_PATH
from prompt_evaluate import build_evaluation_messages
from prompts_extract import build_extract_messages
from prompts_label import build_label_messages


EXTRACTION_COLUMNS = ["actor", "belief", "order"]
LABEL_COLUMNS = [
    "actor",
    "belief",
    "order",
    "truth-status",
    "knowledge-access",
    "representation",
    "content_type",
    "mental-source",
    "context",
]
JUDGE_COLUMNS = ["Actor", "Belief", "MatchCount"]

LABEL_DIMENSIONS = [
    "order",
    "truth-status",
    "knowledge-access",
    "representation",
    "content_type",
    "mental-source",
    "context",
]

JSONL_LABEL_KEY_TO_OUTPUT = {
    "order": "order",
    "truth_status": "truth-status",
    "knowledge_access": "knowledge-access",
    "representation": "representation",
    "content_type": "content_type",
    "mental_source": "mental-source",
    "context": "context",
}

LABEL_COLUMN_ALIASES = {
    "actor": "actor",
    "belief": "belief",
    "order": "order",
    "truthstatus": "truth-status",
    "truth-status": "truth-status",
    "knowledgeaccess": "knowledge-access",
    "knowledge-access": "knowledge-access",
    "representation": "representation",
    "contenttype": "content_type",
    "content_type": "content_type",
    "contents": "content_type",
    "mentalsource": "mental-source",
    "mental-source": "mental-source",
    "context": "context",
}

LABEL_VALUE_ALIASES = {
    "truth-status": {
        "true": "True",
        "false": "False",
        "unknown": "Unknown",
    },
    "knowledge-access": {
        "private": "Private",
        "shared": "Shared",
        "public": "Public",
        "unknown": "Unknown",
    },
    "representation": {
        "explicit": "Explicit",
        "implicit": "Implicit",
    },
    "content_type": {
        "location": "Location",
        "contents/physicalstate": "Contents/Physical State",
        "contentsphysicalstate": "Contents/Physical State",
        "contents / physical state": "Contents/Physical State",
        "physicalstatecontents": "Contents/Physical State",
        "physical state": "Contents/Physical State",
        "identity/relation": "Identity/Relation",
        "identityrelation": "Identity/Relation",
        "relation": "Identity/Relation",
        "identity": "Identity/Relation",
        "epistemic": "Epistemic",
        "desire/intention": "Desire/Intention",
        "desireintention": "Desire/Intention",
        "emotion": "Emotion",
        "trait/value": "Trait/Value",
        "traitvalue": "Trait/Value",
        "action/event": "Action/Event",
        "actionevent": "Action/Event",
        "action": "Action/Event",
        "event": "Action/Event",
    },
    "mental-source": {
        "narration": "Narration",
        "perception": "Perception",
        "memory": "Memory",
        "testimony": "Testimony",
        "inference": "Inference",
        "imagination": "Imagination",
        "unknown": "Unknown",
    },
    "context": {
        "deceptive": "Deceptive",
        "temporal": "Temporal",
        "counterfactual": "Counterfactual",
        "neutral": "Neutral",
    },
}


def normalize_space(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(value: object) -> str:
    text = normalize_space(value).lower()
    text = text.strip("\"'")
    return text


def parse_story_ids(raw: Optional[str]) -> Optional[set[int]]:
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            for story_id in range(start, end + 1):
                out.add(story_id)
        else:
            out.add(int(token))
    return out or None


def load_records(dataset_path: Path) -> List[dict]:
    records: List[dict] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def select_records(records: Sequence[dict], story_ids: Optional[set[int]], max_stories: int) -> List[dict]:
    selected = []
    for record in records:
        story_id = int(record["story_id"])
        if story_ids and story_id not in story_ids:
            continue
        selected.append(record)
        if max_stories and len(selected) >= max_stories:
            break
    return selected


def ground_truth_belief_rows(record: dict) -> List[dict]:
    rows = []
    for belief in record.get("beliefs", []):
        rows.append(
            {
                "actor": normalize_space(belief.get("actor", "")),
                "belief": normalize_space(belief.get("belief", "")),
                "order": normalize_space(belief.get("labels", {}).get("order", "")),
            }
        )
    return rows


def ground_truth_label_rows(record: dict) -> List[dict]:
    rows = []
    for belief in record.get("beliefs", []):
        labels = belief.get("labels", {})
        row = {
            "actor": normalize_space(belief.get("actor", "")),
            "belief": normalize_space(belief.get("belief", "")),
        }
        for json_key, output_key in JSONL_LABEL_KEY_TO_OUTPUT.items():
            row[output_key] = normalize_space(labels.get(json_key, ""))
        rows.append(canonicalize_label_row(row))
    return rows


def rows_to_pipe_table(rows: Sequence[dict], headers: Sequence[str]) -> str:
    pretty_headers = {
        "actor": "Actor",
        "belief": "Belief",
        "order": "Order",
        "truth-status": "Truth-Status",
        "knowledge-access": "Knowledge-Access",
        "representation": "Representation",
        "content_type": "Content Type",
        "mental-source": "Mental-Source",
        "context": "Context",
    }
    header_line = " | ".join(pretty_headers.get(h, h) for h in headers)
    lines = [header_line]
    for row in rows:
        lines.append(" | ".join(normalize_space(row.get(h, "")) for h in headers))
    return "\n".join(lines)


def rows_to_csv_text(rows: Sequence[dict], headers: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(headers))
    writer.writeheader()
    for row in rows:
        writer.writerow({h: normalize_space(row.get(h, "")) for h in headers})
    return buffer.getvalue().strip()


def rows_to_belief_csv_text(rows: Sequence[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["Actor", "Belief"], lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "Actor": normalize_space(row.get("Actor", row.get("actor", ""))),
                "Belief": normalize_space(row.get("Belief", row.get("belief", ""))),
            }
        )
    return buffer.getvalue()


def parse_pipe_table(text: str) -> List[List[str]]:
    raw_lines = (text or "").splitlines()
    if not raw_lines:
        return []

    def split_row(line: str) -> List[str]:
        cleaned = line.strip()
        if cleaned.startswith("|"):
            cleaned = cleaned[1:]
        if cleaned.endswith("|"):
            cleaned = cleaned[:-1]
        return [cell.strip() for cell in cleaned.split("|")]

    def is_separator(cells: Sequence[str]) -> bool:
        if not cells:
            return False
        return all((not cell) or bool(re.fullmatch(r":?-{3,}:?", cell)) for cell in cells)

    rows: List[List[str]] = []
    for line in raw_lines:
        if "|" not in line:
            continue
        cells = split_row(line)
        if is_separator(cells):
            continue
        rows.append(cells)
    return rows


def canonicalize_label_value(column: str, value: object) -> str:
    text = normalize_space(value)
    if not text:
        return ""

    if column == "order":
        match = re.search(r"(\d+)", text)
        return match.group(1) if match else text

    lookup = LABEL_VALUE_ALIASES.get(column, {})
    key = re.sub(r"[^a-z0-9/ ]+", "", text.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    if key in lookup:
        return lookup[key]

    squashed = key.replace(" ", "")
    if squashed in lookup:
        return lookup[squashed]

    return text


def canonicalize_label_row(row: dict) -> dict:
    out = {
        "actor": normalize_space(row.get("actor", "")),
        "belief": normalize_space(row.get("belief", "")),
    }
    for column in LABEL_DIMENSIONS:
        out[column] = canonicalize_label_value(column, row.get(column, ""))
    return out


def parse_extraction_rows(text: str) -> List[dict]:
    rows = parse_pipe_table(text)
    if not rows:
        return []
    parsed = []
    for index, cells in enumerate(rows):
        if index == 0 and len(cells) >= 3 and normalize_key(cells[0]) == "actor":
            continue
        if len(cells) < 3:
            continue
        parsed.append(
            {
                "actor": normalize_space(cells[0]),
                "belief": normalize_space(cells[1]).rstrip("."),
                "order": normalize_space(cells[2]),
            }
        )
    return parsed


def parse_label_rows(text: str) -> List[dict]:
    rows = parse_pipe_table(text)
    if not rows:
        return []
    parsed = []
    for index, cells in enumerate(rows):
        if index == 0 and len(cells) >= 2 and normalize_key(cells[0]) == "actor":
            continue
        if len(cells) < 9:
            continue
        parsed.append(
            canonicalize_label_row(
                {
                    "actor": cells[0],
                    "belief": normalize_space(cells[1]).rstrip("."),
                    "order": cells[2],
                    "truth-status": cells[3],
                    "knowledge-access": cells[4],
                    "representation": cells[5],
                    "content_type": cells[6],
                    "mental-source": cells[7],
                    "context": cells[8],
                }
            )
        )
    return parsed


def parse_csv_rows(text: str) -> List[dict]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    reader = csv.DictReader(io.StringIO(cleaned))
    rows = []
    for row in reader:
        rows.append({key: normalize_space(value) for key, value in row.items()})
    return rows


def is_heading(line: str, kind: str) -> bool:
    text = normalize_space(line).lower()
    text = text.strip("*").strip(":")
    if kind == "pred":
        return bool(re.match(r"^(prediction|predictions)\b", text))
    return bool(re.match(r"^(ground\s*truth|groundtruth|gt)\b", text))


def find_section(text: str, kind: str) -> str:
    lines = (text or "").splitlines()
    start = None
    for index, line in enumerate(lines):
        if is_heading(line, kind):
            start = index + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        if is_heading(lines[index], "pred") or is_heading(lines[index], "gt"):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def extract_csv_like_table(section_text: str) -> str:
    if not section_text:
        return ""
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    start = None
    for index, line in enumerate(lines):
        lowered = normalize_key(line.replace(",", " "))
        if "actor" in lowered and "matchcount" in lowered:
            start = index
            break
    if start is None:
        return ""
    out_lines = []
    for line in lines[start:]:
        if "," not in line and out_lines:
            break
        if "," in line:
            out_lines.append(line)
    return "\n".join(out_lines)


def parse_judge_output(raw_text: str) -> tuple[List[dict], List[dict]]:
    pred_section = find_section(raw_text, "pred")
    gt_section = find_section(raw_text, "gt")

    pred_csv = extract_csv_like_table(pred_section)
    gt_csv = extract_csv_like_table(gt_section)
    return parse_csv_rows(pred_csv), parse_csv_rows(gt_csv)


def coerce_matchcount(value: object) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def align_matchcounts_to_input(input_rows: Sequence[dict], parsed_rows: Sequence[dict]) -> List[dict]:
    output_rows = []
    pool = []
    for index, row in enumerate(parsed_rows):
        actor = row.get("Actor", row.get("actor", ""))
        belief = row.get("Belief", row.get("belief", ""))
        pool.append(
            (
                normalize_key(actor),
                normalize_key(belief),
                coerce_matchcount(row.get("MatchCount", row.get("matchcount", 0))),
                index,
            )
        )

    used: set[int] = set()
    for row in input_rows:
        actor = normalize_key(row.get("actor", ""))
        belief = normalize_key(row.get("belief", ""))
        matchcount = 0
        candidates = [item for item in pool if item[0] == actor and item[1] == belief and item[3] not in used]
        if candidates:
            candidates.sort(key=lambda item: (-item[2], item[3]))
            chosen = candidates[0]
            matchcount = chosen[2]
            used.add(chosen[3])
        output_rows.append(
            {
                "Actor": normalize_space(row.get("actor", "")),
                "Belief": normalize_space(row.get("belief", "")),
                "MatchCount": str(matchcount),
            }
        )
    return output_rows


def write_rows_csv(path: Path, headers: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({header: normalize_space(row.get(header, "")) for header in headers})


def read_rows_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: normalize_space(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_raw_bundle(path: Path, system_prompt: str, user_prompt: str, output_text: str) -> None:
    bundle = (
        "SYSTEM PROMPT:\n"
        + system_prompt.strip()
        + "\n\nUSER PROMPT:\n"
        + user_prompt.strip()
        + "\n\nOUTPUT:\n"
        + output_text.strip()
        + "\n"
    )
    write_text(path, bundle)


def exact_matchcount_rows(left_rows: Sequence[dict], right_rows: Sequence[dict]) -> List[dict]:
    right_counter = Counter((normalize_key(row.get("actor", "")), normalize_key(row.get("belief", ""))) for row in right_rows)
    output = []
    for row in left_rows:
        key = (normalize_key(row.get("actor", "")), normalize_key(row.get("belief", "")))
        output.append(
            {
                "Actor": normalize_space(row.get("actor", "")),
                "Belief": normalize_space(row.get("belief", "")),
                "MatchCount": str(right_counter.get(key, 0)),
            }
        )
    return output


class GenerationBackend:
    """
    Implement `generate_text()` to connect your preferred model runtime.

    The included backends are:
    - `mock`: deterministic smoke testing
    - `hf`: lightweight Hugging Face text generation
    - `api`: OpenAI-compatible Responses API and Google GenAI calls
    """

    def generate_text(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        record: dict,
        extraction_rows: Optional[Sequence[dict]] = None,
    ) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MockBackend(GenerationBackend):
    def generate_text(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        record: dict,
        extraction_rows: Optional[Sequence[dict]] = None,
    ) -> str:
        if stage == "extract":
            return rows_to_pipe_table(ground_truth_belief_rows(record), EXTRACTION_COLUMNS)

        if stage == "label":
            return rows_to_pipe_table(ground_truth_label_rows(record), LABEL_COLUMNS)

        if stage == "judge":
            pred_rows = list(extraction_rows or [])
            gt_rows = ground_truth_belief_rows(record)
            pred_eval = exact_matchcount_rows(pred_rows, gt_rows)
            gt_eval = exact_matchcount_rows(gt_rows, pred_rows)
            return (
                "Prediction Table:\n"
                + rows_to_csv_text(pred_eval, JUDGE_COLUMNS)
                + "\n\nGround Truth Table:\n"
                + rows_to_csv_text(gt_eval, JUDGE_COLUMNS)
            )

        raise ValueError(f"Unsupported mock stage: {stage}")


class HuggingFaceBackend(GenerationBackend):
    def __init__(
        self,
        *,
        default_model: Optional[str],
        extract_model: Optional[str],
        label_model: Optional[str],
        judge_model: Optional[str],
        device: str,
        max_new_tokens: int,
        load_in_4bit: bool,
        bnb_4bit_compute_dtype: str,
    ) -> None:
        self.default_model = (default_model or "").strip() or None
        self.stage_models = {
            "extract": (extract_model or "").strip() or self.default_model,
            "label": (label_model or "").strip() or self.default_model,
            "judge": (judge_model or "").strip() or self.default_model,
        }
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.current_model_id: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.torch = None

    def _resolve_model(self, stage: str) -> str:
        model_id = self.stage_models.get(stage)
        if not model_id:
            raise ValueError(
                f"No model configured for stage '{stage}'. Use --model or the stage-specific "
                f"flag (for example --{stage}-model)."
            )
        return model_id

    def _load_model(self, model_id: str) -> None:
        if self.current_model_id == model_id and self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "The Hugging Face backend requires `torch` and `transformers` in the active environment."
            ) from exc

        self.close()
        self.torch = torch

        BitsAndBytesConfig = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:
                raise RuntimeError(
                    "4-bit loading requested via --load-in-4bit, but BitsAndBytes support is unavailable. "
                    "Install `bitsandbytes` and ensure your `transformers` build supports quantized loading."
                ) from exc

            if self.device == "cpu":
                raise RuntimeError("--load-in-4bit is not supported with --device cpu.")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if self.device == "auto":
            device_map = "auto"
            torch_dtype = "auto"
        elif self.device == "cpu":
            device_map = None
            torch_dtype = torch.float32
        else:
            device_map = {"": self.device}
            torch_dtype = torch.float16

        quantization_config = None
        if self.load_in_4bit:
            dtype_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            compute_dtype = dtype_map[self.bnb_4bit_compute_dtype]
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            torch_dtype = compute_dtype

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
        model.eval()

        self.current_model_id = model_id
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        record: dict,
        extraction_rows: Optional[Sequence[dict]] = None,
    ) -> str:
        model_id = self._resolve_model(stage)
        self._load_model(model_id)
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.torch is not None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if getattr(self.tokenizer, "chat_template", None):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            toks = {"input_ids": inputs}
        else:
            toks = self.tokenizer(
                system_prompt.strip() + "\n\n" + user_prompt.strip(),
                return_tensors="pt",
                padding=False,
                truncation=False,
            )

        if "attention_mask" not in toks:
            toks["attention_mask"] = self.torch.ones_like(toks["input_ids"])

        target_device = self.model.get_input_embeddings().weight.device
        toks = {key: value.to(target_device) for key, value in toks.items()}

        with self.torch.inference_mode():
            outputs = self.model.generate(
                input_ids=toks["input_ids"],
                attention_mask=toks.get("attention_mask"),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = toks["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def close(self) -> None:
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
        if self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception:
                pass
        if self.torch is not None and getattr(self.torch, "cuda", None) and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None
        self.current_model_id = None


class APIBackend(GenerationBackend):
    def __init__(
        self,
        *,
        api_provider: str,
        default_model: Optional[str],
        extract_model: Optional[str],
        label_model: Optional[str],
        judge_model: Optional[str],
        max_new_tokens: int,
        temperature: float,
        api_key_env: Optional[str],
        api_base_url_env: Optional[str],
    ) -> None:
        self.api_provider = (api_provider or "").strip().lower()
        self.default_model = (default_model or "").strip() or None
        self.stage_models = {
            "extract": (extract_model or "").strip() or self.default_model,
            "label": (label_model or "").strip() or self.default_model,
            "judge": (judge_model or "").strip() or self.default_model,
        }
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.api_key_env = (api_key_env or "").strip() or None
        self.api_base_url_env = (api_base_url_env or "").strip() or None

        if self.api_provider not in {"openai", "google_genai"}:
            raise ValueError("--api-provider must be one of: openai, google_genai")

    def _resolve_model(self, stage: str) -> str:
        model_id = self.stage_models.get(stage)
        if not model_id:
            raise ValueError(
                f"No model configured for stage '{stage}'. Use --model or the stage-specific "
                f"flag (for example --{stage}-model)."
            )
        return model_id

    def _api_key(self) -> str:
        candidates: List[str]
        if self.api_key_env:
            candidates = [self.api_key_env]
        elif self.api_provider == "openai":
            candidates = ["OPENAI_API_KEY"]
        else:
            candidates = ["GOOGLE_API_KEY", "GEMINI_API_KEY"]

        for env_name in candidates:
            value = os.getenv(env_name, "").strip()
            if value:
                return value
        raise RuntimeError(f"Missing API key. Set one of: {', '.join(candidates)}")

    def _api_base_url(self) -> Optional[str]:
        candidates: List[str]
        if self.api_base_url_env:
            candidates = [self.api_base_url_env]
        elif self.api_provider == "openai":
            candidates = ["OPENAI_BASE_URL"]
        else:
            candidates = []

        for env_name in candidates:
            value = os.getenv(env_name, "").strip().rstrip("/")
            if value:
                return value
        return None

    @staticmethod
    def _openai_output_text(response: object) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = getattr(response, "output", None)
        parts: List[str] = []
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    content = item.get("content")
                else:
                    content = getattr(item, "content", None)
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            text = block.get("text")
                        else:
                            text = getattr(block, "text", None)
                        if isinstance(text, str):
                            parts.append(text)
        return "\n".join(part for part in parts if part).strip()

    def _generate_openai(self, model_id: str, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("The API backend with --api-provider openai requires `openai`.") from exc

        base_url = self._api_base_url()
        client_kwargs = {"api_key": self._api_key()}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        request = {
            "model": model_id,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
        }
        if "gpt-5" in model_id.lower():
            request["reasoning"] = {"effort": "low"}

        response = client.responses.create(**request)
        text = self._openai_output_text(response)
        if not text:
            raise RuntimeError("Empty model response. Increase --max-new-tokens for this run.")
        return text

    def _generate_google_genai(self, model_id: str, system_prompt: str, user_prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError("The API backend with --api-provider google_genai requires `google-genai`.") from exc

        client = genai.Client(api_key=self._api_key())
        thinking_config = None
        try:
            thinking_config = types.ThinkingConfig(include_thoughts=False, thinking_budget=0)
        except Exception:
            thinking_config = None

        config = types.GenerateContentConfig(
            system_instruction=[system_prompt.strip()] if system_prompt.strip() else None,
            temperature=float(self.temperature),
            max_output_tokens=int(self.max_new_tokens),
            thinking_config=thinking_config,
        )
        response = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=config,
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError("Empty model response. Increase --max-new-tokens for this run.")
        return text

    def generate_text(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        record: dict,
        extraction_rows: Optional[Sequence[dict]] = None,
    ) -> str:
        model_id = self._resolve_model(stage)
        if self.api_provider == "openai":
            return self._generate_openai(model_id, system_prompt, user_prompt)
        return self._generate_google_genai(model_id, system_prompt, user_prompt)


def build_backend(args: argparse.Namespace) -> GenerationBackend:
    if args.backend == "mock":
        return MockBackend()
    if args.backend == "hf":
        return HuggingFaceBackend(
            default_model=args.model,
            extract_model=args.extract_model,
            label_model=args.label_model,
            judge_model=args.judge_model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        )
    if args.backend == "api":
        return APIBackend(
            api_provider=args.api_provider,
            default_model=args.model,
            extract_model=args.extract_model,
            label_model=args.label_model,
            judge_model=args.judge_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            api_key_env=args.api_key_env,
            api_base_url_env=args.api_base_url_env,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def stage_paths(output_dir: Path, story_id: int) -> dict:
    stem = f"story_{story_id:03d}"
    return {
        "extract_raw": output_dir / "extraction" / "raw" / f"{stem}.txt",
        "extract_csv": output_dir / "extraction" / "csv" / f"{stem}_extract.csv",
        "label_raw": output_dir / "labeling" / "raw" / f"{stem}.txt",
        "label_csv": output_dir / "labeling" / "csv" / f"{stem}_label.csv",
        "judge_raw": output_dir / "judge" / "raw" / f"{stem}.txt",
        "judge_pred_csv": output_dir / "judge" / "csv" / f"{stem}_predictions.csv",
        "judge_gt_csv": output_dir / "judge" / "csv" / f"{stem}_groundtruth.csv",
    }


def maybe_load_rows(path: Path) -> Optional[List[dict]]:
    if path.exists():
        return read_rows_csv(path)
    return None


def all_stage_files_exist(records: Sequence[dict], output_dir: Path, keys: Sequence[str]) -> bool:
    for record in records:
        paths = stage_paths(output_dir, int(record["story_id"]))
        for key in keys:
            if not paths[key].exists():
                return False
    return True


def run_extraction(record: dict, backend: GenerationBackend, output_dir: Path, args: argparse.Namespace) -> List[dict]:
    story_id = int(record["story_id"])
    paths = stage_paths(output_dir, story_id)
    if paths["extract_csv"].exists() and not args.overwrite:
        return read_rows_csv(paths["extract_csv"])

    system_prompt, user_prompt = build_extract_messages(story_id, args.dataset_path)
    raw_text = backend.generate_text(
        stage="extract",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        record=record,
    )
    rows = parse_extraction_rows(raw_text)
    write_raw_bundle(paths["extract_raw"], system_prompt, user_prompt, raw_text)
    write_rows_csv(paths["extract_csv"], EXTRACTION_COLUMNS, rows)
    return rows


def run_labeling(record: dict, backend: GenerationBackend, output_dir: Path, args: argparse.Namespace) -> List[dict]:
    story_id = int(record["story_id"])
    paths = stage_paths(output_dir, story_id)
    if paths["label_csv"].exists() and not args.overwrite:
        return [canonicalize_label_row(row) for row in read_rows_csv(paths["label_csv"])]

    system_prompt, user_prompt = build_label_messages(story_id, args.dataset_path)
    raw_text = backend.generate_text(
        stage="label",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        record=record,
    )
    rows = parse_label_rows(raw_text)
    write_raw_bundle(paths["label_raw"], system_prompt, user_prompt, raw_text)
    write_rows_csv(paths["label_csv"], LABEL_COLUMNS, rows)
    return rows


def run_judge(
    record: dict,
    extraction_rows: Sequence[dict],
    backend: GenerationBackend,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[List[dict], List[dict]]:
    story_id = int(record["story_id"])
    paths = stage_paths(output_dir, story_id)
    if paths["judge_pred_csv"].exists() and paths["judge_gt_csv"].exists() and not args.overwrite:
        return read_rows_csv(paths["judge_pred_csv"]), read_rows_csv(paths["judge_gt_csv"])

    predictions_csv = rows_to_belief_csv_text(extraction_rows)
    system_prompt, user_prompt = build_evaluation_messages(
        story_id,
        predictions_csv,
        dataset_path=args.dataset_path,
        include_story=(not args.no_story_context),
        fewshots=args.judge_fewshots,
    )
    raw_text = backend.generate_text(
        stage="judge",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        record=record,
        extraction_rows=extraction_rows,
    )
    parsed_pred, parsed_gt = parse_judge_output(raw_text)
    aligned_pred = align_matchcounts_to_input(extraction_rows, parsed_pred)
    aligned_gt = align_matchcounts_to_input(ground_truth_belief_rows(record), parsed_gt)
    write_raw_bundle(paths["judge_raw"], system_prompt, user_prompt, raw_text)
    write_rows_csv(paths["judge_pred_csv"], JUDGE_COLUMNS, aligned_pred)
    write_rows_csv(paths["judge_gt_csv"], JUDGE_COLUMNS, aligned_gt)
    return aligned_pred, aligned_gt


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(values))


def pair_indexed_rows(rows: Sequence[dict]) -> Dict[tuple[str, str, int], dict]:
    seen: Counter[tuple[str, str]] = Counter()
    indexed = {}
    for row in rows:
        key_base = (normalize_key(row.get("actor", "")), normalize_key(row.get("belief", "")))
        pair_idx = seen[key_base]
        seen[key_base] += 1
        indexed[(key_base[0], key_base[1], pair_idx)] = row
    return indexed


def compute_stage1_metrics(records: Sequence[dict], output_dir: Path) -> None:
    story_rows = []
    for record in records:
        story_id = int(record["story_id"])
        paths = stage_paths(output_dir, story_id)
        pred_rows = read_rows_csv(paths["judge_pred_csv"])
        gt_rows = read_rows_csv(paths["judge_gt_csv"])

        pred_total = len(pred_rows)
        gt_total = len(gt_rows)
        pred_matched = sum(coerce_matchcount(row.get("MatchCount", 0)) > 0 for row in pred_rows)
        gt_matched = sum(coerce_matchcount(row.get("MatchCount", 0)) > 0 for row in gt_rows)
        precision = (pred_matched / pred_total) if pred_total else 0.0
        recall = (gt_matched / gt_total) if gt_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        story_rows.append(
            {
                "story_id": str(story_id),
                "story_category": record["story_category"],
                "pred_total": str(pred_total),
                "gt_total": str(gt_total),
                "pred_matched": str(pred_matched),
                "gt_matched": str(gt_matched),
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "f1": f"{f1:.6f}",
            }
        )

    metrics_dir = output_dir / "metrics"
    write_rows_csv(
        metrics_dir / "stage1_story_metrics.csv",
        ["story_id", "story_category", "pred_total", "gt_total", "pred_matched", "gt_matched", "precision", "recall", "f1"],
        story_rows,
    )

    overall = {
        "stories": str(len(story_rows)),
        "precision": f"{safe_mean(float(row['precision']) for row in story_rows):.6f}",
        "recall": f"{safe_mean(float(row['recall']) for row in story_rows):.6f}",
        "f1": f"{safe_mean(float(row['f1']) for row in story_rows):.6f}",
    }
    write_rows_csv(metrics_dir / "stage1_overall.csv", ["stories", "precision", "recall", "f1"], [overall])

    by_category: Dict[str, List[dict]] = defaultdict(list)
    for row in story_rows:
        by_category[row["story_category"]].append(row)

    category_rows = []
    for category, rows in sorted(by_category.items()):
        category_rows.append(
            {
                "story_category": category,
                "stories": str(len(rows)),
                "precision": f"{safe_mean(float(row['precision']) for row in rows):.6f}",
                "recall": f"{safe_mean(float(row['recall']) for row in rows):.6f}",
                "f1": f"{safe_mean(float(row['f1']) for row in rows):.6f}",
            }
        )
    write_rows_csv(
        metrics_dir / "stage1_by_category.csv",
        ["story_category", "stories", "precision", "recall", "f1"],
        category_rows,
    )


def compute_stage2_metrics(records: Sequence[dict], output_dir: Path) -> None:
    story_rows = []
    for record in records:
        story_id = int(record["story_id"])
        predicted_rows = [canonicalize_label_row(row) for row in read_rows_csv(stage_paths(output_dir, story_id)["label_csv"])]
        gt_rows = ground_truth_label_rows(record)

        pred_index = pair_indexed_rows(predicted_rows)
        gt_index = pair_indexed_rows(gt_rows)

        dim_scores = {}
        for dimension in LABEL_DIMENSIONS:
            correct = 0
            total = len(gt_index)
            for key, gt_row in gt_index.items():
                pred_row = pred_index.get(key)
                if pred_row and pred_row.get(dimension, "") and pred_row.get(dimension, "") == gt_row.get(dimension, ""):
                    correct += 1
            dim_scores[dimension] = (correct / total) if total else 0.0

        overall_accuracy = safe_mean(dim_scores.values())
        story_row = {
            "story_id": str(story_id),
            "story_category": record["story_category"],
            "pairs": str(len(gt_rows)),
            "matched_pairs": str(sum(1 for key in gt_index if key in pred_index)),
            "overall_accuracy": f"{overall_accuracy:.6f}",
        }
        for dimension in LABEL_DIMENSIONS:
            story_row[f"acc_{dimension}"] = f"{dim_scores[dimension]:.6f}"
        story_rows.append(story_row)

    metrics_dir = output_dir / "metrics"
    story_headers = ["story_id", "story_category", "pairs", "matched_pairs", "overall_accuracy"] + [
        f"acc_{dimension}" for dimension in LABEL_DIMENSIONS
    ]
    write_rows_csv(metrics_dir / "stage2_story_metrics.csv", story_headers, story_rows)

    overall_row = {
        "stories": str(len(story_rows)),
        "overall_accuracy": f"{safe_mean(float(row['overall_accuracy']) for row in story_rows):.6f}",
    }
    for dimension in LABEL_DIMENSIONS:
        overall_row[f"acc_{dimension}"] = f"{safe_mean(float(row[f'acc_{dimension}']) for row in story_rows):.6f}"
    write_rows_csv(
        metrics_dir / "stage2_overall.csv",
        ["stories", "overall_accuracy"] + [f"acc_{dimension}" for dimension in LABEL_DIMENSIONS],
        [overall_row],
    )

    by_category: Dict[str, List[dict]] = defaultdict(list)
    for row in story_rows:
        by_category[row["story_category"]].append(row)

    category_rows = []
    for category, rows in sorted(by_category.items()):
        category_row = {
            "story_category": category,
            "stories": str(len(rows)),
            "overall_accuracy": f"{safe_mean(float(row['overall_accuracy']) for row in rows):.6f}",
        }
        for dimension in LABEL_DIMENSIONS:
            category_row[f"acc_{dimension}"] = f"{safe_mean(float(row[f'acc_{dimension}']) for row in rows):.6f}"
        category_rows.append(category_row)
    write_rows_csv(
        metrics_dir / "stage2_by_category.csv",
        ["story_category", "stories", "overall_accuracy"] + [f"acc_{dimension}" for dimension in LABEL_DIMENSIONS],
        category_rows,
    )


def write_manifest(output_dir: Path, args: argparse.Namespace, records: Sequence[dict]) -> None:
    try:
        dataset_path_value = os.path.relpath(args.dataset_path, output_dir)
    except Exception:
        dataset_path_value = str(args.dataset_path.name)

    manifest = {
        "dataset_path": dataset_path_value,
        "backend": args.backend,
        "stages": args.stages,
        "stories": [int(record["story_id"]) for record in records],
        "story_count": len(records),
        "model": args.model,
        "extract_model": args.extract_model,
        "label_model": args.label_model,
        "judge_model": args.judge_model,
        "judge_fewshots": args.judge_fewshots,
        "include_story_context": not args.no_story_context,
        "load_in_4bit": args.load_in_4bit,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
        "api_provider": args.api_provider,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    write_text(output_dir / "manifest.json", json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniToM replication runner")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=BENCHMARK_PATH,
        help="Path to benchmark_story_belief_labels.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "replication_outputs",
        help="Directory where extraction, labeling, judge, and metric outputs are written",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "hf", "api"],
        default="mock",
        help="Model backend. `mock` is for smoke tests; `hf` runs local Hugging Face models; `api` runs OpenAI-compatible or Gemini API models.",
    )
    parser.add_argument("--model", type=str, default=None, help="Default model id for all stages")
    parser.add_argument("--extract-model", type=str, default=None, help="Stage 1 extraction model id")
    parser.add_argument("--label-model", type=str, default=None, help="Stage 2 labeling model id")
    parser.add_argument("--judge-model", type=str, default=None, help="Semantic judge model id")
    parser.add_argument("--device", type=str, default="auto", help="HF device target, for example auto, cpu, cuda")
    parser.add_argument("--max-new-tokens", type=int, default=6000, help="Maximum generated tokens per model call")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for model calls")
    parser.add_argument(
        "--api-provider",
        choices=["openai", "google_genai"],
        default="openai",
        help="Provider for --backend api. Use openai for GPT-5/OpenAI-compatible Responses API models.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Environment variable containing the API key. Defaults to OPENAI_API_KEY for openai and GOOGLE_API_KEY/GEMINI_API_KEY for google_genai.",
    )
    parser.add_argument(
        "--api-base-url-env",
        type=str,
        default=None,
        help="Optional environment variable containing an OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantized Hugging Face loading via bitsandbytes for larger open-weight models",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Compute dtype to use with --load-in-4bit",
    )
    parser.add_argument("--story-ids", type=str, default=None, help="Optional story id filter, for example 1,2,5-7")
    parser.add_argument("--max-stories", type=int, default=0, help="Optional cap on number of stories to run")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even if CSVs already exist")
    parser.add_argument(
        "--judge-fewshots",
        type=int,
        default=3,
        help="Few-shot example count for the semantic judge prompt (0-3; paper setting is 3)",
    )
    parser.add_argument("--no-story-context", action="store_true", help="Omit the narrative from judge prompts")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["extract", "label", "judge", "metrics"],
        choices=["extract", "label", "judge", "metrics"],
        help="Stages to execute. Metrics reads the outputs already present in --output-dir.",
    )
    args = parser.parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main() -> int:
    args = parse_args()
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    records = load_records(args.dataset_path)
    records = select_records(records, parse_story_ids(args.story_ids), args.max_stories)
    if not records:
        raise SystemExit("No stories selected. Adjust --story-ids or --max-stories.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir, args, records)

    backend = build_backend(args)
    try:
        extraction_cache: Dict[int, List[dict]] = {}

        if "extract" in args.stages:
            for record in records:
                story_id = int(record["story_id"])
                extraction_cache[story_id] = run_extraction(record, backend, args.output_dir, args)
        elif "judge" in args.stages:
            for record in records:
                story_id = int(record["story_id"])
                cached = maybe_load_rows(stage_paths(args.output_dir, story_id)["extract_csv"])
                if cached is None:
                    raise FileNotFoundError(
                        f"Missing extraction CSV for story {story_id}. Run --stages extract first, or include "
                        f"extract in this run: "
                        f"{stage_paths(args.output_dir, story_id)['extract_csv']}"
                    )
                extraction_cache[story_id] = cached

        if "label" in args.stages:
            for record in records:
                run_labeling(record, backend, args.output_dir, args)
        elif "metrics" in args.stages and all_stage_files_exist(records, args.output_dir, ["label_csv"]):
            for record in records:
                label_path = stage_paths(args.output_dir, int(record["story_id"]))["label_csv"]
                if not label_path.exists():
                    raise FileNotFoundError(
                        f"Missing labeling CSV for story {record['story_id']} in metrics-only mode: {label_path}"
                    )

        if "judge" in args.stages:
            for record in records:
                story_id = int(record["story_id"])
                run_judge(record, extraction_cache[story_id], backend, args.output_dir, args)
        elif "metrics" in args.stages and all_stage_files_exist(records, args.output_dir, ["judge_pred_csv", "judge_gt_csv"]):
            for record in records:
                judge_paths = stage_paths(args.output_dir, int(record["story_id"]))
                if not judge_paths["judge_pred_csv"].exists() or not judge_paths["judge_gt_csv"].exists():
                    raise FileNotFoundError(
                        f"Missing judge CSVs for story {record['story_id']} in metrics-only mode: "
                        f"{judge_paths['judge_pred_csv']}"
                    )

    finally:
        backend.close()

    has_stage1_outputs = all_stage_files_exist(records, args.output_dir, ["judge_pred_csv", "judge_gt_csv"])
    has_stage2_outputs = all_stage_files_exist(records, args.output_dir, ["label_csv"])

    if "metrics" in args.stages and has_stage1_outputs:
        compute_stage1_metrics(records, args.output_dir)
    if "metrics" in args.stages and has_stage2_outputs:
        compute_stage2_metrics(records, args.output_dir)

    stage1_summary = args.output_dir / "metrics" / "stage1_overall.csv"
    stage2_summary = args.output_dir / "metrics" / "stage2_overall.csv"

    print(f"Stories processed: {len(records)}")
    print(f"Outputs written to: {args.output_dir}")
    if stage1_summary.exists():
        print(f"Stage 1 summary: {stage1_summary}")
    if stage2_summary.exists():
        print(f"Stage 2 summary: {stage2_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
