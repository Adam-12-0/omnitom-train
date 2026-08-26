#!/usr/bin/env python3
"""Convert OmniToM stories to Stage 1 chat-format SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a Theory of Mind belief modeling system. Given a narrative, extract "
    "the belief structure needed to represent what each relevant actor believes "
    "about the world and about other actors' mental states."
)

USER_TEMPLATE = """Extract all relevant belief propositions from the story. Output only a pipe-separated table with columns: Actor | Belief | Order.

Definitions:
Order 0: world-level facts.
Order 1: an actor's belief about the world.
Order 2: an actor's belief about another actor's belief.
Order 3+: higher-order recursive belief.

Story:
{story}"""

TARGET_HEADER = "Actor | Belief | Order"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL file or JSON array")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.9)
    args = parser.parse_args()
    if not 0.0 < args.train_frac < 1.0:
        parser.error("--train-frac must be greater than 0 and less than 1")
    return args


def load_records(path: Path) -> list[Any]:
    """Load a JSON array or non-empty lines from a JSONL file."""
    path = path.expanduser()
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return []
            if not char.isspace():
                first = char
                break
        handle.seek(0)
        if first == "[":
            value = json.load(handle)
            if not isinstance(value, list):
                raise ValueError("JSON input must contain a top-level array")
            return value

        records: list[Any] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on JSONL line {line_number}: {exc.msg}") from exc
        return records


def normalize_cell(value: Any) -> str:
    """Make a value safe for one cell in a one-line pipe-separated table."""
    text = str(value).strip()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.replace("|", r"\|")


def normalize_story(value: Any) -> str:
    return str(value).strip().replace("\r\n", "\n").replace("\r", "\n")


def convert_record(record: Any) -> tuple[dict[str, Any] | None, int, int, str | None]:
    """Return example, total belief rows, skipped rows, and rejection reason."""
    if not isinstance(record, dict):
        return None, 0, 0, "record is not an object"

    beliefs = record.get("beliefs")
    belief_row_count = len(beliefs) if isinstance(beliefs, list) else 0
    story = normalize_story(record.get("story", ""))
    if not story:
        return None, belief_row_count, 0, "missing or empty story"

    story_id = normalize_cell(record.get("story_id", ""))
    if not story_id:
        return None, belief_row_count, 0, "missing or empty story_id"

    if not isinstance(beliefs, list):
        return None, 0, 0, "beliefs is not an array"

    rows: list[str] = []
    skipped = 0
    for item in beliefs:
        if not isinstance(item, dict):
            skipped += 1
            continue
        labels = item.get("labels")
        actor = normalize_cell(item.get("actor", ""))
        belief = normalize_cell(item.get("belief", ""))
        order_value = labels.get("order") if isinstance(labels, dict) else None
        order = normalize_cell(order_value) if order_value is not None else ""
        if not actor or not belief or not order:
            skipped += 1
            continue
        rows.append(f"{actor} | {belief} | {order}")

    if not rows:
        return None, len(beliefs), skipped, "no valid belief rows"

    target = TARGET_HEADER + "\n" + "\n".join(rows)
    if not target.startswith(TARGET_HEADER):
        raise AssertionError("assistant target header invariant failed")

    category = normalize_cell(record.get("story_category", ""))
    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(story=story)},
            {"role": "assistant", "content": target},
        ],
        "story_id": story_id,
        "story_category": category,
    }
    return example, len(beliefs), skipped, None


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_examples(examples: list[dict[str, Any]], seed: int, train_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    if len(indices) <= 1:
        train_count = len(indices)
    else:
        train_count = max(1, min(len(indices) - 1, int(len(indices) * train_frac)))
    train = [examples[index] for index in indices[:train_count]]
    val = [examples[index] for index in indices[train_count:]]
    return train, val


def render_report_markdown(report: dict[str, Any]) -> str:
    labels = {
        "input_stories": "Number of input stories",
        "usable_stories": "Number of usable stories",
        "train_stories": "Number of train stories",
        "validation_stories": "Number of validation stories",
        "total_belief_rows": "Total belief rows",
        "skipped_belief_rows": "Skipped belief rows",
        "average_target_character_length": "Average target character length",
        "max_target_character_length": "Max target character length",
        "average_beliefs_per_story": "Average beliefs per story",
        "max_beliefs_per_story": "Max beliefs per story",
        "rejected_stories": "Rejected stories",
    }
    lines = ["# OmniToM Stage 1 Conversion Report", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {labels[key]} | {report[key]} |" for key in labels)
    lines.extend(["", f"Seed: `{report['seed']}`  ", f"Train fraction: `{report['train_fraction']}`", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)

    examples: list[dict[str, Any]] = []
    total_belief_rows = 0
    skipped_belief_rows = 0
    rejected_stories = 0
    for record in records:
        example, row_count, skipped, rejection = convert_record(record)
        total_belief_rows += row_count
        skipped_belief_rows += skipped
        if rejection is not None:
            rejected_stories += 1
            continue
        assert example is not None
        examples.append(example)

    train, val = split_examples(examples, args.seed, args.train_frac)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train_sft.jsonl", train)
    write_jsonl(output_dir / "val_sft.jsonl", val)
    write_jsonl(output_dir / "sample_8_sft.jsonl", examples[:8])

    target_lengths = [len(example["messages"][2]["content"]) for example in examples]
    belief_counts = [example["messages"][2]["content"].count("\n") for example in examples]
    report = {
        "input_stories": len(records),
        "usable_stories": len(examples),
        "train_stories": len(train),
        "validation_stories": len(val),
        "total_belief_rows": total_belief_rows,
        "skipped_belief_rows": skipped_belief_rows,
        "average_target_character_length": round(statistics.mean(target_lengths), 2) if target_lengths else 0.0,
        "max_target_character_length": max(target_lengths, default=0),
        "average_beliefs_per_story": round(statistics.mean(belief_counts), 2) if belief_counts else 0.0,
        "max_beliefs_per_story": max(belief_counts, default=0),
        "rejected_stories": rejected_stories,
        "seed": args.seed,
        "train_fraction": args.train_frac,
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "conversion_report.md").write_text(render_report_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
