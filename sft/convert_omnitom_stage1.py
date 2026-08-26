#!/usr/bin/env python3
"""Convert OmniToM Stage 1 annotations into chat-format SFT JSONL."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OMNITOM = ROOT / "omnitom-benchmark-review"
sys.path.insert(0, str(OMNITOM))
from benchmark_prompting import load_story_record  # noqa: E402
from prompts_extract import build_extract_messages  # noqa: E402


def output_table(record: dict) -> str:
    lines = ["Actor | Belief | Order"]
    for row in record.get("beliefs", []):
        actor = str(row.get("actor", "")).strip()
        belief = str(row.get("belief", "")).strip()
        order = str(row.get("labels", {}).get("order", "")).strip()
        if not actor or not belief or not order:
            raise ValueError(f"Malformed Stage 1 row in story {record['story_id']}: {row}")
        lines.append(f"{actor} | {belief} | {order}")
    return "\n".join(lines)


def make_example(record: dict, dataset_path: Path) -> dict:
    system, user = build_extract_messages(int(record["story_id"]), dataset_path)
    return {
        "id": f"omnitom_story_{int(record['story_id']):04d}",
        "story_id": int(record["story_id"]),
        "category": record.get("story_category", ""),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": output_table(record)},
        ],
    }


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=OMNITOM / "benchmark_story_belief_labels.jsonl")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/omnitom_stage1_sft")
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Write only a few examples for pipeline validation")
    args = ap.parse_args()
    records = load_records(args.input)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.smoke:
        records = records[: min(4, len(records))]
    n_valid = max(1, round(len(records) * args.validation_fraction)) if len(records) > 1 else 0
    valid = records[:n_valid]
    train = records[n_valid:]
    if not train and valid:
        train, valid = valid, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", valid)):
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for record in rows:
                f.write(json.dumps(make_example(record, args.input), ensure_ascii=False) + "\n")
    manifest = {
        "source": str(args.input.resolve()), "seed": args.seed,
        "validation_fraction": args.validation_fraction, "smoke": args.smoke,
        "train_examples": len(train), "validation_examples": len(valid),
        "format": "messages: system + existing OmniToM Stage 1 extraction prompt + assistant Actor | Belief | Order table",
        "excluded": ["Stage 2 schema labels", "LLM judge", "GRPO"],
    }
    (args.output_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
