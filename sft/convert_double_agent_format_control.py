#!/usr/bin/env python3
"""Create a Double-Agent JSON-format control with no OmniToM supervision."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "AIDoubleAgentDefenders/datasets_directory/final_datasets/three_layered_dataset.json"
TARGET = json.dumps({
    "attacker_belief_about_target": "unknown",
    "attacker_trust_in_us": "unknown",
    "reflection": "unknown",
    "reply": "unknown",
})


def example(index: int, row: dict) -> dict:
    # The prompt mirrors the defender's required output contract without
    # exposing any labeled beliefs or OmniToM story text.
    return {
        "id": f"double_agent_format_{index:04d}",
        "messages": [
            {"role": "system", "content": "Return a JSON object with exactly these keys: attacker_belief_about_target, attacker_trust_in_us, reflection, reply."},
            {"role": "user", "content": "Produce the required structured defender response."},
            {"role": "assistant", "content": TARGET},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/double_agent_format_control")
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = json.loads(SOURCE.read_text())
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_valid = round(len(rows) * args.validation_fraction)
    valid, train = rows[:n_valid], rows[n_valid:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("validation", valid)):
        with (args.output_dir / f"{name}.jsonl").open("w") as handle:
            for i, row in enumerate(subset):
                handle.write(json.dumps(example(i, row)) + "\n")
    (args.output_dir / "conversion_manifest.json").write_text(json.dumps({
        "source": str(SOURCE), "seed": args.seed, "train_examples": len(train),
        "validation_examples": len(valid), "target": "fixed Double-Agent JSON schema placeholders",
        "excluded": ["OmniToM stories", "OmniToM labels", "belief supervision"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
