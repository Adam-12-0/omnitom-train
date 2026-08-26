import json
import csv
import io
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


BENCHMARK_PATH = Path(__file__).with_name("benchmark_story_belief_labels.jsonl")


def _dataset_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else BENCHMARK_PATH


@lru_cache(maxsize=4)
def _load_benchmark_records(path_str: str) -> Dict[int, dict]:
    path = Path(path_str)
    records: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records[int(row["story_id"])] = row
    return records


def load_story_record(story_id: int, dataset_path: str | Path | None = None) -> dict:
    records = _load_benchmark_records(str(_dataset_path(dataset_path).resolve()))
    sid = int(story_id)
    if sid not in records:
        raise KeyError(f"story_id {sid} not found in {dataset_path or BENCHMARK_PATH}")
    return records[sid]


def story_text(story_id: int, dataset_path: str | Path | None = None) -> str:
    return str(load_story_record(story_id, dataset_path)["story"]).strip()


def story_category(story_id: int, dataset_path: str | Path | None = None) -> str:
    return str(load_story_record(story_id, dataset_path)["story_category"]).strip()


def belief_rows(story_id: int, dataset_path: str | Path | None = None) -> List[dict]:
    record = load_story_record(story_id, dataset_path)
    return list(record.get("beliefs", []))


def actor_belief_rows(story_id: int, dataset_path: str | Path | None = None) -> List[dict]:
    rows = []
    for belief in belief_rows(story_id, dataset_path):
        actor = str(belief.get("actor", "")).strip()
        text = str(belief.get("belief", "")).strip()
        if actor and text:
            rows.append({"actor": actor, "belief": text})
    return rows


def belief_table_pipe(story_id: int, dataset_path: str | Path | None = None) -> str:
    lines = ["Actor | Belief"]
    for row in actor_belief_rows(story_id, dataset_path):
        lines.append(f"{row['actor']} | {row['belief']}")
    return "\n".join(lines)


def belief_table_csv(story_id: int, dataset_path: str | Path | None = None) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["Actor", "Belief"], lineterminator="\n")
    writer.writeheader()
    for row in actor_belief_rows(story_id, dataset_path):
        writer.writerow({"Actor": row["actor"], "Belief": row["belief"]})
    return buffer.getvalue()
