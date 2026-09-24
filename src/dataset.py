"""Loading, label mapping, and splitting for the intent classification data.

The data itself is two fixed JSONL files (`data/intents.jsonl` for training,
`data/eval_set.jsonl` for evaluation) copied verbatim from `01-intent-router` —
see `data/SOURCE.md`. This module does not create a new split; it loads the
two files that already exist and derives the label vocabulary from the
training rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class IntentData:
    train_rows: list[dict]
    eval_rows: list[dict]
    labels: list[str]
    label_to_id: dict[str, int]


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Skips blank lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_label_map(rows: list[dict]) -> tuple[list[str], dict[str, int]]:
    """Derive a sorted, stable label vocabulary and id mapping from rows."""
    labels = sorted({row["label"] for row in rows})
    label_to_id = {label: i for i, label in enumerate(labels)}
    return labels, label_to_id


def load_intent_data(data_dir: Path = DATA_DIR) -> IntentData:
    """Load the fixed train/eval split and derive the label mapping."""
    train_rows = load_jsonl(data_dir / "intents.jsonl")
    eval_rows = load_jsonl(data_dir / "eval_set.jsonl")
    labels, label_to_id = build_label_map(train_rows)
    return IntentData(
        train_rows=train_rows,
        eval_rows=eval_rows,
        labels=labels,
        label_to_id=label_to_id,
    )


def encode_labels(rows: list[dict], label_to_id: dict[str, int]) -> list[int]:
    """Map each row's string label to its integer id."""
    return [label_to_id[row["label"]] for row in rows]
