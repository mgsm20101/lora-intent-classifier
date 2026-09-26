"""Overall and per-class accuracy over predicted/true label ids.

A pure function so it can be unit tested without loading a model or tokenizer.
"""

from __future__ import annotations


def evaluate(predicted: list[int], true: list[int], labels: list[str]) -> dict:
    """Return {"accuracy", "correct", "total", "per_class"} for parallel id lists.

    `per_class` is keyed by label name, each {"correct", "total", "accuracy"}.
    A label with zero eval examples reports accuracy 0.0 (distinguishable via
    `total == 0`). Accuracies are rounded to 4 decimals.
    """
    if len(predicted) != len(true):
        raise ValueError("predicted and true must be the same length")

    counts = {label: [0, 0] for label in labels}  # label -> [correct, total]
    for pred_id, true_id in zip(predicted, true):
        counts[labels[true_id]][0] += int(pred_id == true_id)
        counts[labels[true_id]][1] += 1

    def rate(correct: int, total: int) -> float:
        return round(correct / total, 4) if total else 0.0

    correct = sum(c for c, _ in counts.values())
    return {
        "accuracy": rate(correct, len(true)),
        "correct": correct,
        "total": len(true),
        "per_class": {
            label: {"correct": c, "total": t, "accuracy": rate(c, t)}
            for label, (c, t) in counts.items()
        },
    }
