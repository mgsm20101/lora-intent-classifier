"""Accuracy and per-class accuracy metrics.

Pure functions over predicted/true label ids so they can be unit tested
without loading a model or tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    correct: int
    total: int
    per_class: dict[str, dict[str, float | int]]


def accuracy(predicted: list[int], true: list[int]) -> tuple[int, int]:
    """Return (correct, total) for parallel prediction/label id lists."""
    if len(predicted) != len(true):
        raise ValueError("predicted and true must be the same length")
    correct = sum(1 for p, t in zip(predicted, true) if p == t)
    return correct, len(true)


def per_class_accuracy(
    predicted: list[int], true: list[int], labels: list[str]
) -> dict[str, dict[str, float | int]]:
    """Per-class accuracy, keyed by label name.

    A label with zero eval examples reports accuracy 0.0 rather than
    dividing by zero, and is distinguishable via `total == 0`.
    """
    if len(predicted) != len(true):
        raise ValueError("predicted and true must be the same length")

    per_label_counts = {label: {"correct": 0, "total": 0} for label in labels}
    for pred_id, true_id in zip(predicted, true):
        label = labels[true_id]
        per_label_counts[label]["total"] += 1
        if pred_id == true_id:
            per_label_counts[label]["correct"] += 1

    result: dict[str, dict[str, float | int]] = {}
    for label, counts in per_label_counts.items():
        total = counts["total"]
        acc = counts["correct"] / total if total else 0.0
        result[label] = {
            "correct": counts["correct"],
            "total": total,
            "accuracy": round(acc, 4),
        }
    return result


def evaluate(predicted: list[int], true: list[int], labels: list[str]) -> EvalResult:
    """Full evaluation: overall accuracy plus the per-class breakdown."""
    correct, total = accuracy(predicted, true)
    return EvalResult(
        accuracy=round(correct / total, 4) if total else 0.0,
        correct=correct,
        total=total,
        per_class=per_class_accuracy(predicted, true, labels),
    )
