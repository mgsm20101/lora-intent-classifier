import pytest

from src.eval import evaluate


def test_evaluate_counts_exact_matches() -> None:
    result = evaluate(predicted=[0, 1, 1, 0], true=[0, 1, 0, 0], labels=["a", "b"])

    assert (result["correct"], result["total"], result["accuracy"]) == (3, 4, 0.75)


def test_evaluate_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        evaluate(predicted=[0, 1], true=[0], labels=["a", "b"])


def test_evaluate_breaks_down_by_true_label() -> None:
    labels = ["greeting", "farewell"]
    # true: greeting, greeting, farewell, farewell
    # pred: greeting, farewell, farewell, farewell
    result = evaluate(predicted=[0, 1, 1, 1], true=[0, 0, 1, 1], labels=labels)

    assert result["per_class"]["greeting"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert result["per_class"]["farewell"] == {"correct": 2, "total": 2, "accuracy": 1.0}


def test_evaluate_handles_a_label_with_no_eval_examples() -> None:
    result = evaluate(predicted=[0], true=[0], labels=["greeting", "unused_label"])

    assert result["per_class"]["unused_label"] == {"correct": 0, "total": 0, "accuracy": 0.0}


def test_evaluate_rounds_accuracy_to_four_decimals() -> None:
    result = evaluate(predicted=[0, 1, 1], true=[0, 0, 0], labels=["a", "b"])

    assert result["accuracy"] == 0.3333
    assert result["per_class"]["a"]["accuracy"] == 0.3333
