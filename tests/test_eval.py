import pytest

from src.eval import accuracy, evaluate, per_class_accuracy


def test_accuracy_counts_exact_matches() -> None:
    correct, total = accuracy(predicted=[0, 1, 1, 0], true=[0, 1, 0, 0])

    assert (correct, total) == (3, 4)


def test_accuracy_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        accuracy(predicted=[0, 1], true=[0])


def test_per_class_accuracy_breaks_down_by_true_label() -> None:
    labels = ["greeting", "farewell"]
    # true: greeting, greeting, farewell, farewell
    # pred: greeting, farewell, farewell, farewell
    predicted = [0, 1, 1, 1]
    true = [0, 0, 1, 1]

    result = per_class_accuracy(predicted, true, labels)

    assert result["greeting"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert result["farewell"] == {"correct": 2, "total": 2, "accuracy": 1.0}


def test_per_class_accuracy_handles_a_label_with_no_eval_examples() -> None:
    labels = ["greeting", "unused_label"]

    result = per_class_accuracy(predicted=[0], true=[0], labels=labels)

    assert result["unused_label"] == {"correct": 0, "total": 0, "accuracy": 0.0}


def test_evaluate_returns_overall_and_per_class_results() -> None:
    labels = ["greeting", "farewell"]

    result = evaluate(predicted=[0, 0], true=[0, 1], labels=labels)

    assert result.accuracy == 0.5
    assert result.correct == 1
    assert result.total == 2
    assert result.per_class["greeting"]["accuracy"] == 1.0
    assert result.per_class["farewell"]["accuracy"] == 0.0
