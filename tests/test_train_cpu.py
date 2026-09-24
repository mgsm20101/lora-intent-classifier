"""Tests the result-writing shape only — no training, no model download."""

from src.dataset import IntentData
from src.train_cpu import Run, assemble_payload, results_path


def _fake_data() -> IntentData:
    return IntentData(
        train_rows=[{"text": "hi", "label": "greeting"}] * 4,
        eval_rows=[{"text": "bye", "label": "farewell"}] * 2,
        labels=["farewell", "greeting"],
        label_to_id={"farewell": 0, "greeting": 1},
    )


def _fake_run(configuration: str) -> Run:
    return Run(
        configuration=configuration,
        trainable_params=100,
        total_params=1000,
        trainable_pct=10.0,
        accuracy=0.5,
        correct=1,
        n_eval=2,
        per_class={"greeting": {"correct": 1, "total": 1, "accuracy": 1.0}},
        train_seconds=1.2,
        checkpoint_bytes=2048,
        learning_rate=5e-4,
    )


def test_assemble_payload_includes_all_runs_and_provenance() -> None:
    data = _fake_data()
    runs = [_fake_run("head_only"), _fake_run("lora"), _fake_run("full")]
    provenance = {"seed": 1, "source_commit_sha": "deadbeef00", "worktree_clean": True}

    payload = assemble_payload(data, runs, provenance)

    assert payload["n_train"] == 4
    assert payload["n_eval"] == 2
    assert payload["n_classes"] == 2
    assert payload["chance_accuracy"] == 0.5
    assert [r["configuration"] for r in payload["runs"]] == ["head_only", "lora", "full"]
    assert payload["seed"] == 1
    assert payload["source_commit_sha"] == "deadbeef00"


def test_results_path_uses_first_eight_chars_of_commit_sha() -> None:
    path = results_path({"source_commit_sha": "deadbeef00112233"})

    assert path.name == "runs_deadbeef.json"


def test_results_path_falls_back_when_sha_is_missing() -> None:
    path = results_path({"source_commit_sha": None})

    assert path.name == "runs_nogitsha.json"
