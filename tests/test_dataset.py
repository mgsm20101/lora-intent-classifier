from pathlib import Path

from src.dataset import build_label_map, encode_labels, load_intent_data, load_jsonl


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"text": "hi", "label": "greeting"}\n\n{"text": "bye", "label": "farewell"}\n',
        encoding="utf-8",
    )

    rows = load_jsonl(path)

    assert rows == [
        {"text": "hi", "label": "greeting"},
        {"text": "bye", "label": "farewell"},
    ]


def test_build_label_map_is_sorted_and_stable() -> None:
    rows = [{"label": "billing_issue"}, {"label": "greeting"}, {"label": "greeting"}]

    labels, label_to_id = build_label_map(rows)

    assert labels == ["billing_issue", "greeting"]
    assert label_to_id == {"billing_issue": 0, "greeting": 1}


def test_encode_labels_maps_each_row_to_its_id() -> None:
    label_to_id = {"greeting": 0, "billing_issue": 1}
    rows = [{"label": "billing_issue"}, {"label": "greeting"}]

    assert encode_labels(rows, label_to_id) == [1, 0]


def test_load_intent_data_reads_the_real_fixed_split() -> None:
    data = load_intent_data()

    assert len(data.train_rows) == 96
    assert len(data.eval_rows) == 32
    assert len(data.labels) == 8
    assert data.label_to_id[data.labels[0]] == 0
