"""LoRA on a 135M encoder, on CPU, measured against two honest baselines.

    python -m src.train_cpu
    python -m src.train_cpu --allow-dirty   # local iteration, not for citing

The 4-bit quantization half of QLoRA is not demonstrated anywhere in this
project: `bitsandbytes` needs CUDA, and the torch installed here is the
`+cpu` wheel. This script demonstrates the half that *is* runnable and
measures it properly: **LoRA without quantization, on a 135M encoder,
on CPU.**

**Demonstrated here:** what a low-rank update injected into attention
projections actually buys — how few parameters it trains, how small the
resulting adapter is, and what it costs in accuracy against both a cheaper
and a more expensive alternative.

**Not demonstrated here, and not claimed anywhere:** 4-bit quantization, any
behaviour at 1.5B scale, and anything about QLoRA specifically.

The comparison is the point. A LoRA accuracy number on its own says nothing
— the question a reviewer asks is "compared to what?", and there are two
answers worth having:

  head_only  freeze the encoder, train the classifier. The cheap baseline.
             If LoRA does not beat this, the adapters bought nothing.
  lora       rank-8 adapters on the attention query/value projections.
  full       every parameter trainable. The expensive ceiling.
             If LoRA matches this, the adapters cost almost nothing.

Same data, same seed, same epochs, same eval set. The only variable is
which parameters are allowed to move. Results are written to
`results/runs_<sha8>.json`, tied to the exact commit and worktree state
that produced them — see `src/provenance.py`. This script refuses to run
on a dirty or non-git worktree unless `--allow-dirty` is passed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset import encode_labels, load_intent_data
from src.eval import evaluate
from src.lora_setup import apply_configuration, count_trainable_params
from src.provenance import build_provenance

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ARTIFACTS = ROOT / "runs"

MODEL_ID = "distilbert-base-multilingual-cased"
SEED = 20260923
EPOCHS = 12
BATCH_SIZE = 16
LR = 5e-4          # LoRA and head want a higher LR than a full fine-tune
FULL_LR = 3e-5     # a full fine-tune at 5e-4 diverges; this is not a free choice
MAX_LEN = 64


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _encode(rows, tokenizer, label_to_id: dict[str, int]) -> TensorDataset:
    batch = tokenizer(
        [r["text"] for r in rows],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return TensorDataset(
        batch["input_ids"],
        batch["attention_mask"],
        torch.tensor(encode_labels(rows, label_to_id)),
    )


@dataclass
class Run:
    configuration: str
    trainable_params: int
    total_params: int
    trainable_pct: float
    accuracy: float
    correct: int
    n_eval: int
    per_class: dict
    train_seconds: float
    checkpoint_bytes: int
    learning_rate: float


def _build(configuration: str, labels: list[str]):
    """Fresh weights every time — a model carried over from a previous
    configuration would make the second one look better for free."""
    _seed_everything(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(labels)
    )
    lr = FULL_LR if configuration == "full" else LR
    return apply_configuration(model, configuration), lr


@torch.no_grad()
def _predict(model, loader) -> tuple[list[int], list[int]]:
    model.eval()
    predicted, true = [], []
    for input_ids, mask, labels in loader:
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        predicted.extend(logits.argmax(-1).tolist())
        true.extend(labels.tolist())
    return predicted, true


def _checkpoint_size(model, configuration: str) -> int:
    """What you would actually ship for this configuration.

    For LoRA that is the adapter, not the base model — which is the entire
    operational argument for it, so measuring the full state dict here would
    erase the thing being demonstrated.
    """
    out = ARTIFACTS / configuration
    out.mkdir(parents=True, exist_ok=True)
    if configuration == "lora":
        model.save_pretrained(out)
        return sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    path = out / "model.pt"
    torch.save({k: v for k, v in model.state_dict().items()}, path)
    return path.stat().st_size


def train_one(configuration: str, train_rows, eval_rows, labels, label_to_id, tokenizer) -> Run:
    model, lr = _build(configuration, labels)
    trainable, total = count_trainable_params(model)

    train_loader = DataLoader(
        _encode(train_rows, tokenizer, label_to_id), batch_size=BATCH_SIZE, shuffle=True
    )
    eval_loader = DataLoader(_encode(eval_rows, tokenizer, label_to_id), batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    print(
        f"\n  {configuration}: {trainable:,} trainable of {total:,} "
        f"({100 * trainable / total:.2f}%), lr={lr}",
        flush=True,
    )

    start = time.perf_counter()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        for input_ids, mask, batch_labels in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=mask, labels=batch_labels)
            out.loss.backward()
            optimizer.step()
            running += float(out.loss)
        if epoch % 4 == 0 or epoch == 1:
            print(f"    epoch {epoch:>2}  loss {running / len(train_loader):.4f}", flush=True)
        model.train()
    seconds = time.perf_counter() - start

    predicted, true = _predict(model, eval_loader)
    result = evaluate(predicted, true, labels)
    size = _checkpoint_size(model, configuration)
    print(
        f"    accuracy {result.correct}/{result.total} = {result.accuracy:.3f}   "
        f"{seconds:.1f}s   checkpoint {size / 1e6:.1f} MB",
        flush=True,
    )

    return Run(
        configuration=configuration,
        trainable_params=trainable,
        total_params=total,
        trainable_pct=round(100 * trainable / total, 3),
        accuracy=result.accuracy,
        correct=result.correct,
        n_eval=result.total,
        per_class=result.per_class,
        train_seconds=round(seconds, 1),
        checkpoint_bytes=size,
        learning_rate=lr,
    )


def assemble_payload(data, runs: list[Run], provenance: dict) -> dict:
    """Build the JSON-serialisable payload written to results/runs_<sha8>.json.

    Split out from `main` so the result-writing shape can be unit tested
    without running any training.
    """
    return {
        "model": MODEL_ID,
        "device": "cpu",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
        "n_train": len(data.train_rows),
        "n_eval": len(data.eval_rows),
        "n_classes": len(data.labels),
        "chance_accuracy": round(1 / len(data.labels), 3),
        "runs": [asdict(r) for r in runs],
        **provenance,
    }


def results_path(provenance: dict) -> Path:
    sha8 = (provenance["source_commit_sha"] or "nogitsha")[:8]
    return RESULTS_DIR / f"runs_{sha8}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="skip the clean-worktree/in-git-repo check (local iteration only)",
    )
    args = parser.parse_args(argv)

    provenance = build_provenance(ROOT, seed=SEED, allow_dirty=args.allow_dirty)

    data = load_intent_data()
    print(f"model      : {MODEL_ID}")
    print(f"device     : cpu (torch {provenance['library_versions']['torch']})")
    print(
        f"train/eval : {len(data.train_rows)}/{len(data.eval_rows)} · "
        f"{len(data.labels)} classes · chance = {1 / len(data.labels):.3f}"
    )
    print(f"epochs={EPOCHS} batch={BATCH_SIZE} max_len={MAX_LEN} seed={SEED}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    runs = [
        train_one(c, data.train_rows, data.eval_rows, data.labels, data.label_to_id, tokenizer)
        for c in ("head_only", "lora", "full")
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = results_path(provenance)
    out_path.write_text(
        json.dumps(assemble_payload(data, runs, provenance), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")

    print(
        f"\n  {'configuration':<12}{'trainable':>12}{'%':>8}{'acc':>8}"
        f"{'sec':>8}{'ckpt MB':>10}"
    )
    print("  " + "-" * 58)
    for r in runs:
        print(
            f"  {r.configuration:<12}{r.trainable_params:>12,}{r.trainable_pct:>8.2f}"
            f"{r.accuracy:>8.3f}{r.train_seconds:>8.1f}{r.checkpoint_bytes / 1e6:>10.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
