# LoRA on a 135M multilingual encoder, CPU — the 4-bit quantization half of QLoRA is not demonstrated

This project measures what a LoRA adapter actually buys on a small
classification task, on CPU, against two honest baselines. It does **not**
demonstrate 4-bit quantization, and does not claim to.

## Problem

"LoRA trains fewer parameters" is a sentence anyone can repeat. The question
an interviewer actually asks is **how much fewer, at what cost in accuracy,
against what alternative** — and that needs a measurement with baselines on
both sides, not a single number floating alone.

The original plan was QLoRA (4-bit NF4 quantization + LoRA) on
Qwen2.5-1.5B-Instruct on a Colab T4. That path was never run against this
repository: `bitsandbytes`'s 4-bit kernels need CUDA, and the only torch
available in this environment is the `+cpu` wheel. Rather than publish
projected numbers for an unrun path, this project was reduced to what is
actually runnable and measured it properly.

## Architecture

Three configurations of the same model on the same data. The only variable is
**which parameters are allowed to move**:

```
distilbert-base-multilingual-cased (135M)  ·  8-class intent classification
96 train / 32 eval  ·  fixed seed  ·  12 epochs  ·  chance = 0.125

  head_only   encoder frozen, classifier only     the cheap floor
  lora        r=8 alpha=16 on q_lin + v_lin        the thing being measured
  full        every parameter trainable            the expensive ceiling

Fresh weights per configuration — nothing inherits another run's progress.
```

`head_only` exists to answer "did the adapters buy anything?" If LoRA could
not beat a frozen encoder, the low-rank updates would be decoration. `full`
is the ceiling: if LoRA gets close to it while training a small fraction of
the parameters and shipping a much smaller artifact, that is the entire
operational argument for LoRA — measured, not quoted.

Code layout:

```
src/
  dataset.py       load the fixed train/eval JSONL split, derive label ids
  lora_setup.py     LoRA config + per-configuration parameter wiring
  eval.py           accuracy and per-class accuracy, pure functions
  provenance.py     git commit/worktree state, refusal-to-run-dirty guard
  train_cpu.py      entry point: trains all three configurations, writes results
tests/              unit tests — no model download, no HF network
data/
  intents.jsonl     96 training examples (see data/SOURCE.md)
  eval_set.jsonl    32 eval examples
results/
  runs_<sha8>.json  written by a measured run, named after its source commit
docs/
  results.md        write-up, filled in after the measured run
```

## Run

```bash
pip install -r requirements.txt
python -m src.train_cpu
```

No GPU, no `bitsandbytes`, no dataset download — the base model comes from
the local Hugging Face cache and the data ships in `data/`. `train_cpu.py`
refuses to run on a dirty or non-git worktree, so that every result it
writes can be traced to an exact commit; pass `--allow-dirty` to bypass this
for local iteration (never for results meant to be cited).

## Eval

Accuracy on 32 held-out examples, balanced across 8 classes, scored by exact
label match — no judge, no partial credit. `src/eval.py` also reports
per-class accuracy, so a configuration that does well only on the easy
classes is visible rather than averaged away.

## Results

<!-- RESULTS: filled from results/runs_<sha8>.json after the measured run -->

No numbers are quoted here. This repository's own `src/train_cpu.py` will
refuse to write a results file that isn't tied to a clean, committed
worktree, and no run has been made against this repository's history at the
time of writing. Once it has, the results table, per-class breakdown, and
provenance (commit sha, worktree state, library versions, hardware,
timestamp) live in [`docs/results.md`](docs/results.md) and
`results/runs_<sha8>.json`.

## Limitations

- **32 eval examples.** At this sample size, a difference of one or two
  correct answers between configurations is not separable from noise —
  read any close comparison with that in mind.
- **One seed, one hyperparameter setting per run.** Rank, alpha, and learning
  rate are not swept or tuned.
- **Easy, short, English-leaning task on a multilingual model.** Nothing
  here transfers to long-context or generative fine-tuning.
- **Timings are indicative**, not benchmark-grade: one process on a shared
  desktop, no warm-up, no repetitions.

## What this project does not prove

- **4-bit quantization.** Not run, not measured, not claimed. The `Q` in
  QLoRA is absent from every number this project produces.
- **Anything at 1.5B-parameter scale.** The measured model is 135M
  parameters.
- **Anything about QLoRA specifically.** The original QLoRA/Qwen2.5-1.5B
  design was never executed against this codebase and is not part of it.
- **Style transfer or generative fine-tuning.** The measured task is
  8-class intent classification, not text generation.

## Reproduction

`results/runs_<sha8>.json` records the model id, seed, epochs, batch size,
max length, LoRA config, class count, chance accuracy, per-configuration
results, and a provenance block (`source_commit_sha`, `worktree_clean`,
library versions, hardware, timestamp) — enough to check the numbers against
the exact conditions that produced them.

```bash
python -m src.train_cpu && cat results/runs_*.json
```

## Licensing

- **Code**: MIT — see [`LICENSE`](LICENSE).
- **Model**: `distilbert-base-multilingual-cased`, Apache-2.0 (per its Hugging Face
  model card, checked 2026-09-24). The weights are downloaded, not redistributed.
- **Data**: `data/intents.jsonl` and `data/eval_set.jsonl` are the same data as
  [`intent-router`](https://github.com/mgsm20101/intent-router) — see [`data/SOURCE.md`](data/SOURCE.md) for provenance.

## Testing

```bash
pip install -r requirements-ci.txt
python -m pytest -q
```

Tests are deterministic and need no model download or network access
(`HF_HUB_OFFLINE=1` is set in CI): dataset loading and label mapping, eval
metrics, LoRA config construction and parameter counting against a tiny
randomly initialised model, and the result-writing and git-provenance/refusal
logic (with `git` calls mocked).
