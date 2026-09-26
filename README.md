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

## Structure

### Entry points

| Command | Reads | Writes |
|---|---|---|
| `python -m src.train_cpu` (the measured run) | `data/intents.jsonl`, `data/eval_set.jsonl`, the base model from the local Hugging Face cache, `git rev-parse HEAD` / `git status` | `results/runs_<sha8>.json`; checkpoints under `runs/<configuration>/` (git-ignored) |
| `python -m src.train_cpu --allow-dirty` | same | same, but skips the clean-tree check — local iteration only, never cited |
| `python -m pytest -q` | `tests/`, `data/` | nothing (no model download, no network) |

### Run flow

```
python -m src.train_cpu
└─ src/train_cpu.py:main
   ├─ src/provenance.py:build_provenance     HEAD sha + clean check; raises DirtyWorktreeError on a dirty tree
   ├─ src/dataset.py:load_intent_data        data/intents.jsonl + data/eval_set.jsonl, label ids
   ├─ src/train_cpu.py:train_one             once each for head_only, lora, full
   │  ├─ _build → src/lora_setup.py:apply_configuration   fresh weights, choose trainable params
   │  ├─ src/lora_setup.py:count_trainable_params
   │  ├─ AdamW loop, 12 epochs
   │  ├─ _predict → src/eval.py:evaluate                   accuracy + per-class on the 32 eval rows
   │  └─ _checkpoint_size                                  writes runs/<configuration>/, returns bytes
   ├─ src/train_cpu.py:assemble_payload      config + the three runs + provenance
   └─ src/train_cpu.py:results_path          → results/runs_<sha8>.json
```

### Code map

```
src/
  __init__.py        empty; makes `python -m src.train_cpu` work
  train_cpu.py       the only entry point: trains the three configurations, writes the results JSON
  lora_setup.py      LoRA config (r=8, alpha=16, q_lin+v_lin) and which parameters each configuration trains
  dataset.py         loads the fixed train/eval JSONL split, derives label ids
  eval.py            evaluate(): overall and per-class accuracy, a pure function
  provenance.py      HEAD sha, clean/dirty worktree, library versions; refuses a dirty tree
data/
  intents.jsonl      96 training rows
  eval_set.jsonl     32 eval rows (4 per class)
  SOURCE.md          where the rows came from
results/             runs_<sha8>.json, one file per measured run, named after its source commit
tests/               unit tests, one file per src module; no model download, no network
docs/results.md      full write-up of the measured run, including per-class results
```

### Read the code in this order

1. `src/train_cpu.py` — `main`, then `train_one`: the whole experiment in one file.
2. `src/lora_setup.py` — the only thing that differs between the three configurations.
3. `src/eval.py` — how accuracy is scored.
4. `src/provenance.py` — why a run refuses to start on a dirty tree.
5. `src/dataset.py` — data loading; nothing surprising.

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

Measured at commit `d962ba5d` on a clean tree — raw file
[`results/runs_d962ba5d.json`](results/runs_d962ba5d.json). 96 training rows, 32 eval rows,
8 balanced classes (chance 0.125), 12 epochs, seed 20260923,
CPU only (`torch 2.14.0+cpu`, `transformers 5.16.1`,
`peft 0.21.0`).

| configuration | trainable parameters | accuracy | 95% CI (Wilson) | train time | saved artifact |
|---|---:|---:|---:|---:|---:|
| `head_only` | 596,744 (0.44%) | **0.562** (18/32) | 0.39–0.72 | 129 s | 541.4 MB |
| `lora` | 744,200 (0.55%) | **0.719** (23/32) | 0.55–0.84 | 267 s | 3.0 MB |
| `full` | 135,330,824 (100.00%) | **0.750** (24/32) | 0.58–0.87 | 408 s | 541.4 MB |

The Wilson intervals were computed by hand from `correct`/`n_eval` in the JSON; no code
in this repository produces them. With 32 eval rows they overlap heavily: the ordering is
a direction, not a measured size. Per-class results and the full reading are in
[`docs/results.md`](docs/results.md).

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

`results/runs_<sha8>.json` records the model id, device, epochs, batch size, max
length, train/eval row counts, class count, chance accuracy, per-configuration
results (parameter counts, accuracy, per-class breakdown, train time, checkpoint
size, learning rate), and a provenance block (`seed`, `library_versions`,
`hardware`, `timestamp`, `source_commit_sha`, `worktree_clean`) — enough to check
the numbers against the exact conditions that produced them. The LoRA
hyperparameters (rank, alpha, dropout, target modules) are not in the JSON; they
are fixed in `src/lora_setup.py` at the recorded commit.

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
