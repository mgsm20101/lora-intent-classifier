# Results

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

**Read with the intervals.** LoRA recovers most of the gap between training only the
classifier head and training everything (23 vs 18 vs 24 correct of 32) while moving
0.55% of the parameters and saving a 3 MB adapter instead of a 541 MB model. But with 32
eval rows the intervals overlap heavily: the ordering is a direction, not a measured size,
and LoRA vs full is one question apart.

The same numbers came out of an earlier run of the same loop before this repository
existed; with a fixed seed on CPU the run is deterministic.

Per class (correct of 4):

| class | head_only | lora | full |
|---|---:|---:|---:|
| account_update | 3/4 | 3/4 | 3/4 |
| billing_issue | 4/4 | 4/4 | 2/4 |
| cancel_subscription | 1/4 | 4/4 | 4/4 |
| complaint | 2/4 | 2/4 | 3/4 |
| greeting | 1/4 | 2/4 | 3/4 |
| product_inquiry | 4/4 | 3/4 | 4/4 |
| refund_request | 1/4 | 3/4 | 3/4 |
| technical_support | 2/4 | 2/4 | 2/4 |

No run has been made against this repository's history yet. `src/train_cpu.py`
refuses to execute on a dirty or non-git worktree (unless `--allow-dirty` is
passed), specifically so that every number that ends up here can be traced
back to the exact commit that produced it — see `src/provenance.py`.

Once the measured run has been made, this file holds:

* Per-configuration (`head_only`, `lora`, `full`) accuracy on the 32-example
  eval set, with the per-class breakdown from `src/eval.py`.
* Trainable parameter count and share of the base model for each
  configuration, from `src/lora_setup.py`.
* On-disk size of what each configuration would actually ship — the LoRA
  adapter for `lora`, the full state dict for `head_only` and `full`.
* Train wall time per configuration.
* The run's provenance block: seed, library versions, hardware string,
  timestamp, `source_commit_sha`, and `worktree_clean`, all written
  automatically to `results/runs_<sha8>.json` by the same script run.

Until that file exists, any number quoted here or in the README would be
unsourced, so none are quoted. See the README's **What this project does not
prove** section for the boundaries of the comparison this repo is designed
to make.
