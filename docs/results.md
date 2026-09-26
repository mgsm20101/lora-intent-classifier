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

The Wilson intervals were computed by hand from `correct`/`n_eval` in the JSON;
no code in this repository produces them.

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
