# Results

<!-- RESULTS: filled from results/runs_<sha8>.json after the measured run -->

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
