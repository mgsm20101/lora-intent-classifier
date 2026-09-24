# Where this data came from

`intents.jsonl` (96 rows) and `eval_set.jsonl` (32 rows) are copied verbatim
from [`intent-router`](https://github.com/mgsm20101/intent-router) (`data/`). Eight balanced intent classes, so chance
accuracy is 0.125.

Copied rather than imported across projects so this one runs from its own
checkout with no path assumptions about a sibling directory.

**It is reused on purpose.** `intent-router` already measured a DistilBERT
encoder on exactly these rows, which means the LoRA numbers here land next to
an existing reference point instead of floating alone.

That said, the comparison that matters is the one *inside* this project —
`head_only` vs `lora` vs `full`, same seed, same epochs, same eval set. Those
three differ only in which parameters are allowed to move, so the difference
between them is attributable. intent-router's number was produced by a different pipeline
and is context, not a baseline.
