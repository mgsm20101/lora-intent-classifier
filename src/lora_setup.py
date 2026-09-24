"""LoRA config construction and per-configuration parameter wiring.

Kept independent of `from_pretrained` so it can be unit tested against a
freshly constructed, randomly initialised tiny model — no weight download,
no network.
"""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_lin", "v_lin"]  # DistilBERT's attention projection names
SAVE_MODULES = ["classifier", "pre_classifier"]


def build_lora_config(
    r: int = LORA_R,
    alpha: int = LORA_ALPHA,
    targets: list[str] | None = None,
    dropout: float = LORA_DROPOUT,
) -> LoraConfig:
    """Build the LoRA config used across this project's experiments."""
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(targets or LORA_TARGETS),
        # The classifier head is randomly initialised and must train
        # regardless of configuration; peft saves it alongside the adapter.
        modules_to_save=list(SAVE_MODULES),
    )


def count_trainable_params(model) -> tuple[int, int]:
    """Return (trainable_params, total_params) for a torch module."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def apply_configuration(model, configuration: str):
    """Wire up `model` for one of the three experiment arms.

    `model` must already be constructed (from a checkpoint or, for tests,
    from a bare config) with the right number of output labels.
    """
    if configuration == "full":
        return model

    if configuration == "head_only":
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name or "pre_classifier" in name
        return model

    if configuration == "lora":
        return get_peft_model(model, build_lora_config())

    raise ValueError(f"unknown configuration: {configuration!r}")
