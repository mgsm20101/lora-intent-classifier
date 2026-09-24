"""Tests build a tiny, randomly initialised DistilBERT from a config object
(2 layers, dim 32) so no weights are downloaded and no HF network is used.
"""

from transformers import DistilBertConfig, DistilBertForSequenceClassification

from src.lora_setup import apply_configuration, build_lora_config, count_trainable_params

TINY_CONFIG = DistilBertConfig(
    vocab_size=99, dim=32, n_layers=2, n_heads=2, hidden_dim=37, num_labels=3
)


def _tiny_model():
    return DistilBertForSequenceClassification(TINY_CONFIG)


def test_build_lora_config_uses_project_defaults() -> None:
    config = build_lora_config()

    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.target_modules == {"q_lin", "v_lin"}
    assert config.modules_to_save == ["classifier", "pre_classifier"]


def test_full_configuration_trains_every_parameter() -> None:
    model = apply_configuration(_tiny_model(), "full")

    trainable, total = count_trainable_params(model)

    assert trainable == total


def test_head_only_configuration_trains_only_the_classifier() -> None:
    model = apply_configuration(_tiny_model(), "head_only")

    trainable, total = count_trainable_params(model)

    assert 0 < trainable < total
    for name, param in model.named_parameters():
        expected = "classifier" in name or "pre_classifier" in name
        assert param.requires_grad == expected


def test_lora_configuration_trains_fewer_params_than_full_but_more_than_frozen() -> None:
    head_only_trainable, _ = count_trainable_params(apply_configuration(_tiny_model(), "head_only"))
    lora_trainable, lora_total = count_trainable_params(apply_configuration(_tiny_model(), "lora"))
    full_trainable, full_total = count_trainable_params(apply_configuration(_tiny_model(), "full"))

    assert lora_total > full_total  # LoRA wraps the base model and adds adapter params
    assert head_only_trainable < lora_trainable < full_trainable


def test_unknown_configuration_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        apply_configuration(_tiny_model(), "quantized_4bit")
