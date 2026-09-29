from __future__ import annotations

import json
from pathlib import Path

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
    Qwen3Config,
    Qwen3ForCausalLM,
)

_MODEL_CLASSES = {
    "llama": (LlamaConfig, LlamaForCausalLM),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM),
}
_MODEL_FIELDS = {
    "model_type",
    "architectures",
    "hidden_size",
    "intermediate_size",
    "hidden_act",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_sequence_length",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "rope_parameters",
    "initializer_range",
    "tie_word_embeddings",
    "attention_bias",
    "attention_dropout",
    "mlp_bias",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "tokenizer_id",
    "use_cache",
}


def resolve_model_config(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    # Recipes point at configs by repo-relative path; fall back to the packaged copy.
    bundled = Path(__file__).resolve().parent / "configs" / "model" / path.name
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(f"Model configuration not found: {path.name}")


def read_model_config(path: str | Path) -> dict:
    return json.loads(resolve_model_config(path).read_text())


def export_model_config(config: dict) -> dict:
    """Keep architecture and tokenizer fields; exclude source and run metadata."""
    result = {key: value for key, value in config.items() if key in _MODEL_FIELDS}
    tokenizer = result.get("tokenizer_id", "")
    if tokenizer and (Path(tokenizer).is_absolute() or Path(tokenizer).exists()):
        result.pop("tokenizer_id")
    return result


def write_model_config(source: str | Path | dict, destination: str | Path) -> None:
    if isinstance(source, (str, Path)):
        source = read_model_config(source)
    Path(destination).write_text(json.dumps(export_model_config(source), indent=2) + "\n")


def build_model_from_config(
    config: dict, *, attn_implementation: str | None = None, **overrides
) -> PreTrainedModel:
    config = dict(config)
    if "max_sequence_length" in config and "max_position_embeddings" not in config:
        config["max_position_embeddings"] = config.pop("max_sequence_length")
    if config.get("pad_token_id") == -1:
        config["pad_token_id"] = None
    config["use_cache"] = False
    if attn_implementation is not None:
        config["attn_implementation"] = attn_implementation
    config.update({key: value for key, value in overrides.items() if value is not None})
    model_type = config.get("model_type", "llama")
    if model_type not in _MODEL_CLASSES:
        raise ValueError(f"Unsupported model type: {model_type}")
    config_cls, model_cls = _MODEL_CLASSES[model_type]
    return model_cls(config_cls(**config))


def build_model(
    config: str | Path | dict, *, attn_implementation: str | None = None
) -> PreTrainedModel:
    if isinstance(config, (str, Path)):
        config = read_model_config(config)
    return build_model_from_config(config, attn_implementation=attn_implementation)
