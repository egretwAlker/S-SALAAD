"""Greedy generation from a compressed model.

The prefix is recomputed each step (no KV cache): this demonstrates that the
compressed model runs, and is not a serving-speed benchmark.
"""

from __future__ import annotations

import torch


@torch.inference_mode()
def generate(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("Expected a nonempty batch of input token IDs.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative.")
    if input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt and output exceed the configured context length.")
    sequence = input_ids
    for _ in range(max_new_tokens):
        logits = model(input_ids=sequence, use_cache=False).logits[:, -1]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite inference logits.")
        sequence = torch.cat((sequence, logits.argmax(-1, keepdim=True)), dim=1)
    return sequence
