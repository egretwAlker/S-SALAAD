"""Next-token evaluation, used by the serving CLI."""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn

from ssalaad.train.data import PackedLoader, StreamLoader, make_loader

# Perplexity is measured on held-out tokens of the corpus each family was
# trained on, with that corpus' own protocol: C4 documents are scored whole,
# FineWeb-Edu with the sliding window (seq_len 512, stride 256, so every scored
# token sees at least 256 tokens of context), the HuggingFace perplexity recipe:
# https://huggingface.co/docs/transformers/perplexity
#
# FineWeb-Edu is streamed as Parquet. If the process aborts at exit with
# "Fatal Python error: PyGILState_Release" (exit code 134) or hangs after the
# results are printed, that is a pyarrow/datasets shutdown bug
# (https://github.com/apache/arrow/issues/49942), not a fault in the model or
# the score, which are already printed and the bundle already saved. Download a
# FineWeb-Edu shard and evaluate locally to avoid it.
_CORPORA = {
    "llama": {
        "dataset": "allenai/c4",
        "dataset_config": "en",
        "split": "validation",
        "seq_len": 256,
        "mode": "padded",
    },
    "qwen3": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-350BT",
        "split": "train",
        "seq_len": 512,
        "mode": "stream",
        "stride": 256,
    },
}


def make_eval_loader(
    model_config: dict, microbatch_size: int, seed: int
) -> PackedLoader | StreamLoader:
    model_type = model_config.get("model_type", "llama")
    if model_type not in _CORPORA:
        raise SystemExit(
            f"[serve] no evaluation corpus for model_type={model_type!r}; "
            f"known: {sorted(_CORPORA)}"
        )
    tokenizer = model_config.get("tokenizer_id")
    if not tokenizer:
        raise SystemExit(
            "[serve] the bundle carries no tokenizer_id, so the evaluation "
            "corpus cannot be tokenized."
        )
    return make_loader(
        **_CORPORA[model_type],
        tokenizer=tokenizer,
        microbatch_size=microbatch_size,
        seed=seed,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: PackedLoader | StreamLoader,
    device: torch.device,
    val_tokens: int,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Mean next-token loss and perplexity over a ``val_tokens`` budget.

    Batch geometry comes from the loader, so the window and the budget cannot
    disagree. The budget is rounded up to whole batches. Positions labelled
    ``-100`` (padding, and the context-only part of a sliding window) do not
    contribute.
    """
    if val_tokens <= 0:
        raise ValueError("val_tokens must be positive.")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    tokens_per_batch = loader.microbatch_size * loader.seq_len
    n_iters = (val_tokens + tokens_per_batch - 1) // tokens_per_batch
    for _ in range(n_iters):
        ids_cpu, labels_cpu, _ = loader.next_batch()
        ids = ids_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)
        ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype
            else contextlib.nullcontext()
        )
        with ctx:
            out = model(input_ids=ids, labels=labels)
        n_valid = int((labels[:, 1:] != -100).sum().item())
        total_loss += out.loss.item() * n_valid
        total_tokens += n_valid
    if total_tokens == 0:
        raise ValueError("Evaluation requires at least one valid next-token target.")
    avg = total_loss / total_tokens
    return {"val/loss": avg, "val/ppl": math.exp(min(avg, 20.0))}
