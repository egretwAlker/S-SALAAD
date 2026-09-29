from __future__ import annotations

import torch
from torch import nn

from ssalaad.train.model import build_model_from_config
from .artifact import load_artifact
from .ls_linear import LRSLinear


def build_efficient_model(path, *, device="cpu", dtype=None):
    """Load a complete compressed model without its training checkpoint."""
    device = torch.device(device)
    # The dedicated kernels: TileLang on CUDA (bf16), AVX2 on CPU (fp32).
    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU sparse inference requires fp32.")
    payload = load_artifact(path)
    model = build_model_from_config(payload["model_config"], attn_implementation="sdpa")
    layers = payload["layers"]
    missing, unexpected = model.load_state_dict(
        payload["base_state_dict"], strict=False
    )
    expected = {name + ".weight" for name in layers}
    if set(missing) != expected or unexpected:
        raise ValueError(
            "The bundle does not contain exactly the required model weights."
        )
    model.to(dtype=dtype)
    for name, data in layers.items():
        old = model.get_submodule(name)
        if not isinstance(old, (nn.Linear, nn.Embedding)):
            raise TypeError(f"Unsupported compressed module: {name}")
        if "W" in data:
            weight = data["W"]
        elif isinstance(old, nn.Embedding):
            weight = (
                data["A"].float() @ data["B"].float() + data["S"].to_dense().float()
            )
        else:
            new = LRSLinear(
                data["A"].to(device=device, dtype=dtype),
                data["B"].to(device=device, dtype=dtype),
                data["S"].to(device=device, dtype=dtype),
                in_features=old.in_features,
                out_features=old.out_features,
                bias=old.bias.detach().to(device=device, dtype=dtype)
                if old.bias is not None
                else None,
                block_p=data.get("block_p"),
                block_q=data.get("block_q"),
                nm_n=data.get("nm_n"),
                nm_m=data.get("nm_m"),
            )
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, new)
            continue
        if weight.shape != old.weight.shape:
            raise ValueError(f"Weight shape mismatch: {name}")
        with torch.no_grad():
            old.weight.copy_(weight.to(dtype=dtype))
    return model.to(device=device).eval(), payload["model_config"], payload["meta"]
