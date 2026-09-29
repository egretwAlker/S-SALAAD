from __future__ import annotations
from pathlib import Path
from typing import Any
import torch


def factor_from_svd(
    U: torch.Tensor, sigma: torch.Tensor, Vh: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-``rank`` factors of ``U diag(sigma) Vh``, with sqrt(sigma) in each.

    Splitting the singular values evenly keeps ``A`` and ``B`` on the same
    scale, which matters once they are stored in bf16.
    """
    r = min(rank, sigma.shape[0])
    sqrt_sigma = sigma[:r].sqrt()
    return (U[:, :r] * sqrt_sigma, sqrt_sigma.unsqueeze(1) * Vh[:r, :])


def dense_to_bsr(S: torch.Tensor, block_p: int, block_q: int) -> torch.Tensor:
    (n, m) = S.shape
    if n % block_p or m % block_q:
        raise ValueError(
            f"shape ({n}, {m}) not divisible by block ({block_p}, {block_q})"
        )
    (rb, cb) = (n // block_p, m // block_q)
    blocks = S.reshape(rb, block_p, cb, block_q).permute(0, 2, 1, 3).contiguous()
    block_l1 = blocks.reshape(rb, cb, -1).abs().sum(dim=-1)
    mask = block_l1 > 0
    nz_per_row = mask.sum(dim=1).to(torch.int64)
    crow_indices = torch.cat(
        [torch.zeros(1, dtype=torch.int64, device=S.device), nz_per_row.cumsum(dim=0)]
    )
    rc = torch.nonzero(mask, as_tuple=False)
    col_indices = rc[:, 1].to(torch.int64)
    values = blocks[mask].contiguous()
    return torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(n, m))


def dense_to_csr(S: torch.Tensor) -> torch.Tensor:
    return S.to_sparse_csr()


def _dense_to_nm_packed_cpu(S: torch.Tensor) -> torch.Tensor:
    from torch.sparse import to_sparse_semi_structured

    s_cuda = S.to(torch.bfloat16).contiguous().cuda()
    semi = to_sparse_semi_structured(s_cuda)
    cls = type(semi)
    packed_cpu = semi.packed.cpu()
    meta_cpu = semi.meta.cpu() if semi.meta is not None else None
    result = cls(
        S.shape,
        packed=packed_cpu,
        meta=meta_cpu,
        packed_t=None,
        meta_t=None,
        compressed_swizzled_bitmask=None,
    )
    del s_cuda, semi
    return result


def encode_s_for_artifact(
    S: torch.Tensor,
    *,
    block_p: int | None = None,
    block_q: int | None = None,
    nm_n: int | None = None,
    nm_m: int | None = None,
) -> torch.Tensor:
    if block_p is not None and block_q is not None and (block_p, block_q) != (1, 1):
        return dense_to_bsr(S, int(block_p), int(block_q))
    # Element-wise (1, 1) and N:M both store their live entries as CSR.
    return dense_to_csr(S)


def save_artifact(
    path: str | Path,
    layers: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    *,
    model: torch.nn.Module,
    model_config: dict,
) -> None:
    """Save compressed layers and every remaining model tensor in one bundle."""
    from ssalaad.train.model import export_model_config

    allowed_meta = {"mode", "param_count", "p_tgt_million"}
    compressed_keys = {name + ".weight" for name in layers}
    base_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
        if key not in compressed_keys
    }
    payload = {
        "format_version": 1,
        "layers": layers,
        "base_state_dict": base_state,
        "model_config": export_model_config(model_config),
        "meta": {key: value for key, value in meta.items() if key in allowed_meta},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a standalone bundle; legacy partial artifacts are rejected."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "layers", "base_state_dict", "model_config", "meta"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(
            "Expected a complete model bundle produced by the compression entry point."
        )
    if payload["format_version"] != 1:
        raise ValueError("Unsupported model bundle version.")
    return payload
