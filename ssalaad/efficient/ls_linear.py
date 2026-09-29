from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LRSLinearBase(nn.Module):
    kind: str = ""

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        (n, r) = A.shape
        (rB, m) = B.shape
        if rB != r:
            raise ValueError(f"A.shape[1]={r} must match B.shape[0]={rB}")
        if (n, m) != (out_features, in_features):
            raise ValueError(
                f"L = A·B shape ({n},{m}) must equal (out_features={out_features}, in_features={in_features})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.r = int(r)
        self.register_buffer("B", B.contiguous())
        self.register_buffer("A", A.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.bias = None

    def _lr_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.B), self.A)

    def _maybe_add_bias_(self, y: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            y.add_(self.bias.to(y.dtype))
        return y


class _LRSLinearLR(_LRSLinearBase):
    kind = "lr_only"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._maybe_add_bias_(self._lr_forward(x))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.r}, S=zero (skipped), bias={self.bias is not None}"


class _LRSLinearLRBlockBase(_LRSLinearBase):
    """Block-sparse S in BSR layout; subclasses supply the kernel.

    Both kernels accumulate S·x into the dense low-rank output in place, so the
    sparse path never allocates a second buffer.
    """

    kind = "block_sparse"

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        S_sparse: torch.Tensor,
        *,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None,
        block_p: int,
        block_q: int,
    ) -> None:
        super().__init__(
            A, B, in_features=in_features, out_features=out_features, bias=bias
        )
        if tuple(S_sparse.shape) != (out_features, in_features):
            raise ValueError(
                f"S shape {tuple(S_sparse.shape)} must equal "
                f"(out_features, in_features)=({out_features},{in_features})"
            )
        self.block_p = int(block_p)
        self.block_q = int(block_q)
        self.S_nnz = int(S_sparse._nnz())
        crow = S_sparse.crow_indices()
        self.register_buffer("S_crow", crow.to(torch.int32).clone())
        self.register_buffer("S_col", S_sparse.col_indices().to(torch.int32).clone())
        self.register_buffer(
            "S_values", S_sparse.values().transpose(1, 2).contiguous().clone()
        )
        # Block-rows holding no block are skipped outright by both kernels.
        active = (crow[1:] - crow[:-1]).nonzero(as_tuple=False).squeeze(-1)
        self.register_buffer("S_active_rows", active.to(torch.int32).clone())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.r}, S_nnz={self.S_nnz}, "
            f"block=({self.block_p}×{self.block_q}), bias={self.bias is not None}"
        )


class _LRSLinearLRBlockCUDA(_LRSLinearLRBlockBase):
    """TileLang block-sparse kernel, on Tensor Cores."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.S_values.device.type != "cuda":
            raise ValueError("The TileLang block kernel requires CUDA.")
        if self.block_p % 16 or self.block_q % 16:
            raise ValueError(
                "The TileLang block kernel requires block dimensions divisible by 16."
            )
        if self.S_values.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("The TileLang block kernel requires fp16 or bf16.")
        from ssalaad.tilelang_sparse.addmm_inplace import get_tuned_config

        self._row_multiple = get_tuned_config(self.block_p, self.block_q)["BM"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        rows = x_2d.shape[0]
        padding = (-rows) % self._row_multiple
        if padding:
            x_2d = F.pad(x_2d, (0, 0, 0, padding))
        y = F.linear(F.linear(x_2d, self.B), self.A)
        torch.ops.ssalaad.tilelang_bsr_addmm_inplace_dense_(
            y, x_2d, self.S_crow, self.S_col, self.S_values, self.S_active_rows
        )
        return self._maybe_add_bias_(
            y[:rows].reshape(*orig_shape[:-1], self.out_features)
        )


class _LRSLinearLRBlockCPU(_LRSLinearLRBlockBase):
    """AVX2/FMA block-sparse kernel, float32."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from ssalaad.cpu_sparse import SUPPORTED_BLOCK_SIZES

        if self.block_p != self.block_q or self.block_p not in SUPPORTED_BLOCK_SIZES:
            raise ValueError(
                f"The AVX2 block kernel needs square blocks of size "
                f"{SUPPORTED_BLOCK_SIZES}, got {self.block_p}x{self.block_q}."
            )
        if self.S_values.dtype != torch.float32:
            raise ValueError("The AVX2 block kernel requires fp32.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from ssalaad.cpu_sparse import bsr_addmm_avx2_

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        y = F.linear(F.linear(x_2d, self.B), self.A).contiguous()
        bsr_addmm_avx2_(
            y, x_2d, self.S_crow, self.S_col, self.S_values, self.S_active_rows
        )
        return self._maybe_add_bias_(y.reshape(*orig_shape[:-1], self.out_features))


class _LRSLinearLRNM(_LRSLinearBase):
    kind = "nm"

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        S: torch.Tensor,
        *,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            A, B, in_features=in_features, out_features=out_features, bias=bias
        )
        self.S_nnz = S.packed.numel()
        self._nm_shape: tuple[int, int] = (out_features, in_features)
        self._nm_semi_cls = type(S)
        self.register_buffer("_nm_packed", S.packed.contiguous(), persistent=False)
        meta = S.meta
        self.register_buffer(
            "_nm_meta",
            meta.contiguous() if meta is not None else None,
            persistent=False,
        )
        self.register_buffer("_S_semi", None, persistent=False)

    def _apply(self, fn, recurse: bool = True) -> "_LRSLinearLRNM":
        result = super()._apply(fn, recurse=recurse)
        if result._nm_packed is not None and result._nm_packed.is_cuda:
            result._S_semi = result._nm_semi_cls(
                torch.Size(result._nm_shape),
                packed=result._nm_packed,
                meta=result._nm_meta,
                packed_t=None,
                meta_t=None,
                compressed_swizzled_bitmask=None,
            )
            result._nm_packed = None
            result._nm_meta = None
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        if self._S_semi is None:
            raise RuntimeError(
                "N:M layer has no semi-structured S — call .to('cuda') first"
            )
        latent = F.linear(x_2d, self.B)
        y = F.linear(x_2d, self._S_semi)
        y.addmm_(latent, self.A.t())
        y = y.reshape(*orig_shape[:-1], self.out_features)
        return self._maybe_add_bias_(y)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.r}, S_nnz={self.S_nnz}, N:M semi-structured, bias={self.bias is not None}"


def LRSLinear(
    A,
    B,
    S_sparse,
    *,
    in_features,
    out_features,
    bias=None,
    block_p=None,
    block_q=None,
    nm_n=None,
    nm_m=None,
):
    """Compute x @ (A @ B + S).T with the dedicated sparse kernels.

    Block-sparse S runs on the AVX2 kernel (CPU) or the TileLang kernel (CUDA);
    N:M S runs on the packed 2:4 kernel (CUDA only). There is no dense or CSR
    fallback, so a pattern the kernels cannot run is rejected rather than
    silently served slowly.
    """
    if S_sparse.layout == torch.strided:
        nnz = int(torch.count_nonzero(S_sparse))
    else:
        nnz = S_sparse._nnz()
    kwargs = dict(in_features=in_features, out_features=out_features, bias=bias)
    if nnz == 0:
        return _LRSLinearLR(A, B, **kwargs)
    if nm_n is not None:
        if (nm_n, nm_m) != (2, 4) or not A.is_cuda:
            raise ValueError("N:M inference requires CUDA and a 2:4 mask.")
        from ssalaad.efficient.artifact import _dense_to_nm_packed_cpu

        packed = _dense_to_nm_packed_cpu(S_sparse.to_dense())
        return _LRSLinearLRNM(A.cpu(), B.cpu(), packed, **kwargs).to(A.device)
    if block_p is None or block_q is None:
        raise ValueError("Sparse S needs a recorded block shape to pick a kernel.")
    if A.is_cuda:
        return _LRSLinearLRBlockCUDA(
            A, B, S_sparse, block_p=block_p, block_q=block_q, **kwargs
        )
    from ssalaad.cpu_sparse import SUPPORTED_BLOCK_SIZES, is_available

    if block_p != block_q or block_p not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"The AVX2 kernel supports square {SUPPORTED_BLOCK_SIZES} blocks, "
            f"got ({block_p}, {block_q})."
        )
    if not is_available():
        raise ValueError(
            "The AVX2 block kernel failed to build; a C++ toolchain is required."
        )
    return _LRSLinearLRBlockCPU(
        A.float(),
        B.float(),
        S_sparse.float(),
        block_p=block_p,
        block_q=block_q,
        **{**kwargs, "bias": None if bias is None else bias.float()},
    )
