from __future__ import annotations
import torch
import tilelang
import tilelang.language as T

_TORCH_TO_TL_DTYPE = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_TILELANG_JIT = tilelang.jit(
    target="cuda", pass_configs={"tl.disable_safe_memory_legalize": True}
)
(_NNZ, _ACTIVE_NB) = T.dynamic("NNZ, ACTIVE_NB")


def _pick_warp_tiles(block_p: int, block_q: int) -> tuple[int, int, int]:
    WMMA = 16
    BM = max(WMMA, ((2 * block_p - 1) // WMMA + 1) * WMMA)
    (best_wm, best_wn, best_area) = (WMMA, WMMA, 0)
    for floor in (4, 1):
        for wm in range(WMMA, BM + 1, WMMA):
            for wn in range(WMMA, block_p + 1, WMMA):
                if BM % wm or block_p % wn:
                    continue
                if BM // wm * (block_p // wn) < floor:
                    continue
                area = wm * wn
                if area > best_area or (area == best_area and wm > best_wm):
                    (best_wm, best_wn, best_area) = (wm, wn, area)
        if best_area:
            return (BM, best_wm, best_wn)
    return (BM, best_wm, best_wn)


def get_tuned_config(block_p: int, block_q: int) -> dict:
    (BM, wm, wn) = _pick_warp_tiles(block_p, block_q)
    return {"BM": BM, "WARP_TILE_M": wm, "WARP_TILE_N": wn}


def _bsr_body(
    out,
    x,
    crow,
    col,
    values,
    active_rows,
    block_p,
    block_q,
    NB,
    NB_K,
    BM=32,
    WARP_TILE_M=32,
    WARP_TILE_N=16,
    dtype="float16",
):
    (M, N, K) = T.const("M, N, K")
    threads = BM // WARP_TILE_M * (block_p // WARP_TILE_N) * 32
    out: T.Tensor((M, N), dtype)
    x: T.Tensor((M, K), dtype)
    crow: T.Tensor((NB + 1,), T.int32)
    col: T.Tensor((_NNZ,), T.int32)
    values: T.Tensor((_NNZ, block_q, block_p), dtype)
    active_rows: T.Tensor((_ACTIVE_NB,), T.int32)
    with T.Kernel(_ACTIVE_NB, T.ceildiv(M, BM), threads=threads) as (by, bx):
        x_sh = T.alloc_shared((BM, block_q), dtype)
        w_sh = T.alloc_shared((block_q, block_p), dtype)
        acc = T.alloc_fragment((BM, block_p), T.float32)
        actual_by = active_rows[by]
        row_start = crow[actual_by]
        row_end = crow[actual_by + 1]
        for i, j in T.Parallel(BM, block_p):
            acc[i, j] = out[bx * BM + i, actual_by * block_p + j]
        for ji in range(row_end - row_start):
            col_j = col[row_start + ji]
            T.assume(col_j >= 0)
            T.assume(col_j < NB_K)
            T.copy(x[bx * BM, col_j * block_q], x_sh)
            T.copy(values[row_start + ji, 0, 0], w_sh)
            T.gemm(x_sh, w_sh, acc, transpose_B=False)
        for i, j in T.Parallel(BM, block_p):
            out[bx * BM + i, actual_by * block_p + j] = acc[i, j]


_body_bsr = _TILELANG_JIT(_bsr_body)


def _tilelang_bsr_addmm_inplace_dense_impl(
    out_inplace: torch.Tensor,
    x: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    active_rows: torch.Tensor,
) -> None:
    block_q = int(values.shape[1])
    block_p = int(values.shape[2])
    NB = int(crow_indices.shape[0]) - 1
    NB_K = int(x.shape[1]) // block_q
    dtype_str = _TORCH_TO_TL_DTYPE.get(out_inplace.dtype)
    if dtype_str is None:
        raise TypeError(
            f"TileLang BSR kernel supports float16/bfloat16, got {out_inplace.dtype}"
        )
    if dtype_str == "bfloat16":
        (major, _) = torch.cuda.get_device_capability(out_inplace.device)
        if major < 8:
            raise RuntimeError(
                f"TileLang BSR kernel: bfloat16 MMA requires sm_80+ (Ampere), but device has sm_{major}x. Use float16 on this GPU."
            )
    cfg = get_tuned_config(block_p, block_q)
    _body_bsr(
        out_inplace,
        x,
        crow_indices,
        col_indices,
        values,
        active_rows,
        block_p,
        block_q,
        NB,
        NB_K,
        BM=cfg["BM"],
        WARP_TILE_M=cfg["WARP_TILE_M"],
        WARP_TILE_N=cfg["WARP_TILE_N"],
        dtype=dtype_str,
    )


_lib = torch.library.Library("ssalaad", "DEF")
_lib.define(
    "tilelang_bsr_addmm_inplace_dense_(  Tensor(a!) out_inplace, Tensor x,   Tensor crow_indices, Tensor col_indices, Tensor values, Tensor active_rows) -> ()"
)
_lib.impl(
    "tilelang_bsr_addmm_inplace_dense_", _tilelang_bsr_addmm_inplace_dense_impl, "CUDA"
)


@torch.library.register_fake("ssalaad::tilelang_bsr_addmm_inplace_dense_")
def _fake(out_inplace, x, crow_indices, col_indices, values, active_rows) -> None:
    return None
