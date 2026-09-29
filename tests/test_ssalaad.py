"""One test file covering the three stages: training, EGAR, inference.

Run with `python -m pytest -q`. These checks validate functionality, not
full-scale training results.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ssalaad.cpu_sparse import SUPPORTED_BLOCK_SIZES, bsr_addmm_avx2_, is_available
from ssalaad.efficient.artifact import (
    dense_to_bsr,
    factor_from_svd,
    load_artifact,
    save_artifact,
)
from ssalaad.efficient.ls_linear import LRSLinear
from ssalaad.efficient.model import build_efficient_model
from ssalaad.inference.egar import (
    _apply_block_override,
    _layer_param_count,
    _parse_block_size,
    _project_S,
    _project_S_nm,
    stage1_allocate,
)
from ssalaad.inference.generate import generate
from ssalaad.train.model import build_model_from_config

# A one-layer model built inline, so the tests stay fast and depend on no
# shipped config. The recipes under train/configs/model/ are the real ones.
TINY = {
    "model_type": "llama",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "tie_word_embeddings": False,
}


def build_tiny():
    return build_model_from_config(TINY)
from ssalaad.train.ssalaad.context import SSalaadContext
from ssalaad.train.ssalaad.solver import SSalaadSolver, _isvd_step

# ---------------------------------------------------------------- training --


def _solver(buffer_dtype=torch.float32, *, shape=(64, 96), seed=0, **kw):
    torch.manual_seed(seed)
    w = nn.Parameter(torch.randn(*shape) * 0.3)
    return SSalaadSolver(
        w,
        rho=kw.pop("rho", 1e-6),
        target_rank_ratio=kw.pop("target_rank_ratio", 0.15),
        target_density=kw.pop("target_density", 0.05),
        gamma=0.999,
        delta_alpha=0.2,
        delta_beta=0.002,
        buffer_dtype=buffer_dtype,
        **kw,
    )


def _admm_step(s):
    SSalaadContext({"s": s}).run_admm_round()
    return s._last_stats


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_solver_buffers_keep_their_storage_dtype(dtype):
    s = _solver(dtype)
    assert s.M.dtype == s.S.dtype == s.Y_div_rho.dtype == dtype
    assert torch.equal(s.S, torch.zeros_like(s.S))  # S starts at 0
    _admm_step(s)
    assert s.M.dtype == s.S.dtype == s.Y_div_rho.dtype == dtype


def test_bf16_solver_tracks_an_fp32_reference():
    fp32, bf16 = _solver(torch.float32, seed=123), _solver(torch.bfloat16, seed=123)
    _admm_step(fp32)
    _admm_step(bf16)
    for a, b in ((fp32.M, bf16.M), (fp32.S, bf16.S), (fp32.Y_div_rho, bf16.Y_div_rho)):
        rel = ((a.float() - b.float()).norm() / a.float().norm().clamp(min=1e-12)).item()
        assert rel < 0.05


def test_solver_state_survives_a_save_load_roundtrip():
    s = _solver(torch.bfloat16)
    _admm_step(s)
    restored = _solver(torch.float32)
    restored.load_state_dict(s.state_dict())
    assert restored.M.dtype == torch.float32
    torch.testing.assert_close(restored.S.float(), s.S.float())


def test_coupled_penalty_reaches_the_weight():
    s = _solver()
    s.coupled_penalty().backward()
    assert s.w_param.grad is not None and torch.isfinite(s.w_param.grad).all()


def test_nm_beta_is_per_group_and_thresholds_per_group():
    s = _solver(
        shape=(16, 16),
        nm_n=2,
        nm_m=4,
        target_density=0.5,
        target_rank_ratio=0.5,
        rho=1.0,
    )
    assert s.beta.shape == (16, 4) and s.alpha.ndim == 0
    s.beta[:, 0:2] = 1e-9  # keep these groups
    s.beta[:, 2:4] = 1e9  # prune these
    _, S, _ = s._ls_compute()
    assert int((S[:, 0:8] != 0).sum()) > 0
    assert int((S[:, 8:16] != 0).sum()) == 0


@pytest.mark.parametrize("shape,transposed", [((64, 32), False), ((32, 64), True)])
def test_isvd_converges_to_the_true_singular_values(shape, transposed):
    """Repeated orthogonal iteration recovers the exact SVD."""
    n, m = shape
    sv = [10.0, 5.0, 2.0, 1.0]
    torch.manual_seed(0)
    U = torch.linalg.qr(torch.randn(n, len(sv)))[0]
    V = torch.linalg.qr(torch.randn(m, len(sv)))[0]
    M = U * torch.tensor(sv) @ V.T
    Q = torch.eye(n if transposed else m)
    for _ in range(40):
        U_k, s_k, Vh_k, Q = _isvd_step(M, Q, transposed=transposed)
    ranked = s_k.sort(descending=True).values
    for i, true in enumerate(sv):
        assert abs(ranked[i].item() - true) / true < 1e-3
    assert ranked[len(sv)].item() < 1e-4  # no leakage into trailing directions
    assert ((M - U_k @ torch.diag(s_k) @ Vh_k).norm() / M.norm()).item() < 1e-4


# -------------------------------------------------------------------- EGAR --


def _sparse_layer(n=64, m=32, nnz=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.zeros(n, m)
    S.view(-1)[torch.randperm(n * m, generator=g)[:nnz]] = torch.randn(nnz, generator=g)
    info = {
        "block_p": 1,
        "block_q": 1,
        "nm_n": None,
        "nm_m": None,
        "last_rank_ratio": 1.0,
        "last_density": nnz / (n * m),
    }
    return torch.randn(n, m, generator=g), S, info


def test_parse_block_size():
    assert _parse_block_size(None) is None
    assert _parse_block_size([1, 1]) == (1, 1)  # element-wise is the (1,1) case
    assert _parse_block_size([16, 16]) == (16, 16)


def test_block_override_rescores_density():
    _, S, info = _sparse_layer()
    _apply_block_override({"l": S}, {"l": info}, (8, 8))
    assert (info["block_p"], info["block_q"]) == (8, 8)
    # Scattered nonzeros light up more capacity once charged by whole blocks.
    assert info["last_density"] > 40 / (64 * 32)
    info.update(block_p=8, block_q=8, last_density=0.5)
    _apply_block_override({"l": S}, {"l": info}, (1, 1))
    assert info["last_density"] == float((S != 0).float().mean())


def test_block_override_falls_back_on_indivisible_shape():
    _, S, info = _sparse_layer(n=64, m=30)
    _apply_block_override({"l": S}, {"l": info}, (8, 8))
    assert (info["block_p"], info["block_q"]) == (1, 1)


def test_stage1_meets_the_budget_in_whole_blocks():
    L_buf, S_buf, layer_info = {}, {}, {}
    for i in range(2):
        L_buf[f"l{i}"], S_buf[f"l{i}"], layer_info[f"l{i}"] = _sparse_layer(seed=i)
    _apply_block_override(S_buf, layer_info, (8, 8))
    P_rest, P_tgt = 1000, 4000
    N_L, N_S, P = stage1_allocate(L_buf, S_buf, layer_info, P_tgt, P_rest)
    assert P < P_tgt
    assert P == P_rest + sum(
        _layer_param_count(N_L[n], N_S[n], *L_buf[n].shape, 8, 8) for n in L_buf
    )
    for name in L_buf:
        n, m = L_buf[name].shape
        blk = S_buf[name].reshape(n // 8, 8, m // 8, 8).permute(0, 2, 1, 3)
        assert int((blk.abs().amax(dim=(2, 3)) > 0).sum()) <= N_S[name]


def test_stage1_rejects_a_budget_below_the_structural_floor():
    L, S, info = _sparse_layer()
    with pytest.raises(SystemExit, match="structural floor"):
        stage1_allocate({"l": L}, {"l": S}, {"l": info}, 500, 1000)


def test_stage2_projections_keep_the_largest_blocks_and_groups():
    M = torch.zeros(8, 8)
    M[0:4, 0:4] = 3.0
    M[4:8, 4:8] = 1.0
    kept = _project_S(M, 1, 4, 4)
    assert torch.equal(kept[0:4, 0:4], M[0:4, 0:4])
    assert float(kept[4:8, 4:8].abs().max()) == 0.0
    R = torch.tensor([[4.0, 1.0, 3.0, 2.0, 0.5, 9.0, 0.1, 7.0]])
    assert _project_S_nm(R, 2, 4)[0].tolist() == [4, 0, 3, 0, 0, 9, 0, 7]


# --------------------------------------------------------------- inference --


@pytest.mark.parametrize("layout", ["bsr", "zero", "dense"])
def test_bundle_reproduces_the_dense_model(tmp_path, layout):
    torch.manual_seed(7)
    config = TINY
    model = build_tiny().eval()
    name = "model.layers.0.self_attn.q_proj"
    A, B = torch.randn(32, 4) * 0.01, torch.randn(4, 32) * 0.01
    S = torch.zeros(32, 32)
    if layout != "zero":
        S[:16, :16] = torch.randn(16, 16) * 0.01
    with torch.no_grad():
        model.get_submodule(name).weight.copy_(A @ B + S)
    ids = torch.tensor([[1, 3, 4, 5]])
    expected = model(input_ids=ids).logits
    if layout == "dense":
        layer = {"W": (A @ B + S).to(torch.bfloat16)}
    else:
        layer = {
            "A": A.to(torch.bfloat16),
            "B": B.to(torch.bfloat16),
            "S": dense_to_bsr(S.to(torch.bfloat16), 16, 16),
            "block_p": 16,
            "block_q": 16,
        }
    bundle = tmp_path / "model.pt"
    save_artifact(
        bundle, {name: layer}, {"mode": "egar"}, model=model, model_config=config
    )
    restored, _, meta = build_efficient_model(bundle)
    assert meta == {"mode": "egar"}
    torch.testing.assert_close(restored(input_ids=ids).logits, expected, atol=0.1, rtol=0.1)


def test_bundle_drops_unlisted_metadata(tmp_path):
    """Nothing about the packaging machine may leak into a bundle."""
    model = build_tiny()
    bundle = tmp_path / "model.pt"
    save_artifact(
        bundle,
        {},
        {"mode": "egar", "trace": "/private/source", "hostname": "private-host"},
        model=model,
        model_config=TINY,
    )
    assert load_artifact(bundle)["meta"] == {"mode": "egar"}


def test_factorization_preserves_a_low_rank_matrix():
    torch.manual_seed(0)
    matrix = torch.randn(16, 3) @ torch.randn(3, 32)
    svd = torch.linalg.svd(matrix, full_matrices=False)
    A, B = factor_from_svd(*svd, 3)
    torch.testing.assert_close(A @ B, matrix, atol=1e-4, rtol=1e-4)
    empty_A, empty_B = factor_from_svd(*svd, 0)  # an all-pruned layer
    assert empty_A.shape == (16, 0) and empty_B.shape == (0, 32)


def test_generation_appends_the_requested_number_of_tokens():
    model = build_tiny().eval()
    ids = torch.tensor([[1, 3, 4, 5]])
    assert generate(model, ids, 4).shape == (1, 8)
    with pytest.raises(ValueError):
        generate(model, ids, 10_000)  # past the context limit


# ------------------------------------------------------------ AVX2 kernel --


def _block_sparse(n, m, block, coords, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.zeros(n, m)
    for br, bc in coords:
        S[br * block : (br + 1) * block, bc * block : (bc + 1) * block] = torch.randn(
            block, block, generator=g
        )
    return S


def _avx2_addmm(out, x, S, block):
    bsr = S.to_sparse_bsr(blocksize=(block, block))
    crow = bsr.crow_indices().to(torch.int32)
    bsr_addmm_avx2_(
        out,
        x.contiguous(),
        crow,
        bsr.col_indices().to(torch.int32),
        bsr.values().transpose(1, 2).contiguous(),
        (crow[1:] - crow[:-1]).nonzero().squeeze(-1).to(torch.int32),
    )


@pytest.mark.skipif(not is_available(), reason="AVX2 kernel did not build")
@pytest.mark.parametrize("block", SUPPORTED_BLOCK_SIZES)
def test_avx2_kernel_matches_a_dense_matmul(block):
    S = _block_sparse(block * 4, block * 6, block, [(0, 1), (1, 0), (1, 3), (2, 5), (3, 2)])
    x = torch.randn(5, block * 6)
    out = torch.zeros(5, block * 4)
    _avx2_addmm(out, x, S, block)
    torch.testing.assert_close(out, x @ S.T, rtol=0, atol=1e-4)


@pytest.mark.skipif(not is_available(), reason="AVX2 kernel did not build")
def test_avx2_kernel_skips_empty_block_rows():
    block = 16
    S = _block_sparse(block * 3, block * 3, block, [(2, 0)])
    x = torch.randn(3, block * 3)
    out = torch.full((3, block * 3), 7.0)
    _avx2_addmm(out, x, S, block)
    assert torch.equal(out[:, : block * 2], torch.full((3, block * 2), 7.0))


def test_block_kernel_rejects_unsupported_block_shape():
    """Only square 8/16/32/64 blocks have a kernel; anything else must fail loudly."""
    torch.manual_seed(0)
    A, B = torch.randn(64, 8), torch.randn(8, 96)
    S = _block_sparse(64, 96, 8, [(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="square"):
        LRSLinear(
            A,
            B,
            S.to_sparse_bsr(blocksize=(8, 8)),
            in_features=96,
            out_features=64,
            block_p=8,
            block_q=16,
        )
