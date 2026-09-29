"""EGAR — energy-guided alternating refinement.

Compresses a trained checkpoint to a target parameter budget with no
calibration data and no further training, preserving the sparsity structure
induced during pretraining.

Stage I  (`stage1_allocate`)  picks, per projection, how many singular
directions ``N_L`` and how many sparse blocks ``N_S`` to keep, by raising a
shared per-projection threshold ``tau`` over the energy scores until the model
fits the budget.

Stage II (`stage2_refine`)  alternately re-projects ``L`` and ``S`` onto those
capacities to reduce the reconstruction error ``||X - L - S||_F``, with the
dense weight ``X`` as the anchor.

Run it with ``python -m ssalaad.compress --ckpt ... --p-tgt ...``.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ssalaad import OUTPUT_ROOT
from ssalaad.efficient.artifact import (
    encode_s_for_artifact,
    factor_from_svd,
    save_artifact,
)
from ssalaad.inference.evaluate import evaluate, make_eval_loader
from ssalaad.train.model import build_model, read_model_config

# Stage I minimum base step epsilon (paper Fig. 2).
_EPSILON = 1e-8
# Stage I is a threshold search, not an optimizer; this only bounds a runaway.
_MAX_SEARCH_ROUNDS = 10000
# Stage II iteration cap (it normally stops when the error plateaus).
_MAX_ITERS = 32
# Fixes streaming and evaluation order.
_SEED = 42


def _parse_block_size(spec: list[int] | None) -> tuple[int, int] | None:
    """None means "keep the checkpoint's own granularity"; (1, 1) is element-wise."""
    if spec is None:
        return None
    return max(1, int(spec[0])), max(1, int(spec[1]))


@torch.no_grad()
def _apply_block_override(
    S_buf: dict[str, torch.Tensor],
    layer_info: dict[str, dict],
    block_size: tuple[int | None, int | None],
) -> None:
    """Re-score a trained S at a different granularity.

    Lets an unstructured checkpoint be compressed as if it were block-sparse.
    """
    bp, bq = block_size
    for name, info in layer_info.items():
        S = S_buf[name]
        n, m = S.shape
        lbp, lbq = bp, bq
        if n % lbp or m % lbq:
            lbp = lbq = 1
        info["block_p"], info["block_q"] = lbp, lbq
        info["nm_n"] = info["nm_m"] = None
        blk = S.reshape(n // lbp, lbp, m // lbq, lbq).permute(0, 2, 1, 3)
        info["last_density"] = float((blk.abs().amax(dim=(2, 3)) > 0).float().mean())


def _layer_param_count(n_rank: int, n_blk: int, n: int, m: int, bp: int, bq: int) -> int:
    """Stored parameters of one projection: r(n+m) for L plus the live S entries.

    Element-wise sparsity is the (1, 1) case, so one formula covers every
    pattern: block (p, q), row (1, m), column (n, 1) and element-wise (1, 1).
    """
    return n_rank * (n + m) + n_blk * bp * bq


@dataclass
class _Scores:
    """Per-projection energy scores that Stage I thresholds."""

    U: torch.Tensor
    sigma: torch.Tensor
    energy_L: torch.Tensor  # E_j^L = sigma_j^2 / (n + m)
    Vh: torch.Tensor
    n: int
    m: int
    n_sv: int  # candidate directions, i.e. the checkpoint's own retained rank
    # E_d^S = ||S^d||_F^2 / (p q), or None for N:M, whose capacity is fixed.
    energy_S: torch.Tensor | None
    n_blk_total: int


@torch.no_grad()
def _score_projections(
    L_buf: dict[str, torch.Tensor],
    S_buf: dict[str, torch.Tensor],
    layer_info: dict[str, dict],
) -> dict[str, _Scores]:
    """SVD every L, score every S block, and mask both down to the operating
    point the checkpoint was trained at."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scores: dict[str, _Scores] = {}
    # Batch the SVDs by shape — same-shaped projections are the common case.
    by_shape: dict[tuple[int, int], list[str]] = defaultdict(list)
    for name, L in L_buf.items():
        by_shape[L.shape[0], L.shape[1]].append(name)
    for (n, m), names in by_shape.items():
        batch = torch.stack([L_buf[x].float() for x in names]).to(device)
        U_b, sigma_b, Vh_b = torch.linalg.svd(batch, full_matrices=False)
        U_b, sigma_b, Vh_b = U_b.cpu(), sigma_b.cpu(), Vh_b.cpu()
        del batch
        for i, name in enumerate(names):
            info = layer_info[name]
            # Only the directions the checkpoint actually kept are candidates.
            n_sv = min(n, m)
            k = max(0, min(int(round((info.get("last_rank_ratio") or 0.0) * n_sv)), n_sv))
            sigma = sigma_b[i][:k]
            S = S_buf[name].float().cpu()
            bp, bq = info["block_p"], info["block_q"]
            if info.get("nm_n") is not None:
                energy_S, n_blk_total = None, 0
            else:
                # E_d^S = ||S^d||_F^2 / (pq); at (1, 1) this is exactly S_ij^2.
                blk = S.reshape(n // bp, bp, m // bq, bq).permute(0, 2, 1, 3)
                energy_S = blk.pow(2).sum(dim=(2, 3)).div(bp * bq)
                n_blk_total = energy_S.numel()
            if energy_S is not None and n_blk_total:
                # Same for S: only currently live blocks are candidates.
                keep = max(
                    0,
                    min(
                        int(round((info.get("last_density") or 0.0) * n_blk_total)),
                        n_blk_total,
                    ),
                )
                if keep < n_blk_total:
                    kth = (
                        energy_S.flatten().topk(keep, largest=True).values[-1]
                        if keep > 0
                        else float("inf")
                    )
                    energy_S[energy_S < kth] = float("-inf")
            scores[name] = _Scores(
                U=U_b[i][:, :k],
                sigma=sigma,
                energy_L=sigma.pow(2).div(n + m),
                Vh=Vh_b[i][:k],
                n=n,
                m=m,
                n_sv=k,
                energy_S=energy_S,
                n_blk_total=n_blk_total,
            )
    return scores


@torch.no_grad()
def stage1_allocate(
    L_buf: dict[str, torch.Tensor],
    S_buf: dict[str, torch.Tensor],
    layer_info: dict[str, dict],
    P_tgt: int,
    P_rest: int,
) -> tuple[dict[str, int], dict[str, int], int]:
    """Stage I: energy-guided budget allocation.

    Each projection carries its own threshold ``tau``, shared by its low-rank
    and sparse components; candidates scoring above ``tau`` are retained. All
    thresholds rise together in steps scaled by that projection's remaining
    capacity ``G_tau``, doubling until the model fits and then backing off.

    Returns ``(N_L, N_S, P)`` with ``P <= P_tgt``, and truncates ``L_buf`` and
    ``S_buf`` in place to the retained directions and blocks.
    """
    scores = _score_projections(L_buf, S_buf, layer_info)
    # N:M carries a fixed number of nonzeros that no threshold can prune.
    nm_nnz = {
        name: layer_info[name]["nm_n"] * (S_buf[name].numel() // layer_info[name]["nm_m"])
        for name in L_buf
        if layer_info[name].get("nm_n") is not None
    }

    def _budget_at(tau: dict[str, float]) -> tuple[int, dict, dict]:
        """Model size, per-projection counts, and remaining capacity G_tau."""
        P = P_rest
        counts: dict[str, tuple[int, int]] = {}
        G_tau: dict[str, float] = {}
        for name, sc in scores.items():
            info = layer_info[name]
            t = tau[name]
            n_rank = int((sc.energy_L > t).sum())
            if sc.energy_S is None:  # N:M — allocate N_L only
                cost = n_rank * (sc.n + sc.m) + nm_nnz[name]
                n_blk = 0
                G_tau[name] = n_rank / max(1, sc.n_sv)
            else:
                n_blk = int((sc.energy_S > t).sum())
                cost = _layer_param_count(
                    n_rank, n_blk, sc.n, sc.m, info["block_p"], info["block_q"]
                )
                G_tau[name] = n_rank / max(1, sc.n_sv) + n_blk / max(1, sc.n_blk_total)
            # A projection never costs more than storing it densely.
            P += min(cost, sc.n * sc.m)
            counts[name] = (n_rank, n_blk)
        return P, counts, G_tau

    P_floor, _, _ = _budget_at({name: math.inf for name in L_buf})
    if P_floor >= P_tgt:
        n_fixed = sum(1 for sc in scores.values() if sc.energy_S is None)
        fixed_note = (
            f"\n    {n_fixed} projection(s) carry a fixed N:M support that no tau can prune"
            if n_fixed
            else ""
        )
        raise SystemExit(
            f"[egar] --p-tgt {P_tgt / 1e6:.2f}M is below this checkpoint's structural "
            f"floor of {P_floor / 1e6:.2f}M.\n"
            f"    P_rest (untracked)  = {P_rest / 1e6:>8.2f}M\n"
            f"    minimum reachable P = {P_floor / 1e6:>8.2f}M{fixed_note}\n"
            f"  Raise --p-tgt above {P_floor / 1e6:.2f}M."
        )

    tau = {name: 0.0 for name in L_buf}
    last_step = {name: 0.0 for name in L_buf}
    factor = 0.0
    counts: dict[str, tuple[int, int]] = {}
    for _ in range(1, _MAX_SEARCH_ROUNDS):
        P, counts, G_tau = _budget_at(tau)
        if P < P_tgt:
            if factor <= 1:
                break
            # Overshot on a doubled step: undo it and restart from the base step.
            for name in tau:
                tau[name] -= last_step[name]
            factor = 0.0
            continue
        factor = 1.0 if factor == 0 else factor * 2.0
        for name in tau:
            step = factor * G_tau[name] * _EPSILON
            tau[name] += step
            last_step[name] = step
    else:
        raise SystemExit("[egar] Stage I did not converge.")

    # Apply the thresholds: truncate L to its retained directions, mask S.
    for name, sc in scores.items():
        info = layer_info[name]
        keep_L = sc.energy_L > tau[name]
        L_buf[name] = (
            sc.U[:, keep_L] * sc.sigma[keep_L] @ sc.Vh[keep_L]
            if int(keep_L.sum()) > 0
            else torch.zeros(sc.n, sc.m, dtype=torch.float32)
        )
        if sc.energy_S is None:
            continue
        keep_S = (sc.energy_S > tau[name]).float()
        S = S_buf[name].float().cpu()
        bp, bq = info["block_p"], info["block_q"]
        blk = S.reshape(sc.n // bp, bp, sc.m // bq, bq).permute(0, 2, 1, 3)
        S_buf[name] = (
            (blk * keep_S.unsqueeze(2).unsqueeze(3))
            .permute(0, 2, 1, 3)
            .reshape(sc.n, sc.m)
            .contiguous()
        )
    N_L = {name: c[0] for name, c in counts.items()}
    N_S = {name: c[1] for name, c in counts.items()}
    return N_L, N_S, P


@torch.no_grad()
def _project_S_nm(R: torch.Tensor, nm_n: int, nm_m: int) -> torch.Tensor:
    """Keep the ``nm_n`` largest entries in each consecutive group of ``nm_m``."""
    n, m = R.shape
    R3 = R.float().view(n, m // nm_m, nm_m)
    _, top_idx = R3.abs().topk(nm_n, dim=-1)
    mask = torch.zeros_like(R3, dtype=torch.bool)
    mask.scatter_(dim=-1, index=top_idx, value=True)
    return (R3 * mask.float()).view(n, m).contiguous()


@torch.no_grad()
def _project_S(M: torch.Tensor, n_keep: int, block_p: int, block_q: int) -> torch.Tensor:
    """Keep the ``n_keep`` blocks of largest Frobenius norm.

    At (1, 1) the blocks are single entries, so this is element-wise top-k.
    """
    M_f = M.float()
    n, m = M_f.shape
    rb, cb = n // block_p, m // block_q
    n_keep = min(max(n_keep, 0), rb * cb)
    if n_keep == 0:
        return torch.zeros_like(M_f)
    blk = M_f.reshape(rb, block_p, cb, block_q).permute(0, 2, 1, 3)
    _, order = blk.pow(2).sum(dim=(2, 3)).flatten().topk(n_keep, largest=True)
    keep = torch.zeros(rb * cb, dtype=torch.bool)
    keep[order] = True
    keep = keep.reshape(rb, cb).float()
    return (
        (blk * keep.unsqueeze(2).unsqueeze(3))
        .permute(0, 2, 1, 3)
        .reshape(n, m)
        .contiguous()
    )


def _reconstruction_error(
    L_buf: dict[str, torch.Tensor],
    S_buf: dict[str, torch.Tensor],
    X_buf: dict[str, torch.Tensor],
    dense_layers: set[str],
) -> float:
    """Normalized ``||X - L - S||_F / ||X||_F`` over the compressed projections."""
    live = [n for n in L_buf if n not in dense_layers]
    num = sum((L_buf[n] + S_buf[n] - X_buf[n]).norm().item() ** 2 for n in live)
    den = sum(X_buf[n].norm().item() ** 2 for n in live)
    return math.sqrt(num / max(den, 1e-12))


def _deploy(
    L_buf: dict[str, torch.Tensor],
    S_buf: dict[str, torch.Tensor],
    X_buf: dict[str, torch.Tensor],
    dense_layers: set[str],
    param_lookup: dict[str, torch.Tensor],
) -> None:
    """Write the current L + S back into the model's weights."""
    with torch.no_grad():
        for name in L_buf:
            p = param_lookup[name + ".weight"]
            val = X_buf[name] if name in dense_layers else L_buf[name] + S_buf[name]
            p.data.copy_(val.to(dtype=p.dtype, device=p.device))


def _log_allocation(
    L_buf: dict[str, torch.Tensor],
    N_L: dict[str, int],
    N_S: dict[str, int],
    layer_info: dict[str, dict],
    P: int,
) -> None:
    """One line: mean rank ratio and density the checkpoint had, and Stage I kept."""
    rank_kept, dens_kept, rank_ckpt, dens_ckpt = [], [], [], []
    for name, L in L_buf.items():
        n, m = L.shape
        info = layer_info[name]
        rank_kept.append(N_L[name] / min(n, m))
        rank_ckpt.append(info["last_rank_ratio"] or 0.0)
        dens_ckpt.append(info["last_density"] or 0.0)
        if info.get("nm_n") is not None:  # N:M keeps a fixed fraction
            dens_kept.append(info["nm_n"] / info["nm_m"])
        else:
            dens_kept.append(N_S[name] * info["block_p"] * info["block_q"] / (n * m))
    mean = lambda xs: sum(xs) / len(xs)
    print(
        f"[stage1] mean rank ratio {mean(rank_ckpt):.3f} -> {mean(rank_kept):.3f}, "
        f"mean density {mean(dens_ckpt):.3f} -> {mean(dens_kept):.3f}  "
        f"(checkpoint -> kept, over {len(L_buf)} projections)  P={P / 1e6:.2f}M"
    )


@torch.no_grad()
def stage2_refine(
    L_buf: dict[str, torch.Tensor],
    S_buf: dict[str, torch.Tensor],
    layer_info: dict[str, dict],
    N_L: dict[str, int],
    N_S: dict[str, int],
    dense_layers: set[str],
    model: nn.Module,
    P_rest: int,
    max_iters: int,
    device: torch.device,
) -> int:
    """Stage II: alternating refinement.

    With the dense weight ``X`` as anchor, alternate
    ``L <- Proj_rank(X - S)`` and ``S <- Proj_sparse(X - L)`` within the
    capacities Stage I allocated. Each projection minimizes ``||X - L - S||_F``
    with the other component fixed, so the error is non-increasing; we stop
    when it plateaus.
    """
    param_lookup = dict(model.named_parameters())
    nm_nnz_fixed = sum(
        layer_info[n]["nm_n"] * (S_buf[n].numel() // layer_info[n]["nm_m"])
        for n in L_buf
        if layer_info[n].get("nm_n") is not None and n not in dense_layers
    )

    def _cost(name: str) -> int:
        n, m = L_buf[name].shape
        if name in dense_layers:
            return n * m
        if layer_info[name].get("nm_n") is not None:
            return N_L[name] * (n + m)
        return _layer_param_count(
            N_L[name], N_S[name], n, m, layer_info[name]["block_p"], layer_info[name]["block_q"]
        )

    P_final = P_rest + nm_nnz_fixed + sum(_cost(name) for name in L_buf)
    for name in L_buf:
        L_buf[name] = L_buf[name].cpu().float()
        S_buf[name] = S_buf[name].cpu().float()
    X_buf = {
        n: param_lookup[n + ".weight"].detach().cpu().float().clone() for n in L_buf
    }
    by_shape: dict[tuple[int, int], list[str]] = defaultdict(list)
    for name in L_buf:
        if name not in dense_layers:
            by_shape[tuple(X_buf[name].shape)].append(name)

    iteration = 0

    def _log(tag: str) -> float:
        err = _reconstruction_error(L_buf, S_buf, X_buf, dense_layers)
        print(f"[stage2 iter {iteration}/{max_iters} | {tag}]  err={err:.4f}")
        return err

    _deploy(L_buf, S_buf, X_buf, dense_layers, param_lookup)
    prev_err = _log("init")

    def _plateaued(err: float) -> bool:
        nonlocal prev_err
        done = abs(round(err, 4) - round(prev_err, 4)) < 1.5e-4
        prev_err = err
        return done

    for iteration in range(1, max_iters + 1):
        # L <- rank-N_L truncation of (X - S), batched by shape.
        for names in by_shape.values():
            batch = torch.stack([X_buf[n] - S_buf[n] for n in names]).to(device)
            U, sigma, Vh = torch.linalg.svd(batch, full_matrices=False)
            U, sigma, Vh = U.cpu(), sigma.cpu(), Vh.cpu()
            del batch
            for i, name in enumerate(names):
                k = min(N_L[name], sigma.shape[1])
                L_buf[name] = U[i, :, :k] * sigma[i, :k] @ Vh[i, :k, :]
            del U, sigma, Vh
        _deploy(L_buf, S_buf, X_buf, dense_layers, param_lookup)
        if _plateaued(_log("after L")):
            break
        # S <- sparse projection of (X - L) within the allocated capacity.
        for name in L_buf:
            if name in dense_layers:
                continue
            info = layer_info[name]
            R = X_buf[name] - L_buf[name]
            if info.get("nm_n") is not None:
                S_buf[name] = _project_S_nm(R, info["nm_n"], info["nm_m"])
            else:
                S_buf[name] = _project_S(R, N_S[name], info["block_p"], info["block_q"])
        _deploy(L_buf, S_buf, X_buf, dense_layers, param_lookup)
        if _plateaued(_log("after S")):
            break
    else:
        return P_final
    return P_final


@torch.no_grad()
def _load_components(
    ckpt: dict, model: nn.Module, args: argparse.Namespace, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, dict]]:
    """Read L, S and their structure from a trained checkpoint."""
    L_buf: dict[str, torch.Tensor] = {}
    S_buf: dict[str, torch.Tensor] = {}
    layer_info: dict[str, dict] = {}
    if "ssalaad" not in ckpt:
        raise SystemExit(
            "[egar] checkpoint has no solver state; it must come from ssalaad.train."
        )
    param_keys = set(dict(model.named_parameters()))
    for name, sd in ckpt["ssalaad"]["solvers"].items():
        if name + ".weight" not in param_keys:
            continue
        L_buf[name] = sd["L"].float().to(device)
        S_buf[name] = sd["S"].float().to(device)
        layer_info[name] = {
            # A checkpoint trained element-wise stores no block size; (1, 1) is
            # the same thing expressed in the one parameterization.
            "block_p": sd.get("block_p") or 1,
            "block_q": sd.get("block_q") or 1,
            "nm_n": sd.get("nm_n"),
            "nm_m": sd.get("nm_m"),
            "last_rank_ratio": sd.get("last_rank_ratio"),
            "last_density": sd.get("last_density"),
        }
    if args.block_size is not None:
        _apply_block_override(S_buf, layer_info, _parse_block_size(args.block_size))
    if not L_buf:
        raise SystemExit("[egar] no tracked projections found")
    if args.no_s:  # L-only special case: S == 0
        for S in S_buf.values():
            S.zero_()
    return L_buf, S_buf, layer_info


@torch.no_grad()
def _save_bundle(
    path: str,
    L_buf: dict[str, torch.Tensor],
    S_buf: dict[str, torch.Tensor],
    layer_info: dict[str, dict],
    N_L: dict[str, int],
    dense_layers: set[str],
    model: nn.Module,
    model_config: str,
    P_final: int,
    p_tgt_million: float,
) -> None:
    """Factor each L into A @ B and write a standalone inference bundle."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    by_shape: dict[tuple, list[str]] = defaultdict(list)
    for name in L_buf:
        if name not in dense_layers:
            by_shape[tuple(L_buf[name].shape)].append(name)
    A_of: dict[str, torch.Tensor] = {}
    B_of: dict[str, torch.Tensor] = {}
    for names in by_shape.values():
        batch = torch.stack([L_buf[n].float() for n in names]).to(device)
        U, sigma, Vh = torch.linalg.svd(batch, full_matrices=False)
        for i, name in enumerate(names):
            A, B = factor_from_svd(U[i], sigma[i], Vh[i], N_L[name])
            A_of[name] = A.to(torch.bfloat16).cpu()
            B_of[name] = B.to(torch.bfloat16).cpu()
        del batch, U, sigma, Vh
    layers: dict[str, dict] = {}
    for name in L_buf:
        if name in dense_layers:
            W = model.get_submodule(name).weight
            layers[name] = {"W": W.detach().to(torch.bfloat16).cpu()}
            continue
        info = layer_info[name]
        bp, bq = info.get("block_p"), info.get("block_q")
        nm_n, nm_m = info.get("nm_n"), info.get("nm_m")
        layers[name] = {
            "A": A_of[name],
            "B": B_of[name],
            "S": encode_s_for_artifact(
                S_buf[name].to(torch.bfloat16).cpu(),
                block_p=bp,
                block_q=bq,
                nm_n=nm_n,
                nm_m=nm_m,
            ),
            "block_p": bp,
            "block_q": bq,
            "nm_n": nm_n,
            "nm_m": nm_m,
        }
    save_artifact(
        path,
        layers,
        meta={
            "mode": "egar",
            "param_count": int(P_final),
            "p_tgt_million": float(p_tgt_million),
        },
        model=model,
        model_config=read_model_config(model_config),
    )
    print(f"[egar] bundle: {path} ({len(layers)} projections)")


def _serve(bundle: str, args: argparse.Namespace) -> None:
    """Load the bundle back the way a deployment would, and score it.

    Everything after this point uses only the bundle: no checkpoint and no
    training code.
    """
    from ssalaad.efficient.model import build_efficient_model

    model, config, meta = build_efficient_model(bundle, device=args.device)
    print(
        f"[serve] {bundle} on {args.device}, "
        f"{meta['param_count'] / 1e6:.2f}M parameters"
    )
    loader = make_eval_loader(config, args.batch_size, _SEED)
    result = evaluate(model, loader, torch.device(args.device), args.val_tokens)
    print(f"[serve] val/loss={result['val/loss']:.4f} val/ppl={result['val/ppl']:.2f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Compress a trained checkpoint with EGAR, then run the result."
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Training checkpoint (checkpoint.pth), or an already-compressed bundle, "
        "which is served as-is.",
    )
    parser.add_argument("--p-tgt", type=float, help="Target parameters, in millions.")
    parser.add_argument(
        "--out",
        default=str(OUTPUT_ROOT / "model.pt"),
        help="Where to write the bundle.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N.")
    # Serving: perplexity on the corpus the model family was trained on
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-tokens", type=int, default=100_000)
    # EGAR
    parser.add_argument("--no-s", action="store_true", help="L-only: fix S = 0.")
    parser.add_argument(
        "--block-size",
        type=int,
        nargs=2,
        metavar=("P", "Q"),
        help="Re-score S of a trained checkpoint at a different granularity.",
    )
    parser.add_argument("--model-config", help="Default: model_config.json beside --ckpt.")
    args = parser.parse_args(argv)

    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "format_version" in ckpt:  # already a bundle: serve it unchanged
        del ckpt
        _serve(args.ckpt, args)
        return
    if args.p_tgt is None:
        parser.error("--p-tgt is required when compressing a training checkpoint")
    if args.model_config is None:
        local = Path(args.ckpt).parent / "model_config.json"
        if not local.is_file():
            parser.error(
                "model_config.json must sit beside the checkpoint, or pass --model-config."
            )
        args.model_config = str(local)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    print(f"[egar] loading checkpoint: {args.ckpt}")
    model = build_model(args.model_config)
    model.load_state_dict(
        {k.replace("._orig_mod.", "."): v for k, v in ckpt["model"].items()}
    )
    model.to(device=device, dtype=torch.float32)
    model.eval()

    L_buf, S_buf, layer_info = _load_components(ckpt, model, args, device)
    P_model = sum(p.numel() for p in model.parameters())
    P_rest = P_model - sum(L.numel() for L in L_buf.values())
    P_tgt = int(args.p_tgt * 1e6)
    print(
        f"[egar] tracked={len(L_buf)}  P_model={P_model / 1e6:.2f}M  "
        f"P_rest={P_rest / 1e6:.2f}M  P_tgt={args.p_tgt:.2f}M"
    )

    N_L, N_S, P = stage1_allocate(L_buf, S_buf, layer_info, P_tgt, P_rest)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log_allocation(L_buf, N_L, N_S, layer_info, P)

    # Projections whose L + S would cost more than the dense weight are stored dense.
    dense_layers = {
        name
        for name in L_buf
        if _layer_param_count(
            N_L[name],
            N_S[name],
            *L_buf[name].shape,
            layer_info[name]["block_p"],
            layer_info[name]["block_q"],
        )
        > L_buf[name].numel()
    }
    if dense_layers:
        print(f"[egar] {len(dense_layers)} projection(s) cheaper dense; storing X")

    P_final = stage2_refine(
        L_buf,
        S_buf,
        layer_info,
        N_L,
        N_S,
        dense_layers,
        model,
        P_rest,
        _MAX_ITERS,
        device,
    )
    print(f"[egar] P_final={P_final / 1e6:.2f}M")
    _save_bundle(
        args.out,
        L_buf,
        S_buf,
        layer_info,
        N_L,
        dense_layers,
        model,
        args.model_config,
        P_final,
        args.p_tgt,
    )
    del model, L_buf, S_buf
    _serve(args.out, args)


if __name__ == "__main__":
    main()
