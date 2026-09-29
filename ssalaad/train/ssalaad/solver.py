from __future__ import annotations
import torch
import torch.nn as nn


def _block_thresh(
    Z: torch.Tensor, threshold, p: int, q: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    (n, m) = Z.shape
    (rb, cb) = (n // p, m // q)
    blocks = Z.reshape(rb, p, cb, q).permute(0, 2, 1, 3)
    total_blocks = rb * cb
    norms = blocks.reshape(rb, cb, -1).norm(dim=2)
    scale = torch.clamp(1.0 - threshold / norms.clamp(min=1e-12), min=0.0)
    S_blocks = blocks * scale.unsqueeze(-1).unsqueeze(-1)
    nonzero_blocks = (scale > 0).sum()
    S = S_blocks.permute(0, 2, 1, 3).reshape(n, m).contiguous()
    return (S, nonzero_blocks, total_blocks)


def _effective_rank(s: torch.Tensor, gamma: float) -> torch.Tensor:
    """Singular directions needed to cover a fraction ``gamma`` of the energy."""
    energy = (s * s).sort(descending=True).values
    total = energy.sum()
    cum = torch.cumsum(energy, dim=0)
    idx = torch.searchsorted(cum, gamma * total.clamp(min=torch.finfo(s.dtype).tiny))
    k = torch.clamp(idx + 1, max=s.numel())
    return torch.where(total > 0, k, torch.zeros_like(k))


def _isvd_step(
    M: torch.Tensor, Q: torch.Tensor, transposed: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """I-SVD: one step of orthogonal iteration.

    Applies M^T M to the cached basis Q and re-orthogonalizes by QR, giving the
    new basis; the single product M Q then yields both remaining factors, since
    its column norms are the singular values and its normalized columns are U.
    Warm-starting from the previous basis tracks the SVD across ADMM iterations
    at a fraction of the cost of an exact one.
    """
    eps = 1e-12
    if not transposed:
        (Q_new, _) = torch.linalg.qr(M.T @ (M @ Q))
        MQ = M @ Q_new
        s = MQ.norm(dim=0)
        U = MQ / s.clamp(min=eps)
        return (U, s, Q_new.T, Q_new)
    else:
        (Q_new, _) = torch.linalg.qr(M @ (M.T @ Q))
        MTQ = M.T @ Q_new
        s = MTQ.norm(dim=0)
        U_prime = MTQ / s.clamp(min=eps)
        return (Q_new, s, U_prime.T, Q_new)


class SSalaadSolver:
    def __init__(
        self,
        w_param: nn.Parameter,
        rho: float,
        target_rank_ratio: float,
        target_density: float,
        gamma: float,
        delta_alpha: float,
        delta_beta: float,
        block_p: int = 1,
        block_q: int = 1,
        buffer_dtype: torch.dtype = torch.float32,
        nm_n: int | None = None,
        nm_m: int | None = None,
        no_s: bool = False,
    ) -> None:
        self.w_param = w_param
        self.rho = float(rho)
        self.target_rank_ratio = float(target_rank_ratio)
        self.target_density = float(target_density)
        self.gamma = float(gamma)
        self.delta_alpha = float(delta_alpha)
        self.delta_beta = float(delta_beta)
        self.block_p = block_p
        self.block_q = block_q
        self._buffer_dtype = buffer_dtype
        self.nm_n = nm_n
        self.nm_m = nm_m
        (n, m) = w_param.shape
        if n % block_p or m % block_q:
            raise ValueError(
                f"Weight shape ({n}, {m}) not divisible by block ({block_p}, {block_q})."
            )
        if nm_m is not None:
            if (block_p, block_q) != (1, 1):
                raise ValueError("block_p/block_q and nm_n/nm_m are mutually exclusive")
            if nm_n is None or nm_n <= 0 or nm_m <= 0 or (nm_n >= nm_m):
                raise ValueError(
                    f"NM mode requires 0 < nm_n < nm_m; got nm_n={nm_n}, nm_m={nm_m}"
                )
            (n, m) = w_param.shape
            if m % nm_m:
                raise ValueError(
                    f"Weight inner dim ({m}) not divisible by nm_m ({nm_m})"
                )
            if n < 16 or n % 16 != 0 or m < 16 or (m % 16 != 0):
                raise ValueError(
                    f"NM mode weight shape ({n}, {m}) violates the sparse Tensor Core (cuSPARSELt) requirement: both dims must be ≥ 16 AND multiples of 16 (Tensor Core MMA tile alignment). Use a model configuration with aligned dimensions."
                )
            if self.target_density != 0.5:
                raise ValueError(
                    f"NM mode requires target_density=0.5, got {self.target_density}."
                )
        self.alpha = torch.zeros((), device=w_param.device, dtype=torch.float32)
        if nm_m is not None:
            (n, m) = w_param.shape
            self.beta = torch.zeros(
                (n, m // nm_m), device=w_param.device, dtype=torch.float32
            )
        else:
            self.beta = torch.zeros((), device=w_param.device, dtype=torch.float32)
        with torch.no_grad():
            W = w_param.detach().float()
            (U, s, Vh) = torch.linalg.svd(W, full_matrices=False)
            k = max(1, int(len(s) * self.target_rank_ratio))
            L0 = (U[:, :k] @ torch.diag(s[:k]) @ Vh[:k, :]).contiguous()
            self.M = L0.to(buffer_dtype)
            k0 = _effective_rank(s[:k], self.gamma)
            gamma_L0 = float((k0.float() / float(min(W.shape))).item())
            # I-SVD warm start: iterate on whichever side is smaller, and seed
            # the cached basis with this exact decomposition.
            self._svd_transposed = W.shape[1] > W.shape[0]
            self._svd_Q = (U if self._svd_transposed else Vh.T).contiguous()
        self.S = torch.zeros_like(self.M)
        self.Y_div_rho = torch.zeros_like(self.M)
        self.last_rank_ratio: float = gamma_L0
        self.last_density: float = 0.0
        self._last_stats: dict = {}
        self._no_s: bool = no_s

    def coupled_penalty(self) -> torch.Tensor:
        diff = self.w_param - self.M.float()
        return 0.5 * self.rho * torch.sum(diff * diff)

    @torch.no_grad()
    def compute_stats(self) -> dict:
        L = (self.M + self.Y_div_rho - self.S).float()
        S = self.S.float()
        X = self.w_param.detach()
        (U, s, Vh) = torch.linalg.svd(L, full_matrices=False)
        k = _effective_rank(s, self.gamma)
        gamma_L = k / min(L.shape)
        upsilon_S_controller: torch.Tensor | None = None
        if self.nm_m is not None:
            nonzero_count = (S != 0).sum()
            total_count = S.numel()
            (n_dim, m_dim) = S.shape
            S3 = S.view(n_dim, m_dim // self.nm_m, self.nm_m)
            upsilon_S_controller = (S3 != 0).to(torch.float32).mean(dim=-1)
        else:
            (n, m) = S.shape
            total_count = n // self.block_p * (m // self.block_q)
            blocks = S.reshape(
                n // self.block_p, self.block_p, m // self.block_q, self.block_q
            )
            block_norms = (
                blocks.permute(0, 2, 1, 3)
                .reshape(-1, self.block_p * self.block_q)
                .norm(dim=1)
            )
            # At (1, 1) the block norms are |S_ij|, so this counts live entries.
            nonzero_count = (block_norms > 0).sum()
        upsilon_S = nonzero_count / total_count
        self._last_stats = {
            "diff": (X - L - S).norm(),
            "diff_xlsy": (X - L - S + self.Y_div_rho).norm(),
            "y_norm": self.Y_div_rho.norm() * self.rho,
            "alpha": self.alpha,
            "beta": self.beta.mean() if self.beta.ndim > 0 else self.beta,
            "rho": self.rho,
            "gamma_L": gamma_L,
            "upsilon_S": upsilon_S,
            "rank": k,
            "total_rank": min(self.M.shape),
            "nonzero": nonzero_count,
            "total_elements": total_count,
        }
        if self.nm_m is not None:
            self._last_stats["upsilon_S_controller"] = upsilon_S_controller
        return self._last_stats

    @torch.no_grad()
    def _nm_prox(
        self, Z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        (n_dim, m_dim) = Z.shape
        Z3 = Z.view(n_dim, m_dim // self.nm_m, self.nm_m)
        thr = (self.beta / self.rho).unsqueeze(-1)
        S3 = torch.sign(Z3) * torch.clamp(Z3.abs() - thr, min=0.0)
        S = S3.view(n_dim, m_dim).contiguous()
        upsilon_per_group = (S3 != 0).to(torch.float32).mean(dim=-1)
        return (S, upsilon_per_group, (S != 0).sum(), S.numel())

    @torch.no_grad()
    def _ls_compute(self) -> tuple[torch.Tensor, torch.Tensor, dict]:
        X = self.w_param.detach()
        Z = X - self.S + self.Y_div_rho
        (U, s, Vh, self._svd_Q) = _isvd_step(Z, self._svd_Q, self._svd_transposed)
        s_thresh = torch.clamp(s - self.alpha / self.rho, min=0.0)
        L = U * s_thresh @ Vh
        k = _effective_rank(s_thresh, self.gamma)
        gamma_L = k / min(X.shape)
        Z = X - L + self.Y_div_rho
        if self._no_s:
            S = torch.zeros_like(Z)
            nonzero_count = torch.zeros((), dtype=torch.int64, device=Z.device)
            total_count = Z.numel()
            upsilon_S_controller = None
        elif self.nm_m is not None:
            (S, upsilon_S_controller, nonzero_count, total_count) = self._nm_prox(Z)
        elif self.block_p == 1 and self.block_q == 1:
            # (1, 1) blocks are single entries, so the group-norm prox reduces to
            # scalar soft-thresholding. Same result as _block_thresh, no reshape.
            S = torch.sign(Z) * torch.clamp(Z.abs() - self.beta / self.rho, min=0.0)
            nonzero_count = (S != 0).sum()
            total_count = S.numel()
            upsilon_S_controller = None
        else:
            (S, nonzero_count, total_count) = _block_thresh(
                Z, self.beta / self.rho, self.block_p, self.block_q
            )
            upsilon_S_controller = None
        upsilon_S = nonzero_count / total_count
        stats = {
            "diff": (X - L - S).norm(),
            "diff_xlsy": (X - L - S + self.Y_div_rho).norm(),
            "y_norm": self.Y_div_rho.norm() * self.rho,
            "alpha": self.alpha,
            "beta": self.beta.mean() if self.beta.ndim > 0 else self.beta,
            "rho": self.rho,
            "gamma_L": gamma_L,
            "upsilon_S": upsilon_S,
            "rank": k,
            "total_rank": min(X.shape),
            "nonzero": nonzero_count,
            "total_elements": total_count,
        }
        if self.nm_m is not None:
            stats["upsilon_S_controller"] = upsilon_S_controller
        return (L, S, stats)

    @torch.no_grad()
    def step_ls_y_fp32(self) -> dict:
        (L, S, stats) = self._ls_compute()
        X = self.w_param.detach()
        Y_div_rho_new = self.Y_div_rho + (X - L - S)
        M_new = L + S - Y_div_rho_new
        self.M = M_new.to(self._buffer_dtype)
        self.S = S.to(self._buffer_dtype)
        self.Y_div_rho = Y_div_rho_new.to(self._buffer_dtype)
        stats["y_norm"] = Y_div_rho_new.norm() * self.rho
        self._last_stats = stats
        return stats

    def ic_delta(self) -> tuple[torch.Tensor, torch.Tensor]:
        g = self._last_stats["gamma_L"]
        d_alpha = self.rho * (g - self.target_rank_ratio) * self.delta_alpha
        if self._no_s:
            return (d_alpha, torch.zeros_like(d_alpha))
        u = (
            self._last_stats["upsilon_S_controller"]
            if self.nm_m is not None
            else self._last_stats["upsilon_S"]
        )
        d_beta = self.rho * (u - self.target_density) * self.delta_beta
        return (d_alpha, d_beta)

    def state_dict(self) -> dict:
        L_buf = self.M + self.Y_div_rho - self.S
        beta_save: float | torch.Tensor
        if self.beta.ndim > 0:
            beta_save = self.beta.detach().to(dtype=torch.float32, device="cpu")
        else:
            beta_save = float(self.beta.item())
        d = {
            "L": L_buf.cpu(),
            "S": self.S.cpu(),
            "Y_div_rho": self.Y_div_rho.cpu(),
            "alpha": float(self.alpha.item()),
            "beta": beta_save,
            "rho": self.rho,
            "block_p": self.block_p,
            "block_q": self.block_q,
            "nm_n": self.nm_n,
            "nm_m": self.nm_m,
            "last_rank_ratio": self.last_rank_ratio,
            "last_density": self.last_density,
        }
        d["svd_Q"] = self._svd_Q.cpu()
        return d

    def load_state_dict(self, state: dict) -> None:
        device = self.w_param.device
        L_buf = state["L"].to(device=device, dtype=self._buffer_dtype)
        S_buf = state["S"].to(device=device, dtype=self._buffer_dtype)
        if "Y_div_rho" not in state:
            raise ValueError(
                "checkpoint is missing 'Y_div_rho' — this is a pre-Y/ρ-fold checkpoint format that is no longer supported."
            )
        Y_div_rho_buf = state["Y_div_rho"].to(device=device, dtype=self._buffer_dtype)
        self.M = L_buf + S_buf - Y_div_rho_buf
        self.S = S_buf
        self.Y_div_rho = Y_div_rho_buf
        a = state["alpha"]
        a = a.item() if isinstance(a, torch.Tensor) else a
        self.alpha = torch.tensor(float(a), device=device, dtype=torch.float32)
        b = state["beta"]
        if self.nm_m is not None:
            if isinstance(b, torch.Tensor) and b.ndim == 2:
                self.beta = b.to(device=device, dtype=torch.float32)
            else:
                b_val = float(b.item()) if isinstance(b, torch.Tensor) else float(b)
                (n, m) = self.w_param.shape
                self.beta = torch.full(
                    (n, m // self.nm_m), b_val, device=device, dtype=torch.float32
                )
        else:
            if isinstance(b, torch.Tensor) and b.ndim > 0:
                b_val = b.mean().item()
            elif isinstance(b, torch.Tensor):
                b_val = b.item()
            else:
                b_val = b
            self.beta = torch.tensor(float(b_val), device=device, dtype=torch.float32)
        self.rho = state["rho"]
        self._svd_Q = state["svd_Q"].to(device)
        if "last_rank_ratio" in state and "last_density" in state:
            self.last_rank_ratio = float(state["last_rank_ratio"])
            self.last_density = float(state["last_density"])
        else:
            stats = self.compute_stats()
            self.last_rank_ratio = float(stats["gamma_L"].item())
            self.last_density = float(stats["upsilon_S"].item())
