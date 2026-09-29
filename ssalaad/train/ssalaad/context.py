from __future__ import annotations
import re
from typing import Any
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from .solver import SSalaadSolver

_ROLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile("^model\\.embed_tokens$"), "embed"),
    (re.compile("^lm_head$"), "lm_head"),
    (re.compile("^model\\.layers\\.\\d+\\.self_attn\\.q_proj$"), "attn_q_proj"),
    (re.compile("^model\\.layers\\.\\d+\\.self_attn\\.k_proj$"), "attn_k_proj"),
    (re.compile("^model\\.layers\\.\\d+\\.self_attn\\.v_proj$"), "attn_v_proj"),
    (re.compile("^model\\.layers\\.\\d+\\.self_attn\\.o_proj$"), "attn_o_proj"),
    (re.compile("^model\\.layers\\.\\d+\\.mlp\\.gate_proj$"), "mlp_gate_proj"),
    (re.compile("^model\\.layers\\.\\d+\\.mlp\\.up_proj$"), "mlp_up_proj"),
    (re.compile("^model\\.layers\\.\\d+\\.mlp\\.down_proj$"), "mlp_down_proj"),
]


def _role_of(name: str) -> str | None:
    for pat, role in _ROLE_PATTERNS:
        if pat.match(name):
            return role
    return None


def _parse_block_dim(v: Any) -> int | str:
    if str(v).lower() == "full":
        return "full"
    return int(v)


def _resolve_params(cfg_mode: DictConfig, buffer_dtype: torch.dtype) -> dict[str, Any]:
    params: dict[str, Any] = {
        "rho": float(cfg_mode.globals.rho),
        "target_rank_ratio": float(cfg_mode.globals.target_rank_ratio),
        "target_density": float(cfg_mode.globals.target_density),
        "gamma": float(cfg_mode.globals.gamma),
        "delta_alpha": float(cfg_mode.globals.delta_alpha),
        "delta_beta": float(cfg_mode.globals.delta_beta),
        "buffer_dtype": buffer_dtype,
    }
    # (1, 1) is element-wise, (p, q) block, (1, "full") row, ("full", 1) column.
    params["block_p"] = _parse_block_dim(cfg_mode.globals.get("block_p", 1))
    params["block_q"] = _parse_block_dim(cfg_mode.globals.get("block_q", 1))
    nm_n = cfg_mode.globals.get("nm_n", None)
    nm_m = cfg_mode.globals.get("nm_m", None)
    if nm_n is not None and nm_m is not None:
        params["nm_n"] = int(nm_n)
        params["nm_m"] = int(nm_m)
    no_s = cfg_mode.globals.get("no_s", False)
    if no_s:
        params["no_s"] = True
    return params


class SSalaadContext:
    def __init__(self, solvers: dict[str, SSalaadSolver], J: int = 1) -> None:
        self.J = J
        self.solvers = solvers
        self._last_penalty_value: float = 0.0

    @torch.no_grad()
    def log_initial_state(self) -> dict[str, Any]:
        for s in self.solvers.values():
            s.compute_stats()
        return self.logging_scalars()

    def coupled_penalty(self) -> torch.Tensor:
        total = None
        for s in self.solvers.values():
            term = s.coupled_penalty()
            total = term if total is None else total + term
        if total is None:
            return (
                torch.zeros((), device=next(iter(self.solvers.values())).w_param.device)
                if self.solvers
                else torch.zeros(())
            )
        return total

    def record_penalty(self, value: float) -> None:
        self._last_penalty_value = value

    @torch.no_grad()
    def run_admm_round(self) -> None:
        for j in range(self.J):
            for s in self.solvers.values():
                s.step_ls_y_fp32()
        solver_list = list(self.solvers.values())
        if solver_list:
            gl_us = (
                torch.stack(
                    [
                        torch.stack(
                            [s._last_stats["gamma_L"], s._last_stats["upsilon_S"]]
                        )
                        for s in solver_list
                    ]
                )
                .float()
                .cpu()
                .tolist()
            )
        else:
            gl_us = []
        for s, (gl, us) in zip(solver_list, gl_us):
            s.last_rank_ratio = gl
            s.last_density = us
            (d_alpha, d_beta) = s.ic_delta()
            s.alpha = s.alpha + d_alpha
            s.beta = s.beta + d_beta

    def logging_scalars(self) -> dict[str, Any]:
        """Per-layer residual, rank ratio and density, plus their run-level means.

        Every layer's scalars are stacked into one tensor, so a round costs a
        single device-to-host copy rather than one per layer.
        """
        out: dict[str, Any] = {"train/penalty": self._last_penalty_value}
        named = [(n, s._last_stats) for (n, s) in self.solvers.items() if s._last_stats]
        if not named:
            return out
        fields = ("diff", "rank", "nonzero")
        rows = (
            torch.stack(
                [torch.stack([st[f].float() for f in fields]) for (_, st) in named]
            )
            .cpu()
            .tolist()
        )
        (diffs, rank_ratios, densities) = ([], [], [])
        for (name, st), (diff, rank, nonzero) in zip(named, rows):
            rank_ratio = rank / max(1, st["total_rank"])
            density = nonzero / max(1, st["total_elements"])
            out[f"layer/{name}/diff"] = diff
            out[f"layer/{name}/rank_ratio"] = rank_ratio
            out[f"layer/{name}/non_zero_ratio"] = density
            diffs.append(diff)
            rank_ratios.append(rank_ratio)
            densities.append(density)
        out["train/layer_diff"] = sum(diffs)
        out["train/rank_ratio"] = sum(rank_ratios) / len(rank_ratios)
        out["train/density"] = sum(densities) / len(densities)
        # N:M holds a fixed count per group; report how many groups sit on target.
        at_target = [
            st["upsilon_S_controller"]
            for (_, st) in named
            if "upsilon_S_controller" in st
        ]
        if at_target:
            hits = torch.stack([(u == 0.5).to(torch.float32).mean() for u in at_target])
            out["train/nm_at_target_global"] = float(hits.mean())
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "solvers": {name: s.state_dict() for name, s in self.solvers.items()},
            "last_penalty_value": self._last_penalty_value,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, s_state in state["solvers"].items():
            if name in self.solvers:
                self.solvers[name].load_state_dict(s_state)
        self._last_penalty_value = state.get("last_penalty_value", 0.0)


def build_ssalaad_context(model: nn.Module, cfg: DictConfig) -> SSalaadContext:
    cfg_mode = cfg.mode
    # Solver buffers follow the bf16 mixed-precision recipe on CUDA; fp32 on CPU.
    buffer_dtype = torch.bfloat16 if str(cfg.trainer.device) == "cuda" else torch.float32
    tracked_roles = set(OmegaConf.to_container(cfg_mode.tracked_roles))
    candidates: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight") or not isinstance(module.weight, nn.Parameter):
            continue
        role = _role_of(name)
        if role is None or role not in tracked_roles:
            continue
        candidates.append((name, module))
    if not candidates:
        raise SystemExit(
            "S-SALAAD mode: no tracked layers matched; check tracked_roles and _ROLE_PATTERNS."
        )
    solvers: dict[str, SSalaadSolver] = {}
    for name, module in candidates:
        params = _resolve_params(cfg_mode, buffer_dtype)
        (n, m) = module.weight.shape
        if params.get("block_p") == "full":
            params["block_p"] = n
        if params.get("block_q") == "full":
            params["block_q"] = m
        solvers[name] = SSalaadSolver(module.weight, **params)
    return SSalaadContext(solvers, J=int(cfg_mode.globals.J))
