from __future__ import annotations
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from ssalaad import OUTPUT_ROOT
from .data import make_loader
from .kd_data import KDLoader
from .kd_loss import kd_topk_loss
from .logsink import LogSink, log_startup
from .model import build_model, write_model_config
from .ssalaad import build_ssalaad_context

OmegaConf.register_new_resolver("output_dir", lambda name: str(OUTPUT_ROOT / name))

# Shared by every recipe, so they live here instead of in five copies of the
# same YAML: AdamW with no weight decay, a cosine schedule with warmup, and a
# gradient-norm clip.
_BETAS = (0.9, 0.95)
_EPS = 1.0e-8
_WEIGHT_DECAY = 0.0
_GRAD_CLIP = 1.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=_BETAS,
        eps=_EPS,
        weight_decay=_WEIGHT_DECAY,
        fused=next(model.parameters()).is_cuda,
    )


def _build_scheduler(optim: torch.optim.Optimizer, cfg: DictConfig):
    warmup = cfg.warmup_steps
    total = cfg.total_steps
    min_ratio = cfg.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if step >= total:
            return min_ratio
        progress = (step - warmup) / max(1, total - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


def _detach_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            t = torch.empty(obj.shape, dtype=obj.dtype, pin_memory=True)
            t.copy_(obj.detach(), non_blocking=True)
            return t
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for (k, v) in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)((_detach_to_cpu(x) for x in obj))
    return obj


def _write_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pth.tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: dict,
    ssalaad_state: dict | None = None,
) -> None:
    """Write the trained model beside its config, for ``ssalaad.compress``."""
    write_model_config(model_config, path.parent / "model_config.json")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    state = {"model": model.state_dict()}
    if ssalaad_state is not None:
        state["ssalaad"] = ssalaad_state
    _write_checkpoint(_detach_to_cpu(state), path)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device_type = str(cfg.trainer.device)
    if device_type == "cuda" and (not torch.cuda.is_available()):
        raise SystemExit("CUDA is unavailable; use trainer.device=cpu.")
    torch.set_float32_matmul_precision("high")
    # bf16 mixed precision on CUDA; CPU falls back to fp32.
    autocast_dtype = torch.bfloat16 if device_type == "cuda" else None
    kd_enabled = bool(cfg.get("kd", {}).get("enabled", False))
    kd_alpha_ce = float(cfg.kd.alpha_ce) if kd_enabled else 0.0
    device = torch.device("cuda" if device_type == "cuda" else "cpu")
    _set_seed(cfg.seed)
    run_dir = Path(HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sink: LogSink = LogSink(run_dir)
    log_startup(sink, cfg)
    if cfg.data.batch_size % cfg.data.microbatch_size != 0:
        raise SystemExit(
            f"batch_size={cfg.data.batch_size} must be divisible by "
            f"microbatch_size={cfg.data.microbatch_size}"
        )
    grad_accum = cfg.data.batch_size // cfg.data.microbatch_size
    tokens_per_step = cfg.data.batch_size * cfg.data.seq_len
    # One ADMM round per K outer steps, which is also the logging period.
    log_every = int(cfg.mode.globals.K)
    model_config = OmegaConf.to_container(cfg.model_config, resolve=True)
    model = build_model(model_config, attn_implementation="sdpa").to(device)
    model.train()
    ssalaad_ctx = build_ssalaad_context(model, cfg)
    sink.info(f"[ssalaad] tracking {len(ssalaad_ctx.solvers)} layers")
    optim = _build_optimizer(model, cfg.optim)
    sched = _build_scheduler(optim, cfg.schedule)
    if kd_enabled:
        sink.info(
            f"[data] streaming teacher logits from {cfg.data.dataset}/{cfg.data.path}"
        )
        train_loader = KDLoader(
            repo=cfg.data.dataset,
            prefix=cfg.data.path,
            seq_len=cfg.data.seq_len,
            microbatch_size=cfg.data.microbatch_size,
            seed=cfg.seed,
        )
    else:
        sink.info(
            f"[data] streaming {cfg.data.dataset} ({cfg.data.split}), "
            f"tokenizer={cfg.data.tokenizer}"
        )
        train_loader = make_loader(
            dataset=cfg.data.dataset,
            dataset_config=cfg.data.dataset_config,
            split=cfg.data.split,
            tokenizer=cfg.data.tokenizer,
            seq_len=cfg.data.seq_len,
            microbatch_size=cfg.data.microbatch_size,
            seed=cfg.seed,
        )
    sink.info(f"[data] batch_size={cfg.data.batch_size} grad_accum={grad_accum}")
    checkpoint_path = run_dir / "checkpoint.pth"
    step = 0
    tokens_processed = 0
    wall_start = time.time()
    total_steps = cfg.schedule.total_steps
    effective_total = total_steps
    init_metrics = ssalaad_ctx.log_initial_state()
    sink.info("[ssalaad] step 0 initial state (L₀=SVD_trunc, S₀=0):")
    for k, v in sorted(init_metrics.items()):
        if not k.startswith("layer/") or not k.endswith("/diff"):
            continue
        name = k[len("layer/") : -len("/diff")]
        diff = v
        rank_r = init_metrics.get(f"layer/{name}/rank_ratio", float("nan"))
        nz_r = init_metrics.get(f"layer/{name}/non_zero_ratio", float("nan"))
        sink.info(
            f"  {name}: rank_ratio={rank_r:.4f} nz_ratio={nz_r:.4f} diff={diff:.4f}"
        )
    step_start = time.time()
    pbar = tqdm(
        initial=step,
        total=effective_total,
        desc="training",
        unit="step",
        dynamic_ncols=True,
        smoothing=0.05,
        disable=not sys.stderr.isatty(),
    )
    _prefetch_pool = ThreadPoolExecutor(max_workers=1)
    _prefetch = _prefetch_pool.submit(train_loader.next_batch)
    while step < effective_total:
        optim.zero_grad(set_to_none=True)
        loss_sum = 0.0
        kd_ce_sum = 0.0
        kd_kl_sum = 0.0
        penalty_sum = 0.0
        for _ in range(grad_accum):
            batch = _prefetch.result()
            _prefetch = _prefetch_pool.submit(train_loader.next_batch)
            ids = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            pos = (
                batch[2].to(device, non_blocking=True) if batch[2] is not None else None
            )
            with (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype
                else nullcontext()
            ):
                if kd_enabled:
                    hidden = model.model(
                        input_ids=ids, position_ids=pos
                    ).last_hidden_state
                    logits = model.lm_head(hidden)
                    (top_idx, top_logprob, kl_mask) = (
                        batch[j].to(device, non_blocking=True) for j in (3, 4, 5)
                    )
                    (ce_loss, kd_ce, kd_kl) = kd_topk_loss(
                        logits, labels, top_idx, top_logprob, kl_mask, kd_alpha_ce
                    )
                else:
                    ce_loss = model(
                        input_ids=ids, labels=labels, position_ids=pos
                    ).loss
                    kd_ce = kd_kl = None
                loss = ce_loss / grad_accum
                penalty = ssalaad_ctx.coupled_penalty() / grad_accum
                loss = loss + penalty
                penalty_sum += float(penalty.item())
                loss.backward()
            loss_sum += float(ce_loss.item())
            if kd_enabled:
                kd_ce_sum += float(kd_ce.item())
                kd_kl_sum += float(kd_kl.item())
        train_loss = loss_sum / grad_accum
        ssalaad_ctx.record_penalty(penalty_sum)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=_GRAD_CLIP
        )
        optim.step()
        sched.step()
        step += 1
        tokens_processed += tokens_per_step
        if step % log_every == 0:
            ssalaad_ctx.run_admm_round()
            elapsed = time.time() - step_start
            wallclock_ms = elapsed * 1000 / log_every
            tokens_per_sec = (
                log_every * tokens_per_step / max(elapsed, 1e-09)
            )
            lr = sched.get_last_lr()[0]
            metrics = {
                "train/loss": train_loss,
                "train/lr": lr,
                "train/tokens_M": tokens_processed / 1000000.0,
                "train/grad_norm": float(grad_norm),
                "train/step_time_ms": wallclock_ms,
                "train/tokens_per_sec": tokens_per_sec,
            }
            if kd_enabled:
                train_ce = kd_ce_sum / grad_accum
                metrics["train/ce"] = train_ce
                metrics["train/kl"] = kd_kl_sum / grad_accum
                if train_ce <= 5.0:
                    metrics["train/ppl"] = math.exp(train_ce)
            if device.type == "cuda":
                metrics["cuda/max_memory_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                )
                metrics["cuda/max_memory_reserved_gib"] = (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                )
                torch.cuda.reset_peak_memory_stats(device)
            metrics.update(ssalaad_ctx.logging_scalars())
            pbar.set_postfix(
                loss=f"{train_loss:.4f}",
                lr=f"{lr:.6f}",
                grad=f"{float(grad_norm):.3f}",
                tps=f"{tokens_per_sec:.0f}",
            )
            sink.log_step(step, metrics)
            step_start = time.time()
        pbar.update(1)
    pbar.close()
    _save_checkpoint(
        checkpoint_path,
        model,
        model_config,
        ssalaad_state=ssalaad_ctx.state_dict(),
    )
    wall = time.time() - wall_start
    sink.info(f"done. wall={wall:.1f}s tokens={tokens_processed:,}")
    sink.finish()
    _prefetch_pool.shutdown(wait=True)
