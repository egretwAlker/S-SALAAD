from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig


class LogSink:
    """Prints progress, and appends per-step metrics to ``metrics.jsonl``."""

    def __init__(self, run_dir: Path) -> None:
        self._metrics = (run_dir / "metrics.jsonl").open("a", buffering=1)

    def info(self, message: str) -> None:
        print(message, flush=True)

    def log_step(self, step: int, metrics: dict) -> None:
        record = {
            "step": step,
            **{k: v for k, v in metrics.items() if not k.startswith("layer/")},
        }
        self._metrics.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def finish(self) -> None:
        self._metrics.close()


def log_startup(sink: LogSink, cfg: DictConfig) -> None:
    precision = "bf16-mixed" if str(cfg.trainer.device) == "cuda" else "fp32"
    sink.info(
        f"device={cfg.trainer.device} precision={precision} seed={cfg.seed}"
    )
