"""AVX2/FMA CPU kernel for the block-sparse path.

Computes ``out += x @ S.T`` in place for a block-sparse ``S``, so the sparse
contribution accumulates straight into the dense low-rank output with no second
buffer. The extension is compiled on first use by ``torch.utils.cpp_extension``
and cached, so a C++ compiler is needed at runtime, not at install time.

``is_available()`` reports whether the build succeeded; callers fall back to
PyTorch's sparse matmul when it did not.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
import tempfile

import torch
from torch.utils.cpp_extension import load

SUPPORTED_BLOCK_SIZES = (8, 16, 32, 64)


@lru_cache(maxsize=1)
def _extension():
    src = Path(__file__).with_suffix(".cpp")
    # Key the build directory on the source, never on the host CPU: -mavx2 with
    # no -march=native keeps one binary valid across every AVX2 machine.
    tag = hashlib.sha1(src.read_bytes()).hexdigest()[:12]
    build_dir = Path(tempfile.gettempdir()) / f"ssalaad_bsr_addmm_avx2_{tag}"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"ssalaad_bsr_addmm_avx2_{tag}",
        sources=[str(src)],
        extra_cflags=["-O3", "-mavx2", "-mfma"],
        build_directory=str(build_dir),
        verbose=False,
    )


@lru_cache(maxsize=1)
def is_available() -> bool:
    """True when the extension compiles and loads on this machine."""
    try:
        _extension()
        return True
    except Exception as exc:  # no compiler, no AVX2, or a build failure
        print(f"[cpu_sparse] AVX2 kernel unavailable ({exc.__class__.__name__}: {exc})")
        return False


def bsr_addmm_avx2_(
    out: torch.Tensor,
    x: torch.Tensor,
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    active_rows: torch.Tensor,
) -> None:
    """In-place ``out += x @ S.T``.

    All tensors are CPU and contiguous; ``out``, ``x`` and ``values`` are
    float32, the index tensors int32. ``values`` is ``(nnz, block_q, block_p)``
    and ``active_rows`` lists the block-rows that hold at least one block, so
    empty rows cost nothing.
    """
    _extension().bsr_addmm_avx2_(out, x, crow, col, values, active_rows)
