"""Teacher logits streamed from the HuggingFace Hub.

Each shard is a large safetensors file of ``(input_ids, top_idx, top_logprob)``
rows. Rows are fetched one at a time with HTTP range requests, so a shard is
never downloaded whole and nothing is written to disk.
"""
from __future__ import annotations

import json
from typing import Iterator

import numpy as np
import torch

_ROW_KEYS = ("input_ids", "top_idx", "top_logprob")
_NUMPY_DTYPES = {"I32": "<i4", "I64": "<i8", "F16": "<f2", "F32": "<f4", "F64": "<f8"}


def _repo_path(repo: str, prefix: str, name: str) -> str:
    return f"datasets/{repo}/{prefix.strip('/')}/{name}"


def _read_manifest(repo: str, prefix: str) -> dict:
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(_repo_path(repo, prefix, "manifest.json"), "rb") as handle:
        return json.loads(handle.read())


class _RemoteRows:
    """Row-addressed reader over one remote safetensors file."""

    def __init__(self, path: str) -> None:
        from huggingface_hub import HfFileSystem

        self._handle = HfFileSystem().open(path, "rb")
        header_len = int.from_bytes(self._handle.read(8), "little")
        self._header = json.loads(self._handle.read(header_len))
        self._data_start = 8 + header_len

    def num_rows(self) -> int:
        return int(self._header["input_ids"]["shape"][0])

    def row(self, key: str, index: int) -> torch.Tensor:
        entry = self._header[key]
        dtype = np.dtype(_NUMPY_DTYPES[entry["dtype"]])
        shape = entry["shape"][1:]
        width = int(np.prod(shape)) if shape else 1
        start = self._data_start + entry["data_offsets"][0] + index * width * dtype.itemsize
        self._handle.seek(start)
        raw = self._handle.read(width * dtype.itemsize)
        return torch.from_numpy(np.frombuffer(raw, dtype=dtype).copy()).reshape(shape)

    def close(self) -> None:
        self._handle.close()


class _LogitRowSource:
    """Shuffled rows, drawn one shard at a time and re-shuffled every epoch."""

    def __init__(self, repo: str, prefix: str, chunks: list[str], seed: int) -> None:
        self._repo = repo
        self._prefix = prefix
        self._chunks = chunks
        self._seed = int(seed)
        self._epoch = 0
        self._chunk_pos = 0
        self._row_pos = 0
        self._epoch_order = self._shuffle_epoch()
        self._loaded: int | None = None
        self._reader: _RemoteRows | None = None
        self._perm: np.ndarray | None = None

    def _shuffle_epoch(self) -> np.ndarray:
        return np.random.default_rng([self._seed, self._epoch]).permutation(
            len(self._chunks)
        )

    def _row_perm(self, chunk_index: int, num_rows: int) -> np.ndarray:
        return np.random.default_rng([self._seed, self._epoch, chunk_index]).permutation(
            num_rows
        )

    def _load_current_chunk(self) -> None:
        if self._reader is not None:
            self._reader.close()
        index = int(self._epoch_order[self._chunk_pos])
        path = _repo_path(self._repo, self._prefix, f"{self._chunks[index]}.safetensors")
        self._reader = _RemoteRows(path)
        self._perm = self._row_perm(index, self._reader.num_rows())
        self._loaded = self._chunk_pos

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        while True:
            if self._chunk_pos >= len(self._epoch_order):
                self._epoch += 1
                self._chunk_pos = 0
                self._row_pos = 0
                self._epoch_order = self._shuffle_epoch()
                self._loaded = None
            if self._loaded != self._chunk_pos:
                self._load_current_chunk()
            assert self._reader is not None and self._perm is not None
            if self._row_pos >= self._perm.shape[0]:
                self._chunk_pos += 1
                self._row_pos = 0
                continue
            row = int(self._perm[self._row_pos])
            self._row_pos += 1
            return {key: self._reader.row(key, row) for key in _ROW_KEYS}


class KDLoader:
    def __init__(
        self,
        repo: str,
        prefix: str,
        seq_len: int,
        microbatch_size: int,
        seed: int,
    ) -> None:
        manifest = _read_manifest(repo, prefix)
        if int(manifest["seq_len"]) != int(seq_len):
            raise ValueError(
                f"data.seq_len={seq_len} != manifest seq_len={manifest['seq_len']}"
            )
        self.seq_len = int(seq_len)
        self.microbatch_size = int(microbatch_size)
        self.separator_id = int(manifest["separator_id"])
        self.top_k = int(manifest["top_k"])
        self.total_tokens = int(manifest["total_tokens"])
        self._source = _LogitRowSource(repo, prefix, list(manifest["chunks"]), seed)

    def _derive(self, input_ids: torch.Tensor):
        is_start = input_ids == self.separator_id
        seq = torch.arange(self.seq_len, dtype=torch.int64)
        arange = seq.expand_as(input_ids)
        start_col = torch.where(is_start, arange, torch.zeros_like(arange))
        recent_start = torch.cummax(start_col, dim=1).values
        position_ids = arange - recent_start
        labels = input_ids.to(torch.int64).clone()
        interior_start = is_start.clone()
        interior_start[:, 0] = False
        labels[interior_start] = -100
        next_is_start = torch.zeros_like(is_start)
        next_is_start[:, :-1] = is_start[:, 1:]
        kl_mask = ~next_is_start
        return (position_ids, labels, kl_mask)

    def next_batch(self):
        (ids_rows, idx_rows, lp_rows) = ([], [], [])
        for _ in range(self.microbatch_size):
            row = next(self._source)
            ids_rows.append(row["input_ids"])
            idx_rows.append(row["top_idx"])
            lp_rows.append(row["top_logprob"])
        input_ids = torch.stack(ids_rows).to(torch.int64)
        top_idx = torch.stack(idx_rows).to(torch.int64)
        top_logprob = torch.stack(lp_rows)
        (position_ids, labels, kl_mask) = self._derive(input_ids)
        tensors = [input_ids, labels, position_ids, top_idx, top_logprob, kl_mask]
        if torch.cuda.is_available():
            tensors = [t.pin_memory() for t in tensors]
        return tuple(tensors)
