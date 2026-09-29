"""Training data streamed from the HuggingFace Hub.

Text is streamed with ``datasets`` and tokenized on the fly; nothing is
pre-materialised locally. Documents are packed into fixed-length rows: the
first token after a document boundary is masked out of the loss and position
ids restart there, so packing never lets one document predict the next.
"""
from __future__ import annotations

import atexit
from typing import Iterator

import numpy as np
import torch


class _TokenizedDocSource:
    """One token array per streamed document.

    A tokenizer with a BOS (Llama, Nemotron) delimits documents by a leading BOS;
    one without (T5) uses its own default, a trailing EOS. Both match the
    convention the models were trained with. Training shuffles the stream;
    evaluation reads it in file order, so the same documents are scored each time.
    """

    def __init__(
        self,
        dataset: str,
        dataset_config: str | None,
        split: str,
        text_column: str,
        tokenizer: str,
        seed: int,
        shuffle: bool,
    ) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        model_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bos_id = model_tokenizer.bos_token_id
        bos = model_tokenizer.bos_token_id
        if bos is not None:
            self._encode = lambda text: [
                bos,
                *model_tokenizer(text, add_special_tokens=False)["input_ids"],
            ]
        else:
            self._encode = lambda text: model_tokenizer(text)["input_ids"]
        self._text_column = text_column
        stream = load_dataset(dataset, dataset_config, split=split, streaming=True)
        self._rows = iter(stream.shuffle(seed=seed) if shuffle else stream)
        # Close on the way out: tearing this iterator down during interpreter
        # finalization fails inside `datasets`.
        atexit.register(self.close)

    def close(self) -> None:
        close = getattr(self._rows, "close", None)
        if close is not None:
            close()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        while True:
            text = next(self._rows)[self._text_column]
            ids = np.asarray(self._encode(text), dtype=np.int64)
            if ids.size:
                return ids


class PackedLoader:
    _PAD_ID = 0

    def __init__(
        self,
        source: Iterator[np.ndarray],
        seq_len: int,
        microbatch_size: int,
        pack: bool = True,
    ) -> None:
        self._source = source
        self.seq_len = seq_len
        self.microbatch_size = microbatch_size
        self._pack = pack
        self._buf = np.empty(0, dtype=np.int64)
        self._doc_starts: list[int] = []

    def _refill(self, need: int) -> None:
        chunks: list[np.ndarray] = []
        if self._buf.size > 0:
            chunks.append(self._buf)
        total = self._buf.size
        while total < need:
            doc = next(self._source)
            self._doc_starts.append(total)
            chunks.append(doc)
            total += doc.size
        self._buf = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if self._buf.dtype != np.int64:
            self._buf = self._buf.astype(np.int64)

    def _advance(self, consumed: int) -> None:
        self._buf = self._buf[consumed:]
        self._doc_starts = [s - consumed for s in self._doc_starts if s >= consumed]

    def _next_batch_packed(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = self.seq_len
        ids = np.empty((self.microbatch_size, seq_len), dtype=np.int64)
        labels = np.empty((self.microbatch_size, seq_len), dtype=np.int64)
        position_ids = np.empty((self.microbatch_size, seq_len), dtype=np.int64)
        for i in range(self.microbatch_size):
            self._refill(seq_len)
            ids[i, :] = self._buf[:seq_len]
            labels[i, :] = self._buf[:seq_len]
            starts = [s for s in self._doc_starts if s < seq_len]
            for j in range(len(starts)):
                s = starts[j]
                e = starts[j + 1] if j + 1 < len(starts) else seq_len
                position_ids[i, s:e] = np.arange(e - s, dtype=np.int64)
            for s in starts[1:]:
                labels[i, s] = -100
            cut = None
            for s in self._doc_starts:
                if s >= seq_len:
                    cut = s
                    break
            if cut is not None:
                self._advance(cut)
            else:
                self._buf = np.empty(0, dtype=np.int64)
                self._doc_starts = []
        ids_t = torch.from_numpy(ids)
        labels_t = torch.from_numpy(labels)
        pos_t = torch.from_numpy(position_ids)
        if torch.cuda.is_available():
            ids_t = ids_t.pin_memory()
            labels_t = labels_t.pin_memory()
            pos_t = pos_t.pin_memory()
        return (ids_t, labels_t, pos_t)

    def _next_batch_padded(self) -> tuple[torch.Tensor, torch.Tensor, None]:
        seq_len = self.seq_len
        ids = np.full((self.microbatch_size, seq_len), self._PAD_ID, dtype=np.int64)
        labels = np.full((self.microbatch_size, seq_len), -100, dtype=np.int64)
        for i in range(self.microbatch_size):
            doc = next(self._source)
            length = min(int(doc.size), seq_len)
            ids[i, :length] = doc[:length]
            labels[i, :length] = doc[:length]
        ids_t = torch.from_numpy(ids)
        labels_t = torch.from_numpy(labels)
        if torch.cuda.is_available():
            ids_t = ids_t.pin_memory()
            labels_t = labels_t.pin_memory()
        return (ids_t, labels_t, None)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self._next_batch_packed() if self._pack else self._next_batch_padded()


class StreamLoader:
    """Sliding-window perplexity over the corpus as one flat token stream.

    Windows of ``seq_len`` advance by ``stride``. The first window scores every
    position; later windows score only their last ``stride`` tokens, so each
    scored token sees at least ``seq_len - stride`` tokens of context and no
    position is scored twice. Documents run on into one another, and every
    window starts with a BOS anchor. This is the FineWeb-Edu evaluation in the
    paper, following the HuggingFace sliding-window perplexity recipe:
    https://huggingface.co/docs/transformers/perplexity
    """

    def __init__(
        self,
        source: Iterator[np.ndarray],
        seq_len: int,
        microbatch_size: int,
        stride: int,
        bos_id: int | None = None,
    ) -> None:
        self._source = source
        self.seq_len = int(seq_len)
        self.microbatch_size = int(microbatch_size)
        self._stride = int(stride)
        self._bos_id = bos_id
        self._offset = 1 if bos_id is not None else 0
        self._corpus_len = self.seq_len - self._offset
        self._buf = np.empty(0, dtype=np.int64)
        self._first = True

    def _window(self) -> np.ndarray:
        while self._buf.size < self._corpus_len:
            self._buf = np.concatenate((self._buf, next(self._source)))
        window = self._buf[: self._corpus_len].copy()
        self._buf = self._buf[self._stride :]
        return window

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, None]:
        ids = np.empty((self.microbatch_size, self.seq_len), dtype=np.int64)
        labels = np.full((self.microbatch_size, self.seq_len), -100, dtype=np.int64)
        for i in range(self.microbatch_size):
            window = self._window()
            if self._offset:
                ids[i, 0] = self._bos_id
            ids[i, self._offset :] = window
            scored = self._corpus_len if self._first else min(self._stride, self._corpus_len)
            labels[i, self.seq_len - scored :] = window[self._corpus_len - scored :]
            self._first = False
        ids_t = torch.from_numpy(ids)
        labels_t = torch.from_numpy(labels)
        if torch.cuda.is_available():
            ids_t = ids_t.pin_memory()
            labels_t = labels_t.pin_memory()
        return (ids_t, labels_t, None)


def make_loader(
    *,
    dataset: str,
    split: str,
    tokenizer: str,
    seq_len: int,
    microbatch_size: int,
    seed: int,
    text_column: str = "text",
    dataset_config: str | None = None,
    mode: str = "packed",
    stride: int | None = None,
) -> PackedLoader | StreamLoader:
    """Build a loader over a streamed corpus.

    ``packed`` for training, ``padded`` for per-document evaluation, or
    ``stream`` for sliding-window evaluation with ``stride``.
    """
    source = _TokenizedDocSource(
        dataset, dataset_config, split, text_column, tokenizer, seed,
        shuffle=(mode == "packed"),
    )
    if mode == "stream":
        return StreamLoader(
            source, seq_len, microbatch_size, stride or seq_len, bos_id=source.bos_id
        )
    if mode in ("packed", "padded"):
        return PackedLoader(
            source, seq_len, microbatch_size, pack=(mode == "packed")
        )
    raise ValueError(f"unknown loader mode {mode!r}; expected packed|padded|stream")
