# S-SALAAD

Structured sparse-and-low-rank training, compression and inference.

## Install

```bash
pip install -e .
```

## Train

```bash
python -m ssalaad.train model=llama_60m_b16
```

## Compress and serve

```bash
python -m ssalaad.compress --ckpt outputs/llama_60m/checkpoint.pth --p-tgt 50 --device cuda
```

## Test

```bash
python -m pytest
```
