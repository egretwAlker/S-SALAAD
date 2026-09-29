"""Low-rank plus sparse training, compression, and inference."""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Every output -- training runs and compressed bundles -- lands under this folder.
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"

# Keep the HuggingFace cache beside the outputs instead of in ~/.cache.
os.environ.setdefault("HF_HOME", str(OUTPUT_ROOT / "hf_cache"))
