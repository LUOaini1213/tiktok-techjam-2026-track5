"""Write tiny RGB fixtures so infer.py can be smoke-tested."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import enable_utf8_stdout


def main() -> None:
    enable_utf8_stdout()
    out = ROOT / "samples"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    real = (rng.random((256, 256, 3)) * 40 + np.linspace(40, 180, 256)[:, None, None]).clip(0, 255)
    aigc = np.zeros((256, 256, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:256, 0:256]
    aigc[..., 0] = (np.sin(xx / 8.0) * 0.5 + 0.5) * 255
    aigc[..., 1] = (np.cos(yy / 11.0) * 0.5 + 0.5) * 255
    aigc[..., 2] = ((xx + yy) % 64) * 4
    Image.fromarray(real.astype(np.uint8)).save(out / "real_noise.jpg", quality=95)
    Image.fromarray(aigc.astype(np.uint8)).save(out / "synth_pattern.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
