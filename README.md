# RepostGuard

TikTok TechJam 2026 Track 5 — **Robust Detection of AI-Generated Images Under Real-World Transformations**.

AIGC detectors that look strong on clean lab images often collapse after a TikTok-style repost: JPEG re-encode, thumbnail resize, filter jitter, or avatar crop. RepostGuard is a hackathon-scale detector that treats those transforms as the actual test, not an afterthought.

**Model (well under the 2B limit):** frozen OpenAI CLIP `ViT-B/32` (~88M) + 28-D forensic stats (NPR residual, block DCT, FFT rings) computed at **native resolution** + logistic regression. Inference optionally averages the original view with JPEG-70 and 0.5× down/up (TTA); the averaged CLIP embedding is re-L2-normalized.

> **Retrain required after the forensic change.** Forensic features moved from a 128×128 downscale (which low-pass-filtered away the high-frequency fingerprints) to a native-resolution center crop, so the feature width changed (512 CLIP + 28 forensic = 540-D). The committed `artifacts/repostguard.joblib` is stale until you re-run `extract_features.py` → `train.py`; `infer.py` will refuse to run on mismatched weights.

**Training data:** SID-Set binary labels: `0` real vs `1` AIGC-positive (`1` full-synthetic **and** `2` tampered). Official JPEG/blur/resize views plus clean/degraded consistency rows; tampered samples upweighted (`src/train_signal.py`). Features via the same TTA path as inference (`embed_for_score`). The official WildFake demonstration split is **never used for training**.

## Quick start

```bat
python -m pip install -r requirements.txt
python scripts/make_samples.py
python scripts/download_data.py
python scripts/extract_features.py --split train --augment
python scripts/extract_features.py --split val
python scripts/train.py
python infer.py --input_dir samples --output preds.json
python app.py
```

Required submission command:

```bat
python infer.py --input_dir PATH_TO_IMAGES --output preds.json
```

Output JSON:

```json
[
  {"image_path": "C:\\data\\a.jpg", "pred": 0.92},
  {"image_path": "C:\\data\\b.jpg", "pred": 0.11}
]
```

`pred` is calibrated-style P(AIGC) in `[0, 1]`.

`artifacts/repostguard.joblib` is a **SID-Set subset** classifier (`n_train=256`, labels 0 vs 1 only; WildFake unused). See `artifacts/WEIGHTS.txt`. Do not commit giant npz feature caches.

Encoder: frozen CLIP **ViT-B/32** (`openai/clip-vit-base-patch32`, ~88M, well under the 2B limit).

## This machine

Python 3.11 with **PyTorch 2.7.1+cu126** on **NVIDIA GeForce GTX 1650** (4GB). Embedding uses CUDA when `torch.cuda.is_available()`.

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

CPU extraction of ~16k CLIP features is slow but viable. CUDA will cut Block 1 from hours to minutes. Batch size lives in `configs/default.yaml` (`batch_size: 8`).

## Reproduce the robustness table

```bat
python scripts/make_tables.py
python scripts/error_analysis.py
```

Transforms match the problem statement: JPEG 90/70/50/30, Gaussian blur σ 0.5/1.0/2.0, resize 0.5× and 0.25× then upsample, Gaussian noise σ 0.02/0.05/0.10, color jitter ±20%, center crop 80%.

`results/robustness_table.csv` is the compact clean-vs-transformed summary. `results/error_analysis.json` lists representative false positives and false negatives, including JPEG-30 score drift.

## Design choices

| Choice | Why |
|---|---|
| Frozen CLIP + linear head | Strong cross-generator baseline (UnivFD-style); fits 4GB; no 2B-class model |
| Forensic DCT/NPR branch | CLIP embeddings lose high-frequency traces after JPEG/thumbnailing |
| Train-time official augmentations | Aligns the classifier with redistribution, not just clean SID-Set |
| TTA (clean + JPEG70 + 0.5×) | Recovers some of the drop the table is scored on |
| Drop tampered class | Track asks AIGC vs authentic; inpainting is a different forensic task |

## Limits (what we would do with more time)

- Unknown commercial generators (Flux, Midjourney v7) still shift CLIP geometry.
- Extreme 0.25× thumbnails and JPEG-30 remain the hardest rows in the table.
- Local edits / face swaps are out of scope by construction.
- False positives on heavily filtered real photos hurt creators; we keep the decision threshold at 0.5 and report FPR.

## Team

- Member A — model, features, robustness evaluation
- Member B — Gradio demo, video, Devpost write-up

## Tools

VS Code / this repo, Python, PyTorch, Hugging Face `transformers` (CLIP), scikit-learn, OpenCV, SciPy, pandas, Gradio. Optional: `datasets` for SID-Set streaming.

## License

Hackathon prototype. SID-Set and CLIP weights keep their upstream licenses.
