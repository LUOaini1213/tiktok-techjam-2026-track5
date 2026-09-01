# RepostGuard

TikTok TechJam 2026 Track 5 — **Robust Detection of AI-Generated Images Under Real-World Transformations**.

AIGC detectors that look strong on clean lab images often collapse after a TikTok-style repost: JPEG re-encode, thumbnail resize, filter jitter, or avatar crop. RepostGuard is a hackathon-scale detector that treats those transforms as the actual test, not an afterthought.

## Project overview

**Model (well under the 2B parameter limit):** frozen OpenAI CLIP `ViT-B/32` (~88M) + a 28-D forensic vector + a logistic-regression head.

- **CLIP features, A/B-selected.** From a single CLIP forward we cache **two** image-feature variants and let cross-validation pick the winner: the **pre-projection** feature (`vision_model(...).pooler_output`, the post-LayerNorm CLS token, 768-D) and the **projected** feature (`visual_projection(pooler_output)`, 512-D, UnivFD's usual probe input). Pre-projection features usually win for linear-probe fake detection (Cozzolino et al., arXiv:2312.00195), but we verify it by GroupKFold(5) CV AUC on our own data rather than assuming.
- **Native-resolution forensic branch (28-D).** NPR-style residual stats, 8×8 block-DCT high/low energy ratios (aligned to the JPEG grid), FFT radial rings expressed as fractions of Nyquist (including the ≥0.75 bands that carry GAN/upsampling spectral peaks), plus color and Laplacian stats. Computed on a **native-scale 256 center crop** — never a bilinear downscale, which would low-pass-filter away exactly the high-frequency fingerprints this branch exists to measure.
- **Test-time augmentation (TTA).** Inference averages the original view with a JPEG-70 view and a 0.5× down/up view; the averaged CLIP embedding is re-L2-normalized so its scale does not depend on the view count. Train, eval, and infer all go through the **same** `embed_for_score` path so cached features match scored features exactly.
- **Sigmoid calibration.** After the head is fit, we sigmoid-calibrate it on a group-disjoint 30% slice of the validation set and report metrics only on the held-out 70% test slice, so `pred` is a usable P(AIGC) probability rather than a raw margin.

**Training data:** SID-Set (`saberzl/SID_Set`) binary labels: `0` real vs `1` AIGC-positive (`1` full-synthetic **and** `2` tampered, upweighted 5×). Official JPEG/blur/resize views plus clean/degraded consistency rows. The official WildFake demonstration subset is **never used for training** — it is only a reference benchmark (see below).

## Problem-statement alignment

- **Transform grid matches exactly.** Our evaluation presets are precisely the Track-5 transforms: JPEG 90/70/50/30, Gaussian blur σ 0.5/1.0/2.0, resize 0.5×/0.25× then upsample, Gaussian noise σ 0.02/0.05/0.10, color jitter ±20%, center crop 80% (`src/augment.py`).
- **Under 2B parameters.** Frozen CLIP ViT-B/32 (~88M) + a linear head.
- **Submission contract.** `python infer.py --input_dir PATH --output preds.json` emits `[{"image_path", "pred"∈[0,1]}]`, one row per input file, robust to unreadable/truncated files.
- **Allowed data only.** Trained on SID-Set (a listed, properly licensed resource). WildFake is used strictly as a demonstration benchmark, never for training.

## Setup & installation

CPU-only works; a CUDA GPU makes feature extraction much faster (the code auto-uses CUDA when available).

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Steps to reproduce results

```bat
python scripts/make_samples.py                              :: tiny fixtures
python scripts/download_data.py --train_per_class 2000 --val_per_class 1000
python scripts/extract_features.py --split train --augment  :: caches BOTH CLIP variants
python scripts/extract_features.py --split val
python scripts/train.py                                     :: A/B CV, calibrate, write bundle
python scripts/make_tables.py                               :: robustness table + CIs
python scripts/make_chart.py                                :: robustness bar chart PNG
python scripts/error_analysis.py                            :: FP/FN note
python scripts/eval_demo.py                                 :: WildFake demonstration benchmark
python scripts/update_readme.py                             :: fill the README result sections
python infer.py --input_dir samples --output preds.json     :: required submission command
python app.py                                               :: optional Gradio demo
```

`download_data.py` asserts every per-class quota is met and exits non-zero on a partial download, so the validation split can never silently truncate. If no weights exist yet, `scripts/smoke_train.py` fits a 2-image sanity classifier so `infer.py` always runs.

Output JSON:

```json
[
  {"image_path": "C:\\data\\a.jpg", "pred": 0.92},
  {"image_path": "C:\\data\\b.jpg", "pred": 0.11}
]
```

## Robustness evaluation summary

Clean vs each Track-5 transform on the held-out validation **test** slice (calibration images excluded), with 95% stratified-bootstrap AUC confidence intervals. Full machine-readable table: `results/robustness_table.csv`; chart: `results/robustness_chart.png`.

<!-- ROBUSTNESS_TABLE -->
| Transform | n | Accuracy | ROC AUC | 95% CI |
|---|---:|---:|---:|---|
| `clean` | 1400 | 0.9021 | 0.9634 | [0.9540, 0.9719] |
| `jpeg_90` | 1400 | 0.9093 | 0.9674 | [0.9587, 0.9755] |
| `jpeg_70` | 1400 | 0.9157 | 0.9728 | [0.9649, 0.9796] |
| `jpeg_50` | 1400 | 0.9107 | 0.9682 | [0.9597, 0.9758] |
| `jpeg_30` | 1400 | 0.8764 | 0.9547 | [0.9446, 0.9642] |
| `blur_0.5` | 1400 | 0.8986 | 0.9618 | [0.9523, 0.9706] |
| `blur_1.0` | 1400 | 0.8629 | 0.9567 | [0.9464, 0.9660] |
| `blur_2.0` | 1400 | 0.8707 | 0.9500 | [0.9388, 0.9605] |
| `resize_0.5` | 1400 | 0.8650 | 0.9567 | [0.9464, 0.9662] |
| `resize_0.25` | 1400 | 0.8679 | 0.9455 | [0.9338, 0.9564] |
| `noise_0.02` | 1400 | 0.8714 | 0.9244 | [0.9095, 0.9381] |
| `noise_0.05` | 1400 | 0.8164 | 0.8851 | [0.8669, 0.9025] |
| `noise_0.10` | 1400 | 0.6529 | 0.8117 | [0.7879, 0.8339] |
| `jitter_0.20` | 1400 | 0.8714 | 0.9362 | [0.9230, 0.9483] |
| `crop_0.80` | 1400 | 0.8850 | 0.9521 | [0.9418, 0.9620] |

**Clean AUC 0.9634; mean AUC across the 14 transformed conditions 0.9388 (-0.0246 vs clean); worst condition `noise_0.10` at 0.8117 (-0.1517 vs clean).** Every row is the same held-out test slice (1400 images, calibration images excluded) re-scored through the shipped inference path after the transform, so clean and transformed numbers are directly comparable.

![Robustness: clean vs social-media transforms](results/robustness_chart.png)
<!-- /ROBUSTNESS_TABLE -->

## WildFake demonstration benchmark (never used in training)

The official demonstration subset (`techjam-aigc/wildfake-eval-subset`, `default` config = COCO val2017 reals + DALL·E-3 Advanced fakes) scored through the **shipped** inference path on a balanced subsample. This measures cross-source generalization to a generator family absent from SID-Set training. Full table: `results/demo_benchmark.csv`.

<!-- DEMO_TABLE -->
_(not generated yet -- run the pipeline in 'Steps to reproduce results'.)_
<!-- /DEMO_TABLE -->

## Error analysis note

Representative false positives (authentic images flagged AIGC — creator harm), false negatives (generated images missed, especially after JPEG-30), and the score drift under JPEG-30, in `results/error_analysis.json` (`scripts/error_analysis.py`).

<!-- ERROR_ANALYSIS -->
_(not generated yet -- run the pipeline in 'Steps to reproduce results'.)_
<!-- /ERROR_ANALYSIS -->

**Trade-offs.** We keep the decision threshold at 0.5 and report FPR@95%TPR so the creator-harm cost of false positives is explicit. The tampered-class upweight raises recall on locally-edited images at some cost to clean precision. TTA recovers part of the transform-induced AUC drop but triples inference cost per image.

## Design choices

| Choice | Why |
|---|---|
| Frozen CLIP + linear head | Strong cross-generator baseline (UnivFD-style); no 2B-class model; trains in seconds |
| Pre- vs projected CLIP, chosen by CV | Pre-projection usually wins for fake detection; we verify rather than assume |
| Native-resolution forensic branch | CLIP loses high-frequency traces; a 128×128 downscale aliases them away |
| Train-time official augmentations | Aligns the classifier with redistribution, not just clean SID-Set |
| TTA (clean + JPEG-70 + 0.5×) | Recovers part of the transform-induced drop the table is scored on |
| Sigmoid calibration on held-out val | `pred` is a usable probability, evaluated without calibration leakage |

## Limitations & what we would improve with more time

- **Generator diversity.** Trained on SID-Set only. Unknown commercial generators (Flux, Midjourney v7, SD3) still shift CLIP geometry. The listed resources (CIFAKE, WildFake-train) are allowed for training and would be the first addition — mixing generator families is the highest-leverage next step.
- **Hardest rows.** Extreme 0.25× thumbnails and JPEG-30 remain the weakest transforms; a small learned frequency head or a second forensic scale could help.
- **Local edits / face swaps** are only partially covered (tampered class upweight); a dedicated localization branch is out of scope here.
- **Scale.** We trained on a 2000/class subset for the deadline; `configs/default.yaml` targets 8000/class on a CUDA box (see below).
- **False positives on heavily filtered real photos** hurt creators; a per-creator threshold or an abstain band would reduce that harm.

## Full-scale reproduction on a CUDA machine

The pipeline is CPU/GPU agnostic (`get_device()` auto-selects CUDA). On a GPU box:

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python scripts/download_data.py --train_per_class 8000 --val_per_class 1000
python scripts/extract_features.py --split train --augment
python scripts/extract_features.py --split val
python scripts/train.py
```

ViT-B/32 CPU extraction runs ≈20–60 img/s; a modest GPU (e.g. GTX 1650, fp16) cuts extraction from hours to minutes. For a stronger encoder, swap `clip_model_id` to `openai/clip-vit-large-patch14` in `configs/default.yaml` (~15–40 img/s fp16 on a GTX 1650, still well under 2B); the pre-projection variant becomes 1024-D and everything else is unchanged.

## Team member contributions

- **Member A** — model, dual-CLIP features, native forensic branch, robustness evaluation, calibration.
- **Member B** — Gradio demo, demonstration-benchmark eval, video, Devpost write-up.

## Tools

Python, PyTorch (CPU or CUDA), Hugging Face `transformers` (CLIP) + `datasets` (SID-Set / demo subset streaming), scikit-learn, OpenCV, SciPy, NumPy, pandas, matplotlib, Gradio.

## License

Hackathon prototype. SID-Set, the WildFake demonstration subset, and CLIP weights keep their upstream licenses.
