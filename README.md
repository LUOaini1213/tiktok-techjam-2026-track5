# RepostGuard

[![tests](https://github.com/LUOaini1213/tiktok-techjam-2026-track5/actions/workflows/tests.yml/badge.svg)](https://github.com/LUOaini1213/tiktok-techjam-2026-track5/actions/workflows/tests.yml)

**AI-generated-image detection that is scored on the repost, not the original.** A frozen
CLIP ViT-B/32 fused with a frozen DINOv2-small (110M parameters together, CPU-only) plus a
native-resolution forensic branch, evaluated on all 15 real-world transforms a social platform
applies -- JPEG, thumbnail, blur, noise, colour jitter, crop -- with bootstrap confidence
intervals, a published baseline, a 2x2 ablation that explains *why* it holds, a leakage control
that shows it is not cheating, and a post-deadline v2 whose every change is re-measured on the
same held-out images as the deadline build.

| | AUC | TPR@1%FPR |
|---|---:|---:|
| clean | 0.981 | 0.734 |
| mean over 14 transforms | 0.977 | 0.645 |
| worst transform (0.25x thumbnail) | 0.968 | 0.604 |
| real vs fully-synthetic only, mean over transforms | 0.989 | 0.763 |
| cross-source, unseen generator family (WildFake), clean | 0.966 | -- |

v2 numbers on the held-out 1400-image slice of SID-Set, calibration images excluded. The
deadline build (v1: clean 0.963, mean 0.958, worst 0.941) already beat a UnivFD-style linear
probe on 15 / 15 transforms in-distribution and by +0.12 to +0.25 AUC on an unseen generator
family; v2 then improves every one of the 15 conditions on the same images (+0.019 AUC on
average, 14 of 15 beyond overlapping bootstrap CIs). Built solo for TikTok TechJam 2026
(Track 5) in about four days on a laptop with no GPU; everything after the deadline is on the
same branch history.

![Ablation: features x training views vs a UnivFD-style baseline](results/ablation_chart.png)

**Four findings worth reading this repo for**

1. **The forensic branch is the entire cross-source generalisation** -- a CLIP-only probe
   scores 0.54 on thumbnails of an unseen generator (chance is 0.5); the forensic branch
   lifts it to 0.79.
2. **The same branch is what collapses under noise** (0.925 -> 0.810 at sigma 0.10), and a
   noise training view is what makes it safe. Alone they are worth -0.006 and +0.004; together
   +0.014. The interaction is the result, and it corrected our own first explanation.
3. **It is not reading JPEG history.** Re-encoding both classes identically moves every head
   by <= +0.0009 AUC; a bare blockiness scalar separates the classes at only 0.586.
4. **Four times the data plus a second frozen encoder is worth +0.019 AUC everywhere.** The
   post-deadline v2 (8000 images per class, a jitter training view, CLIP + DINOv2-small fusion
   chosen by CV) lifts all 15 conditions by +0.013 to +0.027 AUC and the mean TPR at 1% FPR
   from 0.50 to 0.65; the previously worst row, colour jitter, gains the most.

**Quickstart**

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python infer.py --input_dir samples --output preds.json     :: directory in, JSON out
```

Our weights ship in the repo (`artifacts/repostguard.joblib`, 28 KB); the two frozen encoders it
sits on top of are public checkpoints, so the **first** run downloads CLIP ViT-B/32 and
DINOv2-small (~440 MB together) from the Hugging Face hub and caches them. After that a run of
the two sample images takes about 30 s on a CPU laptop. `python app.py` opens a Gradio demo with
JPEG / blur / noise / crop sliders. The [3-minute video](https://youtu.be/wbeGLieLZ9c) is
rendered from the committed result CSVs by `scripts/make_video.py`; the uploaded cut is the
submitted one and therefore quotes the **v1** numbers, so re-running the script against the
current CSVs produces the v2 figures instead.

## Project overview

**Model (well under the 2B parameter limit):** frozen OpenAI CLIP `ViT-B/32` (~88M) + frozen `facebook/dinov2-small` CLS (~22M) + a 28-D forensic vector + a logistic-regression head. (The deadline build, v1, used CLIP alone; see *What changed after the deadline*.)

- **Feature variant, A/B-selected by cross-validation.** From a single CLIP forward we cache the **pre-projection** feature (`vision_model(...).pooler_output`, the post-LayerNorm CLS token, 768-D) and the **projected** feature (`visual_projection(pooler_output)`, 512-D, UnivFD's usual probe input); v2 adds the **DINOv2-small CLS** embedding (384-D) and the **fusion** of CLIP-proj + DINOv2 (896-D). Which variant ships is not asserted, it is picked by GroupKFold CV AUC on our own data: v1 chose `proj` (0.9536 vs `pre` 0.9414, 5-fold), v2 chose `fuse` (0.9806 vs `proj` 0.9776 vs DINOv2-only 0.9244, 3-fold on 96 000 rows). Pre-projection features are reported to win for linear-probe fake detection (Cozzolino et al., arXiv:2312.00195); on this data they did not, which is why the choice is measured rather than assumed.
- **Native-resolution forensic branch (28-D).** NPR-style residual stats, 8×8 block-DCT high/low energy ratios (aligned to the JPEG grid), FFT radial rings expressed as fractions of Nyquist (including the ≥0.75 bands that carry GAN/upsampling spectral peaks), plus color and Laplacian stats. Computed on a **native-scale 256 center crop** — never a bilinear downscale, which would low-pass-filter away exactly the high-frequency fingerprints this branch exists to measure.
- **Test-time augmentation (TTA).** Inference averages the original view with a JPEG-70 view and a 0.5× down/up view; the averaged CLIP embedding is re-L2-normalized so its scale does not depend on the view count. Train, eval, and infer all go through the **same** `embed_for_score` path so cached features match scored features exactly.
- **Sigmoid calibration.** After the head is fit, we sigmoid-calibrate it on a group-disjoint 30% slice of the validation set and report metrics only on the held-out 70% test slice, so `pred` is a usable P(AIGC) probability rather than a raw margin.

**Training data:** SID-Set (`saberzl/SID_Set`) binary labels: `0` real vs `1` AIGC-positive (`1` full-synthetic **and** `2` tampered, upweighted 5×); **8000 real + 8000 AIGC training images in v2** (2000 per class in the deadline build). One degraded view per official family — JPEG, blur, resize, **noise**, **jitter** — plus the clean view: 6 rows per image, each row embedded through the same TTA path as inference. The noise and jitter views were added after the robustness table showed the families we scored but never trained on were the weakest rows; the 2×2 ablation below shows the forensic branch caused the noise collapse and the noise view is what makes the branch safe (see *What actually makes it robust*). The official WildFake demonstration subset is **never used for training** — it is only a reference benchmark (see below).

## Scope

- **Transform grid.** The evaluation presets are the fifteen official Track-5 transforms: JPEG 90/70/50/30, Gaussian blur σ 0.5/1.0/2.0, resize 0.5×/0.25× then upsample, Gaussian noise σ 0.02/0.05/0.10, color jitter ±20%, center crop 80% (`src/augment.py`).
- **Size.** Frozen CLIP ViT-B/32 (~88M) + frozen DINOv2-small (~22M) + a linear head, about 110M in total; the competition cap was 2B.
- **Interface.** `python infer.py --input_dir PATH --output preds.json` emits `[{"image_path", "pred"∈[0,1]}]`, one row per input file, robust to unreadable/truncated files.
- **Data.** Trained on SID-Set only. The WildFake demonstration subset is a cross-source benchmark and is never used for training.

## Setup & installation

CPU-only works; a CUDA GPU makes feature extraction much faster (the code auto-uses CUDA when available).

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Steps to reproduce results

```bat
python scripts/make_samples.py                              :: tiny fixtures
python scripts/download_data.py --train_per_class 8000 --val_per_class 1000   :: v2 scale (v1 used 2000/class)
:: feature cache: 6 views/train image (clean/jpeg/blur/resize/noise/jitter), variants pre/proj/dino/fuse.
:: One process per shard; each holds CLIP + DINOv2 (~1.5 GB, single-core bound). Merge refuses a short cache.
for /L %i in (0,1,4) do start /b python scripts/extract_features.py --split train --augment --shard %i/5 --out artifacts/_feat/train_%i.npz
python scripts/extract_features.py --merge artifacts/_feat/train_0.npz artifacts/_feat/train_1.npz artifacts/_feat/train_2.npz artifacts/_feat/train_3.npz artifacts/_feat/train_4.npz --out artifacts/features_train.npz
python scripts/extract_features.py --split val
python scripts/train.py --variants proj,dino,fuse --cv_splits 3   :: GroupKFold A/B over variants, calibrate, write bundle
python scripts/make_tables.py                               :: robustness table + CIs
python scripts/make_tables.py --crops 4                     :: eval-time multi-crop TTA variant of the table
python scripts/make_chart.py                                :: robustness bar chart PNG
python scripts/error_analysis.py                            :: FP/FN note
python scripts/eval_demo.py                                 :: WildFake demonstration benchmark
python scripts/compare_ab.py                                :: A/B vs results/baseline/
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
| Transform | n | Accuracy | ROC AUC | 95% CI | TPR@1%FPR |
|---|---:|---:|---:|---|---:|
| `clean` | 1400 | 0.9286 | 0.9812 | [0.9749, 0.9868] | 0.7338 |
| `jpeg_90` | 1400 | 0.9414 | 0.9832 | [0.9769, 0.9884] | 0.7235 |
| `jpeg_70` | 1400 | 0.9450 | 0.9867 | [0.9814, 0.9912] | 0.7750 |
| `jpeg_50` | 1400 | 0.9371 | 0.9825 | [0.9767, 0.9876] | 0.7074 |
| `jpeg_30` | 1400 | 0.9121 | 0.9731 | [0.9656, 0.9802] | 0.5500 |
| `blur_0.5` | 1400 | 0.9250 | 0.9814 | [0.9750, 0.9868] | 0.6882 |
| `blur_1.0` | 1400 | 0.8636 | 0.9789 | [0.9719, 0.9850] | 0.7176 |
| `blur_2.0` | 1400 | 0.9021 | 0.9727 | [0.9646, 0.9794] | 0.6706 |
| `resize_0.5` | 1400 | 0.8879 | 0.9794 | [0.9723, 0.9853] | 0.7176 |
| `resize_0.25` | 1400 | 0.8950 | 0.9675 | [0.9584, 0.9750] | 0.6044 |
| `noise_0.02` | 1400 | 0.9257 | 0.9782 | [0.9709, 0.9848] | 0.6015 |
| `noise_0.05` | 1400 | 0.9171 | 0.9756 | [0.9678, 0.9825] | 0.5000 |
| `noise_0.10` | 1400 | 0.9193 | 0.9689 | [0.9600, 0.9768] | 0.5426 |
| `jitter_0.20` | 1400 | 0.9036 | 0.9676 | [0.9593, 0.9752] | 0.5397 |
| `crop_0.80` | 1400 | 0.9300 | 0.9792 | [0.9726, 0.9850] | 0.6941 |

**Clean AUC 0.9812; mean AUC across the 14 transformed conditions 0.9768 (-0.0044 vs clean); worst condition `resize_0.25` at 0.9675 (-0.0137 vs clean).** Every row is the same held-out test slice (1400 images, calibration images excluded) re-scored through the shipped inference path after the transform, so clean and transformed numbers are directly comparable.

![Robustness: clean vs social-media transforms](results/robustness_chart.png)
<!-- /ROBUSTNESS_TABLE -->

### What changed after the deadline: v1 -> v2 on the same 1400 images

The deadline build (v1, kept intact under `results/v1/` with its head and per-image scores)
trained on 2000 images per class with five training views. v2 keeps the evaluation protocol --
same held-out slice, same 15 transforms, same scoring path -- and changes four things:

1. **Full data.** The 8000-per-class SID-Set subset: 16 000 training images, 96 000 feature rows.
2. **Every scored family except crop is now a training view.** A colour-jitter view joins
   JPEG / blur / resize / noise (six views per image); `jitter_0.20` was v1's worst row.
3. **DINOv2-small fused with CLIP.** A 384-D DINOv2-small CLS embedding (22M parameters,
   clean view only) is concatenated with the 512-D CLIP projection and the 28-D forensic vector.
   The variant is chosen by 3-fold GroupKFold CV, not assumed: CLIP-only 0.9776, DINOv2-only
   0.9244, fusion **0.9806** CV AUC. DINOv2 alone is clearly worse; as a second view of the
   same image it still adds signal on top of CLIP.
4. **Content-seeded evaluation noise.** The Gaussian-noise rows are seeded from the pixel
   content, so the table is bit-reproducible across machines and re-runs.

<!-- AB_V1V2 -->
| Transform | v1 AUC (deadline build) | v2 AUC | delta AUC | CIs disjoint |
|---|---:|---:|---:|---|
| `clean` | 0.9629 | 0.9812 | +0.0183 | yes |
| `jpeg_90` | 0.9658 | 0.9832 | +0.0174 | yes |
| `jpeg_70` | 0.9736 | 0.9867 | +0.0131 | yes |
| `jpeg_50` | 0.9667 | 0.9825 | +0.0158 | yes |
| `jpeg_30` | 0.9557 | 0.9731 | +0.0174 | yes |
| `blur_0.5` | 0.9638 | 0.9814 | +0.0176 | yes |
| `blur_1.0` | 0.9622 | 0.9789 | +0.0167 | yes |
| `blur_2.0` | 0.9536 | 0.9727 | +0.0191 | yes |
| `resize_0.5` | 0.9621 | 0.9794 | +0.0173 | yes |
| `resize_0.25` | 0.9472 | 0.9675 | +0.0203 | yes |
| `noise_0.02` | 0.9572 | 0.9782 | +0.0210 | yes |
| `noise_0.05` | 0.9563 | 0.9756 | +0.0193 | yes |
| `noise_0.10` | 0.9512 | 0.9689 | +0.0177 | no |
| `jitter_0.20` | 0.9411 | 0.9676 | +0.0265 | yes |
| `crop_0.80` | 0.9578 | 0.9792 | +0.0214 | yes |

Mean change +0.0186 AUC over 15 conditions (largest gain `jitter_0.20` +0.0265, smallest `jpeg_70` +0.0131); 14 condition(s) improved beyond overlapping bootstrap CIs, 0 regressed beyond them. Same 1400 held-out images, same transforms, same scoring path; only the training data and the feature variant differ.
<!-- /AB_V1V2 -->

**Multi-crop test-time augmentation** was implemented as an evaluation-time option and measured
separately rather than folded into the head. `--crops N` adds N corner/centre crops to the view
set that is averaged before scoring, so at `--crops 4` the CLIP mean is over 7 views instead of
3 and the DINOv2 mean over 5 instead of 1; the forensic vector is computed once on the full
image and is unaffected. The head, however, was fit on 3-view CLIP / 1-view DINOv2 embeddings,
so changing the view count at score time moves the input distribution away from the one the
linear head was calibrated on. That is the honest reason to report this as its own A/B rather
than quietly switching it on (`results/ab_crops4.csv`):

<!-- CROPS_TABLE -->
_(being measured -- `python scripts/make_tables.py --crops 4` then `scripts/compare_ab.py`.)_
<!-- /CROPS_TABLE -->

### Which task definition? The same scores, split by class

Every headline number above scores **real vs all AIGC**, and in SID-Set "AIGC" includes the
*tampered* class -- real photographs with a locally generated edit. Most published detectors,
and most other entries in this track, score real vs **fully synthetic** only, which is the
easier problem. Since the per-image scores are persisted, both definitions come from the
same run (`scripts/per_class_table.py`, `results/robustness_by_class.csv`):

<!-- BY_CLASS -->
| Transform | AUC: real vs **all AIGC** (headline) | AUC: real vs **fully synthetic** | AUC: real vs **tampered** | TPR@1%FPR all | TPR@1%FPR synthetic |
|---|---:|---:|---:|---:|---:|
| `clean` | 0.9812 | 0.9902 [0.9860, 0.9939] | 0.9727 | 0.7338 | 0.8198 |
| `jpeg_90` | 0.9832 | 0.9917 [0.9879, 0.9951] | 0.9750 | 0.7235 | 0.8108 |
| `jpeg_70` | 0.9867 | 0.9945 [0.9917, 0.9968] | 0.9792 | 0.7750 | 0.8679 |
| `jpeg_50` | 0.9825 | 0.9937 [0.9906, 0.9964] | 0.9717 | 0.7074 | 0.8498 |
| `jpeg_30` | 0.9731 | 0.9871 [0.9824, 0.9917] | 0.9598 | 0.5500 | 0.6877 |
| `blur_0.5` | 0.9814 | 0.9900 [0.9857, 0.9938] | 0.9732 | 0.6882 | 0.7748 |
| `blur_1.0` | 0.9789 | 0.9908 [0.9860, 0.9947] | 0.9676 | 0.7176 | 0.8438 |
| `blur_2.0` | 0.9727 | 0.9917 [0.9869, 0.9956] | 0.9544 | 0.6706 | 0.8559 |
| `resize_0.5` | 0.9794 | 0.9922 [0.9877, 0.9958] | 0.9671 | 0.7176 | 0.8589 |
| `resize_0.25` | 0.9675 | 0.9911 [0.9864, 0.9951] | 0.9449 | 0.6044 | 0.8258 |
| `noise_0.02` | 0.9782 | 0.9882 [0.9828, 0.9929] | 0.9687 | 0.6015 | 0.6697 |
| `noise_0.05` | 0.9756 | 0.9872 [0.9819, 0.9921] | 0.9645 | 0.5000 | 0.5946 |
| `noise_0.10` | 0.9689 | 0.9838 [0.9780, 0.9892] | 0.9546 | 0.5426 | 0.6727 |
| `jitter_0.20` | 0.9676 | 0.9784 [0.9713, 0.9852] | 0.9571 | 0.5397 | 0.5886 |
| `crop_0.80` | 0.9792 | 0.9883 [0.9833, 0.9927] | 0.9704 | 0.6941 | 0.7748 |

n per row: 1400 (all), 1053 (real + fully synthetic), 1067 (real + tampered). Mean over transformed rows: fully synthetic 0.9892, tampered 0.9649.
<!-- /BY_CLASS -->

Read the middle column when comparing against a number that was computed the usual way, and
the right-hand column for where the model is actually weakest: locally edited photographs.

### What actually makes it robust: a 2x2 ablation against a published baseline

The robustness table is also how we found, and then correctly diagnosed, our own worst
behaviour. Our first account of it was wrong, and the ablation below is what corrected it.

Two axes, crossed: **features** (CLIP-only vs CLIP + native-resolution forensic) and
**training views** (the four we started with, clean/jpeg/blur/resize, vs five with a noise
view). The CLIP-only / 4-view corner is a **UnivFD-style linear probe** (Ojha et al., CVPR
2023) -- a published method, not a strawman -- and the reference the deadline configuration
(v1) had to beat. All four heads use v1's data scale (2000 images per class); the ablation
is about mechanism, and the mechanism carried over unchanged into v2. Every transformed image is embedded once and scored by all four heads from
column slices of the same vector, so the whole grid costs one sweep
(`scripts/ablation.py`, `results/ablation_table.csv`).

<!-- ABLATION -->
Mean AUC over the 14 transformed conditions (held-out SID-Set test slice, n = 1400 per cell):

| | 4 views (jpeg/blur/resize) | 5 views (+ noise) |
|---|---:|---:|
| **CLIP-only** | 0.9445 *(UnivFD-style baseline)* | 0.9480 |
| **CLIP + forensic** | **0.9389** (down) | **0.9583** *(v1, the deadline build)* |

Worst single transform: baseline 0.9237 (`jitter_0.20`), forensic alone **0.8096** (`noise_0.10`), v1 0.9411 (`jitter_0.20`). Clean AUC: baseline 0.9501, v1 0.9630. (All four heads here are 2000-image, CLIP-only-or-CLIP+forensic configurations; the v2 head is compared against v1 in *What changed after the deadline*.)

Cross-source (WildFake demo subset, a generator family absent from training, n = 2000 per cell):

| transform | UnivFD-style | + noise view | + forensic | v1 (both) |
|---|---:|---:|---:|---:|
| `clean` | 0.8094 | 0.7799 | 0.9189 | **0.9334** |
| `jpeg_30` | 0.7957 | 0.7895 | 0.9134 | **0.9327** |
| `resize_0.25` | 0.5392 | 0.5085 | 0.7938 | **0.7536** |
| `noise_0.05` | 0.7741 | 0.8137 | 0.8518 | **0.9290** |

![Ablation: features x training views vs a UnivFD-style baseline](results/ablation_chart.png)
<!-- /ABLATION -->

**What the grid says, in order of importance:**

1. **Cross-source, the forensic branch is the whole generalization story.** On a generator
   family the model never trained on, a UnivFD-style probe scores 0.81 clean and **0.54 on
   quarter-scale thumbnails -- barely above chance**. The forensic branch lifts that by
   +0.11 and +0.25. Low-level residual / DCT / spectral statistics of the generation
   process transfer across generators; the semantic embedding of "what our training fakes
   look like" does not.
2. **In-distribution, the forensic branch alone *hurts*.** It adds +0.013 on clean and on
   every jpeg/blur/resize/crop row, and collapses under additive noise (0.9250 -> 0.8096
   at sigma 0.10). Its 28 dimensions are exactly the high-frequency statistics broadband
   noise swamps. CLIP-only never had this problem. Our earlier write-up blamed "noise was
   scored but never trained"; that was the fix, not the cause.
3. **The noise view is what makes the forensic branch deployable.** Alone it is worth
   +0.0035; forensic alone is worth -0.0056; together they are worth +0.0138. The
   interaction is the finding: a high-frequency forensic branch is only safe when the
   training distribution contains the corruption that destroys it.
4. **Shipped beats the published baseline on 15 / 15 in-distribution rows** (worst row
   0.9237 -> 0.9411) and by +0.12 to +0.21 on every cross-source row. Honest caveat: no
   single in-distribution row's bootstrap CIs are disjoint -- consistently better, not
   significantly better per row.
5. `jitter_0.20` is the weakest row for the baseline and for us alike; it moves the CLIP
   embedding itself and the colour-agnostic forensic branch cannot help.

For the record, the current head against the head we had before the noise view, every
transform, same 1400-image slice (`results/baseline/robustness_table.csv` vs
`results/robustness_table.csv`, `scripts/compare_ab.py`). This table now folds in everything
that changed since that snapshot -- the noise view, then v2's data, jitter view and DINOv2
fusion; the isolated effect of the noise view is the 4-vs-5-view column above, and the
isolated effect of v2 is the v1 -> v2 table:

<!-- AB_TABLE -->
| Transform | baseline AUC | shipped AUC | delta AUC | CIs disjoint |
|---|---:|---:|---:|---|
| `clean` | 0.9634 | 0.9812 | +0.0178 | yes |
| `jpeg_90` | 0.9674 | 0.9832 | +0.0158 | yes |
| `jpeg_70` | 0.9728 | 0.9867 | +0.0139 | yes |
| `jpeg_50` | 0.9682 | 0.9825 | +0.0143 | yes |
| `jpeg_30` | 0.9547 | 0.9731 | +0.0184 | yes |
| `blur_0.5` | 0.9618 | 0.9814 | +0.0196 | yes |
| `blur_1.0` | 0.9567 | 0.9789 | +0.0222 | yes |
| `blur_2.0` | 0.9500 | 0.9727 | +0.0227 | yes |
| `resize_0.5` | 0.9567 | 0.9794 | +0.0227 | yes |
| `resize_0.25` | 0.9455 | 0.9675 | +0.0220 | yes |
| `noise_0.02` | 0.9244 | 0.9782 | +0.0538 | yes |
| `noise_0.05` | 0.8851 | 0.9756 | +0.0905 | yes |
| `noise_0.10` | 0.8117 | 0.9689 | +0.1572 | yes |
| `jitter_0.20` | 0.9362 | 0.9676 | +0.0314 | yes |
| `crop_0.80` | 0.9521 | 0.9792 | +0.0271 | yes |

15 transform(s) improved beyond overlapping bootstrap CIs; 0 regressed beyond them.
<!-- /AB_TABLE -->

### Is it reading compression history? A leakage control

Several Track-5 teams found that in SID-Set the authentic images ship as JPEG and the
synthetic ones as PNG, so a detector can score well by reading *compression history*. Our
download pipeline stores both classes as JPEG Q95, which equalises the file container but
not the history: reals are then double-compressed, fakes single-compressed, and our
forensic branch contains block-DCT statistics aligned to the JPEG grid -- the feature that
would notice. So we ran the control (`scripts/leakage_control.py`):

<!-- LEAKAGE -->
On the held-out test slice (1400 images; both classes are stored as JPEG Q95 by our own download pipeline):

**1. Does the shortcut exist in the data?** Hand-written scalars, no model. Separability is the AUROC of the scalar on its own, direction-agnostic; 0.5 means none.

| probe | separability | mean real | mean fake |
|---|---:|---:|---:|
| `blockiness_8px` | **0.5858** | 0.8259 | 1.5580 |
| `laplacian_energy` | **0.6013** | 15.5651 | 11.8971 |

**2. Does the model use it?** Every image re-encoded at JPEG Q95, both classes identically, one and two extra generations (which equalises compression history), then re-scored:

| head | clean AUC | +1 generation | +2 generations | delta (+1) |
|---|---:|---:|---:|---:|
| CLIP-only, 4 views (UnivFD-style) | 0.9501 | 0.9507 | 0.9510 | +0.0006 |
| CLIP-only, 5 views | 0.9511 | 0.9516 | 0.9520 | +0.0005 |
| CLIP + forensic, 4 views | 0.9635 | 0.9640 | 0.9644 | +0.0005 |
| CLIP + forensic, 5 views (v1, deadline build) | 0.9630 | 0.9635 | 0.9637 | +0.0005 |
<!-- /LEAKAGE -->

## WildFake demonstration benchmark (never used in training)

The official demonstration subset (`techjam-aigc/wildfake-eval-subset`, `default` config = COCO val2017 reals + DALL·E-3 Advanced fakes) scored through the **shipped** inference path on a balanced subsample. This measures cross-source generalization to a generator family absent from SID-Set training. Full table: `results/demo_benchmark.csv`.

> **v1 -> v2, matched.** The subsample is seeded, so these 2000 images are the same ones the cross-source ablation table scored (`results/ablation_demo_table.csv`, row `forensic_5v` = the v1 head): clean 0.9334 -> **0.9657**, `noise_0.05` 0.9290 -> **0.9694**, `jpeg_30` 0.9327 -> 0.9250 and `resize_0.25` 0.7536 -> 0.7599 (both within overlapping CIs). The in-distribution gains transfer to the unseen generator on clean and noisy images; heavy JPEG and 0.25x thumbnails of an unseen generator remain the open problem, and thumbnails are still where the model is weakest by a wide margin. (The v1 snapshot in `results/v1/demo_benchmark.csv` was scored on a 1000-image subsample and is not directly comparable.)

<!-- DEMO_TABLE -->
| Transform | n | Accuracy | ROC AUC | 95% CI |
|---|---:|---:|---:|---|
| `clean` | 2000 | 0.8835 | 0.9657 | [0.9586, 0.9722] |
| `jpeg_30` | 2000 | 0.7870 | 0.9250 | [0.9134, 0.9356] |
| `resize_0.25` | 2000 | 0.6660 | 0.7599 | [0.7384, 0.7811] |
| `noise_0.05` | 2000 | 0.8855 | 0.9694 | [0.9623, 0.9756] |

**Cross-source clean AUC 0.9657 on 2000 balanced images from a generator family (DALL-E-3) absent from SID-Set training** -- an out-of-distribution check, not a tuning target.
<!-- /DEMO_TABLE -->

## Error analysis note

Representative false positives (authentic images flagged AIGC — creator harm), false negatives (generated images missed, especially after JPEG-30), and the score drift under JPEG-30, in `results/error_analysis.json` (`scripts/error_analysis.py`).

<!-- ERROR_ANALYSIS -->
On the held-out test slice (1400 images: 720 real, 680 AIGC) at the shipped 0.5 threshold:

- **False positives** (authentic flagged AIGC -- direct creator harm): 71/720 = **9.86% FPR**.
- **False negatives** (generated images missed): 29/680 = **4.26% FNR**.
- **FPR@95%TPR**: 8.19% clean, 12.22% after JPEG-30 -- the price in flagged authentic images if the product insisted on catching 95% of AIGC.
- **JPEG-30 score drift**: real +0.0234, AIGC -0.0061 mean change in P(AIGC); 39 real images flip into false positives and 14 AIGC images flip into false negatives. AUC 0.9812 clean vs 0.9731 after JPEG-30.

The `k` highest-scoring FPs and lowest-scoring FNs, with their per-image clean and JPEG-30 scores, are in `results/error_analysis.json`.
<!-- /ERROR_ANALYSIS -->

**Trade-offs.** We keep the decision threshold at 0.5 and report FPR@95%TPR so the creator-harm cost of false positives is explicit. The tampered-class upweight raises recall on locally-edited images at some cost to clean precision. TTA recovers part of the transform-induced AUC drop but triples inference cost per image.

## Design choices

| Choice | Why |
|---|---|
| Frozen CLIP + linear head | Strong cross-generator baseline (UnivFD-style); no 2B-class model; trains in seconds |
| CLIP pre / proj / DINOv2 / fusion, chosen by CV | The literature favours pre-projection CLIP; our data picked `proj` (v1) and CLIP + DINOv2 fusion (v2), so the choice is measured, not assumed |
| DINOv2-small as a second frozen view (v2) | A different pre-training objective from CLIP. Alone it is weaker (CV AUC 0.924 vs 0.978); fused it adds signal (0.981) for +22M parameters |
| Native-resolution forensic branch | CLIP loses high-frequency traces; a 128×128 downscale aliases them away |
| Train-time official augmentations | Aligns the classifier with redistribution, not just clean SID-Set |
| Every scored family except crop is also a training view | The families we scored but never trained on were measurably our weakest rows; adding the noise view lifted `noise_0.10` AUC by +0.166, and v2 adds the jitter view |
| TTA (clean + JPEG-70 + 0.5×) | Recovers part of the transform-induced drop the table is scored on |
| Sigmoid calibration on held-out val | `pred` is a usable probability, evaluated without calibration leakage |

## Limitations & what we would improve with more time

- **Generator diversity.** Trained on SID-Set only. Unknown commercial generators (Flux, Midjourney v7, SD3) still shift CLIP geometry. The listed resources (CIFAKE, WildFake-train) are allowed for training and would be the first addition — mixing generator families is the highest-leverage next step.
- **Remaining untrained family.** `crop` is the one scored family that is still not a training view (v2 added `jitter`). Crop was addressed at evaluation time instead -- multi-crop TTA, `make_tables.py --crops N`, reported separately in `results/ab_crops4.csv` -- so the crop row stays an honest out-of-training check.
- **Hardest rows.** Extreme 0.25× thumbnails remain the weakest transform; a small learned frequency head or a second forensic scale could help.
- **Local edits / face swaps** are only partially covered (tampered class upweight); a dedicated localization branch is out of scope here.
- **Batched inference.** `embed_for_score` scores one image per forward, so the only way to use all cores is to shard across processes -- and each process holds its own ~1.5 GB copy of CLIP + DINOv2 and is single-core bound (about 10 s per training image for six views: three TTA CLIP forwards each, DINOv2, and the forensic branch at native resolution). On a 16 GB box that caps us at 4-5 workers; more just pages. Batching N images per forward would need one model copy and beat all the shards; we kept the one-image path because it is the single code path shared by train/eval/infer, which is what guarantees cached features and live scores are byte-identical.
- **Scale.** v1 trained on 2000/class for the deadline; v2 uses the full 8000/class subset on the same CPU box (sharded, checkpointed extraction). The next lever is generator diversity, not more SID-Set images.
- **False positives on heavily filtered real photos** hurt creators; a per-creator threshold or an abstain band would reduce that harm.

## Reproduction on a CUDA machine

Everything above was produced on a CPU-only laptop; the pipeline is CPU/GPU agnostic (`get_device()` auto-selects CUDA), and on a GPU box the same commands run in minutes instead of hours:

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python scripts/download_data.py --train_per_class 8000 --val_per_class 1000
python scripts/extract_features.py --split train --augment
python scripts/extract_features.py --split val
python scripts/train.py --variants proj,dino,fuse
```

A modest GPU (e.g. GTX 1650, fp16) cuts the six-view extraction from hours to minutes. For a stronger encoder, swap `clip_model_id` to `openai/clip-vit-large-patch14` in `configs/default.yaml` (~15–40 img/s fp16 on a GTX 1650, still well under 2B); the pre-projection variant becomes 1024-D and everything else is unchanged.

## Authorship

Solo project (one author in the git history). Model, forensic branch, evaluation harness,
ablation, leakage control, video pipeline and write-up are all in this repository; the
commit messages record what was found when, including the two explanations that turned
out to be wrong.

## Tools

Python, PyTorch (CPU or CUDA), Hugging Face `transformers` (CLIP) + `datasets` (SID-Set / demo subset streaming), scikit-learn, OpenCV, SciPy, NumPy, pandas, matplotlib, Gradio.

## Repository layout

```
infer.py                 directory -> preds.json (the shipped entry point)
app.py                   Gradio demo with transform sliders
src/                     features (CLIP + forensic), augment (15 transforms), eval, error analysis
scripts/                 data download, feature cache, train, tables, ablation, leakage control, video
results/                 every table in the README, per-image scores, baseline snapshot
artifacts/               shipped head (repostguard.joblib) + WEIGHTS.txt
docs/                    findings write-up, competitor review, submission sheet, video script
tests/                   52 tests; dataset-dependent ones skip on a clean checkout
```

## License

No license file yet: the code is shared for reading and reproduction. SID-Set, the WildFake
demonstration subset and the CLIP weights keep their upstream licenses.
