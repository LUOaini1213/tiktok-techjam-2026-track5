# RepostGuard

> Research prototype based on the TikTok TechJam 2026 Track 5 brief.
> **Never submitted to Devpost. No contest result or placement is claimed.**

[![tests](https://github.com/LUOaini1213/tiktok-techjam-2026-track5/actions/workflows/tests.yml/badge.svg)](https://github.com/LUOaini1213/tiktok-techjam-2026-track5/actions/workflows/tests.yml)

Detect AI-generated images after compression, thumbnails and repeated sharing,
with a conservative **Likely AIGC / Needs review / Low AIGC signal** decision policy.
Low signal is not proof of authenticity, and a calibrated score is not a universal
probability for an arbitrary deployment population.

## What's new in v3

- **Corrected data accounting.** The old 16,000-file training cache contained only
  11,997 unique files; 19 files overlapped training and validation. The downloader
  now deduplicates resumed streams. The v3 candidate fits on unique originals and excludes
  decoded-pixel/near matches across training and evaluation.
- **Generator diversity, within local CPU resources.** SID-Set is supplemented by
  multiple LatDiff generators, StyleGAN2 and BigGAN from the authors' Community
  Forensics Small dataset. The whole VQDiffusion family is held out. No COCO or
  WildFake reference images are deliberately sampled for training, calibration or development.
  SID includes COCO-derived captions/edits, so shared parent photos cannot be ruled out.
- **Actual repost chains.** Crop → thumbnail → JPEG, noise → JPEG, blur → resize →
  JPEG, and two rounds of sharing. Matching training views use the same parameter
  distributions for both classes; they add no extra inference forwards.
- **Lower-risk decisions.** Probability fitting and threshold selection use
  disjoint calibration originals. Worst-view thresholds count original images,
  not correlated augmentation rows. FPR, recall, false-clear rate, coverage and
  review rate are published together.
- **Locked evaluation.** Grouped CV and final fitting use the same consistency
  recipe. A candidate or small blend is selected on development only, subject to
  clean/thumbnail/SID regression guards. The model checksum is frozen before the
  new sealed test is scored. Paired bootstrap intervals compare matched originals.

CommunityForensics-Small is **CC-BY-NC-SA-4.0**. The v3 model is for
**non-commercial research**; a public repository does not grant commercial rights.
Dataset images are not redistributed.

## Default V3 results

<!-- V3_RESULTS -->
The development-selected model was frozen before these new tests were scored.
Both models see the same originals and transformations within each comparison.

| Test domain | Originals | Conditions | v2 mean AUC | v3 mean AUC | v3 review rate |
|---|---:|---:|---:|---:|---:|
| Fresh SID | 256 | 19 | 0.9846 | 0.9865 | 74.7% |
| Held-out VQDiffusion + CF real images | 192 | 19 | 0.8403 | 0.9501 | 92.5% |
| All newly sealed images | 448 | 19 | 0.9299 | 0.9675 | 82.3% |
| COCO / DALL·E demonstration reference | 400 | 6 | 0.9013 | 0.9160 | 85.1% |

These are descriptive means across conditions, not independent repeated trials.
CF inputs are uniformly fit to 512x512 before evaluation. The reference is advisory,
not a contest score, and is excluded from the sealed aggregate.

At a common binary threshold of 0.5, sealed mean accuracy rises from **84.9% to
90.1%** and FPR falls from 14.9% to 6.9%. The shipped conservative three-way policy
instead gives mean FPR **0.0235%**, AI recall **26.4%**, and review rate **82.3%**.
That is a low-coverage research policy, not an automatic moderation solution or a
universal FPR guarantee. Its one sealed false flag must not be rounded into a claim
of zero errors.

There are regressions: SID local-edit mean AUC decreases from 0.9821 to 0.9784;
reference FPR at 0.5 increases from 14.7% to 22.5%, despite better mean AUC.
The reference quarter-scale condition still has AUC only 0.7924 and shipped-policy
FPR 2.5%. No model or threshold was changed in response to these diagnostics.

[Full matched report and plot](results/v3/REPORT.md) ·
[per-condition confidence intervals](results/v3/evaluation.csv) ·
[paired AUC-difference intervals](results/v3/paired_auc.csv) ·
[deployment checks](results/v3/preflight.json).

Local release checks: **68 passed, 2 skipped**; real CLI inference, unreadable-file
fallback, the repost UI callback and a localhost HTTP demo smoke test all pass.
<!-- /V3_RESULTS -->

The mixed-data candidate uses 12,765 unique training originals; the new protocol has 1,004 calibration originals,
192 external development originals plus a filtered historical SID development slice,
and 448 newly sealed test originals (256 SID + 192 Community Forensics).
A fixed-seed 400-image COCO/DALL·E reference slice is evaluated after the
model is locked and is not part of the sealed aggregate. It is a demonstration
benchmark, not a competition score. Complete methodology: [v3 protocol](docs/v3_protocol.md).

Historical v1/v2 metrics and ablations are preserved in [v2 history](docs/v2_history.md)
and [v2 snapshot](results/v2/). They are not contamination-free final-test claims.
The old JPEG re-encoding control does not prove that source/compression bias is absent.

## V4 research iteration (explicit opt-in)

The bounded CPU experiments compare two new global logistic heads, a 64-unit MLP,
histogram boosting, two local-feature logistic heads and two fixed 50/50 blends.
**None passes the predeclared development gates. V3 remains the default.** The
experimental artifact keeps the V3 classifier and fits a new sigmoid calibration
and decision policy on independent calibration originals; it is not a stronger
backbone or a successful local detector.

The trade-off is visible on the 256-original, six-condition development set: the
new C=.01 global logistic head raises external mean AUC from 0.8937 to 0.9521,
but lowers SID real-versus-local-edit mean AUC from 0.9812 to 0.9629. The allowed
loss was 0.005, so the head is rejected despite lower preview review rate. All
trials, including unsuccessful local features and the bounded MLP's convergence warnings,
are retained in [selection.json](results/v4/selection.json) and
[head_trials.json](results/v4/head_trials.json).

The experimental 1,748-D descriptor extends the exact 924-D global path with
normalized DINO patch dispersion (768-D) and native corner forensic summaries
(56-D). Patches share the CLS forward; no second encoder is added. These pooled
features are **not localization masks or validated explanations**. Because the
selected head ignores them, the shipped opt-in bundle uses a 924-D packaging
adapter with exactly unchanged frozen-test predictions and does not extract the
unused local features during inference.

New calibration uses 1,280 originals, split into 640 probability-fit and 640
final-policy originals (320 real and 320 AI in each half). Development preview
thresholds reuse probability-fit labels and are diagnostic, not certification.
The unique head is selected before the final-policy group is scored; there is
no reselection after final calibration. The model and thresholds are then locked
before extracting the new 384-original, 19-condition test.

The published opt-in metadata attributes the retained classifier to V3's actual
training data, code and weighting, separately from V4 calibration/selection code.
The frozen trial bundle is preserved verbatim, with its provenance-only release
correction recorded in the packaged metadata. No estimator or threshold is refit
during packaging. Monotone probability recalibration changes operating decisions,
not evidence of improved ranking AUC.

The new thresholds cover **clean and crop-resize-JPEG only** under the sampled
calibration distribution. Pointwise 95% bounds are not simultaneous, platform-wide
or unseen-transform guarantees. The independent real test has only 192 originals:
even zero false flags cannot certify a 1% error rate. See the
[locked report](results/v4/REPORT.md), [protocol](docs/v4_protocol.md),
[feature-branch diagnosis](results/v4/diagnosis/ablations.json), and
[deployment checks](results/v4/preflight.json).

After following Quickstart, select the experimental artifact explicitly:

```text
python infer.py --input_dir samples --output preds.json --weights results/v4/repostguard.joblib
python app.py --weights results/v4/repostguard.joblib
```

Its CLI JSON contract is unchanged. The demo labels V4's calibration scope and
keeps unsupported transformations distinct from sampled calibration conditions. The full
CPU experiment is `python scripts/run_v4.py --prepared` **with archived V3
training images/features and the completed V4 acquisition already present**;
`scripts/prepare_v4.py` prepares new sources but does not reconstruct unpublished
legacy caches. A clean clone supports inference and portable evidence tests, not
a claim of one-click, bit-identical full retraining. Source images and large
feature caches are not published. The Community Forensics non-commercial terms
and the no-Devpost-submission statement apply to V4 too.

### V4 matched sealed-test outcome

Both versions score the same **384 originals × 19 conditions**, without using
these results for model or threshold selection. V4's selected fallback and its
independently recalibrated V3 comparison are identical, not separate new models.

| Model / primary policy | Mean AUC | Real false flags | AI flag recall | AI false clear | Review |
|---|---:|---:|---:|---:|---:|
| Default V3 / frozen strict | 0.9044 | 0 observed | 15.0% | 0.33% | 87.2% |
| Opt-in recalibration / conditionwise | 0.9044 | 0 observed | 30.9% | 3.32% | 63.8% |

Rates are means across correlated conditions, not independent repeated trials.
Lower review comes with **more false clearances**: counting each AI original once
if *any* of its 19 views is cleared gives **3/192** for default V3 versus
**28/192** for the opt-in policy. The latter is 14.6%, not the pooled-view 3.32%.
Zero observed real flags on 192 originals still has a one-sided 95% upper bound
of **1.55%**, so this test cannot certify a 1% FPR guarantee.

Generalisation remains the main failure: mean AUC is **0.9893 on fresh SID**, but
only **0.8198 on the named StyleGANXL holdout plus matched CF real sources**.
The opt-in policy clears 26/96 CF AI originals in at least one condition. StyleGAN2
was already training data, so this is not a whole-architecture holdout. These are
matched benchmark observations, not evidence of a stronger production detector.
The paired mean-AUC difference is zero (all four reported domain intervals are
[0, 0]), as expected for monotone recalibration of the retained classifier.

A representative failure is CF AI `0a54ca4f…` at JPEG-90: its opt-in score 0.1045
falls below the low-signal threshold 0.1163. No authenticity claim is justified.
Exact source IDs, transform-specific false positives at 0.5, false negatives and
policy false-clear cases are in [the error note](results/v4/error_analysis.json).
The **default artifact is unchanged**; V4 is available only for explicit research
comparison, not silently promoted.

Local V4 release checks: **88 passed, 2 skipped**. Both pretest/report byte locks,
exact global-only packaging predictions, the real directory-to-JSON CLI,
unreadable-file fallback and the seven-input/three-output localhost demo pass.
The single-image versus batched calibration score drift is below the fixed
1e-5 numerical guard; this is a measured runtime check, not a platform guarantee.

## Quickstart

Python 3.12 or 3.13 in a virtual environment is recommended; the local v3 run uses 3.13.
Install the pinned scikit-learn 1.9.0 dependency before loading the serialized head.

```text
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -c constraints-v3.txt
python infer.py --input_dir samples --output preds.json
python app.py
```

The first run downloads the public CLIP and DINOv2 checkpoints, then caches them.
The [constraints file](constraints-v3.txt) records the measured local package
versions, not a complete cross-platform OS lockfile. Use an isolated environment;
do not install both OpenCV GUI and headless variants into it.
There is no cloud inference service or paid API requirement. CPU inference defaults
to four threads; benchmark your own machine with `--threads N`. CUDA is detected
automatically, but no GPU throughput claim is made here. `--no_tta` is a diagnostic
speed option, not the evaluated default.

The required directory-to-JSON interface is unchanged:

```json
[
  {"image_path": "C:\\data\\a.jpg", "pred": 0.92},
  {"image_path": "C:\\data\\b.jpg", "pred": 0.11}
]
```

One row is emitted per supported image file, including nested folders. EXIF orientation
and RGB conversion are applied. A broken image gets the neutral fallback score 0.5
and a warning on stderr; that fallback is not a successful model prediction. Empty
or missing input folders produce `[]`. Thresholds live in model metadata, so the
machine-readable contract remains exactly `{image_path,pred}`.

## Model and compute

Frozen OpenAI CLIP ViT-B/32 projected features (512-D), frozen DINOv2-small CLS
(384-D), and native-scale residual/DCT/FFT/color statistics (28-D) form a 924-D vector.
CLIP averages clean, JPEG-70 and half-size views; DINOv2 uses the clean view.
Feature extraction, evaluation, CLI and UI share the same scoring path.

The active vision towers have 109,905,792 parameters. The loader also instantiates
CLIP's unused text tower: both full loaded encoders total 173,333,889 parameters,
still far below 2B. Bundle metadata records checkpoint snapshots and runtime versions.
Only the linear head, sigmoid calibration and optional convex blend are trained.
The encoders stay frozen and feature caches make repeat head fitting inexpensive.
Multi-crop inference remains an experiment: the old measured version regressed
on average, so it is not silently enabled.

The Gradio demo offers manual edits and named multi-stage repost scenarios. A named
scenario overrides the sliders rather than stacking hidden extra transforms. The
three-way verdict explicitly distinguishes abstention from a claimed detection.

## Reproduce and verify

The exact v3 run reuses local v2 SID images and the legacy feature cache; those
licensed, large caches are not committed. See [the protocol](docs/v3_protocol.md)
for clean-checkout prerequisites and the limits of bit-identical reproduction.
The [historical commands](docs/v2_history.md#steps-to-reproduce-results) build the
legacy caches; re-downloading with the corrected loader may change that old subset.

```text
python scripts/prepare_v3.py
python scripts/extract_v3.py --roles train,cal,dev
python scripts/train_v3.py
python scripts/extract_v3.py --roles test
python scripts/eval_v3.py
python scripts/validate_v3.py
python -m pip install pytest
python -m pytest tests/ -q
python infer.py --input_dir samples --output preds.json
```

On Windows, `pythonw scripts/run_v3.py` runs the resumable CPU workflow without a
console window. It logs to `logs/v3/` and checkpoints individual feature rows.
It never pushes Git or automatically changes the shipped artifact.
`precache_v3.py` is an optional bounded helper using the same view recipe and cache
fingerprint. The test helper refuses to run before model/manifest locking.
After reviewing the completed report, `python scripts/verify_v3.py --promote`
checks the evidence checksums, installs the already-selected head, runs the full
test suite and exercises the real CLI and UI. It refuses to overwrite a shipped
artifact changed by another task. It does not select a new model from test results.

Evaluation measures all 15 official clean/single-transform conditions and four
ordered chains on the new test originals. AUC confidence intervals and paired
AUC-difference intervals resample matched originals. Per-condition measurements
do not turn repeated views into independent data. Calibration FPR bounds are
distribution-specific; they are not a universal deployment guarantee.

Community Forensics is benchmarked after a common 512x512 bicubic fit for both
classes, before the named transforms. Its scores describe that aligned benchmark,
not native-resolution source files. CLI/UI inputs keep their native geometry by
default; to reproduce a Community Forensics row, first apply the documented fit.

## Limitations and error analysis

- Lower false-flag rates can reduce AI recall and send many cases to review.
  All three must be read together; abstention is not counted as correct detection.
- The fresh test is small and covers a narrow set of real sources and generators.
  It does not establish platform-wide or unseen-generator robustness.
- Uniform RGB geometry and lossless caching do not remove source acquisition,
  original JPEG history, semantic-content bias or unknown encoder-pretraining overlap.
- SID uses COCO-derived edits and captions. Protected-reference bitmap and
  near-duplicate checks do not certify parent-photo disjointness; the official
  reference remains advisory, not a pristine independent generalisation claim.
- SID tampered/local-edit images remain AIGC-positive; whole-image synthesis and
  local edits are not equally difficult. Tampered weight is selected in grouped CV.
  Transformed images keep their source-level label. A crop can remove local-edit
  evidence; this benchmark does not use masks to relabel the remaining pixels.
- Heavy downscaling can destroy useful forensic evidence. There is no guarantee of
  recovery after information has been removed.
- Public encoder weights and datasets retain upstream terms. Community Forensics
  Small makes this iteration non-commercial research; production use needs an
  appropriately licensed replacement and independent validation.

See the matched v3 report for actual false flags, false clearances and 0.5-threshold
false negatives. Historical [error analysis](results/v2/error_analysis.json) is kept
separately, not relabelled as the current model.

Representative current errors (source IDs and all scores are in
[error_analysis.json](results/v3/error_analysis.json)):

- Real reference `0_00902`: score 0.2619 clean, but 0.9952 after quarter-scale
  down/up-sampling, producing a false AI flag. All five reference false flags occur
  in that condition, on five distinct originals.
- Synthetic reference `1_00215`: score 0.0083 clean and 0.0018 at JPEG-30. The latter
  falls below the low-signal threshold, a false clearance. Low signal cannot certify
  authenticity even when the conservative policy is used.
- Held-out CF synthetic `024ad623…`: score 0.0180 clean and 0.0033 with noise 0.10.
  Both are false negatives at 0.5, but both remain **Needs review** under the shipped
  policy. An abstention is not a successful detection.

These paired observations show failure patterns, not a proven causal attribution
to a particular feature branch. Transformed rows of one original are correlated;
they must not be counted as independent evidence for a safety guarantee.

## Video, authorship and tools

The public [three-minute historical demo](https://youtu.be/wbeGLieLZ9c) quotes v1
figures and does not represent v3 or an actual Devpost entry. `python app.py` is the
current interactive demo. No updated YouTube upload is claimed.

Solo project. Repository history records implementation and experiment provenance.
Tools: Python, PyTorch, Hugging Face Transformers/Datasets, scikit-learn, NumPy,
SciPy, OpenCV, matplotlib and Gradio.

## Repository layout

```text
infer.py / app.py        CLI contract and interactive repost demo
src/                    frozen features, transformations, policy and score path
scripts/*v3*.py          provenance, audit, caches, training, locked evaluation
scripts/*v4*.py          bounded trials, independent calibration, opt-in verification
artifacts/              shipped selected head and WEIGHTS.txt
results/v3/             matched report, intervals, selection lock and data manifest
results/v4/             diagnosis, failed trials, new locked report and opt-in head
results/v2/             archived baseline head and historical diagnostics
docs/                   current protocol and preserved historical README
tests/                  contracts, feature consistency, groups, policy and resume
```

No new code license is granted by this update. Upstream datasets and models keep
their licenses; see their source cards and the v3 protocol.
