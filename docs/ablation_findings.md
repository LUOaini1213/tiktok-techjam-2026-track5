# Ablation findings — what actually makes RepostGuard robust

*Analysis branch, produced after the submission deadline. Nothing here changes the shipped
model or any submitted number; it explains them.*

Full table: `results/ablation_table.csv` (60 cells). Chart: `results/ablation_chart.png`.
Script: `scripts/ablation.py` — one CLIP embedding per transformed image, four heads scored
from column slices of it, so the whole 2×2 costs one sweep of the 15-transform grid.

## The 2×2

Two axes, crossed:

- **features** — CLIP-only (512-D) vs CLIP + native-resolution forensic (540-D)
- **training views** — 4 (clean / jpeg / blur / resize) vs 5 (+ noise)

The CLIP-only / 4-view corner is a **UnivFD-style linear probe** (Ojha et al., CVPR 2023): a
linear classifier on frozen CLIP features trained with jpeg/blur augmentation. It is a
published method, so it is the reference the shipped configuration has to beat. All four
heads are fit and sigmoid-calibrated identically on the same slices as the shipped model.

Mean AUC over the 14 transformed conditions, held-out SID-Set test slice (n = 1400 per cell):

| | 4 views | 5 views (+noise) |
|---|---:|---:|
| **CLIP-only** | 0.9445 *(UnivFD baseline)* | 0.9480 |
| **CLIP + forensic** | **0.9389** ↓ | **0.9583** *(shipped)* |

Worst single transform: UnivFD 0.9237 (`jitter_0.20`) · forensic-only **0.8096** (`noise_0.10`) ·
shipped 0.9411 (`jitter_0.20`).

## Finding 1 — the forensic branch alone *hurts* robustness

Added on its own, the forensic branch lowers mean transformed AUC from 0.9445 to 0.9389.
The damage is entirely in the noise family:

| transform | CLIP-only | + forensic | Δ |
|---|---:|---:|---:|
| noise σ0.02 | 0.9417 | 0.9266 | −0.0151 |
| noise σ0.05 | 0.9412 | 0.8842 | −0.0570 |
| noise σ0.10 | 0.9250 | **0.8096** | **−0.1154** |

Everywhere else it helps — clean +0.0136, and +0.010 to +0.017 on every jpeg, blur, resize
and crop row. The mechanism is not mysterious: the 28 forensic dimensions are NPR residual
statistics, 8×8 block-DCT energy ratios and FFT radial-ring energies. Additive Gaussian
noise is broadband high-frequency energy, so it swamps exactly those statistics. A head
trained without noise views learns to trust dimensions that go haywire under noise.

**This corrects the account given in the README and the Devpost write-up.** We said the
noise collapse happened because "noise was scored but never a training view." That is true
but it is not the cause. CLIP-only never collapsed under noise — 0.9250 at σ0.10 with no
noise training at all. The collapse to ~0.81 was *introduced by the forensic branch*.

## Finding 2 — the noise view is what makes the forensic branch safe

| | mean Δ vs UnivFD |
|---|---:|
| forensic alone | −0.0056 |
| noise view alone | +0.0035 |
| both | **+0.0138** |

The components do not add. Their sum is −0.0021; together they deliver +0.0138, an
interaction of roughly +0.016. The noise training view does little for CLIP-only
(+0.0035, since CLIP-only was never fragile) and everything for the forensic head: it
teaches the classifier that forensic dimensions are unreliable when noise-swamped, which
lets it keep their +0.01–0.017 benefit on every other transform without paying the noise
penalty.

Read as engineering rather than as a slogan: **a high-frequency forensic branch is only
deployable if the training distribution includes the corruption that destroys it.**

## Finding 3 — shipped beats the published baseline on every row

Shipped vs UnivFD-style, all 15 transforms:

- 15 / 15 rows higher
- mean over transformed rows +0.0138; worst-case row 0.9237 → 0.9411 (+0.0174)
- clean +0.0129

Honest caveat: **no single row's 95 % bootstrap intervals are disjoint.** The gain is
consistent across the grid but modest per row. The claim we can make is "consistently
better on all 15 official conditions," not "significantly better on condition X."

## Finding 4 — `jitter` is a CLIP weakness, not one we introduced

The shipped model's weakest row is `jitter_0.20` at 0.9411. It is also UnivFD's weakest row
(0.9237). Colour jitter shifts the CLIP embedding itself; the forensic branch is
colour-agnostic and cannot help. Earlier we listed jitter alongside noise as a "scored but
untrained family" as if the fix were the same. The ablation says otherwise: a jitter
training view would help CLIP-only and shipped alike, and it is the right next
experiment, but it is a different mechanism from the noise story.

## What this changes in how the project should be described

1. The Innovation claim moves from "we found a missing training view" (process) to "a
   native-resolution forensic branch is a double-edged component: +0.013 on clean and
   compression, catastrophic under additive noise, and only deployable together with
   noise-view training" (mechanism). The second is a finding; the first was housekeeping.
2. The Technical Execution section gains an external reference. Every number was
   previously compared only to our own earlier head; it can now be compared to UnivFD.
3. The README's "Closing the augmentation gap" section is accurate in its numbers but
   attributes the collapse to the wrong cause. It should say the forensic branch caused it.

## Still unknown — pending the cross-source run

`results/ablation_demo_table.csv` (WildFake demo subset, never trained on) will answer the
question this analysis raises: **does the forensic branch also hurt on a generator family
it has never seen?** Our worst cross-source number is `resize_0.25` = 0.7700 for the shipped
head. If CLIP-only beats shipped there, the forensic dimensions are overfitting to SID-Set's
generators and the right production move may be a smaller forensic weight or a
noise/resize-aware gating of that branch.
