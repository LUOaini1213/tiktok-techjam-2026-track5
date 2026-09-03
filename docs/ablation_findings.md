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

## Finding 5 — cross-source, the forensic branch is the *whole* generalization story

Same four heads on the WildFake demonstration subset (COCO val2017 reals vs DALL·E-3
Advanced fakes — a generator family absent from training), n = 2000 per cell,
`results/ablation_demo_table.csv`:

| transform | UnivFD (CLIP-only) | + noise view | + forensic | shipped (both) |
|---|---:|---:|---:|---:|
| clean | 0.8094 | 0.7799 | 0.9189 | **0.9334** |
| jpeg_30 | 0.7957 | 0.7895 | 0.9134 | **0.9327** |
| resize_0.25 | **0.5392** | 0.5085 | **0.7938** | 0.7536 |
| noise_0.05 | 0.7741 | 0.8137 | 0.8518 | **0.9290** |

The hypothesis this document raised — that the forensic dimensions might be overfitting
SID-Set's generators — is **refuted, and inverted**. It is the CLIP embedding that fails
to transfer: a UnivFD-style probe scores 0.8094 clean on the unseen family and **0.5392 on
quarter-scale thumbnails, barely above chance**. The forensic branch lifts that by +0.11
on clean and +0.25 on thumbnails. Low-level residual / DCT / spectral statistics of the
generation process transfer across generators; the semantic embedding of "what SID-Set
fakes look like" does not.

Shipped beats the published baseline by +0.12 to +0.21 on every cross-source row.

Two honest nuances in this table:

- `resize_0.25` is the one cell where shipped (0.7536) is *not* the best head: forensic
  without the noise view scores 0.7938. The noise view costs about 0.04 there, cross-source
  only. This is now our weakest number anywhere. Note that resize *is* already one of the
  four base training views, so this is not a missing-view problem: a 0.25× down/up pass
  physically removes the high-frequency content the forensic branch measures, and on the
  unseen generator CLIP alone is near chance. Candidates are a second, coarser forensic
  scale or a resize-specific head — not more of the same augmentation.
- The noise view slightly *hurts* CLIP-only cross-source (0.8094 → 0.7799 clean). It only
  helps once the forensic branch is present, where it is decisive on noise (0.8518 → 0.9290).
  Same interaction as in-distribution, larger in magnitude.

This also resolves the sample-size mismatch disclosed in the README's demo-benchmark
section: the "+ forensic, 4 views" column here is the pre-change head at n = 2000
(0.9189 clean, matching the `results/baseline/` snapshot's 0.9187), and "shipped" is the
promoted head at the same n. It is the matched cross-source A/B the README said it did not
have.

## Revised summary of what each component is for

| component | in-distribution | cross-source (unseen generator) |
|---|---|---|
| CLIP-only probe (UnivFD) | 0.9445 mean, solid | 0.81 clean, **0.54 on thumbnails** |
| + native-resolution forensic | +0.013, **but collapses under noise** | **+0.11 clean, +0.25 thumbnails** — the generalization |
| + noise training view | makes the forensic branch safe under noise | same; −0.04 on cross-source thumbnails |

The one-sentence version for the write-up: *a native-resolution forensic branch is what
lets a frozen-CLIP detector transfer to generators it has never seen, and a noise training
view is what stops that same branch collapsing under the corruption it is most sensitive to.*
