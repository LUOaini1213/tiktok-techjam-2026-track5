# Track 5 competitor review — TikTok TechJam 2026

> **Status (2026-09-08).** Written against the v1 deadline build (CLIP only, 4,000 training
> images). The repo has since moved to v2 -- 16,000 training images, a jitter training view,
> CLIP + DINOv2-small fusion -- with every change re-measured on the same 1400 held-out images;
> see README, *What changed after the deadline*. The field survey below is unchanged.

*Surveyed 2026-09-03 during public voting ("1 more day to vote"). 130 Track 5 submissions
enumerated via the Devpost gallery's Track #5 filter (6 pages); 13 read in full. Numbers
are as each team reports them — protocols differ, so cross-team comparison is indicative only.*

## First finding: RepostGuard is not on Devpost

Not in the 130 Track 5 entries, not in the 72 most recent submissions of the full 614-entry
gallery. `RepostGuard-Lite` (2 likes) is a different team — OpenCLIP + CommunityForensics
training, FastAPI/Vue demo; SID-Set "retained only as historical pipeline pilots".

Either the Devpost entry was never filed, or it was filed under a different name without
the Track 5 tag. The handover sheet (`docs/devpost_submission.md`) was written for someone
else to file. **This needs checking in the account's "My projects" page before anything
else in this document matters.**

## The field, by public votes

| likes | project | approach (as stated) | headline numbers (as stated) |
|---:|---|---|---|
| 85 | realityCheCk.TSX | hybrid CLIP + FFT forensics | 0.996 clean, 0.984 across 15 transforms |
| 62 | Detection with Residuals | mLoRC on **DINOv3 H+**, DDA 144k training, novel "Modulated Energy Training" | WildFake 30k / 26 generators: clean 0.9929 AUC, transformed 0.9723; worst JPEG-30 86.1% bAcc; beats LoRC/DDA/DGS-Net baselines; 143 img/s on 3090 Ti |
| 41 | DUET | depth-uniform ensemble with early terminate | — |
| 37 | Robust AI-Gen Detector & Dataset Evaluator | — | — |
| 36 | THEIA | **SigLIP2-giant 1.164B** frozen, stacked-transform training | clean 0.9991, worst cell (noise 0.10) 0.9951, external 0.9917; caught a resolution shortcut (real low-res flagged fake 87%) |
| 33 | Seer | **DINOv3 ViT-L/16** 305M, trained on **2.58M images**, patch heatmap | OpenFake 20 unseen generators 99.84% AUROC; CommunityForensics-Eval 95.79% vs Pangram 97.29%; NTIRE 2026 robust AUC 92.28% (3rd on leaderboard); COCO FPR 0.06% |
| 20 | PixelProof | 21.7M ViT, Chrome extension | 0.9362 on unseen generators |
| 20 | RESONANCE | DINOv3 ViT-S frozen + label-free GRACE adapter + DCT "frequency enricher"; 26-condition lattice incl. compositions | — |
| 19 | RobustFusion | DINOv2+LoRA + SigLIP + Haar HF, 4-tile attention, GroupDRO | cross-dataset (7 sets): clean 93.19%, transformed 89.91%, worst 83.54%; fake recall 94.85% / real 79.53% |
| 18 | SynthFlag | 4-expert CLIP-L / SigLIP ensemble, video extension | **AUC 0.8505**, bAcc 0.8061 |
| 17 | CrossGuard | **DINOv2 ViT-L/14 + LoRA r32** @448px, 306M, H100 | clean 0.9921, 14-cell macro 0.9865, worst resize-0.25 0.9728; TPR 0.867 @ FPR 0.0055; unseen generators 0.9913 |
| 12 | VerifAI | frozen CLIP ViT-L/14 + linear SVM (~1k trained params), **pre-registered** eval | found JPEG-blockiness shortcut (AUROC 0.830 → 0.498 after re-encode); found the WildFake demo default config is separable by image size alone |
| 12 | Telltale | forensic MLP/XGBoost + semantic + localization, leave-one-generator-out | held-out generators 0.686 / 0.850 / 0.822 — published as the honest number |
| 11 | RIFT | frozen CLIP + forensic signals + fixed low-FPR threshold | — |
| 8 | BYTEPRINT | CLIP ViT-B/32 + DINOv2 fused, 173.4M | 15 rungs on SID_Set: mean AUC 0.991, TPR@1%FPR 0.837; found crop-0.8 collapses TPR@1%FPR 0.860 → 0.335; JPEG-container leakage control 0.9025 → 0.9022 |
| 3 | Aquaforge 8 | DINOv2 ViT-L dual towers, 455k-image corpus | OOD test AUC 0.9963, TPR@1%FPR 0.9825 |
| 2 | Adaptive Aigc Forensics | Community-Forensics RGB expert + 26-value signal MLP (449 params), static fusion | source-level split, SHA-256-bound artifacts |
| 1 | WatchDAWG | frozen CLIP + head, 151M | worst-case acc 63% → 83%, mean 83% → 91% |

Likes are a weak signal — WatchDAWG and BYTEPRINT are among the most rigorous write-ups
and have 1 and 8.

## Where RepostGuard sits

Our numbers: clean 0.9629 AUC **including the tampered class as positive** (0.9832 with
tampered excluded — most teams report the easier definition and do not say so), mean
transformed 0.9583, worst 0.9411; TPR@1%FPR 0.546 (0.664 excl. tampered); cross-source
WildFake clean 0.9334. 88M parameters, CPU-only, 4,000 training images.

**On raw AUC we are mid-to-lower pack.** The top of the field trains DINOv2/v3/SigLIP2
backbones with LoRA on 24k–2.58M images on H100s and reports 0.98–0.999. That is not a
gap a linear head on 4k images closes. Two things soften it: (1) the task definition —
our tampered-inclusive number is a harder problem, and (2) several near-perfect numbers
are on protocols other teams showed to be shortcut-prone (below).

**Where we are genuinely competitive:**

- *Feasibility.* 88M, CPU, one-minute refit. Only WatchDAWG (151M), BYTEPRINT (173M) and
  VerifAI (303M frozen) are in the same weight class; we are the smallest that reports a
  full 15-transform grid.
- *The 2×2 ablation on the analysis branch* (forensic × noise-view, in-distribution and
  cross-source) — no other entry reports a component interaction; BYTEPRINT's crop-TPR
  finding is the nearest analogue. This is our best Innovation material and it is **not in
  the submission**.
- *Cross-source clean 0.9334 vs RobustFusion's 0.9319* on their multi-dataset protocol —
  roughly parity with the most-liked entry at a fraction of the compute, though the
  protocols differ.

**Where we are clearly behind — and it is about rigor, not just scale:**

1. **No leakage control.** BYTEPRINT, VerifAI and Telltale all found that SID_Set / WildFake
   reals ship as JPEG and fakes as PNG, and that a hand-written 8×8 blockiness statistic
   alone separates the classes (VerifAI: AUROC 0.830). Our forensic branch contains
   **block-DCT statistics aligned to the JPEG grid** — precisely the feature that would
   read compression history. Our JPEG-30 row barely moving (0.9557 vs 0.9629 clean) is
   evidence against us relying on it, but the standard control — re-encode both classes
   identically, re-measure — was never run. This is the single most important gap to close.
2. **The demo set's size shortcut.** VerifAI showed the WildFake default config is separable
   by image dimensions alone (every real is 200×200). Our forensic branch takes a *native
   256 centre crop*; on a 200×200 image that crop is the whole image plus padding — a
   possible size signature. Our demo clean AUC of 0.9334 is far from 1.0, so we are not
   exploiting it wholesale, but we never checked.
3. **Operating point.** The field reports TPR@1%FPR; we report FPR@95%TPR. Ours is 0.546
   at 1% FPR — well below BYTEPRINT's 0.837 and CrossGuard's 0.867. That is the number a
   platform would deploy on, and it is our weakest.
4. **Honest per-transform reporting is table stakes here, not a differentiator.** At least
   six of the thirteen entries read do it as well as we do.
5. **Training-data scale and backbone.** 4k images, ViT-B/32, one dataset.
6. **Presentation.** Static-slide video against interactive demos, Chrome extensions, heatmaps.

## What I would do next, in order

1. Confirm whether the Devpost entry exists. Nothing else matters if it does not.
2. Run the JPEG-container leakage control (re-encode all val images at Q95, re-score;
   ~10 min on this box). If the number holds, it is a one-line credibility statement the
   top entries all have and we lack. If it drops, we need to know before a judge does.
3. Add TPR@1%FPR to the robustness table — it is already computable from the same scores.
4. Fold the 2×2 ablation into the write-up as the Innovation claim (branch → master only
   if the finalist round happens).
