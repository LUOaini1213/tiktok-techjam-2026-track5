# V4: selective detection and local-edit experiments

This is a non-commercial research iteration, never a TechJam submission. V3 stays
the default while V4 is tested. No old report or locked extractor is rewritten.

## Predeclared experiment

First publish feature-branch grouped-CV ablations and empirical risk/coverage curves
on V3 development/calibration caches. Curves fitted to these labels are explicitly
diagnostic, not held-out performance. The old 448-image V3 test is not used to fit,
calibrate, select or tune V4. The official WildFake/COCO/DALL-E reference remains
demonstration-only, never training/calibration/development data.

V4 compares a bounded set of frozen-feature logistic, small MLP and histogram
gradient-boosting heads. A local trial pools DINOv2 patch tokens and native-scale
corner forensic statistics without introducing a second DINO backbone. Models
are selected on development only. Training views and source identities stay
together; local specialists use explicit source-disjoint training rows.

New original identities, source revisions, source families, role quotas and
content hashes are frozen before fitting. New full-synthesis, local-edit and real
SID images come from later validation shards. New CF models increase training
diversity; a new generator is entirely reserved for the final test. CF excludes
COCO and VQDiffusion. All old V3 identities and protected reference bitmaps/near
matches are blocked from the new images. SID parent-photo or foundation-model
pretraining overlap cannot be ruled out by these checks.

Calibration is split by original into probability fitting and threshold selection.
Compare (1) the existing original worst-view tolerance policy, and (2) an explicitly
different conditionwise tolerance policy. Targets are 1% false AI flags and 5%
false clearances, with one-sided 95% original-count bounds. Report sample counts,
whether each bound is supported, per-condition outcomes, worst-view outcomes,
recall and review together. No universal deployment guarantee is made.

The primary development comparison is lower review at supported conditionwise
error targets, with external AUC loss <=0.005 and SID tampered AUC loss <=0.005
relative to V3 on the same new development originals. If no candidate qualifies,
retain V3 and publish the unsuccessful trials. Select the model/policy and freeze
checksums before final-test feature extraction. New-test results never change
the chosen model or thresholds. Improvements and regressions are both published;
promotion requires separate final integrity, CLI/UI and latency checks.

### Fixed CPU budget (declared before the first V4 fit)

- New training: 256 CF originals, 64 new LatDiff from shard 40 (max 4/model),
  64 guidedDiffusion from shard 67, 64 FFHQ from shard 100 and 64 LandscapesHQ
  from shard 180. Local replay: 384 old SID training originals, 128 per SID label.
- New development: SID shard 11, 64 real/32 synthetic/32 tampered; CF shard 60,
  64 new LatDiff (max 4/model), plus 32 FFHQ/32 LandscapesHQ. Total 256.
- New calibration: SID shards 13-14, 320 real/160 synthetic/160 tampered;
  CF shard 62, 320 new LatDiff (max 16/model), plus 160 FFHQ/160 LandscapesHQ.
  Total 1,280, split equally by source/subclass: 640 probability-fit and 640
  policy images, with 320 real policy images. These counts support a pointwise
  1% upper bound only when the relevant real-error order statistic permits it.
- New final test: SID shard 15, 96 real/48 synthetic/48 tampered; CF shard 80,
  96 StyleGANXL, plus 48 FFHQ/48 LandscapesHQ. Total 384. StyleGANXL is never
  training, development or calibration data. VQDiffusion stays excluded.
- Train/replay/calibration: clean and crop-resize-JPEG, 2 views. Development:
  clean, JPEG-30, blur-1, quarter-resize, noise-.05, crop-resize-JPEG, 6 views.
  Final test: all 15 published conditions and all four ordered repost chains.
  Calibration bounds cover its **two sampled conditions only**, not the unseen
  transform grid. Conditionwise 95% bounds are pointwise, not simultaneous.
- Global heads: LR C=.01/.1; MLP 64 hidden units, max 80 epochs on per-original
  averaged training features (no row-level early-stopping leakage); histogram
  boosting 80 iterations/15 leaves. Local LR C=.01/.1, optionally a fixed 50/50
  probability blend. No unbounded hyperparameter search.
- Feature ablation: 1,200 balanced training originals, three grouped folds.
  Full global heads use deduplicated existing training plus the new CF originals.
  Local heads use the explicit 640-original replay/new-training subset only.
- Exact/near overlap checks block every old V3 original and all 2,000 locally
  protected official reference images. Replay is an explicit training-only
  exception, never counted as new identities. Short quotas are a failure, not
  a reason to silently move source families or reuse test images.

Training data are SID-Set (CC-BY-4.0) and CommunityForensics-Small
(CC-BY-NC-SA-4.0; Jeongsoo Park and Andrew Owens, CVPR 2025). Images and large feature
caches are not redistributed. CF comparisons apply a common 512x512 bicubic fit
before transformations, identically for both labels; CLI inputs do not receive
a hidden dataset-specific fit. Source acquisition/content bias can remain.

References: [selective risk/coverage](https://proceedings.mlr.press/v206/pugnana23a/pugnana23a.pdf),
[Community Forensics](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html).
