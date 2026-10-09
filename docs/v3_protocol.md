# v3: lower false positives and unseen-generator reposts

This is a post-deadline research iteration. This project was never submitted to
TechJam. v1 is the deadline-era development snapshot; v2 and
its head, table and error note are archived under `results/v2/`. No v3 result should
be presented as an official hackathon submission or result.

## Data and the corrected baseline

The local v2 directory contains 16,000 training files, but only 11,997 distinct
file hashes. There are also 19 train/validation overlapping files and one duplicate
validation file. The old count-based stream resume replayed previously saved rows.
The downloader now deduplicates the JPEG payload before saving a replayed record.
Existing data is preserved. v3 selects unique cached source images and excludes
calibration/dev/test exact or conservative near matches to training sources using
decoded RGB identities, pHash and dHash. The historical tables are retained as
historical diagnostics, not relabelled as a contamination-free final test.

New data comes from the authors' `OwensLab/CommunityForensics-Small` dataset:
multiple LatDiff generators, StyleGAN2 and BigGAN for training/development/calibration;
the whole VQDiffusion generator family is reserved for testing. Real images are
restricted to FFHQ and LandscapesHQ. COCO real images are excluded from the added CF
training/calibration sources to protect the official COCO/DALL-E demonstration benchmark. Images and
per-generator licenses are not redistributed in this repository. The Small dataset
is CC-BY-NC-SA-4.0, for non-commercial research; commercial deployment of this v3
artifact requires a different, appropriately licensed training/calibration dataset.
Dataset attribution: Jeongsoo Park and Andrew Owens, *Community Forensics*,
CVPR 2025. [Source card](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small),
[paper](https://arxiv.org/abs/2411.04125),
[CC-BY-NC-SA-4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Our changes are subset selection, source-role partitioning, uniform CF geometry,
repost transformations and head fitting; no ownership of source images is claimed.
The card also points to per-generator terms, which remain relevant.

Fresh SID test images come from pinned validation shard 10, outside the historical
prefix. Source revisions, roles, content hashes, and relative paths are frozen before
cross-validation. Final counts after exclusion are in `results/v3/data_manifest.json`.
The sealed test is small and non-exhaustive; generator/source content bias can remain.
The SIDA paper describes FLUX full-synthetic images from Flickr30k/COCO captions,
and COCO-derived edits (including MagicBrush material). The protected reference is
never deliberately sampled for training/calibration/selection, but shared parent
photos cannot be ruled out by bitmap hashes alone. This qualifies the initial
frozen manifest's overly broad wording about excluding all COCO data; the immutable
manifest is preserved, with the correction in `protocol_audit.json` and model metadata.
An exploratory wider perceptual match is diagnostic only, not proof of parent disjointness.

## Selection and calibration

Both grouped CV and final fitting append the same clean/degraded mean features for
each original image. All views and consistency rows of an original stay in one fold.
A fixed 1,200-original training subset selects `C` from 0.01/0.1/1 and tampered weight
from 1/2/5 using three folds. The selected configuration is fit on all deduplicated
training sources. Existing CLIP-projection + DINOv2 + forensic features are reused.

New training views include randomly cropped, scaled and re-encoded images, alternative
resize interpolation methods, and ordered multi-stage chains. Both classes use the same
parameter distribution. Community Forensics images are decoded RGB, uniformly fit to
512x512, and losslessly cached; source compression history can still remain. This is
an alignment control, not proof that acquisition bias is eliminated.
The Community Forensics numbers apply to these aligned 512x512 inputs, not the
native source files. CLI/UI inference does not silently apply a dataset-specific
fit to arbitrary uploads. A live reproduction must feed the same aligned view;
SID and the official reference retain their native input geometry.

Calibration originals are split into two disjoint, label-stratified halves. One half
fits the sigmoid; the other selects decision thresholds using each original's worst
transformed score. Repeated views do not inflate the number of independent trials.
A conservative one-sided 95% binomial tolerance bound targets 1% false AI flags and
5% false clearances on the calibration distribution. If the sample size cannot support
the target, the metadata explicitly records that the bound was not met.

The UI has three outputs: **Likely AIGC**, **Needs review**, and **Low AIGC signal**.
Low signal does not certify authenticity. The JSON contract stays `{image_path,pred}`;
`pred` remains a continuous score, and operational thresholds are saved in bundle metadata.
Recall, coverage, review rate and false-clear rate are reported with false-positive rate:
a detector that flags almost nothing must not be advertised as an unqualified improvement.
Fixed-0.5 accuracy, FPR and recall are reported separately for both models, so a
ranking/model improvement is not conflated with simply raising a decision threshold.

The mixed-data candidate and 25%/50%/75% candidate blends are compared to a recalibrated
v2 head on development images only. An eligible head must have mean external development
AUC at least v2's, external clean loss no more than 0.01, thumbnail loss no more than
0.005, and SID development clean loss no more than 0.005. The latter uses historical
v2 evaluation images, explicitly reclassified as development, not as a new final test.
The eligible head with the highest external development mean AUC ships. Otherwise the
v2 head with the new calibration/policy ships and the mixed-data trial stays an experiment.
`selection.json` freezes the chosen model checksum before sealed-test extraction.
No promotion or threshold change is allowed in response to sealed-test results.
The historical SID development guard samples by binary label and can overrepresent
fully synthetic files because of filename order. It is not a balanced local-edit
guard. Fresh SID scores are therefore also reported separately for fully synthetic
and tampered positives; those diagnostic test slices never change the locked head.
Deployment installation after integrity/contract checks is not another model
selection step: the development-selected artifact is installed unchanged. A final
`evaluation_lock.json` checksums every generated report, preventing partial or stale
reports from being treated as a completed evaluation.

## Evaluation

All 15 official conditions plus four ordered chains are measured on the same newly
sealed images for v2, the selected v3 artifact and the candidate. The four chains are
crop -> quarter-scale -> JPEG, noise -> JPEG, blur -> resize -> JPEG, and two rounds
of resize/JPEG sharing. Accuracy at 0.5 and AUC are distinguished from the actual
shipped policy's FPR and recall. Per-condition AUC intervals and paired AUC-difference
intervals resample the same original images, not separate model samples. Each condition
has one row per original; means over conditions are descriptive, not independent trials.

FPR confidence bounds remain visible. A small test with zero false positives cannot
establish a universal 1% deployment FPR. WildFake is evaluation-only and is never a
source of training, calibration or threshold-selection labels.

An additional 400-image official reference subset (200 real / 200 AIGC, seed 31009)
is scored on six selected conditions only after the lock. Both models use the same
400 originals. These results must not be directly compared with the old 2,000-image
reference table or included in the 448-image sealed-test aggregate. A separate
protected-reference audit checks all 2,000 cached reference images against training
identities; see `results/v3/protocol_audit.json`.

Model metadata records encoder checkpoint snapshots, exact parameter counts and
library versions. CPU inference defaults to four threads (configurable), matching
the local feature-cache run. No unmeasured GPU throughput claim is made.

## Reproduce on the v2 workspace

The existing SID dataset and `artifacts/features_train.npz` are local licensed caches,
not shipped in Git. On a clean checkout, reproduce the v2 feature cache first using
the archived procedure and record the resulting source hashes; re-downloading with
the fixed loader can change the historical subset and therefore the exact numbers.

```text
python scripts/prepare_v3.py
python scripts/extract_v3.py --roles train,cal,dev
python scripts/train_v3.py
python scripts/extract_v3.py --roles test
python scripts/eval_v3.py
python scripts/validate_v3.py
python -m pytest tests/ -q
```

`pythonw scripts/run_v3.py` runs the first five steps detached on Windows, with
per-image feature checkpoints and bounded retries. Progress and failures are written
to `logs/v3/state.json` and per-stage logs. The workflow never pushes Git or silently
changes the shipped weights. Promotion and the submission-contract check happen after
reviewing the completed reports. Cached feature files are fingerprinted against source
manifest and extractor code; incompatible caches are rejected.

## References

- [Community Forensics, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html)
- [SIDA / SID-Set source construction](https://arxiv.org/html/2412.04292v2)
- [Authors' Small dataset and license](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
- [Aligned Datasets Improve Detection, ICLR 2025](https://openreview.net/pdf/e17314703e142b8fea0685c6de8670019e6245d0.pdf)
- [Real-World Robustness Dataset, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.html)
