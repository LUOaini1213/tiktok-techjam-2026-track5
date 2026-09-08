# Devpost 提交交接单 — TikTok TechJam 2026

> **状态（2026-09-08）**：本交接单记录的是截止日提交的 v1 版本（CLIP 单编码器、2000 张/类）。评审结束后仓库升级到 v2（8000 张/类、加入 jitter 训练视图、CLIP + DINOv2-small 融合、内容种子化的评测噪声），所有数字在同一份 1400 张留出测试集上重新测量，见 README 的 *What changed after the deadline* 小节；v1 的结果、权重和逐图分数完整保留在 `results/v1/`。

**给代填的人：** 下面每一项都是照抄即可，不需要理解内容。按顺序填完点提交。
遇到没列出的可选字段，留空即可。
提交页面：TikTok TechJam 2026 on Devpost → 我的项目 → Edit

---

## 1. Track（赛道选择）

```
Track 5 — Robust Detection of AI-Generated Images Under Real-World Transformations
```

---

## 2. Project name（项目名）

```
RepostGuard — AIGC Detection That Survives the Repost
```

---

## 3. Elevator pitch（一句话简介，196 字符）

```
Built for what actually circulates, not the clean original. Its forensic branch is what generalizes to unseen generators (+0.25 AUC on thumbnails) - and what noise breaks, until we trained for it.
```

---

## 4. Public code repository（公开代码仓库）

```
https://github.com/LUOaini1213/tiktok-techjam-2026-track5
```

---

## 5. Video demo link（演示视频）

```
https://youtu.be/wbeGLieLZ9c
```

> ✅ 已确认可公开访问（标题 "track5 demo"，频道 WENJI LUO）。
>
> **建议顺手改两处**（公众投票 9/1–9/4 期间这条视频会被陌生人看到，目前标题是占位、描述为空）：
>
> YouTube 标题改成：
>
> ```
> RepostGuard — AIGC Detection That Survives the Repost | TikTok TechJam 2026 Track 5
> ```
>
> YouTube 描述粘贴：
>
> ```
> RepostGuard detects AI-generated images after a real-world repost: JPEG re-encoding,
> thumbnail resizing, blur, noise, colour jitter and cropping.
>
> Frozen CLIP ViT-B/32 (88M params, far under the 2B cap) + a native-resolution forensic
> branch + a calibrated linear head. Runs on CPU.
>
> We found our own worst bug by measuring: our evaluation grid has six official transform
> families but our training views had only four. Adding the missing noise view lifted
> worst-case AUC from 0.8117 to 0.9479 (+0.1362), with non-overlapping bootstrap
> confidence intervals and clean AUC unchanged.
>
> Code, full 15-transform robustness table, A/B deltas and error analysis:
> https://github.com/LUOaini1213/tiktok-techjam-2026-track5
>
> 00:00 Problem
> 00:15 Our Solution
> 00:35 Architecture
> 00:55 Live Demo
> 02:20 Results
> 02:45 Impact
> ```
>
> 如果当初传的是「不公开列出 / Unlisted」，建议改成「公开 / Public」，公众投票期更容易被看到。

---

## 6. "Try it out" links（试用链接）

```
https://github.com/LUOaini1213/tiktok-techjam-2026-track5
```

---

## 7. Built with（技术标签，共 24 个，逗号分隔粘贴）

```
python, pytorch, clip, transformers, huggingface, scikit-learn, numpy, scipy, opencv, pillow, matplotlib, joblib, gradio, pytest, claude, anthropic, claude-code, sid-set, wildfake, vision-transformer, logistic-regression, moviepy, edge-tts, ffmpeg
```

---

## 8. Upload a File（上传文件，限 35 MB）

上传这个文件（3.6 MB）：

```
repostguard-track5-submission.zip
```

---

## 9. Image gallery（图片画廊，最多 15 张）

上传这五张，**`01_results.png` 必须放第一张**（它是封面缩略图）：

```
01_results.png
02_robustness.png
03_architecture.png
04_transform_sweep.png
05_tradeoff.png
```

---

## 10. About the project（项目正文）

**Devpost 的正文编辑器支持 Markdown。把下面分隔线之间的全部内容原样复制粘贴进去。**

---8<--- 从这里开始复制 ---8<---

## Inspiration

A creator posts a photo. Three reposts later it has been JPEG re-encoded, shrunk to a thumbnail, colour-filtered and cropped for an avatar. Every AI-image detector we could find was benchmarked on the clean original — the one artifact nobody downstream ever sees.

So we built the detector around the degradation instead of apologising for it afterwards, and we made the transform grid the actual test rather than a robustness appendix.

## What it does

RepostGuard scores any image for P(AI-generated) and holds that score through the transforms a repost actually applies.

**Model, well under the 2B parameter cap:** a frozen OpenAI CLIP ViT-B/32 (~88M) supplying two feature variants from a single forward pass — the pre-projection pooler output (768-D) and the projected embedding (512-D) — concatenated with a 28-D forensic vector computed at **native resolution**, then a logistic-regression head with sigmoid calibration. Which CLIP variant to use is not asserted; GroupKFold(5) cross-validation picks it (proj 0.9536 vs pre 0.9414).

**Submission contract:** `python infer.py --input_dir PATH --output preds.json` emits one `{image_path, pred}` row per file. An unreadable or truncated file gets a fallback score rather than crashing the run, because one corrupt image should not zero an entire submission.

**Headline numbers**, all on the same held-out 1400-image test slice with calibration images excluded, each with a 95% stratified-bootstrap confidence interval:

| | AUC |
|---|---|
| clean | 0.9629 |
| JPEG-30 | 0.9557 |
| blur σ2.0 | 0.9536 |
| 0.25× thumbnail | 0.9472 |
| noise σ0.10 | 0.9479 |
| 80% crop | 0.9578 |
| colour jitter ±20% | 0.9411 |

Across all 14 transformed conditions the mean AUC is 0.9582 against 0.9629 clean. The worst case in the entire grid is 0.9411.

On a cross-source check the model has never trained on — the official WildFake demonstration subset, COCO val2017 reals against DALL·E-3 Advanced fakes — clean AUC is 0.9372 on a balanced 1000-image sample.

Everything runs on CPU. No GPU was used at any point.

## How we built it

Four pieces, plus the discipline that connects them:

1. **A single embed path.** `embed_for_score` is used identically by feature extraction, evaluation and inference, so cached training features and live scores cannot silently diverge.
2. **A native-resolution forensic branch.** NPR-style residual statistics, 8×8 block-DCT high/low energy ratios aligned to the JPEG grid, and FFT radial rings expressed as fractions of Nyquist — all computed on a native-scale 256 centre crop. Downscaling first would low-pass away exactly the high-frequency fingerprints this branch exists to measure.
3. **Test-time augmentation.** Inference averages the original view with a JPEG-70 view and a 0.5× down/up view, re-normalising so the embedding scale is independent of view count.
4. **Leak-free evaluation.** The head is sigmoid-calibrated on a group-disjoint 30% slice of validation and every reported metric comes from the held-out 70%. `artifacts/test_images.json` pins that slice so the robustness table and the error-analysis note are computed on identical images.

**What actually moved the number — and what we got wrong the first time.**

We built the detector, then built a 2×2 ablation to find out which part of it was doing the work: **features** (CLIP-only vs CLIP + native-resolution forensic) crossed with **training views** (the four we started with — clean/jpeg/blur/resize — vs five with a noise view). The CLIP-only / 4-view corner is a UnivFD-style linear probe (Ojha et al., CVPR 2023): a published baseline, not a strawman. All four heads are fit and sigmoid-calibrated identically, and every transformed image is embedded once and scored by all four from column slices of the same vector, so the whole grid costs one sweep.

Mean AUC over the 14 transformed conditions, held-out 1400-image slice:

| | 4 views | 5 views (+ noise) |
|---|---|---|
| CLIP-only | 0.9445 *(UnivFD-style baseline)* | 0.9480 |
| CLIP + forensic | 0.9389 ↓ | **0.9583** *(shipped)* |

Five things the grid says, in order of importance:

1. **Cross-source, the forensic branch is the whole generalization story.** On the WildFake demonstration subset — a generator family absent from training — a UnivFD-style probe scores 0.8094 clean and **0.5392 on quarter-scale thumbnails, barely above chance**. Adding the forensic branch lifts that to 0.9189 and 0.7938. Low-level residual / DCT / spectral statistics of the generation process transfer across generators; the semantic embedding of what our training fakes look like does not. Shipped: 0.9334 clean, +0.12 to +0.21 over the baseline on every cross-source row.
2. **In-distribution, the forensic branch alone *hurts*.** It adds +0.013 on clean and on every jpeg/blur/resize/crop row, and collapses under additive noise: 0.9250 → 0.8096 AUC at σ0.10. Its 28 dimensions are exactly the high-frequency statistics that broadband noise swamps. CLIP-only never had this problem. Our first write-up blamed "noise was scored but never trained"; that was the fix, not the cause.
3. **The noise view is what makes the forensic branch deployable.** Alone it is worth +0.0035; forensic alone is worth −0.0056; together they are worth +0.0138. The interaction is the finding: a high-frequency forensic branch is only safe when the training distribution contains the corruption that destroys it.
4. **Shipped beats the published baseline on 15 / 15 in-distribution rows** (worst row 0.9237 → 0.9411). Honest caveat: no single in-distribution row's bootstrap CIs are disjoint — consistently better, not significantly better per row.
5. **At the deployment operating point the forensic branch matters more than AUC suggests.** TPR@1%FPR on clean: baseline 0.441, shipped 0.546 (0.664 with the tampered class excluded — we report the harder, tampered-inclusive definition throughout).

**It is not reading compression history.** Several teams found SID-Set reals ship as JPEG and fakes as PNG. Our pipeline stores both as JPEG Q95, which equalises the container but leaves reals double-compressed and fakes single — and our forensic branch has JPEG-grid-aligned block-DCT features, the exact thing that would notice. So we ran the control on the held-out slice: a bare 8×8 blockiness scalar separates the classes at only 0.586 AUROC (the shortcut is essentially absent from our data); and re-encoding every image at Q95, both classes identically, for one and two extra generations moves every head by ≤ +0.0009 AUC. The forensic heads do not drop at all.

## Challenges we ran into

**We found our worst bug by measuring, not by debugging.** Nothing was crashing. The table simply showed that the three families we never trained on were the three we were worst at — a pattern invisible without per-transform confidence intervals on a fixed slice.

**A test we wrote destroyed a finished deliverable.** A new test called `evaluate_robustness(transforms=["clean", "jpeg_31"])` without a `table_path`, expecting a rejection. It got one — but only *after* the function had opened the real results file with `"w"`, truncating a completed 15-transform table that had taken half an hour to compute. The fix is that argument validation now happens before the destination is opened, and the test asserts the file is never created. Restored from git; the class of bug is worth more than the incident.

**Demo footage that would have contradicted our own claim.** We were going to use `samples/synth_pattern.png` as the on-screen AI example. Scoring it first showed it is a procedural sine-wave fixture, not real AIGC: it goes 0.6884 clean → **0.0852** under noise. Filming it would have demonstrated the model failing at exactly the transform we were claiming to have fixed. We replaced it with two held-out test images selected by **worst-case margin across all six transforms** rather than by clean score — the AI image holds 0.9866–0.9984 and the authentic one 0.0001–0.0005 across every condition.

**A 16 GB box, discovered the hard way.** Each CLIP evaluation worker holds its own model copy, so ten concurrent workers all died with `MemoryError`. Five is the ceiling here; at that width the pipeline sustains ~12 images/s and the full 15-transform sweep takes about 30 minutes instead of 2.5 hours serial. Twice we also found two competing process sets writing to the same shard files — worth checking before concluding a run is merely slow.

**Reproducibility bugs that only bite other people.** `gaussian_noise` drew from an unseeded generator, so cached noise views could never be reproduced. And because our checkout path contains non-ASCII characters, `print(path)` raised `UnicodeEncodeError` on a legacy Windows code page — *after* the work had finished, turning successful runs into non-zero exits.

## Accomplishments that we're proud of

**The mechanism was found by our own instrumentation — and it corrected our own first explanation.** A robustness table is normally the thing you produce at the end to show you are robust. Ours was the thing that told us we were not, and pointed at the specific cause.

**The A/B is honest about its own weakest link.** The demo benchmark is *not* a matched comparison — the shipped head was scored on 1000 images and the pre-change snapshot on 2000 — and the README says so rather than inviting the reader to compare them. The head-to-head evidence is the 1400-image robustness table where both heads saw identical inputs. The pre-change results are committed in `results/baseline/` so anyone can check the delta themselves.

**False positives are reported, not buried.** At the shipped 0.5 threshold: 12.92% FPR, 6.76% FNR, and FPR@95%TPR of 19.03% clean rising to 22.50% under JPEG-30. Compression alone pushes **42 authentic images** across the threshold. A false positive is a real creator wrongly accused, so it belongs in the write-up next to the AUC.

**The demo video is generated, not recorded.** `scripts/make_video.py` draws its slides with Pillow, synthesizes narration with edge-tts and assembles with MoviePy, reading every on-screen number from the committed CSVs at build time. The timeline is hard-cut to the required six windows and a section whose narration overruns is raised as an error rather than silently truncated — a check that caught three overrunning sections on the first build.

## What we learned

- **Your evaluation grid is a specification for your training data.** Any condition you score but never train on is a blind spot, and it will be measurably your worst row. This generalises well beyond image forensics.
- **A cheap fix can beat a clever one.** Reusing four cached views and adding a fifth bought +0.1362 AUC on the failing condition for ~4k forward passes and a one-minute refit. No architecture change was involved.
- **Validate arguments before opening files for write.** "Fails loudly" and "fails safely" are different properties, and only the second one protects work you already have.
- **Verify demo material against the metric before you film it.** The most persuasive image is worthless if it happens to be the case your model gets wrong.

## What's next

**Finish the job we started.** `jitter` and `crop` are still scored-but-untrained. The jitter feature cache is already extracted and merged; the crop extraction was killed for time. Folding both in is the obvious next experiment and `train.py --extra_features` already accepts them — we shipped only the variant we could verify end-to-end rather than an unmeasured one.

**Generator diversity.** Training is SID-Set only. CIFAKE and WildFake-train are permitted resources and mixing generator families is the highest-leverage addition; unknown commercial generators still shift CLIP geometry.

**The remaining hard row.** 0.25× thumbnails sit at 0.9472, now the joint-weakest condition alongside colour jitter. A small learned frequency head or a second forensic scale is the natural attempt.

**Re-run the demo benchmark at matched n** so the cross-source check becomes a proper A/B rather than a reference point.

---8<--- 复制到这里为止 ---8<---

---

## 附：需要一起发给代填人的文件

| 文件 | 用途 | 位置 |
|---|---|---|
| `repostguard-track5-submission.zip` | 第 8 项上传 | `docs/` |
| `01_results.png` ~ `05_tradeoff.png` | 第 9 项画廊 | `docs/video/gallery/` |
| `track5_demo.mp4` + `.srt` | 已上传 YouTube，链接见第 5 项。字幕可在 YouTube 后台补传 | `build/` |

## 附：如果时间不够

按这个优先级，交上去比填完整更重要：

1. Track + Project name + 仓库链接 + About 正文 → **先点提交**
2. 视频链接、zip、图片、tags → Devpost 截止前可以继续编辑补上

## 附：状态

**十项全部就绪，没有阻塞项。** 照着从第 1 项填到第 10 项，然后提交。
