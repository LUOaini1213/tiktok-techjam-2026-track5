# RepostGuard — 3-minute demo video script

> **状态（2026-09-09）：本项目从未提交到 Devpost。** 这份脚本是照 TikTok TechJam 2026
> Track 5 赛题写的，写作时假定会有评委观看，所以下文多处出现「评委」——**那是当时的写作意图，
> 不代表本项目参加过评审**。视频本身已公开在 YouTube，但没有随任何 Devpost 条目提交。
> 本仓不是参赛作品，不主张任何名次。

照 TikTok TechJam 2026 Track 5 赛题写。Target length **3:00**, public YouTube link.

**录制前准备 (checklist)**
- `python scripts/make_samples.py` 已跑过；`artifacts/repostguard.joblib` 存在
- 准备 2 张图：一张真实照片 (`real.jpg`)、一张 AIGC 图 (`fake.png`)，都不在训练集里
- 终端字号调大到 18pt+，窗口 1920×1080；浏览器只留 Gradio 一个标签页
- 先跑一次 `python app.py` 预热 CLIP 权重，录制时才不会卡 20 秒加载
- 旁白用英文（评委国际化），字幕/画面注解用中英皆可

---

## 0:00–0:15 — Problem

> **旁白.** "A creator posts an original photo. By the time it has been reposted three times, it has been JPEG re-encoded, resized to a thumbnail, and filtered. Every AI-detection tool you can name was benchmarked on the clean original — not on what actually circulates."

**画面**
- 左右分屏：左边原图，右边 `jpeg_30` + `resize_0.25` 之后的同一张图
- 右下角叠字：`JPEG 30 · 0.25× · blur σ2` 
- 最后 3 秒打出问题句：**"Detectors are tested on the original. Users only ever see the repost."**

---

## 0:15–0:35 — Our Solution

> **旁白.** "RepostGuard treats the repost pipeline as the test, not an afterthought. A frozen CLIP ViT-B/32 — 88 million parameters, far under the 2-billion cap — plus a native-resolution forensic branch, plus test-time augmentation. And critically: every official transform is also a *training* view, so the detector has seen degradation before it meets it."

**画面** — 三个要点依次弹出：
1. **Frozen CLIP + linear head** — 88M params, trains in seconds, no fine-tuning
2. **Native-resolution forensic branch (28-D)** — 高频指纹在缩放里会被低通滤掉，所以在 **原始尺度** 的 256 center crop 上算 NPR 残差 / 8×8 DCT / FFT 径向环
3. **Degradation as training signal** — 官方 transform 既是评测项，也是训练视图

> 小字幕可提一句创新点：pre-projection (768-D) vs projected (512-D) CLIP 特征 **不是拍脑袋选的**，用 GroupKFold(5) 交叉验证 A/B 选出来的。

---

## 0:35–0:55 — Architecture

**画面** — 一张数据流图，从左到右画出来：

```
image ─┬─► CLIP ViT-B/32 (frozen) ─► pre 768-D  ─┐
       │                             proj 512-D ─┤
       │                                          ├─► concat ─► TTA average ─► StandardScaler
       └─► native 256 center crop ─► forensic 28-D ┘        (clean + JPEG-70 + 0.5×)      │
                                                                                          ▼
                                          p(AIGC) ◄─ sigmoid calibration ◄─ LogisticRegression
```

> **旁白.** "One CLIP forward gives us both feature variants; cross-validation picks the winner. The forensic branch runs at native resolution in parallel. Inference averages three views, and a sigmoid calibration on a group-disjoint slice makes the output a real probability, not a raw margin."

**强调一句** — train / eval / infer 全部走 **同一条** `embed_for_score` 路径，所以缓存的特征和线上打分逐位一致。

---

## 0:55–2:20 — Live Demo ⭐ (85 秒，最重要)

### (a) 0:55–1:15 — 提交接口 (20s)

终端里真跑，不要剪：

```bash
python infer.py --input_dir samples --output preds.json
```

**画面**：命令执行 → `cat preds.json` 展示输出

> **旁白.** "The required contract: a directory in, JSON out — one row per file, `image_path` and `pred` between zero and one. Unreadable or truncated files get a fallback score instead of crashing the run, so one corrupt image can never zero the whole submission."

### (b) 1:15–1:55 — 鲁棒性现场演示 (40s，全片核心)

浏览器打开 Gradio (`python app.py`)：

1. 传 `real.jpg` → 显示 **Likely authentic**，读出 p(AIGC)
2. 传 `fake.png` → 显示 **Likely AIGC**，读出 p(AIGC)
3. **关键动作**：保持 `fake.png` 不动，依次拖动滑块
   - JPEG quality `100 → 30`
   - Gaussian blur `σ 0 → 2.0`
   - **Gaussian noise `σ 0 → 0.10`**
   - Center crop `1.0 → 0.8`
4. 每拖一次，右侧 "Transformed view" 实时更新，**p(AIGC) 基本不动**

> **旁白.** "This is the whole point. I'm degrading the image live — JPEG 30, blur, noise, an 80-percent crop — the transformed view on the right is visibly damaged, and the score barely moves."

> ⭐ **录制注意**：noise 滑块是本次新加的，而且 noise-augmented 模型**已经 promote 上线**。
> 拖到 σ=0.10 时 p(AIGC) 基本不掉 —— 这正是全片最有说服力的 3 秒（改进前这一档 AUC 只有
> 0.8117，准确率 0.65；现在 0.9479 / 0.88）。一定要给 p(AIGC) 数字一个特写，并停 2 秒。

### (c) 1:55–2:20 — 可复现 (25s)

**画面**：`results/robustness_chart.png` 全屏，然后滚 `results/robustness_table.csv`

> **旁白.** "Every number is reproducible with one command. Fifteen transforms, the full held-out test slice of fourteen hundred images, ninety-five-percent bootstrap confidence intervals — and the calibration images are excluded, so nothing here is inflated by leakage."

```bash
python scripts/make_tables.py     # 15 transforms + bootstrap CIs
python scripts/update_readme.py   # 结果直接写回 README
```

---

## 2:20–2:45 — Results

**画面** — 左右对比表，高亮 noise 三行（`results/ab_summary.csv`）：

| Transform | before | **shipped** | Δ |
|---|---:|---:|---:|
| clean | 0.9634 | **0.9629** | −0.0005 |
| jpeg_30 | 0.9547 | **0.9557** | +0.0010 |
| resize_0.25 | 0.9455 | **0.9472** | +0.0017 |
| **noise_0.10** | 0.8117 | **0.9479** | **+0.1362** |
| *(now worst)* jitter_0.20 | 0.9362 | **0.9411** | +0.0049 |

> **旁白.** "Clean AUC is 0.963, and it holds at 0.956 under JPEG-30. But the number I
> actually care about is this one. We found our own worst bug by measuring instead of
> assuming: our evaluation grid has six transform families, and only four of them were
> ever training views. Noise was scored but never learned — and it collapsed to 0.81.
> We added the missing training view, refit in about a minute, and it went to 0.95.
> Plus 0.136 AUC, non-overlapping confidence intervals, and clean accuracy unchanged."

**必须说的一句（诚实性，评委很吃这套）：**

> "Three transforms improved beyond overlapping bootstrap confidence intervals. Zero
> regressed beyond them. And the worst case across the whole grid went from 0.81 to 0.94."

**画面补充（各 2 秒即可）**
- `results/robustness_chart.png` — 15 个 transform 的 AUC + 95% CI
- 跨源泛化 `results/demo_benchmark.csv`：WildFake demo subset（COCO 真图 + DALL·E-3 假图，**从未参与训练**）clean AUC **0.9372** — 说 "a generator family the model never saw"
- 误报代价：FPR 12.9%、FPR@95%TPR 0.190；JPEG-30 会把 **42** 张真图推过阈值（改进前是 53 张）

> ⚠️ 口播不要把 demo benchmark 说成 A/B：shipped 跑的是 1000 张，baseline 快照是 2000 张，
> 两者 n 不同，不能直接比。A/B 证据是上面那张 1400 张同一切片的 robustness 表。

---

## 2:45–3:00 — Impact

> **旁白.** "Eighty-eight million parameters. It runs on a laptop CPU, no GPU required, and the head retrains in seconds — so a platform can refit it the week a new generator appears, instead of waiting on a training run. We report false positives explicitly, because flagging an authentic image is a direct harm to a real creator. Thanks for watching."

**画面** — 三点收尾 + repo 链接：
- **Scalable** — 88M / <2B，CPU 可跑，换编码器只改 config 一行
- **Honest** — FPR@95%TPR、误报误检案例、局限全部写进 README
- **Extensible** — 新生成器 = 加训练视图 + 重新拟合 linear head（秒级）

最后一帧：`github.com/LUOaini1213/tiktok-techjam-2026-track5`
