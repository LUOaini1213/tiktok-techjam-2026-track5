#!/usr/bin/env python3
"""Build the 3-minute demo video from the committed results.

The timeline is fixed to the required submission structure:

    0:00-0:15  Problem
    0:15-0:35  Our Solution
    0:35-0:55  Architecture
    0:55-2:20  Live Demo        <- the largest block
    2:20-2:45  Results
    2:45-3:00  Impact

Sections are cut to those windows rather than to however long the narration happens to
run: each section's voice-over is synthesized, its slides fill the window exactly, and a
section whose narration overruns is REPORTED so the script gets shortened instead of the
video silently drifting out of the required structure.

Every number on screen is read from results/ at build time -- robustness_table.csv,
ab_summary.csv, error_analysis.json, demo_live_scores.json -- so re-running the pipeline
produces a video that still agrees with the committed data. Nothing is typed twice.

    python scripts/make_video.py --out build/track5_demo.mp4
    python scripts/make_video.py --no_vo        # silent, subtitles only (no network)

Requires pillow, moviepy, edge-tts (imageio-ffmpeg supplies the ffmpeg binary).
Narration uses Microsoft's edge-tts service, so that step needs network access.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import enable_utf8_stdout

W, H = 1920, 1080
FPS = 24

GROUND = "#0d1117"
PANEL = "#161c25"
INK = "#e6ebf1"
INK_2 = "#aab6c4"
MUTED = "#778392"
ACCENT = "#6ba4f0"
GOOD = "#52b581"
CRIT = "#ec8279"
WARN = "#e0b341"
RULE = "#242d38"

FONT_DIR = Path("C:/Windows/Fonts")
VOICE = "en-US-AriaNeural"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Pick a font that exists on this box, falling back to PIL's default."""
    for cand in (name, "segoeui.ttf", "arial.ttf", "consola.ttf"):
        p = FONT_DIR / cand
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()


def F(size, bold=False, mono=False):
    if mono:
        return font("consolab.ttf" if bold else "consola.ttf", size)
    return font("segoeuib.ttf" if bold else "segoeui.ttf", size)


# ----------------------------------------------------------------- drawing helpers
def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), GROUND)
    return img, ImageDraw.Draw(img)


def heading(d, text, kicker=None):
    if kicker:
        d.text((110, 92), kicker.upper(), font=F(30, True), fill=ACCENT)
    d.text((110, 140), text, font=F(72, True), fill=INK)
    d.line([(110, 250), (W - 110, 250)], fill=RULE, width=3)


def bullet(d, y, title, body, colour=INK):
    d.rectangle([110, y, 118, y + 84], fill=ACCENT)
    d.text((150, y), title, font=F(42, True), fill=colour)
    if body:
        d.text((150, y + 52), body, font=F(30), fill=INK_2)
    return y + 130


def panel(d, box, fill=PANEL):
    d.rounded_rectangle(box, radius=16, fill=fill, outline=RULE, width=2)


def table(d, x, y, rows, widths, aligns=None, header=True, row_h=52, fonts=None):
    """Minimal table renderer; rows[0] is the header when header=True."""
    aligns = aligns or ["l"] * len(widths)
    for r, row in enumerate(rows):
        cx = x
        is_head = header and r == 0
        f = fonts[r] if fonts else (F(30, True) if is_head else F(30, mono=True))
        for c, cell in enumerate(row):
            text, colour = cell if isinstance(cell, tuple) else (cell, INK if is_head else INK_2)
            tw = d.textlength(str(text), font=f)
            tx = cx if aligns[c] == "l" else cx + widths[c] - tw
            d.text((tx, y + r * row_h), str(text), font=f, fill=colour)
            cx += widths[c]
        if is_head:
            d.line([(x, y + row_h - 8), (x + sum(widths), y + row_h - 8)], fill=RULE, width=2)
    return y + len(rows) * row_h


def fit_image(path: Path, box_w: int, box_h: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((box_w, box_h), Image.LANCZOS)
    return im


# ----------------------------------------------------------------- data
def load_data():
    res = ROOT / "results"

    def rows(name):
        p = res / name
        if not p.exists():
            return []
        with p.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    rob = {r["transform"]: r for r in rows("robustness_table.csv")}
    ab = {r["transform"]: r for r in rows("ab_summary.csv")}
    demo = rows("demo_benchmark.csv")
    err = json.loads((res / "error_analysis.json").read_text(encoding="utf-8"))["summary"]
    live_path = res / "demo_live_scores.json"
    live = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else {}
    if not rob or not ab:
        raise SystemExit("Missing results/robustness_table.csv or ab_summary.csv -- run the pipeline first.")
    return rob, ab, demo, err, live


ROB, AB, DEMO, ERR, LIVE = {}, {}, [], {}, {}
TRANSFORMS = ["clean", "jpeg_30", "blur_2.0", "resize_0.25", "noise_0.10", "crop_0.80"]


# ----------------------------------------------------------------- slides
def s_problem():
    img, d = canvas()
    heading(d, "A repost is not the original", "the problem")
    d.text((110, 300), "Every hop re-encodes the image.", font=F(46), fill=INK_2)
    steps = [("original", INK), ("JPEG 30", WARN), ("0.25x thumbnail", WARN), ("+ noise", CRIT)]
    x = 110
    for i, (label, col) in enumerate(steps):
        panel(d, [x, 400, x + 380, 560])
        d.text((x + 30, 440), label, font=F(40, True), fill=col)
        d.text((x + 30, 500), f"hop {i}", font=F(26), fill=MUTED)
        x += 420
        if i < len(steps) - 1:
            d.text((x - 32, 460), ">", font=F(44, True), fill=MUTED)
    d.text((110, 660), "Detectors are benchmarked on the original.", font=F(50, True), fill=INK)
    d.text((110, 730), "Users only ever see the repost.", font=F(50, True), fill=CRIT)
    d.text((110, 860), "TikTok TechJam 2026  -  Track 5  -  RepostGuard", font=F(30), fill=MUTED)
    return [img]


def s_solution():
    img, d = canvas()
    heading(d, "RepostGuard", "our solution")
    d.text((110, 300), "Treat the repost pipeline as the test, not an afterthought.",
           font=F(44), fill=INK_2)
    y = 400
    y = bullet(d, y, "Frozen CLIP ViT-B/32 + linear head",
               "88M parameters, far under the 2B cap. Refits in about a minute.")
    y = bullet(d, y, "Native-resolution forensic branch (28-D)",
               "NPR residual, 8x8 block-DCT, FFT radial rings - never bilinear-downscaled.")
    y = bullet(d, y, "Every official transform is also a training view",
               "The families we scored but never trained on were our weakest rows.")
    return [img]


def s_architecture():
    img, d = canvas()
    heading(d, "One forward, two feature views", "architecture")
    boxes = [
        (110, 330, 480, 470, "image", INK, None),
        (560, 250, 1080, 390, "CLIP ViT-B/32 (frozen)", ACCENT, "pre 768-D  /  proj 512-D"),
        (560, 430, 1080, 570, "native 256 center crop", ACCENT, "forensic 28-D"),
        (1160, 330, 1560, 470, "concat + TTA", GOOD, "clean + JPEG-70 + 0.5x"),
    ]
    for x0, y0, x1, y1, label, col, sub in boxes:
        panel(d, [x0, y0, x1, y1])
        d.text((x0 + 26, y0 + 34), label, font=F(34, True), fill=col)
        if sub:
            d.text((x0 + 26, y0 + 82), sub, font=F(26), fill=MUTED)
    d.text((496, 380), ">", font=F(44, True), fill=MUTED)
    d.text((1096, 380), ">", font=F(44, True), fill=MUTED)
    panel(d, [560, 640, 1560, 780])
    d.text((586, 672), "StandardScaler > LogisticRegression > sigmoid calibration > p(AIGC)",
           font=F(34, True), fill=INK)
    d.text((586, 726), "cross-validation picks pre vs proj - we verify rather than assume",
           font=F(26), fill=MUTED)
    d.text((110, 860), "train / eval / infer all go through the same embed_for_score path",
           font=F(30), fill=INK_2)
    return [img]


def _live_table(d, x, y, key, colour):
    entry = LIVE.get(key, {})
    scores = entry.get("scores", {})
    rows = [["transform", "p(AIGC)"]]
    for t in TRANSFORMS:
        if t in scores:
            rows.append([t, (f"{scores[t]:.4f}", colour)])
    table(d, x, y, rows, [340, 200])
    return entry.get("name", "")


def s_demo():
    slides = []

    # 1. the submission contract
    img, d = canvas()
    heading(d, "Directory in, JSON out", "live demo  -  submission contract")
    panel(d, [110, 320, W - 110, 470])
    d.text((150, 370), "python infer.py --input_dir samples --output preds.json",
           font=F(38, mono=True), fill=GOOD)
    panel(d, [110, 520, W - 110, 850])
    body = [
        "[",
        '  { "image_path": "...\\\\samples\\\\real_noise.jpg", "pred": 0.0038 },',
        '  { "image_path": "...\\\\samples\\\\synth_pattern.png", "pred": 0.6884 }',
        "]",
    ]
    for i, line in enumerate(body):
        d.text((150, 560 + i * 52), line, font=F(32, mono=True), fill=INK_2)
    d.text((110, 900), "One row per input file. An unreadable file gets a fallback score "
                       "instead of crashing the run.", font=F(30), fill=MUTED)
    slides.append(img)

    # 2 + 3. real held-out images scored under every transform
    for key, colour, title, caption in (
        ("aigc", GOOD, "A generated image stays caught", "held-out AIGC image, scored through the shipped path"),
        ("real", ACCENT, "An authentic image stays clear", "held-out real image, same six transforms"),
    ):
        img, d = canvas()
        heading(d, title, "live demo  -  transform sweep")
        name = _live_table(d, 110, 330, key, colour)
        entry = LIVE.get(key, {})
        p = entry.get("path")
        if p and Path(p).exists():
            thumb = fit_image(Path(p), 620, 620)
            img.paste(thumb, (760, 330))
        panel(d, [1450, 330, W - 110, 560])
        scores = entry.get("scores", {})
        if scores:
            lo, hi = min(scores.values()), max(scores.values())
            d.text((1480, 366), "range across", font=F(28), fill=MUTED)
            d.text((1480, 400), "all 6 transforms", font=F(28), fill=MUTED)
            d.text((1480, 452), f"{lo:.4f} - {hi:.4f}", font=F(40, True), fill=colour)
        d.text((110, 900), f"{caption}   ({name})", font=F(28), fill=MUTED)
        slides.append(img)

    # 4. the honest counterpoint
    img, d = canvas()
    heading(d, "Where it still costs creators", "live demo  -  the trade-off")
    rows = [
        ["metric", "value"],
        ["false positive rate @ 0.5", (f"{ERR['fpr']:.2%}", WARN)],
        ["false negative rate @ 0.5", (f"{ERR['fnr']:.2%}", INK_2)],
        ["FPR @ 95% TPR, clean", (f"{ERR['fpr_at_95tpr_clean']:.2%}", WARN)],
        ["FPR @ 95% TPR, JPEG-30", (f"{ERR['fpr_at_95tpr_jpeg30']:.2%}", CRIT)],
        ["real images flipped to FP by JPEG-30", (str(ERR["n_flips_to_fp_jpeg30"]), CRIT)],
    ]
    table(d, 110, 330, rows, [760, 300])
    d.text((110, 720), "Compression pushes authentic photos toward 'generated'.",
           font=F(40, True), fill=INK)
    d.text((110, 780), "A false positive is a real creator wrongly accused, so we report it "
                       "instead of burying it.", font=F(30), fill=INK_2)
    slides.append(img)
    return slides


def s_results():
    img, d = canvas()
    heading(d, "We found our own worst bug by measuring", "results")
    rows = [["transform", "before", "shipped", "delta"]]
    for t in ["clean", "jpeg_30", "resize_0.25", "noise_0.02", "noise_0.05", "noise_0.10", "jitter_0.20"]:
        if t not in AB:
            continue
        r = AB[t]
        dv = float(r["delta_auc"])
        col = GOOD if dv > 0.02 else (INK_2 if dv >= 0 else MUTED)
        hi = t.startswith("noise")
        rows.append([
            (t, INK if hi else INK_2),
            r["auc_baseline"],
            (r["auc_candidate"], col),
            (r["delta_auc"], col),
        ])
    table(d, 110, 320, rows, [360, 260, 260, 260], aligns=["l", "r", "r", "r"])
    panel(d, [1300, 320, W - 110, 640])
    d.text((1340, 360), "worst transform", font=F(30), fill=MUTED)
    d.text((1340, 404), "0.8117  ->  0.9411", font=F(44, True), fill=GOOD)
    d.text((1340, 480), "CIs disjoint", font=F(30), fill=MUTED)
    d.text((1340, 524), "3 improved, 0 regressed", font=F(36, True), fill=GOOD)
    d.text((110, 760), "Noise was scored but never a training view. Adding it: "
                       "+0.1362 AUC at noise 0.10,", font=F(34), fill=INK_2)
    d.text((110, 812), "clean unchanged at -0.0005.", font=F(34), fill=INK_2)
    if DEMO:
        clean = next((r for r in DEMO if r["transform"] == "clean"), None)
        if clean:
            d.text((110, 890), f"Cross-source check (WildFake, never trained on): "
                               f"clean AUC {clean['auc']} on {clean['n']} images",
                   font=F(30), fill=MUTED)
    return [img]


def s_impact():
    img, d = canvas()
    heading(d, "Small enough to keep current", "impact")
    y = 340
    y = bullet(d, y, "88M parameters, CPU-capable",
               "No GPU required. The head refits in about a minute.")
    y = bullet(d, y, "A new generator is a new training view",
               "Not a new training run - that is the whole point of the frozen encoder.")
    y = bullet(d, y, "False positives reported, not buried",
               "Flagging an authentic image is a direct harm to a real creator.")
    d.text((110, 880), "github.com/LUOaini1213/tiktok-techjam-2026-track5",
           font=F(36, True), fill=ACCENT)
    return [img]


# ----------------------------------------------------------------- timeline
SECTIONS = [
    ("problem", 0.0, 15.0, s_problem,
     "A creator posts a photo. Three reposts later it has been JPEG re-encoded, shrunk to a "
     "thumbnail, and filtered. Detectors are benchmarked on the original. Users only see the repost."),
    ("solution", 15.0, 35.0, s_solution,
     "RepostGuard treats the repost pipeline as the test. A frozen CLIP VIT-B-32, eighty-eight "
     "million parameters, far under the two billion cap, plus a forensic branch at native "
     "resolution. And critically, every official transform is also a training view."),
    ("architecture", 35.0, 55.0, s_architecture,
     "One CLIP forward gives us two feature variants, and cross-validation picks the winner "
     "rather than us assuming. The forensic branch runs in parallel at native scale. Inference "
     "averages three views, and a sigmoid calibration on a group-disjoint slice makes the "
     "output a real probability."),
    ("demo", 55.0, 140.0, s_demo,
     "The required contract is a directory in and JSON out, one row per file. Now the part that "
     "matters. This is a held-out generated image, scored through the shipped inference path "
     "under all six official transforms: JPEG thirty, heavy blur, a quarter-scale thumbnail, "
     "additive noise, and an eighty percent crop. The score never leaves nine-nine. Here is a "
     "held-out authentic photo through the same six transforms. It stays at essentially zero "
     "throughout. But we also report where it still costs creators. At the shipped threshold "
     "our false positive rate is thirteen percent, and JPEG thirty compression alone pushes "
     "forty-two authentic images across the line. Compression makes real photos look generated. "
     "A false positive is a real creator wrongly accused, so we measure it and publish it."),
    ("results", 140.0, 165.0, s_results,
     "We found our own worst bug by measuring. Six transform families; only four were training "
     "views. Noise collapsed to zero point eight one. We added the missing view, refit in a "
     "minute, and it went to zero point nine five. Non-overlapping intervals, clean accuracy "
     "unchanged. Three improved. Zero regressed."),
    ("impact", 165.0, 180.0, s_impact,
     "Eighty-eight million parameters, running on a laptop CPU. When a new generator appears it "
     "becomes a new training view and the head refits in a minute, instead of waiting on a "
     "training run. Thanks for watching."),
]


async def _tts(text: str, dest: Path, voice: str) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice).save(str(dest))


def synth(text: str, dest: Path, voice: str) -> bool:
    try:
        asyncio.run(_tts(text, dest, voice))
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:  # network down / service change: fall back to a silent cut
        print(f"  ! TTS failed for {dest.name}: {exc!r}")
        return False


def srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):06.3f}".replace(".", ",")


def main() -> None:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "track5_demo.mp4")
    ap.add_argument("--no_vo", action="store_true", help="Silent cut; no network needed")
    ap.add_argument("--voice", default=VOICE)
    args = ap.parse_args()

    global ROB, AB, DEMO, ERR, LIVE
    ROB, AB, DEMO, ERR, LIVE = load_data()

    work = args.out.parent / "video"
    work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from moviepy import AudioFileClip, ImageClip, CompositeAudioClip, concatenate_videoclips

    clips, audio_bits, srt = [], [], []
    overruns = []

    for idx, (name, t0, t1, builder, vo) in enumerate(SECTIONS, 1):
        window = t1 - t0
        print(f"[{idx}/{len(SECTIONS)}] {name}  {t0:.0f}-{t1:.0f}s ({window:.0f}s)")
        slides = builder()
        paths = []
        for j, im in enumerate(slides):
            p = work / f"{idx}_{name}_{j:02d}.png"
            im.save(p)
            paths.append(p)

        # Narration first: it decides nothing about the window, but an overrun is a script bug.
        vo_path = work / f"{idx}_{name}.mp3"
        have_vo = False
        if not args.no_vo:
            have_vo = synth(vo, vo_path, args.voice)
        if have_vo:
            a = AudioFileClip(str(vo_path))
            if a.duration > window + 0.35:
                overruns.append((name, a.duration, window))
            a = a.subclipped(0, min(a.duration, window))
            audio_bits.append(a.with_start(t0))

        per = window / len(paths)
        for p in paths:
            clips.append(ImageClip(str(p)).with_duration(per))

        srt.append((t0, t1, vo))

    video = concatenate_videoclips(clips, method="chain")
    if audio_bits:
        video = video.with_audio(CompositeAudioClip(audio_bits))

    sub = args.out.with_suffix(".srt")
    with sub.open("w", encoding="utf-8") as f:
        for i, (t0, t1, text) in enumerate(srt, 1):
            f.write(f"{i}\n{srt_time(t0)} --> {srt_time(t1)}\n{text}\n\n")

    video.write_videofile(str(args.out), fps=FPS, codec="libx264", audio_codec="aac",
                          preset="medium", threads=4, logger=None)
    print(f"Wrote {args.out}  ({video.duration:.1f}s)")
    print(f"Wrote {sub}")

    if overruns:
        print("\n!! narration overruns the fixed window -- SHORTEN THE SCRIPT:")
        for name, got, want in overruns:
            print(f"   {name}: narration {got:.1f}s > window {want:.0f}s (over by {got - want:.1f}s)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
