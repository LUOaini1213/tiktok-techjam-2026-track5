#!/usr/bin/env python3
"""Console-free orchestrator for the v2 retrain (extraction -> train -> eval -> crops).

Why this exists: two launches of the same chain died with STATUS_CONTROL_C_EXIT while
running under a scheduled task -- a console control event reached the whole process tree.
Processes with no console cannot receive one. So this script is started by pythonw.exe
(no console) and launches every worker with DETACHED_PROCESS (no console), polls for
completion instead of relying on signals, and relaunches a shard from its checkpoint if it
exits without producing its npz. Every stage is idempotent and resumes from what is on
disk, so re-running this script after any interruption is always safe.

Stages, each writing a marker to logs/run_all_v2.log:
  1. extraction: 5 train shards (pool sized by free RAM, ~1 GB per worker) -> merge -> val
  2. train.py --variants proj,dino,fuse --cv_splits 3
  3. robustness table (5 shards) -> chart -> per-class -> A/B vs v1 and vs baseline
  4. demo benchmark (4 transforms) -> error analysis -> README render
  5. multi-crop table (--crops 4, 5 shards) -> A/B vs the no-crop table
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
LOG = ROOT / "logs" / "run_all_v2.log"
MAIN = ROOT / "logs" / "extract_v2_main.log"
PY = str(Path(sys.executable).with_name("python.exe"))  # never pythonw for workers
DETACHED = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | CREATE_NO_WINDOW

SHARDS = ["clean,jpeg_90,jpeg_70", "jpeg_50,jpeg_30,blur_0.5", "blur_1.0,blur_2.0,resize_0.5",
          "resize_0.25,noise_0.02,noise_0.05", "noise_0.10,jitter_0.20,crop_0.80"]
WORKER_MB = 1100
RESERVE_MB = 900
MAX_POOL = 4
THREADS_PER_WORKER = 5


def mark(msg: str, also_main: bool = False) -> None:
    line = f"{msg} {datetime.now():%H:%M:%S}"
    for p in ([LOG, MAIN] if also_main else [LOG]):
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def free_mb() -> int:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    ms = MS(); ms.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return int(ms.ullAvailPhys // (1024 * 1024))


def spawn(args: list[str], log: Path, threads: int) -> subprocess.Popen:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               TOKENIZERS_PARALLELISM="false", PYTHONIOENCODING="utf-8")
    fh = log.open("ab")
    return subprocess.Popen([PY, "-X", "utf8", *args], stdout=fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=str(ROOT), env=env, creationflags=DETACHED)


def run_pool(jobs: list[tuple[str, list[str], Path, Path | None]], threads: int, max_pool: int = MAX_POOL,
             retries: int = 3, tag: str = "") -> None:
    """jobs: (name, args, logfile, expected_output_or_None). Relaunches a job whose process
    exits without its expected output, up to `retries` times. Pool size follows free RAM."""
    pending = [j for j in jobs if not (j[3] and j[3].exists())]
    running: dict[str, tuple[subprocess.Popen, tuple, int]] = {}
    attempts: dict[str, int] = {j[0]: 0 for j in jobs}
    while pending or running:
        for name, (proc, job, _) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            if job[3] is None or job[3].exists():
                mark(f"{tag} {name} done", also_main=True)
            elif attempts[name] < retries:
                mark(f"{tag} {name} exited rc={proc.returncode} without output -> retry {attempts[name] + 1}", also_main=True)
                pending.insert(0, job)
            else:
                mark(f"{tag} {name} FAILED after {retries} retries (rc={proc.returncode})", also_main=True)
                raise SystemExit(f"{tag} {name} failed")
        while pending and len(running) < max_pool:
            if running and free_mb() < RESERVE_MB + WORKER_MB:
                break
            job = pending.pop(0)
            attempts[job[0]] += 1
            proc = spawn(job[1], job[2], threads)
            running[job[0]] = (proc, job, attempts[job[0]])
            mark(f"{tag} {job[0]} start (attempt {attempts[job[0]]}, pool={len(running)}, free={free_mb()}MB)", also_main=True)
            time.sleep(60)  # let it load its models before judging RAM for the next one
        time.sleep(30)


def run_one(args: list[str], log: Path, threads: int = 16, name: str = "") -> None:
    proc = spawn(args, log, threads)
    rc = proc.wait()
    if rc != 0:
        mark(f"{name} FAILED rc={rc}")
        raise SystemExit(f"{name} failed rc={rc}")


def stage_extract() -> None:
    if (ROOT / "artifacts/features_train.npz").exists() and (ROOT / "artifacts/features_val.npz").exists() \
            and LOG.exists() and "EXTRACT_V2_DONE" in LOG.read_text(encoding="utf-8", errors="replace"):
        mark("extract: already complete, skipping"); return
    feat = ROOT / "artifacts/_feat"; feat.mkdir(exist_ok=True)
    jobs = [(f"shard{i}", ["scripts/extract_features.py", "--split", "train", "--augment", "--shard", f"{i}/5",
                           "--out", f"artifacts/_feat/train_{i}.npz"], ROOT / f"logs/extract_train_{i}.log",
             feat / f"train_{i}.npz") for i in range(5)]
    run_pool(jobs, THREADS_PER_WORKER, tag="LIST")
    mark("TRAIN_SHARDS_DONE", also_main=True)
    run_one(["scripts/extract_features.py", "--merge", *[f"artifacts/_feat/train_{i}.npz" for i in range(5)],
             "--out", "artifacts/features_train.npz"], ROOT / "logs/extract_merge.log", name="MERGE")
    run_one(["scripts/extract_features.py", "--split", "val", "--out", "artifacts/features_val.npz"],
            ROOT / "logs/extract_val.log", name="VAL")
    mark("EXTRACT_V2_DONE", also_main=True)


def stage_train() -> None:
    if LOG.exists() and "TRAIN_DONE" in LOG.read_text(encoding="utf-8", errors="replace"):
        mark("train: already done, skipping"); return
    run_one(["scripts/train.py", "--variants", "proj,dino,fuse", "--cv_splits", "3"], ROOT / "logs/train_v2.log", name="TRAIN")
    txt = (ROOT / "logs/train_v2.log").read_text(encoding="utf-8", errors="replace")
    win = [l.strip() for l in txt.splitlines() if "WINNER" in l or "CV AUC" in l]
    mark("TRAIN_DONE " + " | ".join(win))


def table_stage(crops: int, out_csv: str, parts_dir: str, tag: str) -> None:
    if (ROOT / out_csv).exists():
        mark(f"{tag}: {out_csv} exists, skipping"); return
    pd = ROOT / parts_dir; pd.mkdir(exist_ok=True)
    for p in pd.glob("*.csv"):
        p.unlink()
    extra = ["--crops", str(crops)] if crops else []
    jobs = [(f"part{i}", ["scripts/make_tables.py", *extra, "--transforms", SHARDS[i], "--out", f"{parts_dir}/part_{i}.csv"],
             ROOT / f"logs/{tag}_shard_{i}.log", pd / f"part_{i}.csv") for i in range(5)]
    run_pool(jobs, THREADS_PER_WORKER, tag=tag)
    run_one(["scripts/make_tables.py", "--merge", *[f"{parts_dir}/part_{i}.csv" for i in range(5)], "--out", out_csv],
            ROOT / f"logs/{tag}_merge.log", name=f"{tag}_MERGE")


def stage_eval() -> None:
    table_stage(0, "results/robustness_table.csv", "results/_parts", "TABLE")
    run_one(["scripts/make_chart.py"], ROOT / "logs/v2_chart.log", name="CHART")
    mark("TABLE_DONE")
    run_one(["scripts/per_class_table.py"], ROOT / "logs/v2_byclass.log", name="BYCLASS")
    run_one(["scripts/compare_ab.py", "--baseline", "results/v1/robustness_table.csv", "--candidate",
             "results/robustness_table.csv", "--out", "results/ab_v1_v2.csv"], ROOT / "logs/v2_ab.log", name="AB_V1")
    run_one(["scripts/compare_ab.py", "--baseline", "results/baseline/robustness_table.csv", "--candidate",
             "results/robustness_table.csv", "--out", "results/ab_summary.csv"], ROOT / "logs/v2_ab.log", name="AB_BASE")
    dp = ROOT / "results/_demo_parts"; dp.mkdir(exist_ok=True)
    for p in dp.glob("*.csv"):
        p.unlink()
    demo = ["clean", "jpeg_30", "resize_0.25", "noise_0.05"]
    jobs = [(t, ["scripts/eval_demo.py", "--transforms", t, "--out", f"results/_demo_parts/{t}.csv"],
             ROOT / f"logs/v2_demo_{t}.log", dp / f"{t}.csv") for t in demo]
    run_pool(jobs, THREADS_PER_WORKER, tag="DEMO")
    run_one(["scripts/eval_demo.py", "--merge", *[f"results/_demo_parts/{t}.csv" for t in demo]],
            ROOT / "logs/v2_demo_merge.log", name="DEMO_MERGE")
    run_one(["scripts/error_analysis.py", "--k", "8"], ROOT / "logs/v2_error.log", name="ERROR")
    run_one(["scripts/update_readme.py"], ROOT / "logs/v2_readme.log", name="README")
    mark("TRAIN_EVAL_V2_DONE")


def stage_crops() -> None:
    table_stage(4, "results/robustness_table_crops4.csv", "results/_parts_crops", "CROPS")
    run_one(["scripts/compare_ab.py", "--baseline", "results/robustness_table.csv", "--candidate",
             "results/robustness_table_crops4.csv", "--out", "results/ab_crops4.csv"], ROOT / "logs/crops_ab.log", name="AB_CROPS")
    mark("CROPS_V2_DONE")


def main() -> None:
    mark("ALL_START (pythonw, detached workers)", also_main=True)
    stage_extract()
    stage_train()
    stage_eval()
    stage_crops()
    mark("ALL_DONE", also_main=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        mark(f"ALL_FAILED {e}", also_main=True)
        raise
    except Exception as e:  # noqa: BLE001
        mark(f"ALL_FAILED {type(e).__name__}: {e}", also_main=True)
        raise
