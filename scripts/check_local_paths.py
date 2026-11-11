"""Fail if a tracked text file carries an absolute path from a developer machine.

Committed results used to record `<drive>:\\<folder>\\data\\sid_subset\\...` for every
listed image. Such paths say nothing to a reader of a clone, and a home-directory path
would also leak a user name. Results now store repository-relative paths
(`data/sid_subset/...`); this check keeps it that way.

Flags, in every tracked file that decodes as UTF-8 text:
  * Windows drive paths (a drive letter, a colon, then a slash or backslash), and
  * POSIX home directories (`/home/<name>/`, `/Users/<name>/`).
The illustrative `{image_path, pred}` example in the README (`C:\\data\\a.jpg`,
`C:\\data\\b.jpg`) and the system font folder used by scripts/make_video.py are
allowed explicitly.

Usage:  python scripts/check_local_paths.py      # exit 1 and list offenders
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BS = "\\"
DRIVE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:(?:" + re.escape(BS) + r"|/)+(?=[^\s\"'`<>|*?])")
HOME = re.compile(r"(?<![A-Za-z0-9_.])/(?:home|Users)/[A-Za-z0-9_.-]+/")
SEP = r"(?:" + re.escape(BS) + r"|/){1,2}"
# The documented directory-to-JSON example (README.md, docs/v2_history.md) and the
# operating-system font folder scripts/make_video.py looks in; neither is personal data.
ALLOWED = re.compile(r"C:" + SEP + r"data" + SEP + r"[ab]\.jpg|C:" + SEP + r"Windows" + SEP + r"Fonts")


def tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        ).stdout
        names = [n for n in out.decode("utf-8").split("\0") if n]
        return [root / n for n in names]
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", ".venv", "venv", "data", "__pycache__"}
        return [
            p for p in root.rglob("*")
            if p.is_file() and not skip.intersection(p.relative_to(root).parts)
        ]


def offenders(root: Path = ROOT, files: list[Path] | None = None) -> list[str]:
    hits = []
    for path in files if files is not None else tracked_files(root):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            cleaned = ALLOWED.sub("", line)
            for pattern in (DRIVE, HOME):
                m = pattern.search(cleaned)
                if m:
                    rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
                    hits.append(f"{rel}:{lineno}: {cleaned[max(0, m.start() - 20): m.end() + 40].strip()}")
                    break
    return hits


def main() -> int:
    hits = offenders()
    if hits:
        print("Absolute local paths found in tracked files (store repository-relative paths):")
        for hit in hits:
            print("  " + hit)
        return 1
    print("ok: no absolute local paths in tracked text files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
