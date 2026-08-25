#!/usr/bin/env python3
"""Fetch MDL texture materials from an internet repository.

Usage:
    python scripts/fetch_table_materials.py --repo <git-url-or-dir> \
        --dest Assets/Material/table_materials --names "Wood_Pine,Concrete_004"

Supports two kinds of repositories:

1. A git repository (local path or https URL) containing a flat/nested
   collection of MDL materials (each folder = one material with .mdl + maps).
       python scripts/fetch_table_materials.py \
           --repo https://github.com/some/material_repo.git \
           --dest Assets/Material/table_materials \
           --names "Wood_Oak,Carpet_Beige"

2. A plain local directory already checked out:
       python scripts/fetch_table_materials.py \
           --repo /path/to/material_repo --dest Assets/Material/table_materials --names "*"

After downloading, point table_visual.texture.list in the task YAML at the
MDL files under --dest and set table_visual.texture.source: "remote".
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd):
    print(f"[fetch] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Git URL or local dir of MDL materials")
    parser.add_argument("--dest", default="Assets/Material/table_materials", help="Destination dir")
    parser.add_argument("--names", default="*", help="Comma-separated material folder names; '*' copies all")
    parser.add_argument("--clone-depth", type=int, default=1, help="git clone depth (default 1)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dest = (root / args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if Path(args.repo).expanduser().is_dir():
        src = Path(args.repo).expanduser().resolve()
        print(f"[fetch] using local material dir: {src}")
    else:
        tmp_repo = root / ".cache" / "table_material_repo"
        if tmp_repo.exists():
            shutil.rmtree(tmp_repo)
        _run(["git", "clone", "--depth", str(args.clone_depth), args.repo, str(tmp_repo)])
        src = tmp_repo

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if names == ["*"]:
        names = [p.name for p in src.iterdir() if p.is_dir()]

    copied = 0
    for name in names:
        sdir = src / name
        if not sdir.is_dir():
            print(f"[fetch] WARN material folder not found: {name} (under {src})")
            continue
        files = [p for p in sdir.iterdir() if p.suffix.lower() in (".mdl", ".png", ".jpg", ".jpeg", ".exr", ".tga")]
        if not files:
            print(f"[fetch] WARN no material files under {name}")
            continue
        ddir = dest / name
        ddir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, ddir / f.name)
        copied += 1
        print(f"[fetch] copied {name} ({len(files)} files)")

    print(f"[fetch] done: {copied} materials -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
