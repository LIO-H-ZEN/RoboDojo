#!/usr/bin/env python3
"""Regenerate general_pickup eval layouts with a chosen number of clutter.

For each downloaded layout, keep:
  * every rigid labelled "target" (normally 1)
  * the N leftmost clutter rigids (smallest x in default_pos)

and drop the rest, so each scene has exactly len(targets) + N tabletop rigids.
Output goes to a NEW seed dir (default 4); the downloaded layouts are untouched.

Usage:
  python3 scripts/internal/regen_general_pickup_layout.py [--clutter N] [--dst-seed D]
"""
import argparse
import json
from pathlib import Path


def is_target_inst(inst):
    label = inst.get("label")
    return label == "target" or (isinstance(label, list) and "target" in label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clutter", type=int, default=3, help="number of clutter rigids to keep (default 3)")
    parser.add_argument("--dst-seed", type=int, default=4, help="destination eval seed dir (default 4)")
    args = parser.parse_args()

    root = Path(".cache/robodojo_assets_repo/Assets/Eval_Layout/RoboDojo/arx_x5")
    src_dir = root / "0"
    dst_dir = root / str(args.dst_seed)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("general_pickup_*.json"))
    if not files:
        raise SystemExit(f"no layouts found in {src_dir}")

    per_layout = []
    for f in files:
        data = json.loads(f.read_text())
        rigid = data.get("Rigid", {})

        targets = []
        clutters = []
        for cat, insts in rigid.items():
            for inst in insts:
                if is_target_inst(inst):
                    targets.append((cat, inst))
                else:
                    clutters.append((cat, inst))

        # keep the N leftmost clutter (smallest x)
        clutters.sort(key=lambda ci: ci[1].get("default_pos", [0.0, 0.0, 0.0])[0])
        keep_clutter = clutters[: args.clutter]

        new_rigid = {}
        for cat, inst in targets + keep_clutter:
            new_rigid.setdefault(cat, []).append(inst)
        data["Rigid"] = new_rigid

        out = dst_dir / f.name
        out.write_text(json.dumps(data, indent=2))
        per_layout.append(len(targets) + len(keep_clutter))

    print(f"wrote {len(files)} layouts (rigids per layout: {set(per_layout)}) to {dst_dir}")


if __name__ == "__main__":
    main()
