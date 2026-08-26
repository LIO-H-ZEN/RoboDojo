#!/usr/bin/env python3
"""Validate the exact PIPER zero-shot layouts before Isaac Sim starts."""

import argparse
from pathlib import Path

from env.seed_manager import seed_manager
from utils.load_file import is_git_lfs_pointer

PIPER_FILES = (
    "piper.usd",
    "piper.urdf",
    "robot_config.yml",
    "curobo.yml",
)


def parse_layout_ids(raw_layout_ids: str) -> list[int]:
    parts = raw_layout_ids.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"layout ids must be comma-separated non-negative integers: {raw_layout_ids!r}")
    layout_ids = [int(part) for part in parts]
    if len(set(layout_ids)) != len(layout_ids):
        raise ValueError(f"layout ids must be unique: {layout_ids}")
    return layout_ids


def require_hydrated_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required asset is missing: {path}")
    if is_git_lfs_pointer(path):
        raise ValueError(f"required asset is an unresolved Git LFS pointer: {path}")


def validate_assets(assets_root: Path, layout_ids: list[int], seed: int) -> None:
    assets_root = assets_root.resolve()
    for relative_path in PIPER_FILES:
        require_hydrated_file(assets_root / "Robots" / "piper" / relative_path)

    for relative_dir in ("Background", "Material", "Room", "Sensor", "Traj"):
        path = assets_root / relative_dir
        if not path.is_dir():
            raise FileNotFoundError(f"required common asset directory is missing: {path}")

    seed_manager.ASSETS_PATH = str(assets_root)
    manager = seed_manager.SeedManager(
        {
            "num_envs": 1,
            "task_name": "general_pickup_single",
            "config_name": "piper_single",
            "layout_config_name": "arx_x5",
            "layout_task_name": "general_pickup",
            "layout_filter": {
                "target_position": {
                    "label": "target",
                    "xlim": [-0.25, 0.25],
                    "ylim": [-0.25, 0.0],
                }
            },
            "validate_layout_assets": True,
            "eval_num": len(layout_ids),
            "layout_ids": layout_ids,
            "seed": seed,
        }
    )
    manager.init_eval()
    if manager.seed_list != layout_ids:
        raise RuntimeError(f"validated layouts changed unexpectedly: expected={layout_ids} actual={manager.seed_list}")
    print(f"[asset-preflight] PASS assets={assets_root} layouts={layout_ids}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--layout-ids", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    validate_assets(args.assets_root, parse_layout_ids(args.layout_ids), args.seed)


if __name__ == "__main__":
    main()
