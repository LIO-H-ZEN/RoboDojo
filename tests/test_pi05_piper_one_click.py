import json
from pathlib import Path
import re
import subprocess
import tarfile
import uuid

import pytest

from scripts.internal.validate_pi05_piper_eval_assets import parse_layout_ids, validate_assets


def make_asset_tree(root: Path) -> Path:
    assets = root / "Assets"
    piper_dir = assets / "Robots" / "piper"
    piper_dir.mkdir(parents=True)
    for filename in ("piper.usd", "piper.urdf", "robot_config.yml", "curobo.yml"):
        (piper_dir / filename).write_text("hydrated", encoding="utf-8")

    layout_dir = assets / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    layout = {
        "Rigid": {
            "test_object": [
                {
                    "category_idx": 0,
                    "label": "target",
                    "default_pos": [0.0, -0.1, 0.8],
                }
            ]
        }
    }
    (layout_dir / "general_pickup_0.json").write_text(json.dumps(layout), encoding="utf-8")

    object_dir = assets / "Object" / "RoboDojo" / "Rigid" / "test_object" / "00000"
    object_dir.mkdir(parents=True)
    (object_dir / "metadata.json").write_text(json.dumps({"geometry": {"mass": 1.0}}), encoding="utf-8")
    (object_dir / "object.usdz").write_bytes(b"hydrated-usdz")
    return assets


def make_uv_env(root: Path) -> Path:
    env_dir = root / "uv-env"
    bin_dir = env_dir / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text('#!/usr/bin/env bash\nexec python "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    return env_dir


def test_asset_preflight_accepts_hydrated_selected_layout(tmp_path):
    assets = make_asset_tree(tmp_path)
    for common_dir in ("Background", "Material", "Room", "Sensor", "Traj"):
        (assets / common_dir).mkdir()

    validate_assets(assets, [0], seed=0)


def test_asset_preflight_rejects_piper_lfs_pointer(tmp_path):
    assets = make_asset_tree(tmp_path)
    for common_dir in ("Background", "Material", "Room", "Sensor", "Traj"):
        (assets / common_dir).mkdir()
    (assets / "Robots" / "piper" / "piper.usd").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unresolved Git LFS pointer"):
        validate_assets(assets, [0], seed=0)


@pytest.mark.parametrize("raw", ["", "0,", "-1", "0,a", "0,0"])
def test_layout_id_parser_fails_fast(raw):
    with pytest.raises(ValueError):
        parse_layout_ids(raw)


@pytest.mark.parametrize("asset_mode", ["root", "archive"])
def test_one_click_launcher_dry_run_prepares_current_head(tmp_path, asset_mode):
    repo_root = Path(__file__).resolve().parents[1]
    staged_assets = make_asset_tree(tmp_path / "staged")
    archive = tmp_path / "assets.tar"
    with tarfile.open(archive, "w") as tar_file:
        tar_file.add(staged_assets, arcname="Assets")

    common_assets = tmp_path / "common-assets"
    for common_dir in ("Background", "Material", "Room", "Sensor", "Traj"):
        (common_assets / common_dir).mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    uv_env = make_uv_env(tmp_path)
    run_id = f"pytest_pi05_{uuid.uuid4().hex}"
    if asset_mode == "root":
        for common_dir in ("Background", "Material", "Room", "Sensor", "Traj"):
            (staged_assets / common_dir).mkdir()
        asset_args = ["--assets-root", str(staged_assets)]
    else:
        asset_args = [
            "--assets-archive",
            str(archive),
            "--common-assets",
            str(common_assets),
        ]

    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_pi05_piper_zero_shot.sh"),
            "--ckpt",
            str(checkpoint),
            "--policy-env",
            str(uv_env),
            "--eval-env",
            str(uv_env),
            *asset_args,
            "--layout-ids",
            "0",
            "--run-id",
            run_id,
            "--runtime-base",
            str(tmp_path),
            "--allow-dirty",
            "--dry-run",
        ],
        cwd=repo_root.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "[asset-preflight] PASS" in completed.stdout
    assert "task=general_pickup_single env_cfg=piper_single eval_num=1" in completed.stdout
    assert "[pi05-piper] dry-run PASS" in completed.stdout
    runtime_match = re.search(r"runtime=([^ ]+)", completed.stdout)
    assert runtime_match is not None
    assert not Path(runtime_match.group(1)).exists()
