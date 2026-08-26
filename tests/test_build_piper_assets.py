from pathlib import Path

from scripts.internal import build_piper_assets


def test_assets_root_resolves_to_piper_robot_directory(tmp_path):
    assert Path(build_piper_assets.target_dir(tmp_path)) == tmp_path / "Robots" / "piper"
