from pathlib import Path

from scripts.internal import build_piper_assets


def test_assets_root_resolves_to_piper_robot_directory(tmp_path):
    assert Path(build_piper_assets.target_dir(tmp_path)) == tmp_path / "Robots" / "piper"


def test_generated_wrist_camera_mounts_on_imported_usd_link(tmp_path):
    joints = [{"name": f"joint{index}", "type": "revolute" if index <= 6 else "prismatic"} for index in range(1, 9)]

    build_piper_assets.write_robot_config(tmp_path, joints)

    content = (tmp_path / "robot_config.yml").read_text(encoding="utf-8")
    assert "link: root_joint/gripper_base" in content
    assert "ee_link: piper_tcp" in content
    assert "ee_link_is_physical_tcp: true" in content
    assert "gripper_rate_limit: 1.0" in content
