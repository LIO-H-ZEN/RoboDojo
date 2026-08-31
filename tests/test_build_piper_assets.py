from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.internal import build_piper_assets


def test_assets_root_resolves_to_piper_robot_directory(tmp_path):
    assert Path(build_piper_assets.target_dir(tmp_path)) == tmp_path / "Robots" / "piper"


def test_generated_wrist_camera_mounts_on_imported_usd_link(tmp_path):
    joints = [{"name": f"joint{index}", "type": "revolute" if index <= 6 else "prismatic"} for index in range(1, 9)]

    build_piper_assets.write_robot_config(tmp_path, joints)

    content = (tmp_path / "robot_config.yml").read_text(encoding="utf-8")
    assert "link: root_joint/gripper_base" in content


def test_copy_assets_accepts_flat_maniskill_piper_layout(tmp_path):
    source = tmp_path / "source"
    meshes = source / "meshes"
    meshes.mkdir(parents=True)
    (source / "piper_description.urdf").write_text(
        '<robot name="piper"><link name="base_link"/></robot>',
        encoding="utf-8",
    )
    (meshes / "base_link.STL").write_bytes(b"stl")

    target = tmp_path / "target"
    generated = build_piper_assets.copy_assets(source, target)

    assert Path(generated) == target / "piper.urdf"
    assert (target / "meshes" / "base_link.STL").read_bytes() == b"stl"


def test_copy_assets_sanitizes_convex_mesh_names_for_usd_prims(tmp_path):
    source = tmp_path / "source"
    meshes = source / "meshes"
    meshes.mkdir(parents=True)
    (source / "piper_description.urdf").write_text(
        '<robot name="piper"><link name="base_link"><collision><geometry>'
        '<mesh filename="meshes/base_link.convex.stl"/>'
        "</geometry></collision></link></robot>",
        encoding="utf-8",
    )
    (meshes / "base_link.convex.stl").write_bytes(b"convex")

    target = tmp_path / "target"
    build_piper_assets.copy_assets(source, target)

    assert "meshes/base_link_convex.stl" in (target / "piper.urdf").read_text(encoding="utf-8")
    assert (target / "meshes" / "base_link_convex.stl").read_bytes() == b"convex"


def test_gripper_contact_contract_is_all_or_nothing():
    args = SimpleNamespace(
        gripper_static_friction=2.0,
        gripper_dynamic_friction=2.0,
        gripper_torsional_patch_radius=0.02,
        gripper_min_torsional_patch_radius=0.005,
    )
    assert build_piper_assets.gripper_contact_contract(args) == {
        "static_friction": 2.0,
        "dynamic_friction": 2.0,
        "torsional_patch_radius": 0.02,
        "min_torsional_patch_radius": 0.005,
    }
    args.gripper_dynamic_friction = None
    with pytest.raises(ValueError, match="all four"):
        build_piper_assets.gripper_contact_contract(args)
