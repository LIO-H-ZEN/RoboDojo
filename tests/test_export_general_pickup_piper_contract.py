from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.internal.export_general_pickup_piper_contract import (
    parse_layout_ids,
    pose_robodojo_to_maniskill,
    write_contracts,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_repo(root: Path) -> tuple[Path, Path]:
    repo = root / "repo"
    assets = root / "assets"
    config_payloads = {
        "env_cfg/piper_single.yml": {"config_name": "piper_single"},
        "env_cfg/sim/sim_config_piper_policy.yml": {"dt": 0.005},
        "env_cfg/camera/camera_config_piper_policy.yml": {"cam_head": {}},
        "env_cfg/scene/single_arm.yml": {"Table": {}},
        "env_cfg/robot/single_piper.yml": {"robots": []},
        "task/RoboDojo/config/general_pickup.yml": {"Clutter": [{"nums": 10}]},
    }
    for relative_path, payload in config_payloads.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    piper_config = assets / "Robots" / "piper" / "robot_config.yml"
    piper_config.parent.mkdir(parents=True)
    piper_config.write_text(yaml.safe_dump({"ee_link": "link6"}), encoding="utf-8")
    return repo, assets


def add_asset(assets: Path, asset_type: str, category: str, category_idx: int, descriptions: list[str]) -> None:
    asset_dir = assets / "Object" / "RoboDojo" / asset_type / category / f"{category_idx:05d}"
    write_json(
        asset_dir / "metadata.json",
        {
            "geometry": {"oriented_bbox": {"vertices": [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]}},
            "active": {"place": {"default": {"projection_circle": {"center": [0, 0, -0.01, 1, 0, 0, 0]}}}},
        },
    )
    write_json(asset_dir / "description.json", {"description": descriptions})
    (asset_dir / "object.usdz").write_bytes(b"hydrated")


def make_layout(target_x: float = 0.0, clutter_count: int = 1) -> dict:
    rigid = {
        "target_object": [
            {
                "category_idx": 0,
                "label": "target",
                "default_pos": [target_x, -0.1, 0.8],
                "default_ori": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "physics": {"mass": 0.1},
            }
        ]
    }
    for index in range(clutter_count):
        rigid[f"clutter_{index}"] = [
            {
                "category_idx": 0,
                "type": "cluttered",
                "default_pos": [0.1 + index * 0.05, -0.1, 0.8],
                "default_ori": [1.0, 0.0, 0.0, 0.0],
            }
        ]
    return {"Rigid": rigid, "Table": {"default": "table"}}


def test_layout_id_parser_supports_ranges_and_fails_on_duplicates():
    assert parse_layout_ids("0-2,4") == [0, 1, 2, 4]
    with pytest.raises(ValueError, match="unique"):
        parse_layout_ids("0-2,2")


def test_pose_transform_maps_robodojo_base_to_maniskill_base():
    pose = pose_robodojo_to_maniskill([0.0, -0.45, 0.765], [2**-0.5, 0.0, 0.0, 2**-0.5])
    assert pose["position"] == pytest.approx([-0.35, 0.0, 0.0])
    assert pose["orientation_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_export_writes_deterministic_contract_and_records_source_clutter_count(tmp_path):
    repo, assets = make_repo(tmp_path)
    add_asset(assets, "Rigid", "target_object", 0, ["test object"])
    add_asset(assets, "Clutter", "clutter_0", 0, ["red distractor"])
    layout_dir = assets / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    write_json(layout_dir / "general_pickup_0.json", make_layout(clutter_count=1))

    output = tmp_path / "contracts"
    manifest = write_contracts(
        repo_root=repo,
        assets_root=assets,
        output_dir=output,
        layout_ids=[0],
        seed=0,
        expected_clutter_count=10,
    )
    contract = json.loads((output / "general_pickup_filtered_000.json").read_text(encoding="utf-8"))

    assert manifest["contracts"][0]["source_clutter_count"] == 1
    assert contract["source_clutter_count_matches_config"] is False
    assert contract["target"]["instruction_candidates"] == ["Pick up the test object by 10 cm."]
    assert contract["contract_sha256"] == manifest["contracts"][0]["contract_sha256"]


def test_export_fails_when_description_is_missing(tmp_path):
    repo, assets = make_repo(tmp_path)
    add_asset(assets, "Rigid", "target_object", 0, [])
    add_asset(assets, "Clutter", "clutter_0", 0, ["distractor"])
    layout_dir = assets / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    write_json(layout_dir / "general_pickup_0.json", make_layout())

    with pytest.raises(ValueError, match="no non-empty descriptions"):
        write_contracts(
            repo_root=repo,
            assets_root=assets,
            output_dir=tmp_path / "contracts",
            layout_ids=[0],
            seed=0,
            expected_clutter_count=10,
        )


def test_export_allows_clutter_without_description(tmp_path):
    repo, assets = make_repo(tmp_path)
    add_asset(assets, "Rigid", "target_object", 0, ["target"])
    add_asset(assets, "Clutter", "clutter_0", 0, ["unused"])
    (assets / "Object" / "RoboDojo" / "Clutter" / "clutter_0" / "00000" / "description.json").unlink()
    layout_dir = assets / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    write_json(layout_dir / "general_pickup_0.json", make_layout())

    write_contracts(
        repo_root=repo,
        assets_root=assets,
        output_dir=tmp_path / "contracts",
        layout_ids=[0],
        seed=0,
        expected_clutter_count=10,
    )
