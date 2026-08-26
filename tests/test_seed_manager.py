import json

import pytest

from env.seed_manager import seed_manager


def test_layout_config_name_can_reuse_an_existing_layout_family(tmp_path, monkeypatch):
    layout_dir = tmp_path / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    rejected = {
        "Rigid": {"bell": [{"category_idx": 0, "label": "target", "default_pos": [0.4, 0.0, 0.8]}]},
        "Geometry": {"camera_stand": [{"category_idx": 0, "default_pos": [0.0, -0.47, 0.765]}]},
    }
    expected = {
        "Rigid": {"bell": [{"category_idx": 0, "label": "target", "default_pos": [0.1, 0.0, 0.8]}]},
        "Geometry": {"camera_stand": [{"category_idx": 0, "default_pos": [0.0, -0.47, 0.765]}]},
    }
    (layout_dir / "general_pickup_0.json").write_text(json.dumps(rejected), encoding="utf-8")
    (layout_dir / "general_pickup_1.json").write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(seed_manager, "ASSETS_PATH", str(tmp_path))

    manager = seed_manager.SeedManager(
        {
            "num_envs": 1,
            "task_name": "general_pickup_single",
            "config_name": "arx_x5_single",
            "layout_config_name": "arx_x5",
            "layout_task_name": "general_pickup",
            "layout_filter": {
                "target_position": {"label": "target", "xlim": [-0.25, 0.25], "ylim": [-0.2, 0.05]}
            },
            "layout_overrides": {"Geometry": {"camera_stand": {"default_pos": [-0.55, -0.47, 0.715]}}},
            "seed": 0,
        }
    )
    manager.init_eval()

    assert manager.get_seeds() == [0]
    scene = manager.get_seed_scene_info(0)
    assert scene["Rigid"] == expected["Rigid"]
    assert scene["Geometry"]["camera_stand"][0]["default_pos"] == [-0.55, -0.47, 0.715]


def test_layout_ids_select_exact_original_layouts(tmp_path, monkeypatch):
    layout_dir = tmp_path / "Eval_Layout" / "RoboDojo" / "piper_single" / "0"
    layout_dir.mkdir(parents=True)
    for layout_id in range(4):
        layout = {
            "Rigid": {
                "object": [
                    {
                        "category_idx": layout_id,
                        "label": "target",
                        "default_pos": [0.0, 0.0, 0.8],
                    }
                ]
            }
        }
        (layout_dir / f"general_pickup_single_{layout_id}.json").write_text(
            json.dumps(layout), encoding="utf-8"
        )
    monkeypatch.setattr(seed_manager, "ASSETS_PATH", str(tmp_path))

    manager = seed_manager.SeedManager(
        {
            "num_envs": 1,
            "task_name": "general_pickup_single",
            "config_name": "piper_single",
            "layout_ids": [3, 1],
            "seed": 0,
        }
    )
    manager.init_eval()

    assert manager.get_seeds() == [3]
    assert manager.get_seeds() == [1]
    assert manager.get_seeds() is None
    assert manager.get_seed_scene_info(3)["Rigid"]["object"][0]["category_idx"] == 3
    assert manager.get_seed_scene_info(1)["Rigid"]["object"][0]["category_idx"] == 1


@pytest.mark.parametrize("layout_ids", [[], [0, 0], [4], "0,1"])
def test_layout_ids_fail_fast_when_invalid(tmp_path, monkeypatch, layout_ids):
    layout_dir = tmp_path / "Eval_Layout" / "RoboDojo" / "piper_single" / "0"
    layout_dir.mkdir(parents=True)
    (layout_dir / "general_pickup_single_0.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(seed_manager, "ASSETS_PATH", str(tmp_path))
    manager = seed_manager.SeedManager(
        {
            "num_envs": 1,
            "task_name": "general_pickup_single",
            "config_name": "piper_single",
            "layout_ids": layout_ids,
            "seed": 0,
        }
    )

    with pytest.raises(ValueError):
        manager.init_eval()


def test_layout_asset_validation_rejects_unresolved_lfs_pointer(tmp_path, monkeypatch):
    layout_dir = tmp_path / "Eval_Layout" / "RoboDojo" / "piper_single" / "0"
    layout_dir.mkdir(parents=True)
    layout = {
        "Rigid": {
            "car": [
                {
                    "category_idx": 11,
                    "label": "target",
                    "default_pos": [0.0, 0.0, 0.8],
                }
            ]
        }
    }
    (layout_dir / "general_pickup_single_0.json").write_text(json.dumps(layout), encoding="utf-8")
    asset_dir = tmp_path / "Object" / "RoboDojo" / "Rigid" / "car" / "00011"
    asset_dir.mkdir(parents=True)
    lfs_pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
    (asset_dir / "metadata.json").write_text(lfs_pointer, encoding="utf-8")
    (asset_dir / "object.usdz").write_text(lfs_pointer, encoding="utf-8")
    monkeypatch.setattr(seed_manager, "ASSETS_PATH", str(tmp_path))
    manager = seed_manager.SeedManager(
        {
            "num_envs": 1,
            "task_name": "general_pickup_single",
            "config_name": "piper_single",
            "validate_layout_assets": True,
            "eval_num": 1,
            "seed": 0,
        }
    )

    with pytest.raises(ValueError, match="unresolved Git LFS pointer"):
        manager.init_eval()


def test_layout_asset_validation_accepts_hydrated_asset(tmp_path, monkeypatch):
    layout_dir = tmp_path / "Eval_Layout" / "RoboDojo" / "piper_single" / "0"
    layout_dir.mkdir(parents=True)
    layout = {
        "Rigid": {
            "car": [
                {
                    "category_idx": 11,
                    "label": "target",
                    "default_pos": [0.0, 0.0, 0.8],
                }
            ]
        }
    }
    (layout_dir / "general_pickup_single_0.json").write_text(json.dumps(layout), encoding="utf-8")
    asset_dir = tmp_path / "Object" / "RoboDojo" / "Rigid" / "car" / "00011"
    asset_dir.mkdir(parents=True)
    (asset_dir / "metadata.json").write_text(
        json.dumps({"geometry": {"mass": 1.0}}), encoding="utf-8"
    )
    (asset_dir / "object.usdz").write_bytes(b"hydrated-usdz")
    monkeypatch.setattr(seed_manager, "ASSETS_PATH", str(tmp_path))
    manager = seed_manager.SeedManager(
        {
            "num_envs": 1,
            "task_name": "general_pickup_single",
            "config_name": "piper_single",
            "validate_layout_assets": True,
            "eval_num": 1,
            "seed": 0,
        }
    )

    manager.init_eval()

    assert manager.get_seeds() == [0]
