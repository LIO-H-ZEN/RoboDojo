import json

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
