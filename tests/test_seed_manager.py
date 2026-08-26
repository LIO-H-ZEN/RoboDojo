import json

from env.seed_manager import seed_manager


def test_layout_config_name_can_reuse_an_existing_layout_family(tmp_path, monkeypatch):
    layout_dir = tmp_path / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layout_dir.mkdir(parents=True)
    expected = {"Rigid": {"bell": [{"category_idx": 0}]}}
    (layout_dir / "general_pickup_0.json").write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(seed_manager, "ASSETS_PATH", str(tmp_path))

    manager = seed_manager.SeedManager(
        {
            "num_envs": 1,
            "task_name": "general_pickup_single",
            "config_name": "arx_x5_single",
            "layout_config_name": "arx_x5",
            "layout_task_name": "general_pickup",
            "seed": 0,
        }
    )
    manager.init_eval()

    assert manager.get_seeds() == [0]
    assert manager.get_seed_scene_info(0) == expected
