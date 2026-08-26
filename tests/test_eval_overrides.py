from omegaconf import OmegaConf
import pytest

from env.eval_overrides import apply_layout_shard


def test_apply_layout_shard_updates_mapping_and_runtime_config():
    env_cfg = OmegaConf.create({"eval_cfg": {"eval_num": 10}})
    eval_cfg = {"eval_num": 10}

    layout_ids, eval_num = apply_layout_shard(
        env_cfg=env_cfg,
        eval_cfg=eval_cfg,
        eval_num=10,
        raw_layout_ids="5, 6,7,8,9",
    )

    assert layout_ids == [5, 6, 7, 8, 9]
    assert eval_num == 5
    assert eval_cfg == {"eval_num": 5, "layout_ids": [5, 6, 7, 8, 9]}
    assert OmegaConf.to_container(env_cfg.eval_cfg, resolve=True) == {
        "eval_num": 5,
        "layout_ids": [5, 6, 7, 8, 9],
    }


@pytest.mark.parametrize("raw_layout_ids", ["", "1,", "1,,2", "2,2"])
def test_apply_layout_shard_rejects_invalid_values(raw_layout_ids):
    env_cfg = OmegaConf.create({"eval_cfg": {}})

    with pytest.raises(ValueError):
        apply_layout_shard(
            env_cfg=env_cfg,
            eval_cfg={},
            eval_num=10,
            raw_layout_ids=raw_layout_ids,
        )
