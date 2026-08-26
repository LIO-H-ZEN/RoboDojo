from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_piper_policy_observation_frequency_has_an_exact_physics_interval():
    env_cfg = yaml.safe_load((ROOT / "env_cfg" / "piper_single.yml").read_text(encoding="utf-8"))
    sim_name = env_cfg["config"]["sim"]
    sim_cfg = yaml.safe_load((ROOT / "env_cfg" / "sim" / f"{sim_name}.yml").read_text(encoding="utf-8"))

    frequency = env_cfg["observation"]["collect_freq"]
    interval = 1.0 / (sim_cfg["dt"] * frequency)

    assert frequency == 20
    assert interval == 10
    assert sim_cfg["render_interval"] == interval
    assert sim_cfg["scene"]["num_envs"] == 1
