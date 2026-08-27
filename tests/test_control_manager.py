from types import SimpleNamespace

import numpy as np
import pytest

from env.robot_manager.control_manager import MetaControl


class _RobotManager:
    def __init__(self, robot):
        self.robot = robot

    def get_end_effector_real_val(self, robot, env_idx_list):
        return {env_idx_list[0]: np.asarray([0.0184, -0.0184])}

    def get_robot_obs_name(self):
        return ["ee_joint_state"]

    def restore_name(self, name):
        assert name == "ee_joint_state"
        return "arm"

    def get_robot_by_gripper_name(self, name):
        assert name == "ee"
        return self.robot


def _robot(rate_limit=None):
    robot = SimpleNamespace(
        robot_name="test_robot",
        ee_type="gripper",
        gripper_scale=[0.0, 0.035],
        gripper_move={"mimic": ["joint8", -1.0, 0.0]},
    )
    if rate_limit is not None:
        robot.gripper_rate_limit = rate_limit
    return robot


def _close_action(manager):
    return MetaControl({"ee_joint_state": {"position": [0.0], "velocity": [0.0]}}).get_action(manager, 0)


def test_gripper_rate_limit_defaults_to_existing_twenty_percent_step():
    robot = _robot()
    action = _close_action(_RobotManager(robot))

    assert action["ee_joint_state"]["position"] == pytest.approx([0.0114, -0.0114])


def test_gripper_rate_limit_one_preserves_full_close_target():
    robot = _robot(rate_limit=1.0)
    action = _close_action(_RobotManager(robot))

    assert action["ee_joint_state"]["position"] == pytest.approx([0.0, 0.0])


def test_gripper_rate_limit_rejects_invalid_values():
    robot = _robot(rate_limit=0.0)

    with pytest.raises(ValueError, match="gripper_rate_limit"):
        _close_action(_RobotManager(robot))
