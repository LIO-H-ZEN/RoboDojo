import json

import numpy as np

from scripts.internal.check_general_pickup_diagnostics import check_result
from utils.general_pickup_diagnostics import GeneralPickupEpisodeDiagnostics


def test_tracker_reports_approach_close_and_partial_lift():
    tracker = GeneralPickupEpisodeDiagnostics(num_envs=1)
    tracker.reset(
        0,
        target_position=[0.3, 0.0, 0.7],
        tcp_position=[0.0, 0.0, 0.9],
        gripper_opening_m=0.07,
    )
    tracker.record_action(0, arm_action=[0, 0, 0, 0, 0, 0], raw_gripper_command=1.2)
    tracker.record_action(0, arm_action=[0.1, 0, 0, 0, 0, 0], raw_gripper_command=0.0)
    tracker.observe(
        0,
        target_position=[0.3, 0.0, 0.71],
        tcp_position=[0.3, 0.0, 0.72],
        gripper_opening_m=0.01,
    )

    summary = tracker.summary(0)

    assert np.isclose(summary["max_lift_height_m"], 0.01)
    assert np.isclose(summary["min_tcp_target_distance_m"], 0.01)
    assert np.isclose(summary["min_gripper_opening_m"], 0.01)
    assert summary["nearest_tcp_step"] == 1
    assert np.isclose(summary["gripper_opening_at_nearest_tcp_m"], 0.01)
    assert summary["first_near_target_step"] == 1
    assert summary["first_closed_step"] == 1
    assert np.isclose(summary["tcp_target_distance_at_first_close_m"], 0.01)
    assert np.isclose(summary["min_tcp_target_distance_while_closed_m"], 0.01)
    assert np.isclose(summary["max_lift_height_after_first_close_m"], 0.01)
    assert np.isclose(summary["target_displacement_m"], 0.01)
    assert summary["gripper_command_clip_count"] == 1
    assert np.isclose(summary["max_gripper_command_overflow"], 0.2)
    assert np.isclose(summary["max_arm_action_step_l2"], 0.1)
    assert summary["failure_signal"] == "partial-target-lift-or-motion"


def test_tracker_distinguishes_never_approached_target():
    tracker = GeneralPickupEpisodeDiagnostics(num_envs=1)
    tracker.reset(
        0,
        target_position=[0.4, 0.0, 0.7],
        tcp_position=[0.0, 0.0, 0.9],
        gripper_opening_m=0.07,
    )
    tracker.observe(
        0,
        target_position=[0.4, 0.0, 0.7],
        tcp_position=[0.1, 0.0, 0.8],
        gripper_opening_m=0.07,
    )

    assert tracker.summary(0)["failure_signal"] == "tcp-never-near-target"


def test_captured_result_check_is_red_for_failed_lift(tmp_path):
    result = tmp_path / "_result.json"
    result.write_text(
        json.dumps(
            {
                "details": {
                    "0": {
                        "layout_id": 9,
                        "success": False,
                        "diagnostics": {
                            "max_lift_height_m": 0.001,
                            "min_tcp_target_distance_m": 0.2,
                            "min_gripper_opening_m": 0.06,
                            "failure_signal": "tcp-never-near-target",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    failures = check_result(result, layout_id=9, required_lift_m=0.1)

    assert len(failures) == 1
    assert "layout 9 failed" in failures[0]


def test_captured_result_check_is_green_for_successful_lift(tmp_path):
    result = tmp_path / "_result.json"
    result.write_text(
        json.dumps(
            {
                "details": {
                    "0": {
                        "layout_id": 8,
                        "success": True,
                        "diagnostics": {
                            "max_lift_height_m": 0.11,
                            "min_tcp_target_distance_m": 0.01,
                            "min_gripper_opening_m": 0.015,
                            "failure_signal": "lift-threshold-reached",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert check_result(result, layout_id=8, required_lift_m=0.1) == []
