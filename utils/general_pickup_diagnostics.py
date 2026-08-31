"""Lightweight, simulator-independent diagnostics for General Pickup rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DIAGNOSTICS_VERSION = "general_pickup_episode_diagnostics_v2"
NEAR_TARGET_DISTANCE_M = 0.08
CLOSED_GRIPPER_OPENING_M = 0.02


def _vec3(value, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size < 3 or not np.isfinite(result[:3]).all():
        raise ValueError(f"{name} must contain at least three finite values, got {value!r}")
    return result[:3]


def _finite_vector(value, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values, got {value!r}")
    return result


@dataclass
class EpisodeDiagnosticState:
    initial_target_position: np.ndarray
    initial_tcp_position: np.ndarray
    initial_gripper_opening_m: float
    observations: int = 0
    final_target_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    final_tcp_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    final_gripper_opening_m: float = 0.0
    max_target_z_m: float = -np.inf
    min_target_z_m: float = np.inf
    min_tcp_target_distance_m: float = np.inf
    min_tcp_target_xy_distance_m: float = np.inf
    nearest_tcp_step: int = -1
    gripper_opening_at_nearest_tcp_m: float = np.inf
    first_near_target_step: int = -1
    gripper_opening_at_first_near_target_m: float = np.inf
    min_gripper_opening_m: float = np.inf
    min_gripper_step: int = -1
    first_closed_step: int = -1
    tcp_target_distance_at_first_close_m: float = np.inf
    min_tcp_target_distance_while_closed_m: float = np.inf
    max_target_z_after_first_close_m: float = -np.inf
    raw_gripper_command_min: float = np.inf
    raw_gripper_command_max: float = -np.inf
    gripper_command_clip_count: int = 0
    max_gripper_command_overflow: float = 0.0
    arm_action_min: np.ndarray | None = None
    arm_action_max: np.ndarray | None = None
    max_arm_action_step_l2: float = 0.0
    previous_arm_action: np.ndarray | None = None


class GeneralPickupEpisodeDiagnostics:
    """Accumulate small numeric traces without retaining images or simulator objects."""

    def __init__(self, num_envs: int):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self._states: list[EpisodeDiagnosticState | None] = [None] * num_envs

    def reset(self, env_idx: int, *, target_position, tcp_position, gripper_opening_m: float) -> None:
        target = _vec3(target_position, name="target_position")
        tcp = _vec3(tcp_position, name="tcp_position")
        opening = float(gripper_opening_m)
        if not np.isfinite(opening) or opening < 0:
            raise ValueError(f"gripper_opening_m must be finite and non-negative, got {gripper_opening_m!r}")
        self._states[env_idx] = EpisodeDiagnosticState(
            initial_target_position=target.copy(),
            initial_tcp_position=tcp.copy(),
            initial_gripper_opening_m=opening,
        )
        self.observe(
            env_idx,
            target_position=target,
            tcp_position=tcp,
            gripper_opening_m=opening,
        )

    def record_action(self, env_idx: int, *, arm_action, raw_gripper_command: float) -> None:
        state = self._require_state(env_idx)
        arm = _finite_vector(arm_action, name="arm_action")
        command = float(raw_gripper_command)
        if not np.isfinite(command):
            raise ValueError(f"raw_gripper_command must be finite, got {raw_gripper_command!r}")

        if state.arm_action_min is None:
            state.arm_action_min = arm.copy()
            state.arm_action_max = arm.copy()
        else:
            if arm.shape != state.arm_action_min.shape:
                raise ValueError(f"arm_action shape changed from {state.arm_action_min.shape} to {arm.shape}")
            state.arm_action_min = np.minimum(state.arm_action_min, arm)
            state.arm_action_max = np.maximum(state.arm_action_max, arm)

        if state.previous_arm_action is not None:
            state.max_arm_action_step_l2 = max(
                state.max_arm_action_step_l2,
                float(np.linalg.norm(arm - state.previous_arm_action)),
            )
        state.previous_arm_action = arm.copy()
        state.raw_gripper_command_min = min(state.raw_gripper_command_min, command)
        state.raw_gripper_command_max = max(state.raw_gripper_command_max, command)
        overflow = max(0.0, -command, command - 1.0)
        if overflow > 0:
            state.gripper_command_clip_count += 1
            state.max_gripper_command_overflow = max(state.max_gripper_command_overflow, overflow)

    def observe(self, env_idx: int, *, target_position, tcp_position, gripper_opening_m: float) -> None:
        state = self._require_state(env_idx)
        target = _vec3(target_position, name="target_position")
        tcp = _vec3(tcp_position, name="tcp_position")
        opening = float(gripper_opening_m)
        if not np.isfinite(opening) or opening < 0:
            raise ValueError(f"gripper_opening_m must be finite and non-negative, got {gripper_opening_m!r}")

        step = state.observations
        distance = float(np.linalg.norm(tcp - target))
        xy_distance = float(np.linalg.norm(tcp[:2] - target[:2]))
        if distance < state.min_tcp_target_distance_m:
            state.min_tcp_target_distance_m = distance
            state.nearest_tcp_step = step
            state.gripper_opening_at_nearest_tcp_m = opening
        state.min_tcp_target_xy_distance_m = min(state.min_tcp_target_xy_distance_m, xy_distance)
        if distance <= NEAR_TARGET_DISTANCE_M and state.first_near_target_step < 0:
            state.first_near_target_step = step
            state.gripper_opening_at_first_near_target_m = opening
        if opening < state.min_gripper_opening_m:
            state.min_gripper_opening_m = opening
            state.min_gripper_step = step
        if opening <= CLOSED_GRIPPER_OPENING_M:
            if state.first_closed_step < 0:
                state.first_closed_step = step
                state.tcp_target_distance_at_first_close_m = distance
            state.min_tcp_target_distance_while_closed_m = min(
                state.min_tcp_target_distance_while_closed_m,
                distance,
            )
            state.max_target_z_after_first_close_m = max(state.max_target_z_after_first_close_m, float(target[2]))

        state.max_target_z_m = max(state.max_target_z_m, float(target[2]))
        state.min_target_z_m = min(state.min_target_z_m, float(target[2]))
        state.final_target_position = target.copy()
        state.final_tcp_position = tcp.copy()
        state.final_gripper_opening_m = opening
        state.observations += 1

    def summary(self, env_idx: int) -> dict:
        state = self._require_state(env_idx)
        max_lift = state.max_target_z_m - float(state.initial_target_position[2])
        final_lift = float(state.final_target_position[2] - state.initial_target_position[2])
        target_displacement = float(np.linalg.norm(state.final_target_position - state.initial_target_position))
        result = {
            "version": DIAGNOSTICS_VERSION,
            "observations": state.observations,
            "initial_target_position": state.initial_target_position.tolist(),
            "final_target_position": state.final_target_position.tolist(),
            "initial_tcp_position": state.initial_tcp_position.tolist(),
            "final_tcp_position": state.final_tcp_position.tolist(),
            "max_target_z_m": state.max_target_z_m,
            "min_target_z_m": state.min_target_z_m,
            "max_lift_height_m": max_lift,
            "final_lift_height_m": final_lift,
            "min_tcp_target_distance_m": state.min_tcp_target_distance_m,
            "min_tcp_target_xy_distance_m": state.min_tcp_target_xy_distance_m,
            "nearest_tcp_step": state.nearest_tcp_step,
            "gripper_opening_at_nearest_tcp_m": state.gripper_opening_at_nearest_tcp_m,
            "first_near_target_step": state.first_near_target_step,
            "gripper_opening_at_first_near_target_m": (
                state.gripper_opening_at_first_near_target_m if state.first_near_target_step >= 0 else None
            ),
            "initial_gripper_opening_m": state.initial_gripper_opening_m,
            "final_gripper_opening_m": state.final_gripper_opening_m,
            "min_gripper_opening_m": state.min_gripper_opening_m,
            "min_gripper_step": state.min_gripper_step,
            "first_closed_step": state.first_closed_step,
            "tcp_target_distance_at_first_close_m": (
                state.tcp_target_distance_at_first_close_m if state.first_closed_step >= 0 else None
            ),
            "min_tcp_target_distance_while_closed_m": (
                state.min_tcp_target_distance_while_closed_m if state.first_closed_step >= 0 else None
            ),
            "max_lift_height_after_first_close_m": (
                state.max_target_z_after_first_close_m - float(state.initial_target_position[2])
                if state.first_closed_step >= 0
                else None
            ),
            "target_displacement_m": target_displacement,
            "gripper_command_clip_count": state.gripper_command_clip_count,
            "max_gripper_command_overflow": state.max_gripper_command_overflow,
            "max_arm_action_step_l2": state.max_arm_action_step_l2,
        }
        if state.raw_gripper_command_min != np.inf:
            result["raw_gripper_command_min"] = state.raw_gripper_command_min
            result["raw_gripper_command_max"] = state.raw_gripper_command_max
        if state.arm_action_min is not None:
            result["arm_action_min"] = state.arm_action_min.tolist()
            result["arm_action_max"] = state.arm_action_max.tolist()
        result["failure_signal"] = classify_failure_signal(result)
        return result

    def _require_state(self, env_idx: int) -> EpisodeDiagnosticState:
        try:
            state = self._states[env_idx]
        except IndexError as exc:
            raise IndexError(f"env_idx out of range: {env_idx}") from exc
        if state is None:
            raise RuntimeError(f"diagnostics for env {env_idx} were not reset")
        return state


def classify_failure_signal(summary: dict) -> str:
    """Return a conservative evidence label, not a causal diagnosis."""

    max_lift = float(summary["max_lift_height_m"])
    min_distance = float(summary["min_tcp_target_distance_m"])
    min_opening = float(summary["min_gripper_opening_m"])
    if max_lift >= 0.1:
        return "lift-threshold-reached"
    if min_distance > 0.12:
        return "tcp-never-near-target"
    if min_opening > 0.025:
        return "gripper-never-closed-near-target"
    if max_lift >= 0.002:
        return "partial-target-lift-or-motion"
    return "reached-and-closed-without-lift"
