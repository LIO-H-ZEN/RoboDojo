#!/usr/bin/env python3
"""Measure exact convex distance between RoboDojo target and Piper fingers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from general_pickup_direct_replay_common import sha256_file
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import ConvexHull
import transforms3d as t3d


def convex_vertices(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError(f"convex input must be finite Nx3 with N>=4, got {points.shape}")
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    if len(vertices) < 4:
        raise ValueError("convex hull has fewer than four vertices")
    return vertices


def transform_body_points(points: np.ndarray, pose_wxyz: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"pose must be a finite xyz+wxyz vector, got {pose.shape}")
    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"pose quaternion is not normalized: {norm}")
    rotation = t3d.quaternions.quat2mat(quaternion / norm)
    return np.asarray(points, dtype=np.float64) @ rotation.T + pose[:3]


def convex_distance(vertices_a: np.ndarray, vertices_b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    vertices_a = convex_vertices(vertices_a)
    vertices_b = convex_vertices(vertices_b)
    hull_a = ConvexHull(vertices_a)
    hull_b = ConvexHull(vertices_b)
    equations_a = hull_a.equations
    equations_b = hull_b.equations
    constraint_matrix = np.zeros((len(equations_a) + len(equations_b), 6), dtype=np.float64)
    constraint_matrix[: len(equations_a), :3] = equations_a[:, :3]
    constraint_matrix[len(equations_a) :, 3:] = equations_b[:, :3]
    upper = -np.concatenate([equations_a[:, 3], equations_b[:, 3]])
    constraint = LinearConstraint(constraint_matrix, -np.inf, upper)
    initial = np.concatenate([vertices_a.mean(axis=0), vertices_b.mean(axis=0)])

    def objective(value):
        delta = value[:3] - value[3:]
        return 0.5 * float(delta @ delta)

    def gradient(value):
        delta = value[:3] - value[3:]
        return np.concatenate([delta, -delta])

    result = minimize(
        objective,
        initial,
        jac=gradient,
        constraints=[constraint],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"convex distance optimization failed: {result.message}")
    violation = float(np.max(constraint_matrix @ result.x - upper))
    if violation > 1e-8:
        raise RuntimeError(f"convex distance solution violates hull constraints by {violation}")
    point_a = result.x[:3]
    point_b = result.x[3:]
    distance = float(np.linalg.norm(point_a - point_b))
    if not np.isfinite(distance):
        raise RuntimeError("convex distance is not finite")
    return distance, point_a, point_b


def contact_intervals(states: list[int]) -> list[list[int]]:
    if not states:
        return []
    intervals = []
    start = previous = states[0]
    for state in states[1:]:
        if state != previous + 1:
            intervals.append([start, previous])
            start = state
        previous = state
    intervals.append([start, previous])
    return intervals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-state", type=int, default=85)
    parser.add_argument("--contact-tolerance-m", type=float, default=1e-5)
    args = parser.parse_args()
    if args.first_state < 0:
        raise ValueError("first-state must be non-negative")
    if not np.isfinite(args.contact_tolerance_m) or args.contact_tolerance_m <= 0:
        raise ValueError("contact-tolerance-m must be positive and finite")

    with np.load(args.geometry, allow_pickle=False) as geometry:
        geometry_metadata = json.loads(str(geometry["metadata_json"]))
        local_vertices = {
            label: convex_vertices(geometry[f"{label}_points_body"])
            for label in ("target", "left_finger", "right_finger")
        }
    with np.load(args.telemetry, allow_pickle=False) as telemetry:
        poses = {
            "target": np.asarray(telemetry["target_pose_wxyz"], dtype=np.float64),
            "left_finger": np.asarray(telemetry["left_finger_pose_wxyz"], dtype=np.float64),
            "right_finger": np.asarray(telemetry["right_finger_pose_wxyz"], dtype=np.float64),
        }
        opening = np.asarray(telemetry["gripper_opening_m"], dtype=np.float64)
        target_lift = np.asarray(telemetry["target_lift_height_m"], dtype=np.float64)
    state_count = len(poses["target"])
    if any(value.shape != (state_count, 7) for value in poses.values()):
        raise ValueError("telemetry pose arrays must all be Tx7")
    if opening.shape != (state_count,) or target_lift.shape != (state_count,):
        raise ValueError("telemetry scalar arrays must match pose state count")
    if args.first_state >= state_count:
        raise ValueError(f"first-state {args.first_state} is outside {state_count} telemetry states")

    rows = []
    contact_states = {"left_finger": [], "right_finger": [], "both": []}
    for state in range(args.first_state, state_count):
        target_world = transform_body_points(local_vertices["target"], poses["target"][state])
        distances = {}
        closest = {}
        for side in ("left_finger", "right_finger"):
            finger_world = transform_body_points(local_vertices[side], poses[side][state])
            distance, target_point, finger_point = convex_distance(target_world, finger_world)
            distances[side] = distance
            closest[side] = {
                "target": target_point.tolist(),
                "finger": finger_point.tolist(),
            }
            if distance <= args.contact_tolerance_m:
                contact_states[side].append(state)
        if all(distance <= args.contact_tolerance_m for distance in distances.values()):
            contact_states["both"].append(state)
        rows.append(
            {
                "state": state,
                "gripper_opening_m": float(opening[state]),
                "target_lift_height_m": float(target_lift[state]),
                "left_distance_m": distances["left_finger"],
                "right_distance_m": distances["right_finger"],
                "closest_points": closest,
            }
        )

    summary = {
        "schema_version": "general_pickup_collision_geometry_analysis_v1",
        "geometry": str(args.geometry),
        "geometry_sha256": sha256_file(args.geometry),
        "telemetry": str(args.telemetry),
        "telemetry_sha256": sha256_file(args.telemetry),
        "geometry_metadata": geometry_metadata,
        "first_state": args.first_state,
        "last_state": state_count - 1,
        "contact_tolerance_m": args.contact_tolerance_m,
        "hull_vertex_counts": {key: len(value) for key, value in local_vertices.items()},
        "contact_intervals": {key: contact_intervals(value) for key, value in contact_states.items()},
        "minimum_distance_m": {
            "left_finger": min(row["left_distance_m"] for row in rows),
            "right_finger": min(row["right_distance_m"] for row in rows),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
