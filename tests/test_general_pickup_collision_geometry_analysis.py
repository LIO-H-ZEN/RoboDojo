from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "scripts/internal/analyze_general_pickup_collision_geometry.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("general_pickup_collision_geometry_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def cube(center, half_extent=0.5):
    return np.asarray(
        [
            np.asarray(center) + [x, y, z]
            for x in (-half_extent, half_extent)
            for y in (-half_extent, half_extent)
            for z in (-half_extent, half_extent)
        ],
        dtype=np.float64,
    )


def test_convex_distance_is_exact_for_separated_and_overlapping_cubes():
    distance, point_a, point_b = analysis.convex_distance(cube([0, 0, 0]), cube([2, 0, 0]))
    np.testing.assert_allclose(distance, 1.0, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(point_a - point_b), distance, rtol=0.0, atol=1e-12)
    distance, _, _ = analysis.convex_distance(cube([0, 0, 0]), cube([0.75, 0, 0]))
    np.testing.assert_allclose(distance, 0.0, rtol=0.0, atol=1e-9)


def test_transform_body_points_and_contact_intervals():
    pose = np.asarray([1, 2, 3, 1, 0, 0, 0], dtype=np.float64)
    np.testing.assert_allclose(
        analysis.transform_body_points(np.asarray([[0, 0, 0], [1, 0, 0]]), pose),
        [[1, 2, 3], [2, 2, 3]],
        rtol=0.0,
        atol=0.0,
    )
    assert analysis.contact_intervals([2, 3, 4, 7, 9, 10]) == [[2, 4], [7, 7], [9, 10]]
