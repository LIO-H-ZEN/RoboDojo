"""Pure helpers for cross-simulator General Pickup direct-action replay."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

SCHEMA_VERSION = "general_pickup_direct_action_replay_v1"
CONTROL_FREQUENCY_HZ = 20
PHYSICS_DT_S = 0.005
PHYSICS_STEPS_PER_ACTION = 10
GRIPPER_JOINT_LIMIT_M = 0.035
GRIPPER_MEASURED_JOINT_TOLERANCE_M = float(np.finfo(np.float32).eps) * GRIPPER_JOINT_LIMIT_M
PIPER_FINGER_LINKS = {"left": "link7", "right": "link8"}
LAYOUT_OBJECT_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
TARGET_MASS_TOLERANCE_KG = 1e-6
GENERATED_LAYOUT_SCHEME = "generated"
CONTRACT_VERSION = "robodojo_general_pickup_piper_mvp_v1"


def _is_same_or_descendant_prim(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def classify_target_finger_contact(
    *,
    actor_paths: tuple[str, str],
    collider_paths: tuple[str, str],
    target_prim_path: str,
) -> str | None:
    paths = (*actor_paths, *collider_paths)
    if (
        not isinstance(target_prim_path, str)
        or not target_prim_path.startswith("/")
        or any(not isinstance(path, str) or not path.startswith("/") for path in paths)
    ):
        raise ValueError("contact and target prim paths must be absolute strings")
    if not any(_is_same_or_descendant_prim(path, target_prim_path) for path in paths):
        return None

    collider_segments = {segment for path in collider_paths for segment in path.split("/") if segment}
    matching_sides = [side for side, link_name in PIPER_FINGER_LINKS.items() if link_name in collider_segments]
    if len(matching_sides) > 1:
        raise ValueError(f"contact report matched both Piper fingers: {collider_paths!r}")
    return matching_sides[0] if matching_sides else None


def transform_points_between_usd_frames(
    points: np.ndarray,
    *,
    source_local_to_world: np.ndarray,
    destination_local_to_world: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    source = np.asarray(source_local_to_world, dtype=np.float64)
    destination = np.asarray(destination_local_to_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"points must be finite Nx3, got {points.shape}")
    if source.shape != (4, 4) or not np.isfinite(source).all():
        raise ValueError("source_local_to_world must be a finite 4x4 matrix")
    if destination.shape != (4, 4) or not np.isfinite(destination).all():
        raise ValueError("destination_local_to_world must be a finite 4x4 matrix")
    try:
        source_to_destination = source @ np.linalg.inv(destination)
    except np.linalg.LinAlgError as error:
        raise ValueError("destination_local_to_world must be invertible") from error
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    transformed = homogeneous @ source_to_destination
    weights = transformed[:, 3]
    if not np.isfinite(transformed).all() or np.any(np.abs(weights) <= np.finfo(np.float64).eps):
        raise ValueError("USD frame transform produced invalid homogeneous points")
    return transformed[:, :3] / weights[:, None]


def _target_layout_instance(layout: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(layout, dict):
        raise ValueError("layout must be an object")
    matches = []
    for object_type in LAYOUT_OBJECT_TYPES:
        categories = layout.get(object_type, {})
        if not isinstance(categories, dict):
            raise ValueError(f"layout {object_type} must be an object")
        for category, instances in categories.items():
            if not isinstance(instances, list):
                raise ValueError(f"layout {object_type}/{category} must be a list")
            matches.extend(instance for instance in instances if instance.get("label") == "target")
    if len(matches) != 1:
        raise ValueError(f"layout must contain exactly one target instance, got {len(matches)}")
    return matches[0]


def resolve_target_mass_kg(layout: dict[str, Any]) -> float:
    target = _target_layout_instance(layout)
    physics = target.get("physics")
    if not isinstance(physics, dict):
        raise ValueError("target physics must be an object")
    mass = physics.get("mass")
    if isinstance(mass, bool) or not isinstance(mass, int | float | np.integer | np.floating):
        raise ValueError("target mass must be a finite number")
    mass = float(mass)
    if not np.isfinite(mass):
        raise ValueError("target mass must be a finite number")
    if mass <= 0:
        raise ValueError("target mass must be positive")
    return mass


def validate_runtime_target_mass_kg(expected_mass_kg: float, runtime_mass_kg: float | None) -> dict[str, float]:
    expected = float(expected_mass_kg)
    if not np.isfinite(expected) or expected <= 0:
        raise ValueError("expected target mass must be positive and finite")
    if (
        runtime_mass_kg is None
        or isinstance(runtime_mass_kg, bool)
        or not isinstance(runtime_mass_kg, int | float | np.integer | np.floating)
    ):
        raise ValueError("runtime target mass must be a finite number")
    runtime = float(runtime_mass_kg)
    if not np.isfinite(runtime):
        raise ValueError("runtime target mass must be a finite number")
    if not np.isclose(runtime, expected, rtol=0.0, atol=TARGET_MASS_TOLERANCE_KG):
        raise ValueError(f"runtime target mass mismatch: expected {expected}, got {runtime}")
    return {"expected_mass_kg": expected, "runtime_mass_kg": runtime}


def prepare_layout_target_contact_sensor(layout: dict[str, Any], *, collision_approximation: str | None = None) -> None:
    target = _target_layout_instance(layout)
    physics = target.setdefault("physics", {})
    if not isinstance(physics, dict):
        raise ValueError("target physics must be an object")
    if "friction" in physics:
        friction = physics["friction"]
        if isinstance(friction, bool) or not isinstance(friction, int | float) or not np.isfinite(friction):
            raise ValueError("target friction must be a finite number")
        friction = float(friction)
        if friction < 0:
            raise ValueError("target friction must be non-negative")
        for key in ("static_friction", "dynamic_friction"):
            if key in physics and float(physics[key]) != friction:
                raise ValueError(f"target {key} conflicts with friction")
            physics[key] = friction
    physics["prepare_contact_sensor"] = True
    if collision_approximation is not None:
        if collision_approximation not in {"convexHull", "convexDecomposition"}:
            raise ValueError(f"unsupported target collision approximation: {collision_approximation!r}")
        physics["collision_approximation"] = collision_approximation


def resolve_gripper_contact_contract(
    *,
    static_friction: float | None,
    dynamic_friction: float | None,
    torsional_patch_radius: float | None,
    min_torsional_patch_radius: float | None,
) -> dict[str, float] | None:
    values = {
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
        "torsional_patch_radius": torsional_patch_radius,
        "min_torsional_patch_radius": min_torsional_patch_radius,
    }
    if all(value is None for value in values.values()):
        return None
    if any(value is None for value in values.values()):
        raise ValueError("gripper contact contract requires all four parameters")
    result = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"gripper contact {key} must be finite and non-negative")
        result[key] = float(value)
    if result["min_torsional_patch_radius"] > result["torsional_patch_radius"]:
        raise ValueError("gripper contact min torsional patch radius exceeds patch radius")
    return result


def resolve_replay_action_count(*, total_actions: int, max_actions: int | None) -> int:
    total = int(total_actions)
    if total <= 0:
        raise ValueError(f"total_actions must be positive, got {total_actions!r}")
    if max_actions is None:
        return total
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions <= 0:
        raise ValueError(f"max_actions must be a positive integer, got {max_actions!r}")
    if max_actions > total:
        raise ValueError(f"max_actions {max_actions} exceeds source action count {total}")
    return max_actions


def build_robot_alignment_overrides(
    *,
    disable_self_collisions: bool,
    activate_contact_sensors: bool = False,
    enable_robot_gravity: bool = False,
    arm_stiffness: float | None = None,
    arm_damping: float | None = None,
    arm_effort_limit: float | None = None,
    arm_armature: float | None = None,
    gripper_stiffness: float | None = None,
    gripper_damping: float | None = None,
    gripper_effort_limit: float | None = None,
    gripper_armature: float | None = None,
    robot_usd: str | None = None,
) -> dict[str, Any]:
    if not isinstance(disable_self_collisions, bool):
        raise ValueError("disable_self_collisions must be a boolean")
    if not isinstance(activate_contact_sensors, bool):
        raise ValueError("activate_contact_sensors must be a boolean")
    if not isinstance(enable_robot_gravity, bool):
        raise ValueError("enable_robot_gravity must be a boolean")
    overrides = {}
    if disable_self_collisions:
        overrides["enabled_self_collisions"] = False
    if activate_contact_sensors:
        overrides["activate_contact_sensors"] = True
    if enable_robot_gravity:
        overrides["disable_gravity"] = False
    if robot_usd is not None:
        if not isinstance(robot_usd, str) or not robot_usd.strip():
            raise ValueError("robot_usd must be a non-empty path string")
        overrides["usd_path"] = robot_usd

    actuator_groups = {
        "arm": {
            "stiffness": arm_stiffness,
            "damping": arm_damping,
            "effort_limit_sim": arm_effort_limit,
            "armature": arm_armature,
        },
        "gripper": {
            "stiffness": gripper_stiffness,
            "damping": gripper_damping,
            "effort_limit_sim": gripper_effort_limit,
            "armature": gripper_armature,
        },
    }
    for actuator_name, actuator_values in actuator_groups.items():
        actuator_overrides = {}
        for name, value in actuator_values.items():
            if value is None:
                continue
            numeric_value = float(value)
            if not np.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(f"{actuator_name} actuator {name} must be finite and >= 0.0, got {value!r}")
            if name == "effort_limit_sim" and numeric_value == 0.0:
                raise ValueError(f"{actuator_name} actuator effort_limit_sim must be positive")
            actuator_overrides[name] = numeric_value
        if actuator_overrides:
            overrides[f"{actuator_name}_actuator_overrides"] = actuator_overrides
    return overrides


def get_exact_source_initial_qpos(source_qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(source_qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 8 or qpos.shape[0] == 0 or not np.isfinite(qpos).all():
        raise ValueError(f"source qpos must be finite (T, 8), got {qpos.shape}")
    return qpos[0].copy()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _require_positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_finite_vector(value: Any, *, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a length-{length} list")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(float(item))
    return result


def _validate_contract_sha256(contract: dict[str, Any]) -> None:
    expected = contract.get("contract_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("contract_sha256 must be a 64-character string")
    payload = copy.deepcopy(contract)
    payload.pop("contract_sha256")
    actual = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if actual != expected:
        raise ValueError(f"layout contract hash mismatch: expected {expected}, got {actual}")


def _asset_key(value: dict[str, Any], *, name: str) -> str:
    asset_type = _require_nonempty_string(value.get("asset_type"), name=f"{name}.asset_type")
    category = _require_nonempty_string(value.get("category"), name=f"{name}.category")
    category_idx = _require_nonnegative_integer(value.get("category_idx"), name=f"{name}.category_idx")
    return f"{asset_type}/{category}/{category_idx:05d}"


def _validate_generated_layout_provenance(contract: dict[str, Any], source_uri: str, expected_sha: str) -> None:
    parsed = urlsplit(source_uri)
    profile = _require_nonempty_string(contract.get("profile"), name="profile")
    layout_id = _require_nonnegative_integer(contract.get("filtered_layout_id"), name="filtered_layout_id")
    if parsed.scheme != GENERATED_LAYOUT_SCHEME or parsed.netloc != profile or parsed.path != f"/{layout_id:06d}":
        raise ValueError(f"generated source_layout URI does not match contract profile/layout: {source_uri!r}")
    target = contract.get("target")
    clutter = contract.get("clutter")
    if not isinstance(target, dict) or not isinstance(clutter, list):
        raise ValueError("generated contract requires target object and clutter list")
    clutter_asset_keys = []
    clutter_poses = []
    for index, item in enumerate(clutter):
        if not isinstance(item, dict):
            raise ValueError(f"clutter[{index}] must be an object")
        clutter_asset_keys.append(_asset_key(item, name=f"clutter[{index}]"))
        clutter_poses.append(item.get("pose_maniskill"))
    generation_payload = {
        "generator_version": _require_nonempty_string(contract.get("generator_version"), name="generator_version"),
        "generation_seed": _require_nonnegative_integer(contract.get("generation_seed"), name="generation_seed"),
        "layout_id": layout_id,
        "target_asset_key": _asset_key(target, name="target"),
        "clutter_asset_keys": clutter_asset_keys,
        "target_pose": target.get("pose_maniskill"),
        "clutter_poses": clutter_poses,
    }
    actual_sha = hashlib.sha256(_canonical_json_bytes(generation_payload)).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"generated source layout hash mismatch: expected {expected_sha}, got {actual_sha}")


def _build_contract_object_instance(value: Any, *, name: str, target: bool) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    expected_asset_type = "Rigid" if target else "Clutter"
    if value.get("asset_type") != expected_asset_type or value.get("object_type") != "Rigid":
        raise ValueError(f"{name} must describe a {expected_asset_type} asset with object_type 'Rigid'")
    expected_label = "target" if target else None
    if value.get("label") != expected_label:
        raise ValueError(f"{name}.label must be {expected_label!r}")
    category = _require_nonempty_string(value.get("category"), name=f"{name}.category")
    category_idx = _require_nonnegative_integer(value.get("category_idx"), name=f"{name}.category_idx")
    scale = _require_finite_vector(value.get("scale"), length=3, name=f"{name}.scale")
    if any(item <= 0.0 for item in scale):
        raise ValueError(f"{name}.scale must be positive")
    physics = value.get("physics")
    if not isinstance(physics, dict) or physics.get("type") != "rigid":
        raise ValueError(f"{name}.physics must be a rigid object")
    for field in ("mass", "friction"):
        if field not in physics:
            continue
        number = physics[field]
        if isinstance(number, bool) or not isinstance(number, int | float) or not math.isfinite(number):
            raise ValueError(f"{name}.physics.{field} must be finite")
        invalid = number <= 0.0 if field == "mass" else number < 0.0
        if invalid:
            raise ValueError(f"{name}.physics.{field} is outside its valid range")
    pose = value.get("pose_robodojo")
    if not isinstance(pose, dict):
        raise ValueError(f"{name}.pose_robodojo must be an object")
    position = _require_finite_vector(pose.get("position"), length=3, name=f"{name}.pose_robodojo.position")
    orientation = _require_finite_vector(
        pose.get("orientation_wxyz"), length=4, name=f"{name}.pose_robodojo.orientation_wxyz"
    )
    if not math.isclose(sum(item * item for item in orientation), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name}.pose_robodojo.orientation_wxyz must be normalized")
    instance = {
        "category_idx": category_idx,
        "physics": copy.deepcopy(physics),
        "scale": scale,
        "relative_plane": "Table",
        "default_pos": position,
        "default_ori": orientation,
    }
    if target:
        instance.update({"category": category, "label": "target", "visual": {}})
    else:
        instance.update(
            {
                "type": "cluttered",
                "clutter_idx": 0,
                "yaml_path": "Clutter/clutter.yml",
            }
        )
    return category, instance


def _build_contract_geometry(contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    geometry_specs = contract.get("static_contract", {}).get("configs", {}).get("scene", {}).get("Geometry")
    if not isinstance(geometry_specs, list) or not geometry_specs:
        raise ValueError("generated contract requires static_contract.configs.scene.Geometry")
    geometry: dict[str, list[dict[str, Any]]] = {}
    for spec_index, spec in enumerate(geometry_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"scene.Geometry[{spec_index}] must be an object")
        categories = spec.get("category")
        common = spec.get("common")
        select_mode = spec.get("select_mode")
        if (
            not isinstance(categories, list)
            or not categories
            or not isinstance(common, dict)
            or not isinstance(select_mode, dict)
        ):
            raise ValueError(f"scene.Geometry[{spec_index}] has invalid category/common/select_mode")
        if select_mode.get("mode") != "unique" or select_mode.get("nums") != 1:
            raise ValueError(f"scene.Geometry[{spec_index}] must select exactly one unique instance")
        labels = select_mode.get("label")
        if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str):
            raise ValueError(f"scene.Geometry[{spec_index}].select_mode.label must contain one string")
        qpos = _require_finite_vector(common.get("qpos"), length=4, name=f"scene.Geometry[{spec_index}].common.qpos")
        bounds = {
            axis: _require_finite_vector(common.get(axis), length=2, name=f"scene.Geometry[{spec_index}].common.{axis}")
            for axis in ("xlim", "ylim", "zlim")
        }
        for axis, values in bounds.items():
            if values[0] != values[1]:
                raise ValueError(f"scene.Geometry[{spec_index}].common.{axis} must be fixed")
        relative_plane = _require_nonempty_string(
            common.get("relative_plane"), name=f"scene.Geometry[{spec_index}].common.relative_plane"
        )
        if common.get("rotate_rand") is not False:
            raise ValueError(f"scene.Geometry[{spec_index}].common.rotate_rand must be false")
        if len(categories) != 1 or not isinstance(categories[0], dict):
            raise ValueError(f"scene.Geometry[{spec_index}] must contain exactly one category")
        category = _require_nonempty_string(
            categories[0].get("name"), name=f"scene.Geometry[{spec_index}].category.name"
        )
        indices = categories[0].get("index")
        if not isinstance(indices, list) or len(indices) != 1:
            raise ValueError(f"scene.Geometry[{spec_index}].category.index must contain one value")
        category_idx = _require_nonnegative_integer(indices[0], name=f"scene.Geometry[{spec_index}].category.index[0]")
        geometry.setdefault(category, []).append(
            {
                "category": category,
                "category_idx": category_idx,
                "group": None,
                **copy.deepcopy(bounds),
                "qpos": qpos,
                "rotate_deg": 0,
                "rotate_rand": False,
                "relative_plane": relative_plane,
                "place_tag": None,
                "margin": 0.01,
                "check_mode": "bbox",
                "need_check_stable": True,
                "label": labels[0],
                "default_pos": [bounds["xlim"][0], bounds["ylim"][0], bounds["zlim"][0]],
                "default_ori": qpos,
                "scale": [1.0, 1.0, 1.0],
                "physics": {"type": "geometry"},
                "visual": {},
            }
        )
    return geometry


def build_layout_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    fixtures = contract.get("fixtures")
    required_fixtures = {"Room", "Table", "Ground", "Background", "Light"}
    if not isinstance(fixtures, dict) or set(fixtures) != required_fixtures:
        raise ValueError(f"generated contract fixtures must be exactly {sorted(required_fixtures)}")
    for key in required_fixtures - {"Light"}:
        if not isinstance(fixtures[key], dict):
            raise ValueError(f"generated contract fixture {key} must be an object")
    if fixtures["Light"] is not None and not isinstance(fixtures["Light"], dict):
        raise ValueError("generated contract fixture Light must be null or an object")

    target_category, target_instance = _build_contract_object_instance(
        contract.get("target"), name="target", target=True
    )
    clutter = contract.get("clutter")
    if not isinstance(clutter, list) or not clutter:
        raise ValueError("generated contract clutter must be a non-empty list")
    expected_count = _require_positive_integer(contract.get("expected_clutter_count"), name="expected_clutter_count")
    source_count = _require_positive_integer(contract.get("source_clutter_count"), name="source_clutter_count")
    if (
        len(clutter) != expected_count
        or len(clutter) != source_count
        or contract.get("source_clutter_count_matches_config") is not True
    ):
        raise ValueError("generated contract clutter counts are inconsistent")
    rigid = {target_category: [target_instance]}
    clutter_categories = set()
    for index, value in enumerate(clutter):
        category, instance = _build_contract_object_instance(value, name=f"clutter[{index}]", target=False)
        if category in clutter_categories:
            raise ValueError(f"generated contract clutter category is duplicated: {category}")
        clutter_categories.add(category)
        rigid.setdefault(category, []).append(instance)
    layout = {key: copy.deepcopy(value) for key, value in fixtures.items() if value is not None}
    layout["Rigid"] = rigid
    layout["Geometry"] = _build_contract_geometry(contract)
    return layout


def gripper_action_to_normalized(action: float) -> float:
    value = float(action)
    if not np.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"gripper action must be finite and in [-1, 1], got {action}")
    return (value + 1.0) * 0.5


def gripper_action_to_joint7(action: float) -> float:
    return gripper_action_to_normalized(action) * GRIPPER_JOINT_LIMIT_M


def project_measured_joint7(value: float) -> float:
    measured = float(value)
    tolerance = GRIPPER_MEASURED_JOINT_TOLERANCE_M
    if not np.isfinite(measured) or measured < -tolerance or measured > GRIPPER_JOINT_LIMIT_M + tolerance:
        raise ValueError(
            "current_joint7 must be finite and within numerical tolerance of "
            f"[0, {GRIPPER_JOINT_LIMIT_M}], got {measured!r} "
            f"(tolerance={tolerance!r})"
        )
    return float(np.clip(measured, 0.0, GRIPPER_JOINT_LIMIT_M))


def load_source_episode(path: Path, *, candidate_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        required = {
            "action_command_applied",
            "observation.actual_qpos",
            "expert_stage",
            "metadata_json",
            "prompt",
        }
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"source trajectory is missing arrays: {missing}")
        actions = np.asarray(source["action_command_applied"], dtype=np.float32)
        source_qpos = np.asarray(source["observation.actual_qpos"], dtype=np.float32)
        stages = np.asarray(source["expert_stage"])
        metadata = json.loads(str(source["metadata_json"]))
        prompt = str(source["prompt"])
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"source actions must be finite (T, 7), got {actions.shape}")
    if source_qpos.shape != (actions.shape[0], 8) or not np.isfinite(source_qpos).all():
        raise ValueError(f"source actual qpos must be finite {(actions.shape[0], 8)}, got {source_qpos.shape}")
    if stages.shape != (actions.shape[0],):
        raise ValueError(f"source expert stages must have shape {(actions.shape[0],)}, got {stages.shape}")
    if np.any(actions[:, 6] < -1.0) or np.any(actions[:, 6] > 1.0):
        raise ValueError("source gripper commands leave [-1, 1]")
    if metadata.get("candidate_id") != candidate_id:
        raise ValueError(f"expected candidate {candidate_id!r}, got {metadata.get('candidate_id')!r}")
    layout_id = _require_nonnegative_integer(metadata.get("layout_id"), name="layout_id")
    if metadata.get("control_mode") != "robodojo_pd_joint_pos":
        raise ValueError(f"unexpected source control mode: {metadata.get('control_mode')!r}")
    if not metadata.get("robust_success_10step"):
        raise ValueError("source trajectory is not robust_success_10step")
    return {
        "actions": actions,
        "source_qpos": source_qpos,
        "stages": stages,
        "metadata": metadata,
        "layout_id": layout_id,
        "prompt": prompt,
        "sha256": sha256_file(path),
    }


def load_source_layout(
    contract_path: Path,
    *,
    assets_root: Path,
    expected_layout_id: int,
) -> tuple[dict[str, Any], Path | str, dict[str, Any]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("layout contract must be an object")
    _validate_contract_sha256(contract)
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"unsupported layout contract version: {contract.get('contract_version')!r}")
    source_layout_id = _require_nonnegative_integer(expected_layout_id, name="source layout_id")
    contract_layout_id = _require_nonnegative_integer(contract.get("filtered_layout_id"), name="filtered_layout_id")
    if contract_layout_id != source_layout_id:
        raise ValueError(f"source layout {source_layout_id} does not match contract layout {contract_layout_id}")
    source_contract = contract.get("source_layout")
    if not isinstance(source_contract, dict):
        raise ValueError("contract is missing source_layout")
    relative = source_contract.get("path")
    expected_sha = source_contract.get("sha256")
    if not isinstance(relative, str) or not relative or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("source_layout requires string path and sha256")
    parsed = urlsplit(relative)
    if parsed.scheme == GENERATED_LAYOUT_SCHEME:
        _validate_generated_layout_provenance(contract, relative, expected_sha)
        layout = build_layout_from_contract(contract)
        layout_source: Path | str = relative
    elif parsed.scheme:
        raise ValueError(f"unsupported source_layout URI scheme: {parsed.scheme!r}")
    else:
        layout_path = assets_root / relative
        actual_sha = sha256_file(layout_path)
        if actual_sha != expected_sha:
            raise ValueError(f"RoboDojo source layout hash mismatch: expected {expected_sha}, got {actual_sha}")
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        if not isinstance(layout, dict):
            raise ValueError("RoboDojo source layout must be an object")
        layout_source = layout_path
    overrides = (
        contract.get("static_contract", {}).get("configs", {}).get("environment", {}).get("layout_overrides", {})
    )
    geometry_overrides = overrides.get("Geometry", {})
    for category, category_override in geometry_overrides.items():
        instances = (layout.get("Geometry") or {}).get(category)
        if not isinstance(instances, list) or not instances:
            raise ValueError(f"layout override references missing Geometry/{category}")
        if not isinstance(category_override, dict):
            raise ValueError(f"layout override for Geometry/{category} must be an object")
        for instance in instances:
            for key, value in category_override.items():
                instance[key] = value
    return layout, layout_source, contract


def interpolate_joint_targets(
    current_arm: np.ndarray,
    target_arm: np.ndarray,
    current_joint7: float,
    target_joint7: float,
    *,
    physics_steps: int = PHYSICS_STEPS_PER_ACTION,
) -> list[tuple[np.ndarray, float]]:
    current_arm = np.asarray(current_arm, dtype=np.float64)
    target_arm = np.asarray(target_arm, dtype=np.float64)
    if current_arm.shape != (6,) or target_arm.shape != (6,):
        raise ValueError("arm targets must have shape (6,)")
    if physics_steps <= 0:
        raise ValueError("physics_steps must be positive")
    if not np.isfinite(current_arm).all() or not np.isfinite(target_arm).all():
        raise ValueError("arm targets must be finite")
    current_joint7 = project_measured_joint7(current_joint7)
    target_joint7 = float(target_joint7)
    if not np.isfinite(target_joint7) or not 0.0 <= target_joint7 <= GRIPPER_JOINT_LIMIT_M:
        raise ValueError(f"target_joint7 must be in [0, {GRIPPER_JOINT_LIMIT_M}], got {target_joint7!r}")
    ramp_steps = int(np.floor(physics_steps * 0.8))
    sequence = []
    for index in range(physics_steps):
        if index < ramp_steps:
            alpha = (index + 1) / ramp_steps
            arm = (1.0 - alpha) * current_arm + alpha * target_arm
            joint7 = (1.0 - alpha) * current_joint7 + alpha * target_joint7
        else:
            arm = target_arm.copy()
            joint7 = target_joint7
        sequence.append((arm, float(joint7)))
    return sequence
