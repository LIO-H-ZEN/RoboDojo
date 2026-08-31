#!/usr/bin/env python3
"""Replay ManiSkill General Pickup joint actions directly in RoboDojo."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

from isaaclab.app import AppLauncher

VIDEO_RENDER_WARMUP_FRAMES = 20

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-trajectory", type=Path, required=True)
parser.add_argument("--contract", type=Path, required=True)
parser.add_argument("--assets-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--candidate-id", default="pair-0328-roll-03")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle-steps", type=int, default=200)
parser.add_argument("--camera-name", default="cam_head")
parser.add_argument(
    "--camera-names",
    nargs="+",
    choices=("cam_head", "cam_wrist", "cam_side"),
)
parser.add_argument("--max-actions", type=int)
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--disable-self-collisions", action="store_true")
parser.add_argument(
    "--target-collision-approximation",
    choices=("source", "convexHull", "convexDecomposition"),
    default="source",
)
parser.add_argument("--enable-robot-gravity", action="store_true")
parser.add_argument("--arm-stiffness", type=float)
parser.add_argument("--arm-damping", type=float)
parser.add_argument("--arm-effort-limit", type=float)
parser.add_argument("--arm-armature", type=float)
parser.add_argument("--gripper-stiffness", type=float)
parser.add_argument("--gripper-damping", type=float)
parser.add_argument("--gripper-effort-limit", type=float)
parser.add_argument("--gripper-armature", type=float)
parser.add_argument("--gripper-static-friction", type=float)
parser.add_argument("--gripper-dynamic-friction", type=float)
parser.add_argument("--gripper-torsional-patch-radius", type=float)
parser.add_argument("--gripper-min-torsional-patch-radius", type=float)
parser.add_argument("--robot-usd", type=Path)
parser.add_argument(
    "--initial-state-mode",
    choices=("settled", "source-exact"),
    default="settled",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402
import torch  # noqa: E402
import transforms3d as t3d  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from general_pickup_direct_replay_common import (  # noqa: E402
    CONTROL_FREQUENCY_HZ,
    GRIPPER_MEASURED_JOINT_TOLERANCE_M,
    PHYSICS_DT_S,
    PHYSICS_STEPS_PER_ACTION,
    SCHEMA_VERSION,
    build_robot_alignment_overrides,
    classify_target_finger_contact,
    get_exact_source_initial_qpos,
    gripper_action_to_joint7,
    interpolate_joint_targets,
    load_source_episode,
    load_source_layout,
    prepare_layout_target_contact_sensor,
    resolve_gripper_contact_contract,
    resolve_replay_action_count,
    resolve_target_mass_kg,
    sha256_file,
    transform_points_between_usd_frames,
    validate_runtime_target_mass_kg,
)

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR  # noqa: E402
from utils.load_file import load_yaml  # noqa: E402
from utils.save_file import VideoStreamWriter  # noqa: E402


def build_env_cfg():
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "piper_single.yml"))
    sim_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml"))
    if float(sim_cfg["dt"]) != PHYSICS_DT_S:
        raise ValueError(f"RoboDojo physics dt must be {PHYSICS_DT_S}, got {sim_cfg['dt']}")
    sim_cfg["scene"]["num_envs"] = 1
    sim_cfg["seed"] = [args_cli.seed]
    camera_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml"))
    camera_cfg["default_frequency"] = eval_cfg["observation"]["collect_freq"]
    if int(eval_cfg["observation"]["collect_freq"]) != CONTROL_FREQUENCY_HZ:
        raise ValueError("RoboDojo piper_single observation frequency is not 20 Hz")
    robot_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml"))
    robots = robot_cfg.get("robots")
    if not isinstance(robots, list) or len(robots) != 1 or not isinstance(robots[0], dict):
        raise ValueError("RoboDojo direct replay requires exactly one robot config")
    if args_cli.robot_usd is not None and not args_cli.robot_usd.is_file():
        raise FileNotFoundError(args_cli.robot_usd)
    alignment_overrides = build_robot_alignment_overrides(
        disable_self_collisions=args_cli.disable_self_collisions,
        activate_contact_sensors=True,
        enable_robot_gravity=args_cli.enable_robot_gravity,
        arm_stiffness=args_cli.arm_stiffness,
        arm_damping=args_cli.arm_damping,
        arm_effort_limit=args_cli.arm_effort_limit,
        arm_armature=args_cli.arm_armature,
        gripper_stiffness=args_cli.gripper_stiffness,
        gripper_damping=args_cli.gripper_damping,
        gripper_effort_limit=args_cli.gripper_effort_limit,
        gripper_armature=args_cli.gripper_armature,
        robot_usd=None if args_cli.robot_usd is None else str(args_cli.robot_usd.resolve()),
    )
    robots[0].update(alignment_overrides)
    return OmegaConf.create(
        {
            "sim": sim_cfg,
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
            "camera": camera_cfg,
            "robot": robot_cfg,
            "task_env": load_yaml(
                importlib.import_module(f"task.{BENCHMARK}.task_registry").task_config_path(
                    os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), "general_pickup_single"
                )
            ),
        }
    )


class HeadVideoRecorder:
    def __init__(self, env, app, output_path: Path, camera_names: list[str]):
        self.env = env
        self.app = app
        self.output_path = output_path
        if not camera_names or len(camera_names) != len(set(camera_names)):
            raise ValueError(f"camera names must be non-empty and unique, got {camera_names}")
        self.camera_names = camera_names
        self.annotators = []
        self.writer = None
        self.frames = 0
        self.warmup_frames = 0

    def _initialize(self):
        import omni.replicator.core as rep

        available_names = self.env.camera_manager.camera_names[0]
        missing_names = [name for name in self.camera_names if name not in available_names]
        if missing_names:
            raise ValueError(f"cameras {missing_names} not found; available cameras: {available_names}")
        for camera_name in self.camera_names:
            camera_id = available_names.index(camera_name)
            camera = self.env.camera_manager.cameras[0][camera_id]
            width, height = camera._resolution
            product = rep.create.render_product(camera.prim_path, resolution=(int(width), int(height)))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annotator.attach(product)
            self.annotators.append(annotator)
        warmup_frames = []
        for _ in range(VIDEO_RENDER_WARMUP_FRAMES):
            self.env.render()
            warmup_frames = [np.asarray(annotator.get_data()) for annotator in self.annotators]
        invalid = [
            (name, frame.shape)
            for name, frame in zip(self.camera_names, warmup_frames, strict=True)
            if frame.ndim != 3 or frame.shape[2] < 3 or frame.shape[0] == 0
        ]
        if invalid:
            raise RuntimeError(
                f"cameras did not become ready after "
                f"{VIDEO_RENDER_WARMUP_FRAMES} discarded render warmup frames; "
                f"invalid frames: {invalid}"
            )
        panel_shapes = {frame.shape[:2] for frame in warmup_frames}
        if len(panel_shapes) != 1:
            raise RuntimeError(f"camera panel resolutions differ: {panel_shapes}")
        self.warmup_frames = VIDEO_RENDER_WARMUP_FRAMES

    def capture(self):
        if not self.annotators:
            self._initialize()
        frames = []
        for _ in range(20):
            self.env.render()
            frames = [np.asarray(annotator.get_data()) for annotator in self.annotators]
            if all(frame.ndim == 3 and frame.shape[2] >= 3 and frame.shape[0] > 0 for frame in frames):
                break
        else:
            raise RuntimeError(
                "triptych cameras did not become ready after 20 render attempts; "
                f"last frame shapes were {[frame.shape for frame in frames]}"
            )
        frame = np.ascontiguousarray(np.concatenate([panel[:, :, :3] for panel in frames], axis=1))
        if self.writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = VideoStreamWriter(
                str(self.output_path), frame.shape[0], frame.shape[1], 3, fps=CONTROL_FREQUENCY_HZ
            )
        self.writer.append(frame)
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()


class TargetFingerContactTelemetry:
    def __init__(self, target):
        self.target_prim_path = str(target._prim_path)
        prim = target.stage.GetPrimAtPath(self.target_prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"target prim is invalid: {self.target_prim_path}")
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"target prim is not a rigid body: {self.target_prim_path}")
        rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Get(target.stage, prim.GetPath())
        if not rigid_body_api:
            raise RuntimeError(f"target PhysX rigid body API is unavailable: {self.target_prim_path}")
        rigid_body_api.CreateSleepThresholdAttr().Set(0.0)
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        if not report_api:
            raise RuntimeError(f"failed to apply contact report API to {self.target_prim_path}")
        report_api.CreateThresholdAttr().Set(0.0)
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact_report_event
        )
        if self._subscription is None:
            raise RuntimeError("failed to subscribe to PhysX contact reports")
        self._lifetime_pairs = {}
        self.reset()

    @staticmethod
    def _decode_path(encoded) -> str:
        path = str(PhysicsSchemaTools.intToSdfPath(encoded))
        if not path.startswith("/"):
            raise ValueError(f"PhysX contact report returned a non-absolute prim path: {path!r}")
        return path

    def _on_contact_report_event(self, contact_headers, contact_data) -> None:
        try:
            for header in contact_headers:
                actor_paths = (self._decode_path(header.actor0), self._decode_path(header.actor1))
                collider_paths = (self._decode_path(header.collider0), self._decode_path(header.collider1))
                paths = (*actor_paths, *collider_paths)
                if not any(
                    path == self.target_prim_path or path.startswith(f"{self.target_prim_path}/") for path in paths
                ):
                    continue
                offset = int(header.contact_data_offset)
                count = int(header.num_contact_data)
                if offset < 0 or count < 0 or offset + count > len(contact_data):
                    raise ValueError(
                        f"invalid PhysX contact data range offset={offset}, count={count}, total={len(contact_data)}"
                    )
                impulse_ns = 0.0
                for index in range(offset, offset + count):
                    impulse = np.asarray(contact_data[index].impulse, dtype=np.float64)
                    if impulse.shape != (3,) or not np.isfinite(impulse).all():
                        raise ValueError(f"invalid PhysX contact impulse: {impulse!r}")
                    impulse_ns += float(np.linalg.norm(impulse))
                pair_key = " :: ".join(collider_paths)
                pair = self._lifetime_pairs.setdefault(
                    pair_key,
                    {"collider_paths": list(collider_paths), "headers": 0, "contact_points": 0, "impulse_ns": 0.0},
                )
                pair["headers"] += 1
                pair["contact_points"] += count
                pair["impulse_ns"] += impulse_ns
                self._target_impulse_ns += impulse_ns
                self._target_contact_points += count
                self._target_contact_headers += 1
                side = classify_target_finger_contact(
                    actor_paths=actor_paths,
                    collider_paths=collider_paths,
                    target_prim_path=self.target_prim_path,
                )
                if side is None:
                    continue
                self._impulse_ns[side] += impulse_ns
                self._contact_points[side] += count
                self._contact_headers[side] += 1
        except BaseException as error:
            self._callback_error = error

    def reset(self) -> None:
        self._impulse_ns = {"left": 0.0, "right": 0.0}
        self._contact_points = {"left": 0, "right": 0}
        self._contact_headers = {"left": 0, "right": 0}
        self._target_impulse_ns = 0.0
        self._target_contact_points = 0
        self._target_contact_headers = 0
        self._callback_error = None

    def snapshot_and_reset(self) -> dict:
        if self._callback_error is not None:
            raise RuntimeError("PhysX contact telemetry callback failed") from self._callback_error
        control_dt_s = 1.0 / CONTROL_FREQUENCY_HZ
        sample = {
            "left_contact_impulse_ns": np.float32(self._impulse_ns["left"]),
            "right_contact_impulse_ns": np.float32(self._impulse_ns["right"]),
            "left_contact_force_n": np.float32(self._impulse_ns["left"] / control_dt_s),
            "right_contact_force_n": np.float32(self._impulse_ns["right"] / control_dt_s),
            "left_contact_points": np.int32(self._contact_points["left"]),
            "right_contact_points": np.int32(self._contact_points["right"]),
            "left_contact_headers": np.int32(self._contact_headers["left"]),
            "right_contact_headers": np.int32(self._contact_headers["right"]),
            "target_contact_impulse_ns": np.float32(self._target_impulse_ns),
            "target_contact_force_n": np.float32(self._target_impulse_ns / control_dt_s),
            "target_contact_points": np.int32(self._target_contact_points),
            "target_contact_headers": np.int32(self._target_contact_headers),
        }
        sample["is_grasped"] = bool(sample["left_contact_points"] > 0 and sample["right_contact_points"] > 0)
        self.reset()
        return sample

    def lifetime_pairs(self) -> list[dict]:
        if self._callback_error is not None:
            raise RuntimeError("PhysX contact telemetry callback failed") from self._callback_error
        return [self._lifetime_pairs[key].copy() for key in sorted(self._lifetime_pairs)]

    def close(self) -> None:
        self._subscription = None


def collision_prim_audit(stage, *, target_prim_path: str) -> dict:
    result = {
        "target": [],
        "left_finger": [],
        "right_finger": [],
        "robot": [],
        "collision_filter_prims": [],
        "physics_scenes": [],
        "contact_report_prims": [],
        "rigid_bodies": [],
    }
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
    )
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path in {
            target_prim_path,
            "/World/envs/env_0/robot0/root_joint/link7",
            "/World/envs/env_0/robot0/root_joint/link8",
        }:
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
            mass_api = UsdPhysics.MassAPI(prim)
            result["rigid_bodies"].append(
                {
                    "path": path,
                    "rigid_body_enabled": rigid_body_api.GetRigidBodyEnabledAttr().Get(),
                    "kinematic_enabled": rigid_body_api.GetKinematicEnabledAttr().Get(),
                    "mass_kg": mass_api.GetMassAttr().Get() if mass_api else None,
                }
            )
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            physics_scene_api = PhysxSchema.PhysxSceneAPI(prim)
            result["physics_scenes"].append(
                {
                    "path": path,
                    "invert_collision_group_filter": physics_scene_api.GetInvertCollisionGroupFilterAttr().Get(),
                }
            )
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            contact_report_api = PhysxSchema.PhysxContactReportAPI(prim)
            result["contact_report_prims"].append(
                {
                    "path": path,
                    "threshold_n": contact_report_api.GetThresholdAttr().Get(),
                }
            )
        is_collision_group = prim.GetTypeName() == "PhysicsCollisionGroup"
        filter_relationships = {
            relationship.GetName(): [str(target) for target in relationship.GetTargets()]
            for relationship in prim.GetRelationships()
            if is_collision_group
            or "filter" in relationship.GetName().lower()
            or "collision" in relationship.GetName().lower()
        }
        if is_collision_group or filter_relationships:
            result["collision_filter_prims"].append(
                {
                    "path": path,
                    "type": prim.GetTypeName(),
                    "attributes": {
                        attribute.GetName(): str(attribute.Get())
                        for attribute in prim.GetAttributes()
                        if is_collision_group
                    },
                    "relationships": filter_relationships,
                }
            )
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        segments = {segment for segment in path.split("/") if segment}
        groups = []
        if path == target_prim_path or path.startswith(f"{target_prim_path}/"):
            groups.append("target")
        if "/robot0/" in path:
            groups.append("robot")
        if "link7" in segments:
            groups.append("left_finger")
        if "link8" in segments:
            groups.append("right_finger")
        if not groups:
            continue
        collision_api = UsdPhysics.CollisionAPI(prim)
        approximation = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        row = {
            "path": path,
            "type": prim.GetTypeName(),
            "collision_enabled": collision_api.GetCollisionEnabledAttr().Get() is not False,
            "approximation": approximation,
        }
        if any(group in groups for group in ("target", "left_finger", "right_finger")):
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            row["contact_offset_m"] = physx_collision_api.GetContactOffsetAttr().Get() if physx_collision_api else None
            row["rest_offset_m"] = physx_collision_api.GetRestOffsetAttr().Get() if physx_collision_api else None
            row["torsional_patch_radius_m"] = (
                physx_collision_api.GetTorsionalPatchRadiusAttr().Get() if physx_collision_api else None
            )
            row["min_torsional_patch_radius_m"] = (
                physx_collision_api.GetMinTorsionalPatchRadiusAttr().Get() if physx_collision_api else None
            )
            material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if material:
                material_prim = material.GetPrim()
                material_api = UsdPhysics.MaterialAPI(material_prim)
                physx_material_api = PhysxSchema.PhysxMaterialAPI(material_prim)
                row["physics_material"] = {
                    "path": str(material_prim.GetPath()),
                    "binding_relationship": str(relationship.GetPath()),
                    "static_friction": material_api.GetStaticFrictionAttr().Get(),
                    "dynamic_friction": material_api.GetDynamicFrictionAttr().Get(),
                    "restitution": material_api.GetRestitutionAttr().Get(),
                    "friction_combine_mode": (
                        physx_material_api.GetFrictionCombineModeAttr().Get() if physx_material_api else None
                    ),
                }
            else:
                row["physics_material"] = None
            local_range = bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
            if local_range.IsEmpty():
                raise RuntimeError(f"collision prim has an empty local bound: {path}")
            row["local_bound_min"] = [float(value) for value in local_range.GetMin()]
            row["local_bound_max"] = [float(value) for value in local_range.GetMax()]
        for group in groups:
            result[group].append(row.copy())
    for rows in result.values():
        rows.sort(key=lambda row: row["path"])
    return result


def collision_geometry_snapshot(stage, *, target_prim_path: str) -> tuple[dict[str, np.ndarray], dict]:
    body_paths = {
        "target": target_prim_path,
        "left_finger": "/World/envs/env_0/robot0/root_joint/link7",
        "right_finger": "/World/envs/env_0/robot0/root_joint/link8",
    }
    arrays = {}
    metadata = {"coordinate_frame": "rigid_body_local", "bodies": {}}
    for label, body_path in body_paths.items():
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim or not body_prim.IsValid():
            raise RuntimeError(f"collision geometry rigid body is invalid: {body_path}")
        if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"collision geometry prim is not a rigid body: {body_path}")
        body_xform = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        body_matrix = np.asarray(body_xform, dtype=np.float64)
        point_arrays = []
        point_prim_rows = []
        point_count = 0
        for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.PointBased):
                continue
            collision_ancestor = prim
            while collision_ancestor and collision_ancestor != body_prim:
                if collision_ancestor.HasAPI(UsdPhysics.CollisionAPI):
                    break
                collision_ancestor = collision_ancestor.GetParent()
            if not collision_ancestor or not collision_ancestor.HasAPI(UsdPhysics.CollisionAPI):
                continue
            points = np.asarray(UsdGeom.PointBased(prim).GetPointsAttr().Get(), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
                raise RuntimeError(f"collision PointBased prim has invalid points: {prim.GetPath()}")
            point_xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transformed = transform_points_between_usd_frames(
                points,
                source_local_to_world=np.asarray(point_xform, dtype=np.float64),
                destination_local_to_world=body_matrix,
            )
            expected = body_xform.GetInverse().Transform(point_xform.Transform(Gf.Vec3d(*points[0])))
            if not np.allclose(transformed[0], np.asarray(expected), rtol=0.0, atol=1e-9):
                raise RuntimeError(f"NumPy/Gf USD transform disagreement at {prim.GetPath()}")
            point_arrays.append(transformed)
            point_prim_rows.append(
                {
                    "path": str(prim.GetPath()),
                    "collision_ancestor": str(collision_ancestor.GetPath()),
                    "start": point_count,
                    "stop": point_count + len(transformed),
                }
            )
            point_count += len(transformed)
        if not point_arrays:
            raise RuntimeError(f"no collision PointBased geometry found under {body_path}")
        body_points = np.concatenate(point_arrays, axis=0)
        arrays[f"{label}_points_body"] = body_points
        arrays[f"{label}_body_local_to_world"] = body_matrix
        metadata["bodies"][label] = {
            "body_path": body_path,
            "point_count": len(body_points),
            "point_prims": point_prim_rows,
        }
    return arrays, metadata


def verify_gripper_contact_contract(stage, contract: dict[str, float] | None) -> None:
    if contract is None:
        return
    for link_name in ("link7", "link8"):
        link_path = f"/World/envs/env_0/robot0/root_joint/{link_name}"
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim or not link_prim.IsValid():
            raise RuntimeError(f"gripper contact link is invalid: {link_path}")
        collision_prims = [
            prim
            for prim in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(collision_prims) != 1:
            raise RuntimeError(f"expected exactly one collider under {link_path}, got {len(collision_prims)}")
        collision_prim = collision_prims[0]
        physx_collision_api = PhysxSchema.PhysxCollisionAPI(collision_prim)
        if not physx_collision_api:
            raise RuntimeError(f"gripper collider has no PhysxCollisionAPI: {collision_prim.GetPath()}")
        material, _ = UsdShade.MaterialBindingAPI(collision_prim).ComputeBoundMaterial("physics")
        if not material:
            raise RuntimeError(f"gripper collider has no physics material: {collision_prim.GetPath()}")
        material_api = UsdPhysics.MaterialAPI(material.GetPrim())
        actual = {
            "static_friction": material_api.GetStaticFrictionAttr().Get(),
            "dynamic_friction": material_api.GetDynamicFrictionAttr().Get(),
            "torsional_patch_radius": physx_collision_api.GetTorsionalPatchRadiusAttr().Get(),
            "min_torsional_patch_radius": physx_collision_api.GetMinTorsionalPatchRadiusAttr().Get(),
        }
        for key, expected in contract.items():
            if actual[key] is None or not np.isclose(float(actual[key]), expected, rtol=0.0, atol=1e-9):
                raise RuntimeError(
                    f"gripper contact {link_name} {key} mismatch: expected {expected}, got {actual[key]}"
                )


def write_npz_atomic(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sample_state(env, robot, *, initial_target_z: float, success_streak: int, contact_sample: dict) -> dict:
    arm = np.asarray(env.robot_manager.get_joint(robot, env_idx_list=[0])[0], dtype=np.float32)
    gripper = np.asarray(env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0], dtype=np.float32)
    if arm.shape != (6,) or gripper.shape != (2,):
        raise ValueError(f"unexpected RoboDojo qpos shapes: arm={arm.shape}, gripper={gripper.shape}")
    qpos = np.concatenate([arm, gripper]).astype(np.float32)
    ee_pose = np.asarray(
        env.robot_manager.get_real_endpose(robot, env_idx_list=[0], is_relative=True)[0], dtype=np.float32
    )
    rotation = t3d.quaternions.quat2mat(ee_pose[3:7])
    tcp_pose = ee_pose.copy()
    tcp_pose[:3] += rotation[:, 2] * float(robot.gripper_bias)
    left_finger_pose = np.asarray(
        env.robot_manager.get_link_pose(robot, "link7", env_idx_list=[0], is_relative=True)[0],
        dtype=np.float32,
    )
    right_finger_pose = np.asarray(
        env.robot_manager.get_link_pose(robot, "link8", env_idx_list=[0], is_relative=True)[0],
        dtype=np.float32,
    )
    if left_finger_pose.shape != (7,) or right_finger_pose.shape != (7,):
        raise ValueError(
            f"unexpected RoboDojo finger pose shapes: left={left_finger_pose.shape}, right={right_finger_pose.shape}"
        )
    target_position, target_quaternion = env.scene_manager.layout_manager.get_instance_pose(
        env_idx=0, label="target", relative=True
    )
    if target_position is None or target_quaternion is None:
        raise RuntimeError("RoboDojo target pose is unavailable")
    target_pose = np.concatenate(
        [np.asarray(target_position, dtype=np.float32), np.asarray(target_quaternion, dtype=np.float32)]
    )
    lift_height = float(target_pose[2] - initial_target_z)
    return {
        "qpos": qpos,
        "tcp_pose": tcp_pose,
        "left_finger_pose": left_finger_pose,
        "right_finger_pose": right_finger_pose,
        "target_pose": target_pose,
        "target_lift_height_m": np.float32(lift_height),
        "gripper_opening_m": np.float32(np.abs(gripper).sum()),
        **contact_sample,
        "legacy_success_3step": success_streak >= 3,
        "robust_success_10step": success_streak >= 10,
    }


def execute_source_action(env, robot, action: np.ndarray) -> None:
    arm_key = env.robot_manager.process_name(robot.arm_name)
    gripper_key = env.robot_manager.process_name(robot.gripper_name)
    current_arm = np.asarray(env.robot_manager.get_joint(robot, env_idx_list=[0])[0], dtype=np.float64)
    current_joint7 = float(env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0][0])
    target_joint7 = gripper_action_to_joint7(float(action[6]))
    sequence = []
    for arm_target, joint7_target in interpolate_joint_targets(
        current_arm,
        action[:6],
        current_joint7,
        target_joint7,
        physics_steps=PHYSICS_STEPS_PER_ACTION,
    ):
        sequence.append(
            {
                arm_key: {"position": arm_target.tolist()},
                gripper_key: {"position": [joint7_target, -joint7_target]},
            }
        )
    manager = env.robot_manager.control_manager
    manager.push([0], [sequence])
    physics_steps = 0
    while not manager.get_empty([0]):
        env.step(manager.pop(env_idx_list=[0]))
        env.sim_step(render=False)
        physics_steps += 1
    if physics_steps != PHYSICS_STEPS_PER_ACTION:
        raise RuntimeError(f"expected {PHYSICS_STEPS_PER_ACTION} physics steps, executed {physics_steps}")


def write_exact_initial_robot_state(env, robot, source_qpos: np.ndarray) -> None:
    qpos = get_exact_source_initial_qpos(source_qpos)
    articulation = env.robot_manager.robot_key[env.robot_manager.robot_list.index(robot)]
    if articulation.data.joint_pos.shape != (1, 8):
        raise ValueError(f"exact source state requires one 8-DoF robot, got {articulation.data.joint_pos.shape}")
    device = articulation.data.joint_pos.device
    position = torch.as_tensor(qpos, dtype=torch.float32, device=device).unsqueeze(0)
    velocity = torch.zeros_like(position)
    env_ids = torch.tensor([0], dtype=torch.int32, device=device)
    articulation.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
    articulation.set_joint_position_target(position, env_ids=env_ids)
    articulation.set_joint_velocity_target(velocity, env_ids=env_ids)


def main() -> int:
    if args_cli.settle_steps < 0:
        raise ValueError("settle steps must be non-negative")
    source = load_source_episode(args_cli.source_trajectory, candidate_id=args_cli.candidate_id)
    action_count = resolve_replay_action_count(
        total_actions=source["actions"].shape[0], max_actions=args_cli.max_actions
    )
    layout, layout_path, contract = load_source_layout(
        args_cli.contract,
        assets_root=args_cli.assets_root,
        expected_layout_id=source["layout_id"],
    )
    expected_target_mass_kg = resolve_target_mass_kg(layout)
    prepare_layout_target_contact_sensor(
        layout,
        collision_approximation=(
            None if args_cli.target_collision_approximation == "source" else args_cli.target_collision_approximation
        ),
    )
    gripper_contact_contract = resolve_gripper_contact_contract(
        static_friction=args_cli.gripper_static_friction,
        dynamic_friction=args_cli.gripper_dynamic_friction,
        torsional_patch_radius=args_cli.gripper_torsional_patch_radius,
        min_torsional_patch_radius=args_cli.gripper_min_torsional_patch_radius,
    )
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = build_env_cfg()
    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    _, task_class = task_registry.load_task_class("general_pickup_single")
    env = task_class(env_cfg, simulation_app)
    camera_names = args_cli.camera_names or [args_cli.camera_name]
    video_view = "triptych" if len(camera_names) == 3 else camera_names[0]
    video_path = args_cli.output_dir / f"robodojo_layout{source['layout_id']}_{args_cli.candidate_id}_{video_view}.mp4"
    recorder = None if args_cli.no_video else HeadVideoRecorder(env, simulation_app, video_path, camera_names)
    contact_telemetry = None
    states = []
    collision_audit = None
    collision_geometry_arrays = None
    collision_geometry_metadata = None
    target_mass_contract = None
    success_streak = 0
    try:
        env.scene_manager.layout_manager.set_saved_layout(0, layout)
        env.reset(seed=[args_cli.seed])
        verify_gripper_contact_contract(env.scene_manager.stage, gripper_contact_contract)
        env.scene_manager.apply_saved_poses(env_idx_list=[0])
        for _ in range(args_cli.settle_steps):
            env.sim_step(render=False)
        robot = next(robot for robot in env.robot_manager.robot_list if robot.type == "target")
        target_name = env.scene_manager.layout_manager.get_instance_name(env_idx=0, label="target")
        if target_name is None:
            raise RuntimeError("target instance name is unavailable")
        target = env.scene_manager.layout_manager.get_scene_object(env_idx=0, inst_name=target_name)
        if target is None:
            raise RuntimeError(f"target scene object is unavailable: {target_name}")
        target_mass_contract = validate_runtime_target_mass_kg(expected_target_mass_kg, float(target.get_mass()))
        contact_telemetry = TargetFingerContactTelemetry(target)
        collision_audit = collision_prim_audit(target.stage, target_prim_path=contact_telemetry.target_prim_path)
        collision_geometry_arrays, collision_geometry_metadata = collision_geometry_snapshot(
            target.stage, target_prim_path=contact_telemetry.target_prim_path
        )
        if args_cli.initial_state_mode == "source-exact":
            write_exact_initial_robot_state(env, robot, source["source_qpos"])
        env.robot_manager.set_origin_endpose()
        env.robot_manager.set_robot_init_state()
        initial_target_position, _ = env.scene_manager.layout_manager.get_instance_pose(
            env_idx=0, label="target", relative=True
        )
        if initial_target_position is None:
            raise RuntimeError("target position is unavailable after settling")
        initial_target_z = float(initial_target_position[2])
        states.append(
            sample_state(
                env,
                robot,
                initial_target_z=initial_target_z,
                success_streak=success_streak,
                contact_sample=contact_telemetry.snapshot_and_reset(),
            )
        )
        if recorder is not None:
            recorder.capture()
        for action in source["actions"][:action_count]:
            execute_source_action(env, robot, action)
            state = sample_state(
                env,
                robot,
                initial_target_z=initial_target_z,
                success_streak=success_streak,
                contact_sample=contact_telemetry.snapshot_and_reset(),
            )
            if float(state["target_lift_height_m"]) > 0.1:
                success_streak += 1
            else:
                success_streak = 0
            state["legacy_success_3step"] = success_streak >= 3
            state["robust_success_10step"] = success_streak >= 10
            states.append(state)
            if recorder is not None:
                recorder.capture()
    finally:
        if recorder is not None:
            recorder.close()
        if contact_telemetry is not None:
            contact_telemetry.close()
        env.close()

    if collision_geometry_arrays is None or collision_geometry_metadata is None:
        raise RuntimeError("collision geometry snapshot was not captured")
    if target_mass_contract is None:
        raise RuntimeError("runtime target mass contract was not captured")
    collision_geometry_path = args_cli.output_dir / "robodojo_collision_geometry.npz"
    write_npz_atomic(
        collision_geometry_path,
        {
            **collision_geometry_arrays,
            "metadata_json": np.asarray(json.dumps(collision_geometry_metadata, sort_keys=True)),
        },
    )
    collision_geometry_sha256 = sha256_file(collision_geometry_path)
    actions = source["actions"][:action_count]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "simulator": "RoboDojo",
        "layout_id": source["layout_id"],
        "candidate_id": args_cli.candidate_id,
        "prompt": source["prompt"],
        "source_trajectory": str(args_cli.source_trajectory),
        "source_trajectory_sha256": source["sha256"],
        "layout_contract": str(args_cli.contract),
        "layout_contract_sha256": contract["contract_sha256"],
        "source_layout": str(layout_path),
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "physics_frequency_hz": int(round(1.0 / PHYSICS_DT_S)),
        "physics_steps_per_action": PHYSICS_STEPS_PER_ACTION,
        "interpolation_ramp_steps": int(np.floor(PHYSICS_STEPS_PER_ACTION * 0.8)),
        "arm_command_semantics": "absolute_joint_position_radians",
        "source_gripper_semantics": "normalized_minus1_closed_plus1_open",
        "physical_gripper_semantics": "joint7_meters_0_closed_0.035_open",
        "measured_joint7_projection_tolerance_m": GRIPPER_MEASURED_JOINT_TOLERANCE_M,
        "contact_measurement_available": True,
        "contact_measurement": {
            "method": "PhysxContactReportAPI",
            "schema_prepared_before_environment_reset": True,
            "threshold_n": 0.0,
            "force_semantics": "sum_contact_impulse_norm_over_50ms_control_window_divided_by_0.05s",
            "target_prim_path": contact_telemetry.target_prim_path,
            "left_finger_link": "link7",
            "right_finger_link": "link8",
            "lifetime_pairs": contact_telemetry.lifetime_pairs(),
        },
        "collision_prim_audit": collision_audit,
        "collision_geometry_snapshot": {
            "path": str(collision_geometry_path),
            "sha256": collision_geometry_sha256,
            **collision_geometry_metadata,
        },
        "steps": action_count,
        "source_steps": int(source["actions"].shape[0]),
        "settle_steps": args_cli.settle_steps,
        "initial_state_mode": args_cli.initial_state_mode,
        "target_collision_approximation": args_cli.target_collision_approximation,
        "target_mass_contract": target_mass_contract,
        "gripper_contact_contract": gripper_contact_contract,
        "video_enabled": not args_cli.no_video,
        "video_camera_names": [] if recorder is None else recorder.camera_names,
        "video_render_warmup_frames": 0 if recorder is None else recorder.warmup_frames,
        "video_frames": 0 if recorder is None else recorder.frames,
        "robot_alignment_overrides": build_robot_alignment_overrides(
            disable_self_collisions=args_cli.disable_self_collisions,
            activate_contact_sensors=True,
            enable_robot_gravity=args_cli.enable_robot_gravity,
            arm_stiffness=args_cli.arm_stiffness,
            arm_damping=args_cli.arm_damping,
            arm_effort_limit=args_cli.arm_effort_limit,
            arm_armature=args_cli.arm_armature,
            gripper_stiffness=args_cli.gripper_stiffness,
            gripper_damping=args_cli.gripper_damping,
            gripper_effort_limit=args_cli.gripper_effort_limit,
            gripper_armature=args_cli.gripper_armature,
            robot_usd=None if args_cli.robot_usd is None else str(args_cli.robot_usd.resolve()),
        ),
        "robot_usd_sha256": (None if args_cli.robot_usd is None else sha256_file(args_cli.robot_usd)),
        "final_robust_success_10step": bool(states[-1]["robust_success_10step"]),
        "max_target_lift_height_m": max(float(state["target_lift_height_m"]) for state in states),
    }
    arrays = {
        "state_time_s": np.arange(len(states), dtype=np.float64) / CONTROL_FREQUENCY_HZ,
        "action_time_s": np.arange(len(actions), dtype=np.float64) / CONTROL_FREQUENCY_HZ,
        "source_action": actions,
        "source_actual_qpos_pre_action": source["source_qpos"][:action_count],
        "source_expert_stage": source["stages"][:action_count],
        "commanded_arm_qpos": actions[:, :6],
        "commanded_gripper_normalized": actions[:, 6],
        "commanded_gripper_joint7_m": np.asarray(
            [gripper_action_to_joint7(value) for value in actions[:, 6]], dtype=np.float32
        ),
        "qpos": np.stack([state["qpos"] for state in states]),
        "tcp_pose_wxyz": np.stack([state["tcp_pose"] for state in states]),
        "left_finger_pose_wxyz": np.stack([state["left_finger_pose"] for state in states]),
        "right_finger_pose_wxyz": np.stack([state["right_finger_pose"] for state in states]),
        "target_pose_wxyz": np.stack([state["target_pose"] for state in states]),
        "target_lift_height_m": np.asarray([state["target_lift_height_m"] for state in states], dtype=np.float32),
        "gripper_opening_m": np.asarray([state["gripper_opening_m"] for state in states], dtype=np.float32),
        "left_contact_force_n": np.asarray([state["left_contact_force_n"] for state in states], dtype=np.float32),
        "right_contact_force_n": np.asarray([state["right_contact_force_n"] for state in states], dtype=np.float32),
        "left_contact_impulse_ns": np.asarray([state["left_contact_impulse_ns"] for state in states], dtype=np.float32),
        "right_contact_impulse_ns": np.asarray(
            [state["right_contact_impulse_ns"] for state in states], dtype=np.float32
        ),
        "left_contact_points": np.asarray([state["left_contact_points"] for state in states], dtype=np.int32),
        "right_contact_points": np.asarray([state["right_contact_points"] for state in states], dtype=np.int32),
        "left_contact_headers": np.asarray([state["left_contact_headers"] for state in states], dtype=np.int32),
        "right_contact_headers": np.asarray([state["right_contact_headers"] for state in states], dtype=np.int32),
        "target_contact_force_n": np.asarray([state["target_contact_force_n"] for state in states], dtype=np.float32),
        "target_contact_impulse_ns": np.asarray(
            [state["target_contact_impulse_ns"] for state in states], dtype=np.float32
        ),
        "target_contact_points": np.asarray([state["target_contact_points"] for state in states], dtype=np.int32),
        "target_contact_headers": np.asarray([state["target_contact_headers"] for state in states], dtype=np.int32),
        "is_grasped": np.asarray([state["is_grasped"] for state in states], dtype=np.bool_),
        "legacy_success_3step": np.asarray([state["legacy_success_3step"] for state in states], dtype=np.bool_),
        "robust_success_10step": np.asarray([state["robust_success_10step"] for state in states], dtype=np.bool_),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    telemetry_path = args_cli.output_dir / "robodojo_replay_telemetry.npz"
    write_npz_atomic(telemetry_path, arrays)
    (args_cli.output_dir / "robodojo_replay_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                **metadata,
                "telemetry": str(telemetry_path),
                "video": None if recorder is None else str(video_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
