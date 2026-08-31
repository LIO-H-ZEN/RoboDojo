import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "internal" / "general_pickup_direct_replay_common.py"
REPLAY_SCRIPT = ROOT / "scripts" / "internal" / "replay_general_pickup_actions.py"
SPEC = importlib.util.spec_from_file_location("general_pickup_direct_replay_common", SCRIPT)
common = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(common)


def test_runtime_gripper_contract_uses_scene_manager_stage():
    tree = ast.parse(REPLAY_SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "verify_gripper_contact_contract"
    ]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == "env.scene_manager.stage"


def test_video_capture_renders_without_advancing_the_app_loop():
    tree = ast.parse(REPLAY_SCRIPT.read_text(encoding="utf-8"))
    recorder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HeadVideoRecorder")
    capture = next(node for node in recorder.body if isinstance(node, ast.FunctionDef) and node.name == "capture")
    calls = [ast.unparse(node) for node in ast.walk(capture) if isinstance(node, ast.Call)]
    assert "self.env.render()" in calls
    assert "self.app.update()" not in calls


def test_video_capture_discards_fixed_render_warmup_before_recording():
    tree = ast.parse(REPLAY_SCRIPT.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "VIDEO_RENDER_WARMUP_FRAMES"
    }
    assert assignments == {"VIDEO_RENDER_WARMUP_FRAMES": 20}

    recorder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HeadVideoRecorder")
    initialize = next(
        node for node in recorder.body if isinstance(node, ast.FunctionDef) and node.name == "_initialize"
    )
    warmup_loops = [
        node
        for node in ast.walk(initialize)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(VIDEO_RENDER_WARMUP_FRAMES)"
    ]
    assert len(warmup_loops) == 1
    warmup_calls = [ast.unparse(node) for node in ast.walk(warmup_loops[0]) if isinstance(node, ast.Call)]
    assert "self.env.render()" in warmup_calls
    assert "annotator.get_data()" in warmup_calls
    assert not any(call.startswith("self.writer.append(") for call in warmup_calls)

    manifest_values = {
        key.value: value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert ast.unparse(manifest_values["video_render_warmup_frames"]) == (
        "0 if recorder is None else recorder.warmup_frames"
    )


def test_video_capture_concatenates_all_camera_panels_horizontally():
    tree = ast.parse(REPLAY_SCRIPT.read_text(encoding="utf-8"))
    recorder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HeadVideoRecorder")
    capture = next(node for node in recorder.body if isinstance(node, ast.FunctionDef) and node.name == "capture")
    calls = [ast.unparse(node) for node in ast.walk(capture) if isinstance(node, ast.Call)]
    assert "np.concatenate([panel[:, :, :3] for panel in frames], axis=1)" in calls


def test_gripper_mapping_matches_maniskill_contract():
    assert common.gripper_action_to_normalized(-1.0) == pytest.approx(0.0)
    assert common.gripper_action_to_normalized(1.0) == pytest.approx(1.0)
    assert common.gripper_action_to_joint7(-1.0) == pytest.approx(0.0)
    assert common.gripper_action_to_joint7(1.0) == pytest.approx(0.035)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        common.gripper_action_to_normalized(-1.1)


def test_interpolation_is_eight_ramp_steps_then_two_holds():
    sequence = common.interpolate_joint_targets(np.zeros(6), np.ones(6), 0.035, 0.0, physics_steps=10)
    assert len(sequence) == 10
    np.testing.assert_allclose(sequence[0][0], np.full(6, 1.0 / 8.0))
    assert sequence[0][1] == pytest.approx(0.035 * 7.0 / 8.0)
    np.testing.assert_allclose(sequence[7][0], np.ones(6))
    assert sequence[7][1] == pytest.approx(0.0)
    np.testing.assert_allclose(sequence[8][0], np.ones(6))
    np.testing.assert_allclose(sequence[9][0], np.ones(6))
    assert sequence[8][1] == pytest.approx(0.0)
    assert sequence[9][1] == pytest.approx(0.0)


def test_interpolation_projects_float32_measured_joint7_limit_roundoff():
    measured_joint7 = float(np.float32(common.GRIPPER_JOINT_LIMIT_M))
    assert measured_joint7 == 0.03500000014901161
    sequence = common.interpolate_joint_targets(np.zeros(6), np.ones(6), measured_joint7, 0.0, physics_steps=10)
    assert sequence[0][1] <= common.GRIPPER_JOINT_LIMIT_M
    assert sequence[-1][1] == pytest.approx(0.0)


def test_interpolation_does_not_project_commanded_joint7_limit_violation():
    commanded_joint7 = float(np.float32(common.GRIPPER_JOINT_LIMIT_M))
    with pytest.raises(ValueError, match=r"target_joint7.*got 0\.03500000014901161"):
        common.interpolate_joint_targets(np.zeros(6), np.ones(6), common.GRIPPER_JOINT_LIMIT_M, commanded_joint7)


def test_interpolation_fast_fails_on_physical_measured_joint7_violation():
    measured_joint7 = common.GRIPPER_JOINT_LIMIT_M + 2.0 * common.GRIPPER_MEASURED_JOINT_TOLERANCE_M
    with pytest.raises(ValueError, match=r"current_joint7.*numerical tolerance"):
        common.interpolate_joint_targets(np.zeros(6), np.ones(6), measured_joint7, common.GRIPPER_JOINT_LIMIT_M)


def test_interpolation_fast_fails_on_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        common.interpolate_joint_targets(np.zeros(7), np.zeros(6), 0.0, 0.0)


def test_resolve_replay_action_count_supports_strict_prefixes():
    assert common.resolve_replay_action_count(total_actions=134, max_actions=None) == 134
    assert common.resolve_replay_action_count(total_actions=134, max_actions=10) == 10
    assert common.resolve_replay_action_count(total_actions=134, max_actions=134) == 134
    with pytest.raises(ValueError, match="positive"):
        common.resolve_replay_action_count(total_actions=134, max_actions=0)
    with pytest.raises(ValueError, match="exceeds"):
        common.resolve_replay_action_count(total_actions=134, max_actions=135)


def _write_source_episode(path: Path, *, layout_id):
    np.savez_compressed(
        path,
        action_command_applied=np.zeros((1, 7), dtype=np.float32),
        **{
            "observation.actual_qpos": np.zeros((1, 8), dtype=np.float32),
            "expert_stage": np.asarray(["lift"]),
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "candidate_id": "candidate",
                        "control_mode": "robodojo_pd_joint_pos",
                        "layout_id": layout_id,
                        "robust_success_10step": True,
                    }
                )
            ),
            "prompt": np.asarray("Pick up the target by 10 cm."),
        },
    )


def _with_contract_sha(payload):
    result = dict(payload)
    result["contract_sha256"] = hashlib.sha256(common._canonical_json_bytes(result)).hexdigest()
    return result


def _generated_contract(*, layout_id=2007):
    target = {
        "asset_type": "Rigid",
        "object_type": "Rigid",
        "category": "owl",
        "category_idx": 0,
        "label": "target",
        "scale": [1, 1, 1],
        "physics": {"type": "rigid", "mass": 0.35, "friction": 0.55},
        "pose_maniskill": {
            "position": [0.0, -0.1, 0.054],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "pose_robodojo": {
            "position": [0.1, -0.1, 0.819],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }
    clutter = {
        "asset_type": "Clutter",
        "object_type": "Rigid",
        "category": "hammer",
        "category_idx": 2,
        "label": None,
        "scale": [1, 1, 1],
        "physics": {"type": "rigid"},
        "pose_maniskill": {
            "position": [0.2, 0.1, 0.02],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "pose_robodojo": {
            "position": [-0.1, 0.1, 0.785],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }
    generation_payload = {
        "generator_version": "robodojo_mirror_train_v2",
        "generation_seed": 123,
        "layout_id": layout_id,
        "target_asset_key": "Rigid/owl/00000",
        "clutter_asset_keys": ["Clutter/hammer/00002"],
        "target_pose": target["pose_maniskill"],
        "clutter_poses": [clutter["pose_maniskill"]],
    }
    payload = {
        "contract_version": "robodojo_general_pickup_piper_mvp_v1",
        "profile": "mirror_train",
        "generator_version": "robodojo_mirror_train_v2",
        "generation_seed": 123,
        "filtered_layout_id": layout_id,
        "source_layout": {
            "path": f"generated://mirror_train/{layout_id:06d}",
            "sha256": hashlib.sha256(common._canonical_json_bytes(generation_payload)).hexdigest(),
        },
        "expected_clutter_count": 1,
        "source_clutter_count": 1,
        "source_clutter_count_matches_config": True,
        "target": target,
        "clutter": [clutter],
        "fixtures": {
            "Room": {"default": "Simple_Room_nolight"},
            "Table": {"default": "material_0122"},
            "Ground": {"geometry": "cube"},
            "Background": {"category_name": "brown_photostudio_02_4k.hdr"},
            "Light": None,
        },
        "static_contract": {
            "configs": {
                "scene": {
                    "Geometry": [
                        {
                            "category": [{"index": [0], "name": "camera_stand"}],
                            "common": {
                                "qpos": [0.7071067811865476, -0.7071067811865476, 0, 0],
                                "relative_plane": "Ground",
                                "rotate_rand": False,
                                "xlim": [0, 0],
                                "ylim": [-0.47, -0.47],
                                "zlim": [0.765, 0.765],
                            },
                            "select_mode": {
                                "label": ["camera_stand"],
                                "mode": "unique",
                                "nums": 1,
                            },
                        }
                    ]
                },
                "environment": {
                    "layout_overrides": {"Geometry": {"camera_stand": {"default_pos": [-0.55, -0.47, 0.715]}}}
                },
            }
        },
    }
    return _with_contract_sha(payload)


@pytest.mark.parametrize("layout_id", [0, 2007])
def test_load_source_episode_returns_nonnegative_layout_id(tmp_path, layout_id):
    source_path = tmp_path / "source.npz"
    _write_source_episode(source_path, layout_id=layout_id)

    source = common.load_source_episode(source_path, candidate_id="candidate")

    assert source["layout_id"] == layout_id


@pytest.mark.parametrize("layout_id", [True, -1, "2007"])
def test_load_source_episode_rejects_invalid_layout_id(tmp_path, layout_id):
    source_path = tmp_path / "source.npz"
    _write_source_episode(source_path, layout_id=layout_id)

    with pytest.raises(ValueError, match="layout_id must be a non-negative integer"):
        common.load_source_episode(source_path, candidate_id="candidate")


def test_load_source_layout_requires_matching_source_layout_id(tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            _with_contract_sha(
                {
                    "contract_version": "robodojo_general_pickup_piper_mvp_v1",
                    "filtered_layout_id": 2,
                }
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source layout 2007.*contract layout 2"):
        common.load_source_layout(
            contract_path,
            assets_root=tmp_path,
            expected_layout_id=2007,
        )


def test_load_source_layout_builds_generated_contract_without_filesystem_fallback(tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_generated_contract()), encoding="utf-8")

    layout, source_layout, contract = common.load_source_layout(
        contract_path,
        assets_root=tmp_path / "missing-assets",
        expected_layout_id=2007,
    )

    assert source_layout == "generated://mirror_train/002007"
    assert contract["filtered_layout_id"] == 2007
    assert "Light" not in layout
    assert layout["Rigid"]["owl"] == [
        {
            "category": "owl",
            "category_idx": 0,
            "label": "target",
            "visual": {},
            "physics": {"type": "rigid", "mass": 0.35, "friction": 0.55},
            "scale": [1.0, 1.0, 1.0],
            "relative_plane": "Table",
            "default_pos": [0.1, -0.1, 0.819],
            "default_ori": [1.0, 0.0, 0.0, 0.0],
        }
    ]
    assert layout["Rigid"]["hammer"][0]["type"] == "cluttered"
    assert layout["Rigid"]["hammer"][0]["category_idx"] == 2
    assert layout["Geometry"]["camera_stand"][0]["default_pos"] == [-0.55, -0.47, 0.715]


def test_load_source_layout_accepts_generated_layout_zero(tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_generated_contract(layout_id=0)), encoding="utf-8")

    _, source_layout, contract = common.load_source_layout(
        contract_path,
        assets_root=tmp_path / "missing-assets",
        expected_layout_id=0,
    )

    assert source_layout == "generated://mirror_train/000000"
    assert contract["filtered_layout_id"] == 0


def test_load_source_layout_rejects_tampered_generated_provenance(tmp_path):
    contract = _generated_contract()
    contract["source_layout"]["sha256"] = "0" * 64
    contract = _with_contract_sha({key: value for key, value in contract.items() if key != "contract_sha256"})
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="generated source layout hash mismatch"):
        common.load_source_layout(
            contract_path,
            assets_root=tmp_path,
            expected_layout_id=2007,
        )


def test_replay_uses_source_layout_id_for_manifest_and_video_name():
    tree = ast.parse(REPLAY_SCRIPT.read_text(encoding="utf-8"))
    source_layout_references = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript) and ast.unparse(node) == "source['layout_id']"
    ]

    assert len(source_layout_references) >= 3


def test_build_robot_alignment_overrides_only_changes_requested_contracts():
    assert common.build_robot_alignment_overrides(disable_self_collisions=False) == {}
    assert common.build_robot_alignment_overrides(disable_self_collisions=True) == {"enabled_self_collisions": False}
    assert common.build_robot_alignment_overrides(disable_self_collisions=False, enable_robot_gravity=True) == {
        "disable_gravity": False
    }
    assert common.build_robot_alignment_overrides(disable_self_collisions=False, activate_contact_sensors=True) == {
        "activate_contact_sensors": True
    }
    with pytest.raises(ValueError, match="activate_contact_sensors"):
        common.build_robot_alignment_overrides(disable_self_collisions=False, activate_contact_sensors=1)


def test_build_robot_alignment_overrides_validates_actuator_parameters():
    assert common.build_robot_alignment_overrides(
        disable_self_collisions=False,
        arm_stiffness=100,
        arm_damping=5,
        arm_effort_limit=100,
        arm_armature=0,
    ) == {
        "arm_actuator_overrides": {
            "stiffness": 100.0,
            "damping": 5.0,
            "effort_limit_sim": 100.0,
            "armature": 0.0,
        }
    }
    with pytest.raises(ValueError, match="armature"):
        common.build_robot_alignment_overrides(disable_self_collisions=False, arm_armature=-0.01)
    with pytest.raises(ValueError, match="effort_limit_sim must be positive"):
        common.build_robot_alignment_overrides(disable_self_collisions=False, arm_effort_limit=0)


def test_build_robot_alignment_overrides_accepts_explicit_robot_usd():
    assert common.build_robot_alignment_overrides(
        disable_self_collisions=False, robot_usd="/tmp/piper_urdf_inertia.usd"
    ) == {"usd_path": "/tmp/piper_urdf_inertia.usd"}
    with pytest.raises(ValueError, match="non-empty"):
        common.build_robot_alignment_overrides(disable_self_collisions=False, robot_usd="")


def test_build_robot_alignment_overrides_matches_maniskill_gripper_drive():
    assert common.build_robot_alignment_overrides(
        disable_self_collisions=True,
        gripper_stiffness=500,
        gripper_damping=10,
        gripper_effort_limit=40,
        gripper_armature=0,
    ) == {
        "enabled_self_collisions": False,
        "gripper_actuator_overrides": {
            "stiffness": 500.0,
            "damping": 10.0,
            "effort_limit_sim": 40.0,
            "armature": 0.0,
        },
    }
    with pytest.raises(ValueError, match="gripper actuator effort_limit_sim"):
        common.build_robot_alignment_overrides(
            disable_self_collisions=False,
            gripper_effort_limit=0,
        )


def test_get_exact_source_initial_qpos_is_strict_and_copies():
    source = np.arange(24, dtype=np.float32).reshape(3, 8)
    initial = common.get_exact_source_initial_qpos(source)
    np.testing.assert_array_equal(initial, source[0])
    initial[0] = -1.0
    assert source[0, 0] == 0.0
    with pytest.raises(ValueError, match=r"\(T, 8\)"):
        common.get_exact_source_initial_qpos(np.zeros((8,), dtype=np.float32))


def test_classify_target_finger_contact_uses_exact_path_segments():
    target = "/World/envs/env_0/Rigid/car_00011"
    assert (
        common.classify_target_finger_contact(
            actor_paths=("/World/envs/env_0/robot0", target),
            collider_paths=("/World/envs/env_0/robot0/root_joint/link7/collisions/mesh", f"{target}/mesh"),
            target_prim_path=target,
        )
        == "left"
    )
    assert (
        common.classify_target_finger_contact(
            actor_paths=(target, "/World/envs/env_0/robot0"),
            collider_paths=(f"{target}/mesh", "/World/envs/env_0/robot0/root_joint/link8/collisions/mesh"),
            target_prim_path=target,
        )
        == "right"
    )
    assert (
        common.classify_target_finger_contact(
            actor_paths=(target, "/World/envs/env_0/Table"),
            collider_paths=(f"{target}/mesh", "/World/envs/env_0/Table/collision"),
            target_prim_path=target,
        )
        is None
    )
    assert (
        common.classify_target_finger_contact(
            actor_paths=(target + "_lookalike", "/World/envs/env_0/robot0"),
            collider_paths=(target + "_lookalike/mesh", "/World/envs/env_0/robot0/root_joint/link7_extra/mesh"),
            target_prim_path=target,
        )
        is None
    )


def test_classify_target_finger_contact_fast_fails_on_ambiguous_fingers():
    target = "/World/envs/env_0/Rigid/car_00011"
    with pytest.raises(ValueError, match="both Piper fingers"):
        common.classify_target_finger_contact(
            actor_paths=(target, "/World/envs/env_0/robot0"),
            collider_paths=(
                "/World/envs/env_0/robot0/root_joint/link7/link8/mesh",
                f"{target}/mesh",
            ),
            target_prim_path=target,
        )


def test_prepare_layout_target_contact_sensor_is_strict_and_scoped():
    layout = {
        "Rigid": {
            "car": [{"label": "target", "physics": {"mass": 0.45}}],
            "cup": [{"label": None, "physics": {"mass": 0.1}}],
        }
    }
    common.prepare_layout_target_contact_sensor(layout)
    assert layout["Rigid"]["car"][0]["physics"] == {
        "mass": 0.45,
        "prepare_contact_sensor": True,
    }
    assert layout["Rigid"]["cup"][0]["physics"] == {"mass": 0.1}

    common.prepare_layout_target_contact_sensor(layout, collision_approximation="convexHull")
    assert layout["Rigid"]["car"][0]["physics"]["collision_approximation"] == "convexHull"
    with pytest.raises(ValueError, match="unsupported target collision approximation"):
        common.prepare_layout_target_contact_sensor(layout, collision_approximation="boundingCube")

    layout["Rigid"]["car"][0]["physics"]["friction"] = 0.32
    common.prepare_layout_target_contact_sensor(layout)
    assert layout["Rigid"]["car"][0]["physics"]["static_friction"] == 0.32
    assert layout["Rigid"]["car"][0]["physics"]["dynamic_friction"] == 0.32


@pytest.mark.parametrize(
    "layout, error",
    [
        ({"Rigid": {"car": []}}, "exactly one target"),
        (
            {"Rigid": {"car": [{"label": "target"}, {"label": "target"}]}},
            "exactly one target",
        ),
        ({"Rigid": {"car": [{"label": "target", "physics": []}]}}, "physics must be an object"),
    ],
)
def test_prepare_layout_target_contact_sensor_fast_fails(layout, error):
    with pytest.raises(ValueError, match=error):
        common.prepare_layout_target_contact_sensor(layout)


def test_target_mass_contract_uses_layout_and_runtime_physics_view():
    layout = {
        "Rigid": {
            "car": [{"label": "target", "physics": {"mass": 0.45}}],
            "cup": [{"label": None, "physics": {"mass": 0.1}}],
        }
    }
    expected_mass = common.resolve_target_mass_kg(layout)
    assert expected_mass == pytest.approx(0.45)
    assert common.validate_runtime_target_mass_kg(expected_mass, np.float32(0.45)) == {
        "expected_mass_kg": 0.45,
        "runtime_mass_kg": pytest.approx(0.45),
    }


@pytest.mark.parametrize(
    "layout, error",
    [
        ({"Rigid": {"car": []}}, "exactly one target"),
        ({"Rigid": {"car": [{"label": "target"}]}}, "physics must be an object"),
        ({"Rigid": {"car": [{"label": "target", "physics": {}}]}}, "mass"),
        ({"Rigid": {"car": [{"label": "target", "physics": {"mass": True}}]}}, "mass"),
        ({"Rigid": {"car": [{"label": "target", "physics": {"mass": 0}}]}}, "positive"),
    ],
)
def test_resolve_target_mass_contract_fast_fails(layout, error):
    with pytest.raises(ValueError, match=error):
        common.resolve_target_mass_kg(layout)


@pytest.mark.parametrize("runtime_mass", [None, np.nan, 0.44])
def test_runtime_target_mass_contract_fast_fails(runtime_mass):
    with pytest.raises(ValueError, match="runtime target mass"):
        common.validate_runtime_target_mass_kg(0.45, runtime_mass)


def test_transform_points_between_usd_frames_uses_row_vector_convention():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    source = np.eye(4)
    source[3, :3] = [4.0, 5.0, 6.0]
    destination = np.eye(4)
    destination[3, :3] = [1.0, 1.0, 1.0]
    actual = common.transform_points_between_usd_frames(
        points,
        source_local_to_world=source,
        destination_local_to_world=destination,
    )
    np.testing.assert_allclose(actual, points + [3.0, 4.0, 5.0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "points,source,destination,error",
    [
        (np.zeros((2, 2)), np.eye(4), np.eye(4), "finite Nx3"),
        (np.zeros((2, 3)), np.eye(3), np.eye(4), "source_local_to_world"),
        (np.zeros((2, 3)), np.eye(4), np.zeros((4, 4)), "invertible"),
    ],
)
def test_transform_points_between_usd_frames_fast_fails(points, source, destination, error):
    with pytest.raises(ValueError, match=error):
        common.transform_points_between_usd_frames(
            points,
            source_local_to_world=source,
            destination_local_to_world=destination,
        )


def test_resolve_gripper_contact_contract_is_strict():
    assert (
        common.resolve_gripper_contact_contract(
            static_friction=None,
            dynamic_friction=None,
            torsional_patch_radius=None,
            min_torsional_patch_radius=None,
        )
        is None
    )
    assert common.resolve_gripper_contact_contract(
        static_friction=2,
        dynamic_friction=2,
        torsional_patch_radius=0.02,
        min_torsional_patch_radius=0.005,
    ) == {
        "static_friction": 2.0,
        "dynamic_friction": 2.0,
        "torsional_patch_radius": 0.02,
        "min_torsional_patch_radius": 0.005,
    }
    with pytest.raises(ValueError, match="all four"):
        common.resolve_gripper_contact_contract(
            static_friction=2,
            dynamic_friction=None,
            torsional_patch_radius=0.02,
            min_torsional_patch_radius=0.005,
        )
    with pytest.raises(ValueError, match="exceeds"):
        common.resolve_gripper_contact_contract(
            static_friction=2,
            dynamic_friction=2,
            torsional_patch_radius=0.005,
            min_torsional_patch_radius=0.02,
        )
