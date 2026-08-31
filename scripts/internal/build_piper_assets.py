#!/usr/bin/env python3
"""Bootstrap Assets/Robots/piper from a local piper_ros piper_description.

Does three things:
  1. Copies piper_description.urdf (+ meshes) into Assets/Robots/piper/,
     rewriting package:// mesh paths to relative ones.
  2. Writes the embodiment config (robot_config.yml) and a minimal curobo.yml
     from the parsed URDF (joint names, gripper limits/mimic).
  3. Optionally (--convert) converts the URDF to piper.usd with Isaac Sim's
     URDF importer - run this on the server inside the robodojo env.

Usage:
    # local laptop: files + configs only (no Isaac needed)
    python scripts/internal/build_piper_assets.py \
        --piper-src /path/to/piper_ros-noetic/src/piper_description

    # server: also convert to USD
    python scripts/internal/build_piper_assets.py \
        --piper-src /path/to/piper_description --convert --headless
"""

import argparse
import os
import re
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--piper-src", type=str, required=True, help="path to a piper_description checkout (contains urdf/ and meshes/)"
)
parser.add_argument("--assets-root", type=str, default=None, help="RoboDojo Assets dir (default: <repo>/Assets)")
parser.add_argument("--convert", action="store_true", help="also convert the URDF to piper.usd (requires Isaac Sim)")
parser.add_argument("--gripper-static-friction", type=float)
parser.add_argument("--gripper-dynamic-friction", type=float)
parser.add_argument("--gripper-torsional-patch-radius", type=float)
parser.add_argument("--gripper-min-torsional-patch-radius", type=float)
# Extra flags (e.g. --headless) pass through to AppLauncher in the convert
# stage; main() uses parse_known_args so they are not rejected here.

CONVERT_ARGS = []


def gripper_contact_contract(args):
    values = {
        "static_friction": args.gripper_static_friction,
        "dynamic_friction": args.gripper_dynamic_friction,
        "torsional_patch_radius": args.gripper_torsional_patch_radius,
        "min_torsional_patch_radius": args.gripper_min_torsional_patch_radius,
    }
    if all(value is None for value in values.values()):
        return None
    if any(value is None for value in values.values()):
        raise ValueError("all four gripper contact parameters are required")
    for key, value in values.items():
        if value < 0:
            raise ValueError(f"{key} must be non-negative")
    if values["min_torsional_patch_radius"] > values["torsional_patch_radius"]:
        raise ValueError("minimum torsional patch radius exceeds patch radius")
    return values


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def target_dir(assets_root):
    root = assets_root or os.path.join(repo_root(), "Assets")
    return os.path.join(os.path.abspath(root), "Robots", "piper")


def parse_urdf(urdf_path):
    src = open(urdf_path, encoding="utf-8").read()
    joints = []
    for m in re.finditer(r"<joint\s*\n?\s*name=\"([^\"]+)\"\s*\n?\s*type=\"([^\"]+)\"(.*?)</joint>", src, re.S):
        name, jtype, body = m.group(1), m.group(2), m.group(3)
        parent = re.search(r"<parent\s+link=\"([^\"]+)\"", body)
        child = re.search(r"<child\s+link=\"([^\"]+)\"", body)
        limit = re.search(r"<limit\s+lower=\"([^\"]*)\"\s+upper=\"([^\"]*)\"", body)
        joints.append(
            {
                "name": name,
                "type": jtype,
                "parent": parent.group(1) if parent else None,
                "child": child.group(1) if child else None,
                "lower": float(limit.group(1)) if limit else None,
                "upper": float(limit.group(2)) if limit else None,
            }
        )
    return joints


def copy_assets(piper_src, target):
    nested_urdf = os.path.join(piper_src, "urdf", "piper_description.urdf")
    flat_urdf = os.path.join(piper_src, "piper_description.urdf")
    urdf_src = nested_urdf if os.path.isfile(nested_urdf) else flat_urdf
    if not os.path.isfile(urdf_src):
        raise FileNotFoundError(f"piper_description.urdf not found in either {nested_urdf} or {flat_urdf}")
    os.makedirs(os.path.join(target, "meshes"), exist_ok=True)

    src = open(urdf_src, encoding="utf-8").read()
    src = src.replace("package://piper_description/meshes/", "meshes/")
    # Isaac derives USD prim names from mesh basenames. A second dot in names
    # such as `base_link.convex.stl` produces an invalid prim path, so rewrite
    # the deterministic convex suffix in both the URDF and copied files.
    src = src.replace(".convex.stl", "_convex.stl")
    urdf_dst = os.path.join(target, "piper.urdf")
    open(urdf_dst, "w", encoding="utf-8").write(src)

    copied = 0
    for name in os.listdir(os.path.join(piper_src, "meshes")):
        if name.lower().endswith(".stl"):
            target_name = name.replace(".convex.stl", "_convex.stl")
            shutil.copy2(
                os.path.join(piper_src, "meshes", name),
                os.path.join(target, "meshes", target_name),
            )
            copied += 1
    print(f"[piper] urdf -> {urdf_dst} ({copied} STL meshes copied)")
    return urdf_dst


def write_robot_config(target, joints):
    arm = [j["name"] for j in joints if j["type"] == "revolute"]
    gripper = [j["name"] for j in joints if j["type"] == "prismatic"]
    if len(arm) != 6 or len(gripper) != 2:
        raise ValueError(f"Expected 6 revolute + 2 prismatic joints, got arm={arm} gripper={gripper}")
    j7, j8 = gripper[0], gripper[1]
    # joint7 [0, +max], joint8 [-max, 0]: mirrored fingers, joint8 = -joint7
    mimic_mult = -1.0
    content = f"""# Embodiment args for the Piper arm (consumed by env/robot_manager/robot_class/piper.py).
# Generated by scripts/internal/build_piper_assets.py from piper_description.urdf.
urdf_path: "piper.urdf"
ee_joints: "{j7}"
ee_link: "link6"
ee_type: "gripper"
base_link: "base_link"
arm_joints_name: {arm}
gripper_joints_name: ['{j7}', '{j8}']
save_gripper_joints_name: {arm + gripper}
gripper_move:
    base: "{j7}"
    sign: 1.0
    mimic: ["{j8}", {mimic_mult}, 0.]
# link6 -> TCP distance along the approach axis. link6->finger base is
# 0.1358 in the URDF; fingertips add ~4cm. TUNE from first-run videos.
gripper_bias: 0.18
# joint7 range [0, 0.035]; joint8 is the mirrored finger.
gripper_scale: [0.0, 0.035]
delta_matrix: [[1,0,0],[0,1,0],[0,0,1]]
global_trans_matrix: [[1,0,0],[0,-1,0],[0,0,-1]] # for curobo
dual_arm: False
grasp_camera_reference_axis: [1, 0, 0] # in base_link frame
camera:
  - name: cam_wrist
    # Isaac's unmerged fixed-joint import nests every robot link below the
    # synthetic root_joint prim. CameraManager consumes a USD prim path, not
    # an articulation body name.
    link: root_joint/gripper_base
    type: PIPER_POLICY
    mesh: pinhole
    pos: [0.045, 0.0, 0.045]
    look_at_target: [0.0, 0.0, 0.180]
    look_at_up: [0.0, 1.0, 0.0]
"""
    path = os.path.join(target, "robot_config.yml")
    open(path, "w", encoding="utf-8").write(content)
    print(f"[piper] wrote {path}")


def write_curobo_config(target, joints):
    """Mirror the official x5 curobo.yml structure (all 8 joints in cspace).

    Collision spheres are omitted: IK (solve_pose) does not consult them and
    the scripted validation never runs motion planning. If plan_path with
    collision checking is ever needed, regenerate the file with curobo's
    generate_robot_yaml.py sphere fitting and merge the spheres in.
    """
    arm = [j["name"] for j in joints if j["type"] == "revolute"]
    gripper = [j["name"] for j in joints if j["type"] == "prismatic"]
    # Same ordering convention as the x5 config: arm joints, then gripper
    # joints mirrored (joint8 before joint7).
    cspace_joints = arm + gripper[::-1]
    n = len(cspace_joints)
    joint_list = ", ".join(f'"{j}"' for j in cspace_joints)
    ones = ", ".join(["1.0"] * n)
    tens = ", ".join(["10.0"] * n)
    jerks = ", ".join(["500.0"] * n)
    zeros = ", ".join(["0.0"] * n)
    content = f"""# Curobo config for the Piper arm (consumed by
# env/planner_manager/curobo_planner.py), mirroring the official x5
# curobo.yml structure. IK-only: collision spheres omitted (IK solve_pose
# does not use them); regenerate with curobo's generate_robot_yaml.py if
# collision-checked motion planning is ever needed.
robot_cfg:
  kinematics:
    base_link: base_link
    tool_frames:
    - link6
    urdf_path: $RoboDojo_ASSETS/Robots/piper/piper.urdf
    asset_root_path: $RoboDojo_ASSETS/Robots/piper
    use_global_cumul: true
    load_meshes: false
    format_version: 2.0
    cspace:
      joint_names: [{joint_list}]
      default_joint_position: [{zeros}]
      cspace_distance_weight: [{ones}]
      null_space_weight: [{ones}]
      null_space_maximum_distance: [{ones}]
      max_acceleration: [{tens}]
      max_jerk: [{jerks}]
      velocity_scale: [{ones}]
      acceleration_scale: [{ones}]
      jerk_scale: [{ones}]
      position_limit_clip: 0.0

planner:
  frame_bias: [0.0, 0.0, 0.0]
"""
    path = os.path.join(target, "curobo.yml")
    open(path, "w", encoding="utf-8").write(content)
    print(f"[piper] wrote {path}")


def author_gripper_contact_properties(usd_path, contract):
    if contract is None:
        return
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"failed to open Piper composed USD: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"Piper composed USD has no default prim: {usd_path}")
    material = UsdShade.Material.Define(stage, default_prim.GetPath().AppendChild("gripper_physics_material"))
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(contract["static_friction"])
    material_api.CreateDynamicFrictionAttr().Set(contract["dynamic_friction"])
    material_api.CreateRestitutionAttr().Set(0.0)
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material_api.CreateFrictionCombineModeAttr().Set("average")
    physx_material_api.CreateRestitutionCombineModeAttr().Set("average")

    for link_name in ("link7", "link8"):
        matches = [
            prim
            for prim in Usd.PrimRange(default_prim, Usd.TraverseInstanceProxies())
            if link_name in {part for part in str(prim.GetPath()).split("/") if part}
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {link_name} collider in {usd_path}, got {len(matches)}")
        collider = matches[0]
        collider_path = collider.GetPath()
        if collider.IsInstanceProxy():
            instance_root = collider.GetParent()
            while instance_root.IsInstanceProxy():
                instance_root = instance_root.GetParent()
            if not instance_root.IsInstance():
                raise RuntimeError(f"{link_name} collider proxy has no instance root")
            instance_root.SetInstanceable(False)
            collider = stage.GetPrimAtPath(collider_path)
        if not collider or not collider.IsValid() or collider.IsInstanceProxy():
            raise RuntimeError(f"failed to author {link_name} collider at {collider_path}")
        physx_collision_api = PhysxSchema.PhysxCollisionAPI(collider)
        if not physx_collision_api:
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider)
        physx_collision_api.CreateTorsionalPatchRadiusAttr().Set(contract["torsional_patch_radius"])
        physx_collision_api.CreateMinTorsionalPatchRadiusAttr().Set(contract["min_torsional_patch_radius"])
        UsdShade.MaterialBindingAPI.Apply(collider).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
    stage.GetRootLayer().Save()
    print(f"[piper] authored gripper contact contract in {usd_path}: {contract}")


def convert_urdf_to_usd(urdf_path, usd_path, contact_contract=None):
    from isaaclab.app import AppLauncher

    convert_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(convert_parser)
    # Drop our own flags; pass the rest (e.g. --headless) through.
    passthrough = [
        arg for arg in CONVERT_ARGS if arg != "--convert" and not arg.startswith(("--piper-src", "--assets-root"))
    ]
    args, _ = convert_parser.parse_known_args(passthrough)
    app = AppLauncher(args).app

    from isaacsim.asset.importer.urdf import _urdf  # noqa: E402
    import omni.kit.commands  # noqa: E402

    config = _urdf.ImportConfig()
    # Attribute names vary across Isaac versions - set only what exists.
    candidates = {
        "merge_fixed_joints": False,  # keep joint names as authored
        "fix_base": True,
        "fix_base_link": True,
        "self_collision": True,
        "create_physics": True,
        "create_joint_drives": True,
        "import_inertia_tensor": True,
        "import_mass": True,
        # Preserve the authored URDF mass/inertia contract. Recomputing from
        # visual meshes changes link masses by multiples and breaks replay
        # dynamics across simulators even when kinematics are identical.
        "inertia_from_visuals": False,
        "make_default_prim": True,
        "density": 1000.0,
    }
    applied = []
    for attr, value in candidates.items():
        if hasattr(config, attr):
            setattr(config, attr, value)
            applied.append(attr)
    print(f"[piper] import attrs applied: {applied}")
    print(f"[piper] import attrs unavailable: {sorted(set(candidates) - set(applied))}")

    result = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=config,
        dest_path=usd_path,
        get_articulation_root=True,
    )
    print(f"[piper] urdf->usd result={result}")

    if os.path.isfile(usd_path):
        print(f"[piper] SUCCESS: {usd_path} ({os.path.getsize(usd_path)} bytes)")
    else:
        raise RuntimeError(f"Piper USD was not created: {usd_path}")
    author_gripper_contact_properties(usd_path, contact_contract)
    app.close()


def main():
    args, _ = parser.parse_known_args()  # ignore AppLauncher passthrough flags
    global CONVERT_ARGS
    CONVERT_ARGS = sys.argv[1:]
    target = target_dir(args.assets_root)

    urdf_path = copy_assets(args.piper_src, target)
    joints = parse_urdf(urdf_path)
    write_robot_config(target, joints)
    write_curobo_config(target, joints)

    if args.convert:
        convert_urdf_to_usd(
            urdf_path,
            os.path.join(target, "piper.usd"),
            contact_contract=gripper_contact_contract(args),
        )
    else:
        print(f"[piper] files ready in {target}; run with --convert (server, robodojo env) to build piper.usd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
