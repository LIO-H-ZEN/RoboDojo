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
# Extra flags (e.g. --headless) pass through to AppLauncher in the convert
# stage; main() uses parse_known_args so they are not rejected here.

CONVERT_ARGS = []
PIPER_TCP_LINK = "piper_tcp"
PIPER_TCP_OFFSET_M = 0.1358


def ensure_piper_tcp(urdf_text):
    """Add the physical TCP frame used by planning and Isaac observation.

    The upstream piper_description ends at the two finger joints. The contact
    plane is exactly 0.1358m along gripper_base +Z (the same origin used by
    joint7/joint8), so express it once as a fixed, collision-free URDF link
    instead of maintaining a controller-only offset from link6. Its negligible
    positive inertial values keep Isaac's URDF importer from dropping it.
    """
    if re.search(rf'<link\s+name="{PIPER_TCP_LINK}"', urdf_text):
        return urdf_text
    tcp = f'''  <link name="{PIPER_TCP_LINK}">
    <inertial>
      <mass value="0.000001"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.000000001" ixy="0" ixz="0" iyy="0.000000001" iyz="0" izz="0.000000001"/>
    </inertial>
  </link>
  <joint name="gripper_base_to_{PIPER_TCP_LINK}" type="fixed">
    <origin xyz="0 0 {PIPER_TCP_OFFSET_M}" rpy="0 0 0"/>
    <parent link="gripper_base"/>
    <child link="{PIPER_TCP_LINK}"/>
  </joint>
'''
    if "</robot>" not in urdf_text:
        raise ValueError("Piper URDF has no closing </robot> tag")
    return urdf_text.replace("</robot>", tcp + "</robot>", 1)


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
    urdf_src = os.path.join(piper_src, "urdf", "piper_description.urdf")
    if not os.path.isfile(urdf_src):
        raise FileNotFoundError(f"{urdf_src} not found - pass the piper_description dir as --piper-src")
    os.makedirs(os.path.join(target, "meshes"), exist_ok=True)

    src = open(urdf_src, encoding="utf-8").read()
    src = src.replace("package://piper_description/meshes/", "meshes/")
    src = ensure_piper_tcp(src)
    urdf_dst = os.path.join(target, "piper.urdf")
    open(urdf_dst, "w", encoding="utf-8").write(src)

    copied = 0
    for name in os.listdir(os.path.join(piper_src, "meshes")):
        if name.lower().endswith(".stl"):
            shutil.copy2(os.path.join(piper_src, "meshes", name), os.path.join(target, "meshes", name))
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
ee_link: "{PIPER_TCP_LINK}"
ee_link_is_physical_tcp: true
ee_type: "gripper"
base_link: "base_link"
arm_joints_name: {arm}
gripper_joints_name: ['{j7}', '{j8}']
save_gripper_joints_name: {arm + gripper}
gripper_move:
    base: "{j7}"
    sign: 1.0
    mimic: ["{j8}", {mimic_mult}, 0.]
# Legacy link6 -> TCP distance. Physical-TCP consumers use ee_link directly
# and do not apply this value as an additional target offset.
gripper_bias: {PIPER_TCP_OFFSET_M}
# joint7 range [0, 0.035]; joint8 is the mirrored finger.
gripper_scale: [0.0, 0.035]
# Preserve full close-side position error after contact. The control manager's
# default 0.2 limiter remains in effect for every robot without this override.
gripper_rate_limit: 1.0
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
    - {PIPER_TCP_LINK}
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


def postprocess_usd(usd_path):
    """Bake PhysxContactReportAPI onto the finger rigid bodies (link7/link8).

    scripted_single_arm_pickup.py reads per-finger contact forces through
    PhysX rigid contact views. Applying the report API at runtime - after the
    TaskEnv has initialized physics - is ignored by the running PhysX scene
    ("Failed to find contact report API"), so it must live in the asset USD.
    Diagnostic-only: no mass, friction, gain, or collision-geometry change.

    Also reports (no edits) the physics material and friction bound to each
    finger collision prim - if a material is bound, runtime friction edits on
    the CollisionAPI fallback attributes are ignored by PhysX.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print("[piper] post-process FAILED: could not open usd stage")
        return
    patched = 0
    material_lines = []
    subtree_lines = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not (path.endswith("/link7") or path.endswith("/link8")):
            continue
        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim)
        patched += 1
        if prim.IsInstanceable():
            subtree_lines.append(f"{path}: INSTANCEABLE - children hidden from Traverse")
        # Dump the full finger subtree: every prim with its applied schemas,
        # so collision placement/naming is visible whatever the importer did.
        for child in stage.Traverse():
            child_path = str(child.GetPath())
            if not child_path.startswith(path + "/"):
                continue
            schemas = ",".join(child.GetAppliedSchemas()) or "-"
            subtree_lines.append(f"{child_path}: type={child.GetTypeName()} api=[{schemas}]")
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            rel = child.GetRelationship("material:binding:physics")
            targets = list(rel.GetForwardedTargets()) if rel is not None else []
            if not targets:
                material_lines.append(f"{child_path}: no physics material binding")
                continue
            mat = stage.GetPrimAtPath(targets[0])
            static = mat.GetAttribute("physics:staticFriction").Get() if mat else None
            dynamic = mat.GetAttribute("physics:dynamicFriction").Get() if mat else None
            material_lines.append(
                f"{child_path}: material={targets[0]} static={static} dynamic={dynamic}"
            )
    stage.GetRootLayer().Save()
    print(f"[piper] post-process: contact report applied to {patched} finger body prims")
    # Census: every collision prim in the whole robot asset. Shows where
    # collision geometry actually lives if not under link7/link8.
    collision_paths = [
        str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    print(f"[piper] collision census: {len(collision_paths)} collision prims in asset")
    for cp in collision_paths[:40]:
        print(f"[piper] collision: {cp}")
    if len(collision_paths) > 40:
        print(f"[piper] collision: ... and {len(collision_paths) - 40} more")
    for line in subtree_lines:
        print(f"[piper] finger subtree: {line}")
    if not material_lines:
        print("[piper] finger material: NO collision prims found under link7/link8 "
              "- check import config (collision geometry may be missing)")
    for line in material_lines:
        print(f"[piper] finger material: {line}")


def convert_urdf_to_usd(urdf_path, usd_path):
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
        "inertia_from_visuals": True,
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
        postprocess_usd(usd_path)
        print(f"[piper] SUCCESS: {usd_path} ({os.path.getsize(usd_path)} bytes)")
    else:
        print(f"[piper] FAILED: {usd_path} was not created")
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
        convert_urdf_to_usd(urdf_path, os.path.join(target, "piper.usd"))
    else:
        print(f"[piper] files ready in {target}; run with --convert (server, robodojo env) to build piper.usd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
