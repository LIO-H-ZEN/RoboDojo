"""Convert piper URDF to USD using IsaacSim's built-in URDF importer."""
import os
import sys

sys.path.append("/home/ubuntu/RoboDojo")
sys.path.append("/home/ubuntu/RoboDojo/XPolicyLab")

URDF_PATH = "/home/ubuntu/RoboDojo/Assets/Robots/piper/piper.urdf"
DEST_PATH = "/home/ubuntu/RoboDojo/Assets/Robots/piper/piper.usd"

print(f"[convert] starting IsaacSim app...", flush=True)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "num_envs": 1})
simulation_app = app_launcher.app

print("[convert] app started, importing URDF...", flush=True)

import omni.kit.commands  # noqa: E402
from isaacsim.asset.importer.urdf import _urdf  # noqa: E402

config = _urdf.ImportConfig()

# Dump available attributes to a file for debugging
with open("/tmp/urdf_import_config_attrs.txt", "w") as f:
    f.write("\n".join(sorted(a for a in dir(config) if not a.startswith("_"))))

# Set only attributes that exist (detect dynamically)
candidate_attrs = {
    "density": 1000.0,
    "inertia_from_visuals": True,
    "make_default_prim": True,
    "merge_fixed_joints": False,
    "self_collision": True,
    "fix_base_link": True,
    "create_physics": True,
    "create_joint_drives": True,
    "import_inertia_tensor": True,
    "import_mass": True,
    "replace_urdf_meshes_with_collision_sdf": False,
}
set_attrs = []
for attr, value in candidate_attrs.items():
    if hasattr(config, attr):
        setattr(config, attr, value)
        set_attrs.append(attr)
    else:
        print(f"[convert] skip attr (not exists): {attr}", flush=True)

print(f"[convert] set attrs: {set_attrs}", flush=True)
print(f"[convert] config: urdf={URDF_PATH}", flush=True)

result = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=config,
    dest_path=DEST_PATH,
    get_articulation_root=True,
)

print(f"[convert] result={result}", flush=True)

if os.path.exists(DEST_PATH):
    size = os.path.getsize(DEST_PATH)
    print(f"[convert] SUCCESS: piper.usd created ({size} bytes)", flush=True)
else:
    print(f"[convert] FAILED: {DEST_PATH} was not created", flush=True)

simulation_app.close()