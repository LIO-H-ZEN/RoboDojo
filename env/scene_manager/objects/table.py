from __future__ import annotations

import os
from pathlib import Path
import random

from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.utils.prims import (
    get_prim_at_path,
    is_prim_path_valid,
)
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
import numpy as np
import omni.kit.commands
from omni.physx.scripts import physicsUtils
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
import torch


def resolve_path(path: str) -> str | None:
    """Resolve path to absolute path, handling various path formats."""
    if not path:
        return None
    p = Path(path).expanduser()
    if p.exists():
        return str(p.resolve())
    return None


def _ensure_parent_xforms(stage: Usd.Stage, prim_path: str) -> None:
    """Ensure all parent prims exist as Xforms for a given prim path."""
    if not prim_path or not prim_path.startswith("/"):
        return
    parts = [p for p in prim_path.split("/") if p]
    if len(parts) <= 1:
        return
    current = ""
    for name in parts[:-1]:
        current = f"{current}/{name}" if current else f"/{name}"
        prim = stage.GetPrimAtPath(current)
        if not prim or not prim.IsValid():
            UsdGeom.Xform.Define(stage, current)


class Table(SingleGeometryPrim, SingleRigidPrim):
    """
    init and create MDL material in the env[id]
    """

    def __init__(
        self,
        prim_path: str,
        mdl_file_path: str,
        instance_config: str,
        mdl_name: str = "Ceiling_Tiles",
        resolution: int = 10,
    ):
        prim = get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            stage = stage_utils.get_current_stage()
            _ensure_parent_xforms(stage, prim_path)
            omni.kit.commands.execute(
                "CreateMeshPrimCommand",
                prim_type="Cube",
                prim_path=prim_path,
                u_patches=resolution,
                v_patches=resolution,
                w_patches=resolution,
            )
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"Failed to create prim at path {prim_path}")
            physicsUtils.setup_transform_as_scale_orient_translate(prim)

        self.stage = prim.GetStage()
        self._prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]

        self.scale = instance_config.get("scale", [1.0, 1.0, 1.0])
        self.position = instance_config.get("default_pos", [0.0, 0.0, 0.0])
        self.orientation = instance_config.get("default_ori", [1.0, 0.0, 0.0, 0.0])
        self.is_static = bool(instance_config.get("static", True))

        self.materials_prim_path = find_unique_string_name(
            prim_path + "/Material",
            lambda x: not is_prim_path_valid(x),
        )

        self.physics_material_path = find_unique_string_name(
            prim_path + "/physics_material",
            lambda x: not is_prim_path_valid(x),
        )
        self.physics_material = PhysicsMaterial(
            prim_path=self.physics_material_path,
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0,
        )

        SingleGeometryPrim.__init__(
            self,
            prim_path=prim_path,
            name=self.instance_name,
            scale=self.scale,
            collision=True,
            visible=True,
            track_contact_forces=False,
        )
        try:
            self.set_collision_approximation("convexDecomposition")
        except Exception as e:
            print(f"[WARN] Failed to set convex decomposition collision for {prim_path}: {e}")

        # Make the table immovable by default: don't create a rigid body.
        # We still set pose so the collider is placed correctly.
        try:
            physicsUtils.setup_transform_as_scale_orient_translate(self.prim)
            physicsUtils.set_or_add_translate_op(self.prim, translate=Gf.Vec3f([float(v) for v in self.position]))
            if isinstance(self.orientation, (list, tuple)) and len(self.orientation) == 4:
                w, x, y, z = [float(v) for v in self.orientation]
                physicsUtils.set_or_add_orient_op(
                    self.prim,
                    orient=Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))),
                )
        except Exception as e:
            print(f"[WARN] Failed to set static table pose for {prim_path}: {e}")

        if not self.is_static:
            SingleRigidPrim.__init__(
                self,
                prim_path=prim_path,
                name=self.instance_name,
                translation=self.position,
                orientation=self.orientation,
                scale=self.scale,
            )

        self.mdl_file_path = mdl_file_path
        self.mdl_name = mdl_name
        self.instance_config = instance_config

        resolved_mdl_path = resolve_path(self.mdl_file_path)

        if not resolved_mdl_path:
            print(f"Warning: MDL material path not found: {self.mdl_file_path}")
            return

        if not self.mdl_name:
            mdl_name = os.path.splitext(os.path.basename(resolved_mdl_path))[0]

        # Create unique material path under geometry's Looks
        material_path = find_unique_string_name(
            initial_name=self.materials_prim_path,
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        self.materials_prim_path = material_path
        create_mdl_material(resolved_mdl_path, mdl_name, self.materials_prim_path)
        self.apply_material()

        self._default_linear_velocity = [0.0, 0.0, 0.0]
        self._default_angular_velocity = [0.0, 0.0, 0.0]
        self._setup_physics()

    def apply_saved_pose(self):
        self.set_local_pose(translation=self.position, orientation=self.orientation)
        self.set_local_scale(np.array(self.scale))
        if not self.is_static:
            self._apply_default_velocities()

    def relocate_offscreen(self):
        _FAR_CENTER = (100000.0, 100000.0, 100000)
        _FAR_JITTER = 1000.0
        far_pos = (
            _FAR_CENTER[0] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[1] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
            _FAR_CENTER[2] + random.uniform(-_FAR_JITTER, _FAR_JITTER),
        )
        self.set_local_pose(translation=far_pos, orientation=self.orientation)
        self.set_local_scale(np.array(self.scale))
        if not self.is_static:
            self._apply_default_velocities()

    def _apply_default_velocities(self):
        """Re-apply default linear/angular velocity if configured."""
        if self._default_linear_velocity is not None:
            self.set_linear_velocity(torch.tensor(self._default_linear_velocity))
        if self._default_angular_velocity is not None:
            self.set_angular_velocity(torch.tensor(self._default_angular_velocity))

    def _setup_physics(self):
        """Configure physics properties (rigid type, mass) from instance config."""
        if not self.is_static:
            if self.mass <= 0:
                self.mass = 0.05
            self.set_mass(self.mass)

            if self._default_linear_velocity is not None or self._default_angular_velocity is not None:
                self.set_default_state(
                    linear_velocity=self._default_linear_velocity,
                    angular_velocity=self._default_angular_velocity,
                )
            if self.physics_material is not None:
                self.apply_physics_material(physics_material=self.physics_material)

    def set_image_texture(self, image_path):
        """Display an arbitrary image on the table surface (screen effect).

        Builds a UsdPreviewSurface material whose diffuseColor is driven by a
        UsdUVTexture shader pointing at ``image_path`` (PNG/JPG/EXR), then
        binds it onto the table prim and its child meshes. Any prior MDL /
        PreviewSurface material on the table is replaced by the new binding.
        """
        try:
            resolved = resolve_path(image_path)
            if not resolved:
                print(f"[Table.set_image_texture] image path not resolved: {image_path}")
                return False

            stage = omni.usd.get_context().get_stage()
            img_base = os.path.splitext(os.path.basename(resolved))[0].replace(" ", "_")
            mat_path = f"{self.prim_path}/Looks/TableImage_{img_base}"

            # Remove any previous image material prim so a fresh one is created.
            if is_prim_path_valid(mat_path):
                omni.kit.commands.execute("DeletePrimsCommand", paths=[mat_path])

            # --- UsdPreviewSurface material with UsdUVTexture diffuse map ---
            material = UsdShade.Material.Define(stage, mat_path)

            pbr_shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
            pbr_shader.CreateIdAttr("UsdPreviewSurface")
            pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")

            st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/st_reader")
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_input = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
            st_input.Set("st")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(
                st_input
            )

            tex_sampler = UsdShade.Shader.Define(stage, f"{mat_path}/diffuse_texture")
            tex_sampler.CreateIdAttr("UsdUVTexture")
            tex_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(resolved)
            tex_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.ConnectableAPI(), "result"
            )
            tex_sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex_sampler.ConnectableAPI(), "rgb"
            )

            # Bind onto the table prim and child Meshes / GeomSubsets.
            for bind_path in [self.prim_path] + [
                str(child.GetPath())
                for child in prims_utils.get_prim_children(stage.GetPrimAtPath(self.prim_path))
                if child.GetTypeName() in ("Mesh", "GeomSubset")
            ]:
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=bind_path,
                    material_path=mat_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )

            self.materials_prim_path = mat_path
            omni.kit.app.get_app().update()
            print(f"[Table.set_image_texture] {self.instance_name} -> {resolved}")
            return True
        except Exception as e:
            print(f"[Table.set_image_texture] failed: {e}")
            return False

    def update_texture(self, image_path):
        """Fast-path texture swap for the current image material.

        Reuses the UsdUVTexture shader created by :meth:`set_image_texture`
        and only updates its ``file`` input, avoiding a full material rebuild
        (important when playing video frames at high frequency). Falls back
        to :meth:`set_image_texture` if no image material is currently bound.
        """
        try:
            resolved = resolve_path(image_path)
            if not resolved:
                print(f"[Table.update_texture] image path not resolved: {image_path}")
                return False

            stage = omni.usd.get_context().get_stage()
            mat_prim = stage.GetPrimAtPath(self.materials_prim_path)
            texture_shader = None
            if mat_prim and mat_prim.IsValid():
                for child in mat_prim.GetAllChildren():
                    shader = UsdShade.Shader(child)
                    if shader.GetIdAttr().Get() == "UsdUVTexture":
                        texture_shader = shader
                        break

            if texture_shader is not None:
                file_input = texture_shader.GetInput("file")
                if file_input is not None:
                    file_input.Set(resolved)
                    omni.kit.app.get_app().update()
                    return True

            # No usable image material bound yet -> full rebuild path.
            return self.set_image_texture(resolved)
        except Exception as e:
            print(f"[Table.update_texture] failed: {e}")
            return False

    def set_material(self, mdl_path):
        """Switch the table's material to an MDL texture (r,g,b ignored).

        Loads the MDL file (e.g. Ceiling_Tiles.mdl / Carpet_Beige.mdl) and
        binds it to the table prim and its meshes. Returns True on success.
        """
        try:
            resolved = resolve_path(mdl_path)
            if not resolved:
                print(f"[Table.set_material] MDL path not resolved: {mdl_path}")
                return False
            mdl_name = os.path.splitext(os.path.basename(resolved))[0]

            # Create a fresh material prim under Looks (avoid stale prims).
            safe_name = f"Mat_{mdl_name.replace(' ', '_')}"
            mat_path = f"{self.prim_path}/Looks/{safe_name}"
            omni.kit.commands.execute(
                "DeletePrimsCommand",
                paths=[mat_path],
            ) if is_prim_path_valid(mat_path) else None

            create_mdl_material(resolved, mdl_name, mat_path)

            # Track the newly-bound material so set_color tints the ACTIVE one.
            self.materials_prim_path = mat_path

            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.prim_path,
                material_path=mat_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.prim_path)
            children_prims = prims_utils.get_prim_children(prim)
            for child in children_prims:
                if child.GetTypeName() in ("Mesh", "GeomSubset"):
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=str(child.GetPath()),
                        material_path=mat_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )
            omni.kit.app.get_app().update()
            print(f"[Table.set_material] {self.instance_name} -> {mdl_name}")
            return True
        except Exception as e:
            print(f"[Table.set_material] failed: {e}")
            return False

    def set_color(self, rgb):
        """Tint the table's existing MDL material with (r,g,b in 0..1).

        Unlike the previous PreviewSurface replacement (which erased the
        texture), this modifies the MDL's ``diffuse_tint`` shader input so the
        texture detail is preserved under a colour overlay. We re-resolve the
        bound MDL shader each call (safe across scene resets).
        """
        try:
            stage = omni.usd.get_context().get_stage()
            material_prim = stage.GetPrimAtPath(self.materials_prim_path)
            if not material_prim or not material_prim.IsValid():
                # Fallback: attempt to locate any Shader child under Looks.
                looks_path = f"{self.prim_path}/Looks"
                looks_prim = stage.GetPrimAtPath(looks_path)
                if looks_prim and looks_prim.IsValid():
                    for child in looks_prim.GetChildren():
                        material_prim = stage.GetPrimAtPath(str(child.GetPath()))
                        if material_prim and material_prim.IsValid():
                            break

            touched = False
            if material_prim and material_prim.IsValid():
                for child in material_prim.GetAllChildren():
                    if child.GetTypeName() != "Shader":
                        continue
                    shader = UsdShade.Shader(child)
                    tint = shader.GetInput("diffuse_tint")
                    if tint is not None:
                        tint.Set(Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2])))
                        touched = True
                    dcc = shader.GetInput("diffuse_color_constant")
                    if dcc is not None:
                        dcc.Set(Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2])))
                        touched = True

            if not touched:
                print(
                    f"[Table.set_color] no tint input; material={self.materials_prim_path} "
                    f"rgb={rgb} -> using PreviewSurface fallback"
                )
                # Fallback to pure PreviewSurface (last resort; loses texture).
                color_mat_path = f"{self.prim_path}/Looks/TableColor"
                surface = PreviewSurface(color_mat_path)
                surface.set_color(np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])]))
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=self.prim_path,
                    material_path=color_mat_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )
                touched = True

            if touched:
                print(f"[Table.set_color] {self.instance_name} -> {rgb} (tint overlay)")
            return touched
        except Exception as e:
            print(f"[Table.set_color] failed: {e}")
            return False

    def apply_material(self):
        """
        Apply an MDL material to the geometry object.

        Args:
            materials_prim_path: Path to the materials prim
            mdl_name: Name of the material in the MDL file (optional)
        """

        try:
            # Bind material to the prim and its children
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.prim_path,
                material_path=self.materials_prim_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )

            children_prims = prims_utils.get_prim_children(self.prim)
            for prim in children_prims:
                if prim.GetTypeName() in ["Mesh", "GeomSubset"]:
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=prim.GetPath(),
                        material_path=self.materials_prim_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )

            self.visual_material_path = self.materials_prim_path
            self.visual_material = PreviewSurface(self.materials_prim_path)

        except Exception as e:
            print(f"Warning: Failed to apply MDL material {self.materials_prim_path}: {e}")
