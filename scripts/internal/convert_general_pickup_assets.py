#!/usr/bin/env python3
"""Convert the asset subset referenced by General Pickup contracts.

RoboDojo distributes objects as USD/USDZ. ManiSkill consumes a textured GLB
for rendering and authored collision meshes decomposed into deterministic
convex components. Conversion is entirely offline; runtime environments never
parse USD or silently rebuild collision geometry.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

import coacd
import numpy as np
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import trimesh

CONVERTER_VERSION = "robodojo_usdz_to_maniskill_v4_material_graph_coacd"
COLLISION_FILENAME = "collision.ply"
COACD_PARAMETERS = {
    "threshold": 0.05,
    "max_convex_hull": 32,
    "seed": 0,
}

SUPPORTED_SHADER_TYPES = frozenset(
    {
        "UsdPreviewSurface",
        "UsdUVTexture",
        "UsdPrimvarReader_float2",
        "mdl:gltf/pbr.mdl#gltf_material",
        "mdl:gltf/pbr.mdl#gltf_texture_lookup",
    }
)


@dataclass(frozen=True)
class ParsedMaterial:
    path: str
    material: trimesh.visual.material.PBRMaterial
    uv_primvar: str | None
    texture_source: str | None
    shader_types: tuple[str, ...]


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_key(obj: dict[str, Any]) -> str:
    return f"{obj['asset_type']}/{obj['category']}/{int(obj['category_idx']):05d}"


def contract_asset_keys(contract_dir: Path) -> list[str]:
    manifest = json.loads((contract_dir / "manifest.json").read_text(encoding="utf-8"))
    keys = set()
    for row in manifest["contracts"]:
        contract_path = contract_dir / row["contract_file"]
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        keys.add(_asset_key(contract["target"]))
        keys.update(_asset_key(obj) for obj in contract["clutter"])
    if not keys:
        raise ValueError(f"contract manifest references no assets: {contract_dir}")
    return sorted(keys)


def _open_stage(source_path: Path, extract_dir: Path) -> tuple[Usd.Stage, Path]:
    if source_path.suffix.lower() == ".usdz":
        with zipfile.ZipFile(source_path) as archive:
            archive.extractall(extract_dir)
        stage_files = sorted(
            path for path in extract_dir.rglob("*") if path.suffix.lower() in {".usd", ".usda", ".usdc"}
        )
        if not stage_files:
            raise ValueError(f"USDZ archive contains no USD stage: {source_path}")
        stage_path = stage_files[0]
    else:
        stage_path = source_path
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise ValueError(f"failed to open USD stage: {stage_path}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError(f"only Z-up assets are supported: {source_path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isclose(meters_per_unit, 1.0):
        raise ValueError(f"asset is not authored in meters: {source_path} meters_per_unit={meters_per_unit}")
    return stage, stage_path


def _visual_meshes(stage: Usd.Stage) -> list[UsdGeom.Mesh]:
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise ValueError("USD stage contains no mesh")
    visual = [mesh for mesh in meshes if "/visual/" in str(mesh.GetPath()).lower()]
    if not visual:
        visual = [mesh for mesh in meshes if str(mesh.GetPath()).lower().endswith("/visual")]
    return visual or meshes


def _collision_meshes(stage: Usd.Stage) -> list[UsdGeom.Mesh]:
    meshes = []
    unsupported = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_api = UsdPhysics.CollisionAPI(prim)
        enabled = collision_api.GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            unsupported.append(f"{prim.GetPath()} ({prim.GetTypeName()})")
            continue
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
        approximation = mesh_collision_api.GetApproximationAttr().Get()
        if str(approximation) != str(UsdPhysics.Tokens.convexDecomposition):
            raise ValueError(
                "unsupported collision approximation "
                f"{approximation!s} on {prim.GetPath()}; expected convexDecomposition"
            )
        meshes.append(UsdGeom.Mesh(prim))
    if unsupported:
        raise ValueError("enabled USD collision prims must be meshes: " + ", ".join(unsupported))
    if not meshes:
        raise ValueError("USD stage has no enabled USD collision mesh")
    return meshes


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles = []
    offset = 0
    for count in counts:
        count = int(count)
        if count < 3:
            raise ValueError(f"mesh face has fewer than three vertices: {count}")
        face = indices[offset : offset + count]
        for index in range(1, count - 1):
            triangles.append((int(face[0]), int(face[index]), int(face[index + 1])))
        offset += count
    if offset != len(indices):
        raise ValueError("faceVertexCounts and faceVertexIndices disagree")
    return np.asarray(triangles, dtype=np.int64)


def _mesh_topology(
    mesh: UsdGeom.Mesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError(f"invalid USD points array: {points.shape}")
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()), dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    points = (homogeneous @ transform)[:, :3]
    triangles = _triangulate(counts, indices)
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        triangles = triangles[:, ::-1]
    triangle_faces = np.repeat(np.arange(len(counts), dtype=np.int64), counts - 2)
    return points, counts, indices, np.column_stack((triangles, triangle_faces))


def _mesh_arrays(
    mesh: UsdGeom.Mesh,
    *,
    source_face_indices: np.ndarray,
    uv_primvar_name: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    points, counts, indices, triangle_records = _mesh_topology(mesh)
    face_selection = np.zeros(len(counts), dtype=bool)
    face_selection[source_face_indices] = True
    selected = triangle_records[face_selection[triangle_records[:, 3]]]
    triangles = selected[:, :3]
    if len(triangles) == 0:
        raise ValueError(f"material binding selects no faces: {mesh.GetPath()}")
    if uv_primvar_name is None:
        return points, triangles, None

    primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar(uv_primvar_name)
    if not primvar or not primvar.HasValue():
        raise ValueError(f"material requires missing UV primvar {uv_primvar_name!r}: {mesh.GetPath()}")
    values = np.asarray(primvar.ComputeFlattened(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"invalid UV primvar {uv_primvar_name!r} on {mesh.GetPath()}: {values.shape}")
    interpolation = primvar.GetInterpolation()
    if interpolation in {UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying}:
        if len(values) != len(points):
            raise ValueError(f"vertex UV cardinality mismatch on {mesh.GetPath()}: {len(values)} != {len(points)}")
        return points, triangles, values[:, :2]
    if interpolation != UsdGeom.Tokens.faceVarying:
        raise ValueError(f"unsupported UV interpolation {interpolation!r} on {mesh.GetPath()}")
    if len(values) != len(indices):
        raise ValueError(f"faceVarying UV cardinality mismatch on {mesh.GetPath()}: {len(values)} != {len(indices)}")

    expanded_points = []
    expanded_uv = []
    expanded_faces = []
    offset = 0
    selected_faces = set(int(value) for value in source_face_indices)
    for face_index, count_value in enumerate(counts):
        count = int(count_value)
        if face_index not in selected_faces:
            offset += count
            continue
        face_vertices = indices[offset : offset + count]
        face_uv = values[offset : offset + count, :2]
        for corner in range(1, count - 1):
            base = len(expanded_points)
            corner_ids = (0, corner, corner + 1)
            expanded_points.extend(points[face_vertices[list(corner_ids)]])
            expanded_uv.extend(face_uv[list(corner_ids)])
            expanded_faces.append((base, base + 1, base + 2))
        offset += count
    return (
        np.asarray(expanded_points, dtype=np.float64),
        np.asarray(expanded_faces, dtype=np.int64),
        np.asarray(expanded_uv, dtype=np.float64),
    )


def _one_connected_shader(connectable, *, description: str) -> UsdShade.Shader:
    sources, invalid = connectable.GetConnectedSources()
    if invalid or len(sources) != 1:
        raise ValueError(
            f"{description} requires exactly one valid shader connection; "
            f"sources={len(sources)} invalid={list(invalid)}"
        )
    shader = UsdShade.Shader(sources[0].source)
    if not shader:
        raise ValueError(f"{description} is not connected to a shader")
    return shader


def _asset_path(value: Any, stage_path: Path, *, description: str) -> Path:
    if not isinstance(value, Sdf.AssetPath) or not value.path:
        raise ValueError(f"{description} has no asset path")
    resolved = Path(value.resolvedPath) if value.resolvedPath else stage_path.parent / value.path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is missing: {resolved}")
    return resolved


def _image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA").copy()


def _float_input(shader: UsdShade.Shader, name: str, default: float) -> float:
    shader_input = shader.GetInput(name)
    value = shader_input.Get() if shader_input else None
    return default if value is None else float(value)


def _preview_material(material: UsdShade.Material, stage_path: Path) -> ParsedMaterial:
    surface = material.GetSurfaceOutput()
    shader = _one_connected_shader(surface, description=f"material surface {material.GetPath()}")
    shader_id = str(shader.GetIdAttr().Get() or "")
    if shader_id != "UsdPreviewSurface":
        raise ValueError(f"unsupported surface shader {shader_id or '<missing-id>'}: {shader.GetPath()}")

    diffuse = shader.GetInput("diffuseColor")
    if not diffuse:
        raise ValueError(f"UsdPreviewSurface has no diffuseColor: {shader.GetPath()}")
    diffuse_sources, invalid = diffuse.GetConnectedSources()
    if invalid:
        raise ValueError(f"diffuseColor has invalid connections: {list(invalid)}")
    texture = None
    texture_source = None
    uv_primvar = None
    base_color = diffuse.Get()
    shader_types = {shader_id}
    if diffuse_sources:
        texture_shader = _one_connected_shader(diffuse, description=f"diffuseColor {shader.GetPath()}")
        texture_shader_id = str(texture_shader.GetIdAttr().Get() or "")
        shader_types.add(texture_shader_id or "<missing-id>")
        if texture_shader_id != "UsdUVTexture":
            raise ValueError(
                f"unsupported diffuseColor shader {texture_shader_id or '<missing-id>'}: {texture_shader.GetPath()}"
            )
        texture_path = _asset_path(
            texture_shader.GetInput("file").Get(),
            stage_path,
            description=f"UsdUVTexture file {texture_shader.GetPath()}",
        )
        st_input = texture_shader.GetInput("st")
        if not st_input:
            raise ValueError(f"UsdUVTexture has no st input: {texture_shader.GetPath()}")
        reader = _one_connected_shader(st_input, description=f"UsdUVTexture st {texture_shader.GetPath()}")
        reader_id = str(reader.GetIdAttr().Get() or "")
        shader_types.add(reader_id or "<missing-id>")
        if reader_id != "UsdPrimvarReader_float2":
            raise ValueError(f"unsupported UV reader {reader_id or '<missing-id>'}: {reader.GetPath()}")
        varname = reader.GetInput("varname")
        uv_primvar = str(varname.Get() if varname else "")
        if not uv_primvar:
            raise ValueError(f"UV reader has no varname: {reader.GetPath()}")
        scale_input = texture_shader.GetInput("scale")
        scale = scale_input.Get() if scale_input else None
        base_color = tuple(scale) if scale is not None else (1.0, 1.0, 1.0, 1.0)
        bias_input = texture_shader.GetInput("bias")
        bias = bias_input.Get() if bias_input else None
        if bias is not None and not np.allclose(np.asarray(bias), 0.0):
            raise ValueError(f"non-zero UsdUVTexture bias is unsupported: {texture_shader.GetPath()}")
        texture = _image(texture_path)
        texture_source = str(texture_path)
    elif base_color is None:
        base_color = (0.18, 0.18, 0.18)

    rgba = tuple(float(value) for value in base_color[:3]) + (_float_input(shader, "opacity", 1.0),)
    return ParsedMaterial(
        path=str(material.GetPath()),
        material=trimesh.visual.material.PBRMaterial(
            name=str(material.GetPath()),
            baseColorFactor=rgba,
            baseColorTexture=texture,
            metallicFactor=_float_input(shader, "metallic", 0.0),
            roughnessFactor=_float_input(shader, "roughness", 0.5),
        ),
        uv_primvar=uv_primvar,
        texture_source=texture_source,
        shader_types=tuple(sorted(shader_types)),
    )


def _mdl_shader_type(shader: UsdShade.Shader) -> str:
    source_asset = shader.GetSourceAsset("mdl")
    source_path = source_asset.path if isinstance(source_asset, Sdf.AssetPath) else ""
    subidentifier = str(shader.GetSourceAssetSubIdentifier("mdl") or "")
    return f"mdl:{source_path}#{subidentifier}"


def _mdl_gltf_material(material: UsdShade.Material, stage_path: Path) -> ParsedMaterial:
    surface = material.GetSurfaceOutput("mdl")
    shader = _one_connected_shader(surface, description=f"MDL material surface {material.GetPath()}")
    shader_type = _mdl_shader_type(shader)
    if shader_type != "mdl:gltf/pbr.mdl#gltf_material":
        raise ValueError(f"unsupported surface shader {shader_type}: {shader.GetPath()}")

    color_input = shader.GetInput("base_color_factor")
    color = color_input.Get() if color_input else None
    if color is None:
        raise ValueError(f"glTF MDL material has no base_color_factor: {shader.GetPath()}")
    alpha = _float_input(shader, "base_alpha", 1.0)
    texture = None
    texture_source = None
    uv_primvar = None
    shader_types = {shader_type}
    texture_input = shader.GetInput("base_color_texture")
    if texture_input and texture_input.GetConnectedSources()[0]:
        texture_shader = _one_connected_shader(texture_input, description=f"base_color_texture {shader.GetPath()}")
        texture_shader_type = _mdl_shader_type(texture_shader)
        shader_types.add(texture_shader_type)
        if texture_shader_type != "mdl:gltf/pbr.mdl#gltf_texture_lookup":
            raise ValueError(f"unsupported base color texture shader {texture_shader_type}: {texture_shader.GetPath()}")
        texture_path = _asset_path(
            texture_shader.GetInput("texture").Get(),
            stage_path,
            description=f"glTF MDL texture {texture_shader.GetPath()}",
        )
        tex_coord_input = texture_shader.GetInput("tex_coord_index")
        tex_coord_index = int(tex_coord_input.Get() if tex_coord_input else 0)
        uv_primvar = "st" if tex_coord_index == 0 else f"st_{tex_coord_index}"
        texture = _image(texture_path)
        texture_source = str(texture_path)

    return ParsedMaterial(
        path=str(material.GetPath()),
        material=trimesh.visual.material.PBRMaterial(
            name=str(material.GetPath()),
            baseColorFactor=tuple(float(value) for value in color[:3]) + (alpha,),
            baseColorTexture=texture,
            metallicFactor=_float_input(shader, "metallic_factor", 0.0),
            roughnessFactor=_float_input(shader, "roughness_factor", 0.5),
        ),
        uv_primvar=uv_primvar,
        texture_source=texture_source,
        shader_types=tuple(sorted(shader_types)),
    )


def _parse_material(material: UsdShade.Material, stage_path: Path) -> ParsedMaterial:
    if material.GetSurfaceOutput().GetConnectedSources()[0]:
        return _preview_material(material, stage_path)
    if material.GetSurfaceOutput("mdl").GetConnectedSources()[0]:
        return _mdl_gltf_material(material, stage_path)
    outputs = [output.GetFullName() for output in material.GetOutputs()]
    raise ValueError(f"material has no convertible surface output: {material.GetPath()} outputs={outputs}")


def _material_assignments(
    mesh: UsdGeom.Mesh,
) -> list[tuple[UsdShade.Material | None, np.ndarray]]:
    face_count = len(mesh.GetFaceVertexCountsAttr().Get() or [])
    if face_count == 0:
        raise ValueError(f"USD mesh has no faces: {mesh.GetPath()}")
    assignments = []
    covered = np.zeros(face_count, dtype=bool)
    binding_api = UsdShade.MaterialBindingAPI(mesh.GetPrim())
    for subset in binding_api.GetMaterialBindSubsets():
        face_indices = np.asarray(subset.GetIndicesAttr().Get() or [], dtype=np.int64)
        if len(face_indices) == 0:
            raise ValueError(f"empty MaterialSubset: {subset.GetPath()}")
        if np.any(face_indices < 0) or np.any(face_indices >= face_count):
            raise ValueError(f"MaterialSubset face index is out of range: {subset.GetPath()}")
        if np.any(covered[face_indices]):
            raise ValueError(f"overlapping MaterialSubset faces: {subset.GetPath()}")
        material, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
        if not material:
            raise ValueError(f"MaterialSubset has no bound material: {subset.GetPath()}")
        covered[face_indices] = True
        assignments.append((material, face_indices))

    remaining = np.flatnonzero(~covered)
    if len(remaining):
        material, _ = binding_api.ComputeBoundMaterial()
        if not material and assignments:
            raise ValueError(f"mesh has faces outside MaterialSubset bindings: {mesh.GetPath()}")
        assignments.append((material if material else None, remaining))
    return assignments


def _source_usd(asset_dir: Path) -> Path:
    for filename in ("object.usdz", "object.usd", "object.usdc", "object.usda"):
        path = asset_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"asset has no object USD/USDZ: {asset_dir}")


def _usd_collision_mesh(mesh: UsdGeom.Mesh) -> tuple[trimesh.Trimesh, int]:
    vertices, _, _, triangle_records = _mesh_topology(mesh)
    collision = trimesh.Trimesh(
        vertices=vertices,
        faces=triangle_records[:, :3],
        process=False,
        maintain_order=True,
    )
    if not np.all(np.isfinite(collision.vertices)):
        raise ValueError(f"USD collision mesh has non-finite vertices: {mesh.GetPath()}")
    valid_faces = collision.nondegenerate_faces(height=1e-12)
    removed_faces = int(len(valid_faces) - np.count_nonzero(valid_faces))
    if removed_faces:
        collision.update_faces(valid_faces)
        collision.remove_unreferenced_vertices()
    if len(collision.vertices) < 4 or len(collision.faces) < 4:
        raise ValueError(f"USD collision mesh has no usable volume after cleanup: {mesh.GetPath()}")
    return collision, removed_faces


def _coacd_components(
    source_meshes: list[trimesh.Trimesh],
) -> tuple[list[trimesh.Trimesh], str]:
    package_version = importlib.metadata.version("coacd")
    components = []
    for source_index, source in enumerate(source_meshes):
        result = coacd.run_coacd(
            coacd.Mesh(
                np.asarray(source.vertices, dtype=np.float64),
                np.asarray(source.faces, dtype=np.int32),
            ),
            **COACD_PARAMETERS,
        )
        if not result:
            raise RuntimeError(f"CoACD produced no convex hulls for collision mesh {source_index}")
        for hull_index, (vertices, faces) in enumerate(result):
            component = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=True,
            )
            volume = abs(float(component.volume))
            if (
                len(component.vertices) < 4
                or len(component.faces) < 4
                or not component.is_watertight
                or not component.is_volume
                or not np.isfinite(volume)
                or volume <= 0.0
            ):
                raise RuntimeError(f"CoACD produced an invalid convex hull: source={source_index} hull={hull_index}")
            components.append(component)
    return components, package_version


def convert_asset(source_assets_root: Path, output_root: Path, asset_key: str) -> dict[str, Any]:
    source_dir = source_assets_root / "Object" / "RoboDojo" / asset_key
    source_path = _source_usd(source_dir)
    output_dir = output_root / asset_key
    final_manifest = output_dir / "conversion.json"
    source_sha = sha256_file(source_path)
    if final_manifest.is_file():
        existing = json.loads(final_manifest.read_text(encoding="utf-8"))
        if existing.get("converter_version") != CONVERTER_VERSION or existing.get("source_sha256") != source_sha:
            raise RuntimeError(f"stale converted asset must be removed explicitly: {output_dir}")
        for name in ("visual.glb", COLLISION_FILENAME):
            path = output_dir / name
            if not path.is_file() or sha256_file(path) != existing[f"{name}_sha256"]:
                raise RuntimeError(f"converted asset payload is corrupt: {path}")
        return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        extract_dir = temporary_dir / "usd"
        extract_dir.mkdir()
        stage, stage_path = _open_stage(source_path, extract_dir)
        usd_meshes = _visual_meshes(stage)
        usd_collision_meshes = _collision_meshes(stage)
        parsed_materials: dict[str, ParsedMaterial] = {}
        texture_sources = set()
        uv_sets = set()
        shader_types = set()
        visual = trimesh.Scene()
        source_vertices = 0
        source_triangles = 0
        geometry_index = 0
        for usd_mesh in usd_meshes:
            mesh_points, counts, _, triangle_records = _mesh_topology(usd_mesh)
            source_vertices += len(mesh_points)
            source_triangles += len(triangle_records)
            for material, face_indices in _material_assignments(usd_mesh):
                parsed = None
                if material:
                    material_path = str(material.GetPath())
                    if material_path not in parsed_materials:
                        parsed_materials[material_path] = _parse_material(material, stage_path)
                    parsed = parsed_materials[material_path]
                    shader_types.update(parsed.shader_types)
                    if parsed.texture_source is not None:
                        texture_sources.add(parsed.texture_source)
                    if parsed.uv_primvar is not None:
                        uv_sets.add((str(usd_mesh.GetPath()), parsed.uv_primvar))
                vertices, faces, uv = _mesh_arrays(
                    usd_mesh,
                    source_face_indices=face_indices,
                    uv_primvar_name=None if parsed is None else parsed.uv_primvar,
                )
                geometry = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    process=False,
                    maintain_order=True,
                )
                if parsed is not None:
                    geometry.visual = trimesh.visual.texture.TextureVisuals(
                        uv=uv,
                        material=parsed.material,
                    )
                if not np.all(np.isfinite(geometry.vertices)) or len(geometry.faces) == 0:
                    raise ValueError(f"converted visual mesh is invalid: {asset_key} {usd_mesh.GetPath()}")
                geometry_name = f"mesh-{geometry_index:04d}"
                visual.add_geometry(
                    geometry,
                    geom_name=geometry_name,
                    node_name=geometry_name,
                )
                geometry_index += 1
        if not visual.geometry:
            raise ValueError(f"converted visual scene is empty: {asset_key}")
        visual_path = temporary_dir / "visual.glb"
        visual.export(visual_path)

        collision_rows = [_usd_collision_mesh(usd_mesh) for usd_mesh in usd_collision_meshes]
        collision_sources = [row[0] for row in collision_rows]
        collision_degenerate_faces_removed = sum(row[1] for row in collision_rows)
        collision_components, coacd_version = _coacd_components(collision_sources)
        collision = trimesh.util.concatenate(collision_components)
        collision_path = temporary_dir / COLLISION_FILENAME
        collision.export(collision_path, file_type="ply")
        collision_volume = sum(abs(float(component.volume)) for component in collision_components)
        unsupported_shader_types = sorted(shader_types - SUPPORTED_SHADER_TYPES)
        manifest = {
            "converter_version": CONVERTER_VERSION,
            "asset_key": asset_key,
            "source_path": source_path.relative_to(source_assets_root).as_posix(),
            "source_sha256": source_sha,
            "mesh_count": len(usd_meshes),
            "material_count": len(parsed_materials),
            "texture_count": len(texture_sources),
            "uv_set_count": len(uv_sets),
            "unsupported_shader_types": unsupported_shader_types,
            "source_vertices": int(source_vertices),
            "source_triangles": int(source_triangles),
            "collision_source_mesh_count": len(collision_sources),
            "collision_source_vertices": int(sum(len(mesh.vertices) for mesh in collision_sources)),
            "collision_source_triangles": int(sum(len(mesh.faces) for mesh in collision_sources)),
            "collision_source_degenerate_triangles_removed": int(collision_degenerate_faces_removed),
            "collision_decomposition": {
                "algorithm": "coacd",
                "package_version": coacd_version,
                "parameters": COACD_PARAMETERS,
            },
            "collision_hull_count": len(collision_components),
            "collision_vertices": int(len(collision.vertices)),
            "collision_triangles": int(len(collision.faces)),
            "collision_volume": float(collision_volume),
            "visual.glb_sha256": sha256_file(visual_path),
            f"{COLLISION_FILENAME}_sha256": sha256_file(collision_path),
        }
        (temporary_dir / "conversion.json").write_bytes(canonical_json_bytes(manifest))
        os.replace(temporary_dir, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def convert_contract_assets(
    *,
    contract_dir: Path,
    source_assets_root: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    keys = contract_asset_keys(contract_dir)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_asset, source_assets_root, output_root, key): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"asset conversion failed for {key}") from error
            results.append(result)
            print(f"[asset-convert] {len(results)}/{len(keys)} {key}", flush=True)
    results.sort(key=lambda item: item["asset_key"])
    manifest = {
        "converter_version": CONVERTER_VERSION,
        "contract_manifest_sha256": sha256_file(contract_dir / "manifest.json"),
        "asset_count": len(results),
        "mesh_count": sum(item["mesh_count"] for item in results),
        "material_count": sum(item["material_count"] for item in results),
        "texture_count": sum(item["texture_count"] for item in results),
        "uv_set_count": sum(item["uv_set_count"] for item in results),
        "collision_source_mesh_count": sum(item["collision_source_mesh_count"] for item in results),
        "collision_source_degenerate_triangles_removed": sum(
            item["collision_source_degenerate_triangles_removed"] for item in results
        ),
        "collision_hull_count": sum(item["collision_hull_count"] for item in results),
        "collision_vertices": sum(item["collision_vertices"] for item in results),
        "collision_triangles": sum(item["collision_triangles"] for item in results),
        "collision_volume": sum(item["collision_volume"] for item in results),
        "collision_decomposition": {
            "algorithm": "coacd",
            "package_version": importlib.metadata.version("coacd"),
            "parameters": COACD_PARAMETERS,
        },
        "unsupported_shader_types": sorted(
            {shader_type for item in results for shader_type in item["unsupported_shader_types"]}
        ),
        "assets": results,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".manifest.{os.getpid()}.tmp"
    temporary.write_bytes(canonical_json_bytes(manifest))
    os.replace(temporary, output_root / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--source-assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    manifest = convert_contract_assets(
        contract_dir=args.contract_dir.resolve(),
        source_assets_root=args.source_assets_root.resolve(),
        output_root=args.output_root.resolve(),
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
