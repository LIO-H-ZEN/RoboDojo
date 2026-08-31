from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import trimesh

from scripts.internal.convert_general_pickup_assets import (
    COACD_PARAMETERS,
    CONVERTER_VERSION,
    convert_asset,
    convert_contract_assets,
)


def _tetrahedron(stage: Usd.Stage, path: str, *, x_offset: float = 0.0) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            (x_offset - 0.01, -0.01, -0.01),
            (x_offset + 0.01, -0.01, -0.01),
            (x_offset, 0.01, -0.01),
            (x_offset, 0.0, 0.01),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
    return mesh


def _preview_material(
    stage: Usd.Stage,
    path: str,
    *,
    color: tuple[float, float, float] | None = None,
    texture_path: str | None = None,
    uv_name: str = "st",
    texture_shader_name: str = "albedoSampler",
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    preview = UsdShade.Shader.Define(stage, f"{path}/surfaceShader")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    diffuse = preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if texture_path is None:
        if color is None:
            raise ValueError("constant preview material requires color")
        diffuse.Set(Gf.Vec3f(*color))
        return material

    texture = UsdShade.Shader.Define(stage, f"{path}/{texture_shader_name}")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    reader = UsdShade.Shader.Define(stage, f"{path}/textureCoordinates")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(uv_name)
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    diffuse.ConnectToSource(texture.ConnectableAPI(), "rgb")
    return material


def _bind(mesh: UsdGeom.Mesh, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _make_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _collision_tetrahedron(
    stage: Usd.Stage, path: str = "/root/collision/model", *, x_offset: float = 0.0
) -> UsdGeom.Mesh:
    mesh = _tetrahedron(stage, path, x_offset=x_offset)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexDecomposition)
    return mesh


def _pack_usdz(
    stage: Usd.Stage,
    output_path: Path,
    *resources: Path,
    add_default_collision: bool = True,
) -> None:
    if add_default_collision and not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse()):
        _collision_tetrahedron(stage)
    stage.GetRootLayer().Save()
    stage_path = Path(stage.GetRootLayer().realPath)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(stage_path, stage_path.name)
        for resource in resources:
            archive.write(resource, resource.name)
    stage_path.unlink()


def _asset_paths(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    asset_dir = source_root / "Object" / "RoboDojo" / "Rigid" / "test" / "00000"
    asset_dir.mkdir(parents=True)
    return source_root, asset_dir


def _load_scene(path: Path) -> trimesh.Scene:
    scene = trimesh.load(path, force="scene", process=False)
    assert isinstance(scene, trimesh.Scene)
    return scene


def make_usdz(path: Path) -> None:
    stage_path = path.parent / "object.usda"
    stage = _make_stage(stage_path)
    _tetrahedron(stage, "/root/visual/model")
    _collision_tetrahedron(stage)
    _pack_usdz(stage, path, add_default_collision=False)


@pytest.mark.parametrize("uv_name", ["st", "uv"])
def test_preview_texture_follows_shader_connections_and_uv_primvar(tmp_path, uv_name):
    source_root, asset_dir = _asset_paths(tmp_path)
    stage_path = asset_dir / "object.usda"
    stage = _make_stage(stage_path)
    mesh = _tetrahedron(stage, "/root/visual/model")
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        uv_name, Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    primvar.Set([(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.5, 0.5)])
    texture_path = asset_dir / "albedo.png"
    Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(texture_path)
    material = _preview_material(
        stage,
        "/root/Looks/material",
        texture_path=texture_path.name,
        uv_name=uv_name,
        texture_shader_name="arbitraryTextureNode",
    )
    _bind(mesh, material)
    _pack_usdz(stage, asset_dir / "object.usdz", texture_path)

    manifest = convert_asset(source_root, tmp_path / "converted", "Rigid/test/00000")
    scene = _load_scene(tmp_path / "converted/Rigid/test/00000/visual.glb")

    assert manifest["mesh_count"] == 1
    assert manifest["material_count"] == 1
    assert manifest["texture_count"] == 1
    assert manifest["uv_set_count"] == 1
    assert manifest["unsupported_shader_types"] == []
    geometry = next(iter(scene.geometry.values()))
    assert geometry.visual.material.baseColorTexture is not None
    assert geometry.visual.uv.shape == (4, 2)


def test_preview_constant_diffuse_color_is_preserved(tmp_path):
    source_root, asset_dir = _asset_paths(tmp_path)
    stage = _make_stage(asset_dir / "object.usda")
    mesh = _tetrahedron(stage, "/root/visual/model")
    _bind(
        mesh,
        _preview_material(stage, "/root/Looks/red", color=(0.25, 0.5, 0.75)),
    )
    _pack_usdz(stage, asset_dir / "object.usdz")

    manifest = convert_asset(source_root, tmp_path / "converted", "Rigid/test/00000")
    scene = _load_scene(tmp_path / "converted/Rigid/test/00000/visual.glb")

    assert manifest["material_count"] == 1
    assert manifest["texture_count"] == 0
    material = next(iter(scene.geometry.values())).visual.material
    np.testing.assert_allclose(
        np.asarray(material.baseColorFactor[:3], dtype=np.float64) / 255.0,
        [0.25, 0.5, 0.75],
        atol=1 / 255,
    )


def test_indexed_face_varying_uv_is_expanded(tmp_path):
    source_root, asset_dir = _asset_paths(tmp_path)
    stage = _make_stage(asset_dir / "object.usda")
    mesh = _tetrahedron(stage, "/root/visual/model")
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set([(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
    primvar.SetIndices([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    texture_path = asset_dir / "indexed.png"
    Image.new("RGBA", (2, 2), (40, 50, 60, 255)).save(texture_path)
    _bind(
        mesh,
        _preview_material(
            stage,
            "/root/Looks/indexed",
            texture_path=texture_path.name,
            uv_name="st",
        ),
    )
    _pack_usdz(stage, asset_dir / "object.usdz", texture_path)

    convert_asset(source_root, tmp_path / "converted", "Rigid/test/00000")
    scene = _load_scene(tmp_path / "converted/Rigid/test/00000/visual.glb")

    geometry = next(iter(scene.geometry.values()))
    assert len(geometry.vertices) == 12
    assert geometry.visual.uv.shape == (12, 2)


def test_multiple_meshes_and_material_subsets_are_preserved(tmp_path):
    source_root, asset_dir = _asset_paths(tmp_path)
    stage = _make_stage(asset_dir / "object.usda")
    first = _tetrahedron(stage, "/root/visual/first", x_offset=-0.04)
    second = _tetrahedron(stage, "/root/visual/second", x_offset=0.04)
    red = _preview_material(stage, "/root/Looks/red", color=(1.0, 0.0, 0.0))
    green = _preview_material(stage, "/root/Looks/green", color=(0.0, 1.0, 0.0))
    blue = _preview_material(stage, "/root/Looks/blue", color=(0.0, 0.0, 1.0))
    _bind(first, red)
    binding = UsdShade.MaterialBindingAPI.Apply(second.GetPrim())
    green_subset = binding.CreateMaterialBindSubset("green", [0, 1])
    blue_subset = binding.CreateMaterialBindSubset("blue", [2, 3])
    UsdShade.MaterialBindingAPI.Apply(green_subset.GetPrim()).Bind(green)
    UsdShade.MaterialBindingAPI.Apply(blue_subset.GetPrim()).Bind(blue)
    _pack_usdz(stage, asset_dir / "object.usdz")

    manifest = convert_asset(source_root, tmp_path / "converted", "Rigid/test/00000")
    scene = _load_scene(tmp_path / "converted/Rigid/test/00000/visual.glb")

    assert manifest["mesh_count"] == 2
    assert manifest["material_count"] == 3
    assert len(scene.geometry) == 3


def test_bound_material_with_unsupported_surface_shader_fails(tmp_path):
    source_root, asset_dir = _asset_paths(tmp_path)
    stage = _make_stage(asset_dir / "object.usda")
    mesh = _tetrahedron(stage, "/root/visual/model")
    material = UsdShade.Material.Define(stage, "/root/Looks/unsupported")
    shader = UsdShade.Shader.Define(stage, "/root/Looks/unsupported/custom")
    shader.CreateIdAttr("CompanyCustomSurface")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _bind(mesh, material)
    _pack_usdz(stage, asset_dir / "object.usdz")

    with pytest.raises(ValueError, match="unsupported surface shader.*CompanyCustomSurface"):
        convert_asset(source_root, tmp_path / "converted", "Rigid/test/00000")


def test_convert_asset_writes_authored_coacd_collision(tmp_path):
    source_root = tmp_path / "source"
    asset_dir = source_root / "Object" / "RoboDojo" / "Rigid" / "test" / "00000"
    asset_dir.mkdir(parents=True)
    make_usdz(asset_dir / "object.usdz")

    output_root = tmp_path / "converted"
    manifest = convert_asset(source_root, output_root, "Rigid/test/00000")

    visual_path = output_root / "Rigid" / "test" / "00000" / "visual.glb"
    collision_path = output_root / "Rigid" / "test" / "00000" / "collision.ply"
    assert visual_path.is_file()
    assert collision_path.is_file()
    assert manifest["source_vertices"] == 4
    assert manifest["collision_source_mesh_count"] == 1
    assert manifest["collision_decomposition"]["parameters"] == COACD_PARAMETERS
    assert manifest["collision_hull_count"] >= 1
    collision = trimesh.load(collision_path, force="mesh")
    assert collision.is_watertight
    assert collision.is_volume
    assert np.all(np.isfinite(collision.vertices))


def test_convert_asset_uses_collision_api_mesh_instead_of_visual_mesh(tmp_path):
    source_root, asset_dir = _asset_paths(tmp_path)
    stage = _make_stage(asset_dir / "object.usda")
    _tetrahedron(stage, "/root/visual/left", x_offset=-0.10)
    _tetrahedron(stage, "/root/visual/right", x_offset=0.10)
    _collision_tetrahedron(stage, x_offset=0.03)
    _pack_usdz(stage, asset_dir / "object.usdz", add_default_collision=False)

    manifest = convert_asset(source_root, tmp_path / "converted", "Rigid/test/00000")
    collision = trimesh.load(
        tmp_path / "converted/Rigid/test/00000/collision.ply",
        force="mesh",
        process=False,
    )

    assert manifest["mesh_count"] == 2
    assert manifest["collision_source_mesh_count"] == 1
    assert np.allclose(np.asarray(collision.centroid)[0], 0.03, atol=1e-3)


def test_convert_asset_requires_explicit_supported_collision_mesh(tmp_path):
    source_root, asset_dir = _asset_paths(tmp_path)
    stage = _make_stage(asset_dir / "object.usda")
    _tetrahedron(stage, "/root/visual/model")
    _pack_usdz(stage, asset_dir / "object.usdz", add_default_collision=False)

    with pytest.raises(ValueError, match="no enabled USD collision mesh"):
        convert_asset(source_root, tmp_path / "converted", "Rigid/test/00000")


def test_convert_asset_rejects_unsupported_collision_approximation(tmp_path):
    source_root, asset_dir = _asset_paths(tmp_path)
    stage = _make_stage(asset_dir / "object.usda")
    _tetrahedron(stage, "/root/visual/model")
    collision = _tetrahedron(stage, "/root/collision/model")
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    _pack_usdz(stage, asset_dir / "object.usdz", add_default_collision=False)

    with pytest.raises(ValueError, match="unsupported collision approximation.*convexHull"):
        convert_asset(source_root, tmp_path / "converted", "Rigid/test/00000")


def test_convert_asset_removes_and_records_degenerate_collision_faces(tmp_path):
    source_root, asset_dir = _asset_paths(tmp_path)
    stage = _make_stage(asset_dir / "object.usda")
    _tetrahedron(stage, "/root/visual/model")
    collision = _collision_tetrahedron(stage)
    counts = list(collision.GetFaceVertexCountsAttr().Get())
    indices = list(collision.GetFaceVertexIndicesAttr().Get())
    collision.GetFaceVertexCountsAttr().Set([*counts, 3])
    collision.GetFaceVertexIndicesAttr().Set([*indices, 0, 0, 1])
    _pack_usdz(stage, asset_dir / "object.usdz", add_default_collision=False)

    manifest = convert_asset(source_root, tmp_path / "converted", "Rigid/test/00000")

    assert manifest["collision_source_degenerate_triangles_removed"] == 1
    assert manifest["collision_source_triangles"] == 4


def test_convert_asset_reuses_matching_cache(tmp_path):
    source_root = tmp_path / "source"
    asset_dir = source_root / "Object" / "RoboDojo" / "Rigid" / "test" / "00000"
    asset_dir.mkdir(parents=True)
    make_usdz(asset_dir / "object.usdz")
    output_root = tmp_path / "converted"

    first = convert_asset(source_root, output_root, "Rigid/test/00000")
    second = convert_asset(source_root, output_root, "Rigid/test/00000")

    assert first == second
    stored = json.loads((output_root / "Rigid" / "test" / "00000" / "conversion.json").read_text(encoding="utf-8"))
    assert stored == first


def test_contract_manifest_records_aggregate_material_statistics(tmp_path):
    source_root, asset_dir = _asset_paths(tmp_path)
    make_usdz(asset_dir / "object.usdz")
    contract_dir = tmp_path / "contracts"
    contract_dir.mkdir()
    contract = {
        "target": {
            "asset_type": "Rigid",
            "category": "test",
            "category_idx": 0,
        },
        "clutter": [],
    }
    (contract_dir / "layout.json").write_text(json.dumps(contract), encoding="utf-8")
    (contract_dir / "manifest.json").write_text(
        json.dumps({"contracts": [{"contract_file": "layout.json"}]}),
        encoding="utf-8",
    )

    manifest = convert_contract_assets(
        contract_dir=contract_dir,
        source_assets_root=source_root,
        output_root=tmp_path / "converted",
        workers=2,
    )

    assert manifest["converter_version"] == CONVERTER_VERSION
    assert manifest["mesh_count"] == 1
    assert manifest["material_count"] == 0
    assert manifest["texture_count"] == 0
    assert manifest["uv_set_count"] == 0
    assert manifest["unsupported_shader_types"] == []
    assert manifest["collision_decomposition"]["parameters"] == COACD_PARAMETERS
    assert manifest["collision_hull_count"] >= 1
    assert manifest["collision_source_degenerate_triangles_removed"] == 0
