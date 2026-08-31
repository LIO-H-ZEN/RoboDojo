#!/usr/bin/env python3
"""Export deterministic, simulator-neutral PIPER General Pickup contracts.

The exporter intentionally does not import Isaac Sim.  It operates on the
saved evaluation layouts and hydrated RoboDojo assets so it can be used by
ManiSkill data-generation workers and by lightweight CI jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

OBJECT_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
CONTRACT_VERSION = "robodojo_general_pickup_piper_mvp_v1"

PIPER_BASE_ROBODOJO = (0.0, -0.45, 0.765)
PIPER_BASE_MANISKILL = (-0.35, 0.0, 0.0)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_hydrated_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required asset is missing: {path}")
    with path.open("rb") as source:
        if source.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            raise ValueError(f"required asset is an unresolved Git LFS pointer: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    require_hydrated_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    require_hydrated_file(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def parse_layout_ids(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"invalid layout ID expression: {raw!r}")
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2 or not all(value.isdigit() for value in bounds):
                raise ValueError(f"invalid layout ID range: {part!r}")
            start, end = map(int, bounds)
            if end < start:
                raise ValueError(f"layout ID range is reversed: {part!r}")
            values.extend(range(start, end + 1))
        elif part.isdigit():
            values.append(int(part))
        else:
            raise ValueError(f"invalid layout ID: {part!r}")
    if len(values) != len(set(values)):
        raise ValueError(f"layout IDs must be unique: {values}")
    return values


def quat_multiply(lhs: Sequence[float], rhs: Sequence[float]) -> list[float]:
    lw, lx, ly, lz = map(float, lhs)
    rw, rx, ry, rz = map(float, rhs)
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def pose_robodojo_to_maniskill(position: Sequence[float], orientation: Sequence[float]) -> dict[str, list[float]]:
    if len(position) != 3 or len(orientation) != 4:
        raise ValueError(f"invalid pose: position={position} orientation={orientation}")
    dx = float(position[0]) - PIPER_BASE_ROBODOJO[0]
    dy = float(position[1]) - PIPER_BASE_ROBODOJO[1]
    dz = float(position[2]) - PIPER_BASE_ROBODOJO[2]
    position_maniskill = [
        PIPER_BASE_MANISKILL[0] + dy,
        PIPER_BASE_MANISKILL[1] - dx,
        PIPER_BASE_MANISKILL[2] + dz,
    ]
    half = math.sqrt(0.5)
    orientation_maniskill = quat_multiply((half, 0.0, 0.0, -half), orientation)
    return {"position": position_maniskill, "orientation_wxyz": orientation_maniskill}


def iter_layout_objects(layout: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for object_type in OBJECT_TYPES:
        categories = layout.get(object_type, {})
        if not isinstance(categories, dict):
            raise ValueError(f"layout field {object_type} must be a mapping")
        for category, instances in categories.items():
            if not isinstance(instances, list):
                raise ValueError(f"layout field {object_type}.{category} must be a list")
            for instance in instances:
                if not isinstance(instance, dict):
                    raise ValueError(f"layout object {object_type}.{category} must be a mapping")
                yield object_type, str(category), instance


def layout_is_allowed(layout: dict[str, Any]) -> bool:
    targets = [instance for _, _, instance in iter_layout_objects(layout) if instance.get("label") == "target"]
    if len(targets) != 1:
        raise ValueError(f"expected exactly one target, got {len(targets)}")
    position = targets[0].get("default_pos")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError(f"target has invalid default_pos: {position}")
    return -0.25 <= float(position[0]) <= 0.25 and -0.25 <= float(position[1]) <= 0.0


def filtered_layouts(assets_root: Path, seed: int) -> list[Path]:
    layout_dir = assets_root / "Eval_Layout" / "RoboDojo" / "arx_x5" / str(seed)
    if not layout_dir.is_dir():
        raise FileNotFoundError(f"layout directory is missing: {layout_dir}")
    candidates = sorted(
        layout_dir.glob("general_pickup_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    selected = [path for path in candidates if layout_is_allowed(load_json(path))]
    if not selected:
        raise ValueError(f"no filtered General Pickup layouts found in {layout_dir}")
    return selected


def object_asset_dir(assets_root: Path, object_type: str, category: str, instance: dict[str, Any]) -> Path:
    category_idx = instance.get("category_idx")
    if not isinstance(category_idx, int) or category_idx < 0:
        raise ValueError(f"invalid category_idx for {category}: {category_idx}")
    asset_type = "Clutter" if instance.get("type") == "cluttered" else object_type
    return assets_root / "Object" / "RoboDojo" / asset_type / category / f"{category_idx:05d}"


def hash_asset_directory(asset_dir: Path, assets_root: Path) -> list[dict[str, Any]]:
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"object asset directory is missing: {asset_dir}")
    files = []
    for path in sorted(candidate for candidate in asset_dir.rglob("*") if candidate.is_file()):
        require_hydrated_file(path)
        files.append(
            {
                "path": path.relative_to(assets_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise FileNotFoundError(f"object asset directory is empty: {asset_dir}")
    return files


def read_descriptions(asset_dir: Path) -> list[str]:
    payload = load_json(asset_dir / "description.json")
    descriptions = payload.get("description")
    if isinstance(descriptions, str):
        descriptions = [descriptions]
    if not isinstance(descriptions, list):
        raise ValueError(f"description must be a string or list: {asset_dir / 'description.json'}")
    values = [str(value).strip() for value in descriptions if str(value).strip()]
    if not values:
        raise ValueError(f"object has no non-empty descriptions: {asset_dir}")
    return values


def export_object(
    assets_root: Path,
    object_type: str,
    category: str,
    instance: dict[str, Any],
    *,
    require_descriptions: bool,
) -> dict[str, Any]:
    asset_dir = object_asset_dir(assets_root, object_type, category, instance)
    metadata_path = asset_dir / "metadata.json"
    metadata = load_json(metadata_path)
    geometry = metadata.get("geometry")
    if not isinstance(geometry, dict) or not geometry:
        raise ValueError(f"object metadata has no geometry: {metadata_path}")
    position = instance.get("default_pos")
    orientation = instance.get("default_ori")
    if not isinstance(position, list) or not isinstance(orientation, list):
        raise ValueError(f"object {category} has no complete default pose")
    description_path = asset_dir / "description.json"
    descriptions = read_descriptions(asset_dir) if require_descriptions or description_path.is_file() else []
    return {
        "object_type": object_type,
        "asset_type": "Clutter" if instance.get("type") == "cluttered" else object_type,
        "category": category,
        "category_idx": int(instance["category_idx"]),
        "label": instance.get("label"),
        "descriptions": descriptions,
        "instruction_candidates": [f"Pick up the {description} by 10 cm." for description in descriptions],
        "scale": instance.get("scale", [1.0, 1.0, 1.0]),
        "physics": instance.get("physics", metadata.get("physics", {})),
        "pose_robodojo": {"position": position, "orientation_wxyz": orientation},
        "pose_maniskill": pose_robodojo_to_maniskill(position, orientation),
        "metadata": metadata,
        "source_files": hash_asset_directory(asset_dir, assets_root),
    }


def load_static_contract(
    repo_root: Path,
    assets_root: Path,
    piper_assets_root: Path,
) -> dict[str, Any]:
    config_paths = {
        "environment": repo_root / "env_cfg" / "piper_single.yml",
        "simulation": repo_root / "env_cfg" / "sim" / "sim_config_piper_policy.yml",
        "cameras": repo_root / "env_cfg" / "camera" / "camera_config_piper_policy.yml",
        "scene": repo_root / "env_cfg" / "scene" / "single_arm.yml",
        "robot": repo_root / "env_cfg" / "robot" / "single_piper.yml",
        "task": repo_root / "task" / "RoboDojo" / "config" / "general_pickup.yml",
    }
    configs = {name: load_yaml(path) for name, path in config_paths.items()}
    piper_dir = piper_assets_root / "Robots" / "piper"
    robot_config_path = piper_dir / "robot_config.yml"
    configs["piper_asset"] = load_yaml(robot_config_path)
    source_files = [
        {"path": path.relative_to(repo_root).as_posix(), "sha256": sha256_file(path)} for path in config_paths.values()
    ]
    source_files.append(
        {
            "path": robot_config_path.relative_to(piper_assets_root).as_posix(),
            "sha256": sha256_file(robot_config_path),
        }
    )
    return {
        "configs": configs,
        "source_files": source_files,
        "piper_source_files": hash_asset_directory(piper_dir, piper_assets_root),
        "coordinate_contract": {
            "robodojo_piper_base_position": list(PIPER_BASE_ROBODOJO),
            "maniskill_piper_base_position": list(PIPER_BASE_MANISKILL),
            "robodojo_R_maniskill": "Rz(+90deg)",
            "quaternion_order": "wxyz",
        },
        "control": {
            "physics_hz": 200,
            "policy_hz": 20,
            "interpolation_steps": 8,
            "hold_steps": 2,
            "action_horizon": 16,
            "execute_steps": 4,
            "max_policy_steps": 200,
        },
        "task": {
            "instruction_template": "Pick up the <target> by 10 cm.",
            "success": "target_delta_z > 0.1",
        },
    }


def export_layout_contract(
    *,
    repo_root: Path,
    assets_root: Path,
    filtered_layout_id: int,
    source_layout_path: Path,
    expected_clutter_count: int,
    static_contract: dict[str, Any],
) -> dict[str, Any]:
    layout = load_json(source_layout_path)
    targets = []
    clutter = []
    for object_type, category, instance in iter_layout_objects(layout):
        if instance.get("label") == "target":
            targets.append(
                export_object(
                    assets_root,
                    object_type,
                    category,
                    instance,
                    require_descriptions=True,
                )
            )
        elif instance.get("type") == "cluttered":
            clutter.append(
                export_object(
                    assets_root,
                    object_type,
                    category,
                    instance,
                    require_descriptions=False,
                )
            )
    if len(targets) != 1:
        raise ValueError(f"layout {source_layout_path} has {len(targets)} targets")
    categories = [item["category"] for item in clutter]
    if len(categories) != len(set(categories)):
        raise ValueError(f"layout {source_layout_path} violates category-unique clutter: {categories}")
    if not clutter:
        raise ValueError(f"layout {source_layout_path} has no clutter")
    contract = {
        "contract_version": CONTRACT_VERSION,
        "filtered_layout_id": filtered_layout_id,
        "source_layout": {
            "path": source_layout_path.relative_to(assets_root).as_posix(),
            "sha256": sha256_file(source_layout_path),
        },
        "expected_clutter_count": expected_clutter_count,
        "source_clutter_count": len(clutter),
        "source_clutter_count_matches_config": len(clutter) == expected_clutter_count,
        "target": targets[0],
        "clutter": clutter,
        "fixtures": {key: layout.get(key) for key in ("Room", "Table", "Ground", "Background", "Light")},
        "static_contract": static_contract,
    }
    contract["contract_sha256"] = sha256_bytes(canonical_json_bytes(contract))
    return contract


def write_contracts(
    *,
    repo_root: Path,
    assets_root: Path,
    output_dir: Path,
    layout_ids: Sequence[int],
    seed: int,
    expected_clutter_count: int,
    piper_assets_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    assets_root = assets_root.resolve()
    piper_assets_root = assets_root if piper_assets_root is None else piper_assets_root.resolve()
    output_dir = output_dir.resolve()
    selected = filtered_layouts(assets_root, seed)
    invalid = [layout_id for layout_id in layout_ids if layout_id < 0 or layout_id >= len(selected)]
    if invalid:
        raise ValueError(f"filtered layout IDs out of range: {invalid}; available=0..{len(selected) - 1}")
    static_contract = load_static_contract(repo_root, assets_root, piper_assets_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for layout_id in layout_ids:
        contract = export_layout_contract(
            repo_root=repo_root,
            assets_root=assets_root,
            filtered_layout_id=layout_id,
            source_layout_path=selected[layout_id],
            expected_clutter_count=expected_clutter_count,
            static_contract=static_contract,
        )
        filename = f"general_pickup_filtered_{layout_id:03d}.json"
        path = output_dir / filename
        path.write_bytes(canonical_json_bytes(contract))
        rows.append(
            {
                "filtered_layout_id": layout_id,
                "contract_file": filename,
                "contract_sha256": contract["contract_sha256"],
                "source_layout": contract["source_layout"],
                "target": {
                    "category": contract["target"]["category"],
                    "category_idx": contract["target"]["category_idx"],
                },
                "source_clutter_count": contract["source_clutter_count"],
                "source_clutter_count_matches_config": contract["source_clutter_count_matches_config"],
            }
        )
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "assets_root": str(assets_root),
        "piper_assets_root": str(piper_assets_root),
        "seed": seed,
        "layout_ids": list(layout_ids),
        "contracts": rows,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    (output_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=repo_root / ".cache" / "robodojo_assets_repo" / "Assets",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--piper-assets-root",
        type=Path,
        help="optional separate Assets root containing Robots/piper",
    )
    parser.add_argument("--layout-ids", default="0-9", help="filtered IDs, e.g. 0-9 or 0,2,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-clutter-count", type=int, default=10)
    args = parser.parse_args()
    manifest = write_contracts(
        repo_root=repo_root,
        assets_root=args.assets_root,
        output_dir=args.output_dir,
        layout_ids=parse_layout_ids(args.layout_ids),
        seed=args.seed,
        expected_clutter_count=args.expected_clutter_count,
        piper_assets_root=args.piper_assets_root,
    )
    mismatches = [row for row in manifest["contracts"] if not row["source_clutter_count_matches_config"]]
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if mismatches:
        print(
            "[contract-export] WARNING source layouts differ from configured clutter count: "
            + ", ".join(
                f"filtered={row['filtered_layout_id']} count={row['source_clutter_count']}" for row in mismatches
            )
        )


if __name__ == "__main__":
    main()
