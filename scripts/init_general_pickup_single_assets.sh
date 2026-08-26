#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_CACHE_DIR="${ASSET_CACHE_DIR:-/tmp/robodojo_assets_repo}"
LFS_CONCURRENT_TRANSFERS="${LFS_CONCURRENT_TRANSFERS:-32}"
LAYOUT_COUNT="${LAYOUT_COUNT:-10}"
LAYOUT_DIR="Assets/Eval_Layout/RoboDojo/arx_x5/0"

command -v jq >/dev/null 2>&1 || {
    echo "[task_assets] jq is required" >&2
    exit 1
}
git -C "${ASSET_CACHE_DIR}" rev-parse --is-inside-work-tree >/dev/null
git -C "${ASSET_CACHE_DIR}" lfs install --local >/dev/null
git -C "${ASSET_CACHE_DIR}" config lfs.concurrenttransfers "${LFS_CONCURRENT_TRANSFERS}"

# Layout JSONs are small and must be materialized before extracting exact object instances.
git -C "${ASSET_CACHE_DIR}" lfs pull \
    --include="${LAYOUT_DIR}/general_pickup_*.json" --exclude=""

mapfile -t layout_files < <(
    find "${ASSET_CACHE_DIR}/${LAYOUT_DIR}" -maxdepth 1 -type f -name 'general_pickup_*.json' \
        -printf '%p\n' | sort -V | head -n "${LAYOUT_COUNT}"
)
[[ "${#layout_files[@]}" -eq "${LAYOUT_COUNT}" ]] || {
    echo "[task_assets] expected ${LAYOUT_COUNT} layouts, found ${#layout_files[@]}" >&2
    exit 1
}

includes=(
    "Assets/Background/brown_photostudio_02_4k.hdr"
    "Assets/Material/material_0122/**"
    "Assets/Material/material_0564/**"
    "Assets/Room/Simple_Room_nolight/**"
    "Assets/Robots/x5/**"
    "Assets/Object/RoboDojo/Clutter/clutter.yml"
)

for layout_file in "${layout_files[@]}"; do
    while IFS=$'\t' read -r object_type category category_idx; do
        printf -v padded_idx '%05d' "${category_idx}"
        includes+=("Assets/Object/RoboDojo/${object_type}/${category}/${padded_idx}/**")
    done < <(
        jq -r '
            to_entries[]
            | select(.key == "Rigid" or .key == "Geometry")
            | .key as $object_type
            | .value
            | to_entries[]
            | .key as $category
            | .value[]
            | [if .type == "cluttered" then "Clutter" else $object_type end,
               $category,
               .category_idx]
            | @tsv
        ' "${layout_file}"
    )
done

include_csv="$(printf '%s\n' "${includes[@]}" | sort -u | paste -sd, -)"
git -C "${ASSET_CACHE_DIR}" lfs pull --include="${include_csv}" --exclude=""

target="${ROOT_DIR}/Assets"
[[ ! -e "${target}" ]] || {
    echo "[task_assets] target already exists: ${target}" >&2
    exit 1
}
ln -s "${ASSET_CACHE_DIR}/Assets" "${target}"

for required in Robots/x5 Object/RoboDojo Material Room Eval_Layout/RoboDojo/arx_x5; do
    [[ -d "${target}/${required}" ]] || {
        echo "[task_assets] missing required directory: ${target}/${required}" >&2
        exit 1
    }
done

echo "[task_assets] ready: ${target} (${#includes[@]} requested paths)"
