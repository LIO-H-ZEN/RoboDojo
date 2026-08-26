#!/usr/bin/env bash
# One-command PI-0.5 PIPER zero-shot evaluation on an existing GPU worker.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/Pi_05_Piper_LiftAnything_Native"
CHECKPOINT="/tmp/checkpoints/pi05_piper_liftanything_e500/seed42_20260821/9999"
POLICY_ENV="/tmp/openpi-pi05-uv"
EVAL_ENV="/tmp/robodojo-uv"
ASSETS_ARCHIVE="/tmp/pi05_piper_layouts0_9_assets.tar"
ASSETS_ARCHIVE_EXPLICIT="false"
ASSETS_ROOT=""
COMMON_ASSETS="/tmp/robodojo_assets_repo/Assets"
LAYOUT_IDS="0,1,2,3,4,5,6,7,8,9"
RUN_ID="pi05_piper_zero_shot_$(date +%Y%m%d_%H%M%S)_$$"
POLICY_GPU="0"
ENV_GPU="0"
SEED="0"
MIN_FREE_GPU_MIB="30000"
RUNTIME_BASE="${TMPDIR:-/tmp}"
HDFS_OUTPUT=""
ALLOW_DIRTY="false"
DRY_RUN="false"
KEEP_RUNTIME="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_pi05_piper_zero_shot.sh [options]

Runs the native PI-0.5 PIPER policy on general_pickup_single layouts 0-9.
Execute it on an existing GPU worker after creating the uv environments and
the hydrated assets described in README_single_arm_pickup.md.

Inputs:
  --ckpt PATH              Checkpoint directory
  --policy-dir PATH        Native PIPER policy directory
  --policy-env PATH        uv policy environment (contains bin/python)
  --eval-env PATH          uv RoboDojo environment (contains bin/python)
  --assets-root PATH       Use one already-hydrated Assets directory directly
  --assets-archive PATH    Local tar archive, or hdfs:// URI, containing Assets/
  --common-assets PATH     Hydrated common Assets root used for missing top-level dirs
  --layout-ids IDS         Comma-separated filtered layout IDs (default: 0,...,9)
  --run-id ID              Unique evaluation ID
  --seed N                 Layout seed (default: 0)
  --policy-gpu ID          Policy GPU (default: 0)
  --env-gpu ID             Simulator GPU (default: 0)
  --min-free-gpu-mib N     Required free memory on every selected GPU (default: 30000)
  --runtime-base PATH      Parent for the isolated current-HEAD checkout (default: /tmp)
  --hdfs-output URI        Also upload a result tar and SHA256 file to this HDFS directory

Control:
  --dry-run                Prepare and validate everything, then print the eval command
  --keep-runtime           Keep the isolated checkout after exit
  --allow-dirty            Permit source changes; the runtime still uses committed HEAD
  -h, --help               Show this help

All defaults can be overridden explicitly; there is no automatic asset or
checkpoint fallback. One policy+simulator pair runs sequentially per GPU.
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[pi05-piper] missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) need_value "$@"; CHECKPOINT="$2"; shift 2 ;;
    --policy-dir) need_value "$@"; POLICY_DIR="$2"; shift 2 ;;
    --policy-env) need_value "$@"; POLICY_ENV="$2"; shift 2 ;;
    --eval-env) need_value "$@"; EVAL_ENV="$2"; shift 2 ;;
    --assets-root) need_value "$@"; ASSETS_ROOT="$2"; shift 2 ;;
    --assets-archive) need_value "$@"; ASSETS_ARCHIVE="$2"; ASSETS_ARCHIVE_EXPLICIT="true"; shift 2 ;;
    --common-assets) need_value "$@"; COMMON_ASSETS="$2"; shift 2 ;;
    --layout-ids) need_value "$@"; LAYOUT_IDS="$2"; shift 2 ;;
    --run-id) need_value "$@"; RUN_ID="$2"; shift 2 ;;
    --seed) need_value "$@"; SEED="$2"; shift 2 ;;
    --policy-gpu) need_value "$@"; POLICY_GPU="$2"; shift 2 ;;
    --env-gpu) need_value "$@"; ENV_GPU="$2"; shift 2 ;;
    --min-free-gpu-mib) need_value "$@"; MIN_FREE_GPU_MIB="$2"; shift 2 ;;
    --runtime-base) need_value "$@"; RUNTIME_BASE="$2"; shift 2 ;;
    --hdfs-output) need_value "$@"; HDFS_OUTPUT="${2%/}"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --keep-runtime) KEEP_RUNTIME="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[pi05-piper] unknown argument: $1" >&2; exit 2 ;;
  esac
done

for numeric in "${SEED}" "${POLICY_GPU}" "${ENV_GPU}" "${MIN_FREE_GPU_MIB}"; do
  if [[ ! "${numeric}" =~ ^[0-9]+$ ]]; then
    echo "[pi05-piper] expected a non-negative integer, got: ${numeric}" >&2
    exit 2
  fi
done
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[pi05-piper] run id may contain only letters, digits, dot, underscore, and dash: ${RUN_ID}" >&2
  exit 2
fi
if [[ ! "${LAYOUT_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[pi05-piper] invalid --layout-ids: ${LAYOUT_IDS}" >&2
  exit 2
fi
if [[ -n "${ASSETS_ROOT}" && "${ASSETS_ARCHIVE_EXPLICIT}" == "true" ]]; then
  echo "[pi05-piper] --assets-root and --assets-archive are mutually exclusive" >&2
  exit 2
fi

command -v git >/dev/null 2>&1 || { echo "[pi05-piper] git is required" >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo "[pi05-piper] tar is required" >&2; exit 1; }
[[ -d "${RUNTIME_BASE}" ]] || { echo "[pi05-piper] runtime parent is missing: ${RUNTIME_BASE}" >&2; exit 1; }
[[ -e "${CHECKPOINT}" ]] || { echo "[pi05-piper] checkpoint is missing: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${POLICY_DIR}/eval.sh" ]] || { echo "[pi05-piper] policy eval.sh is missing: ${POLICY_DIR}/eval.sh" >&2; exit 1; }
[[ -x "${POLICY_ENV}/bin/python" ]] || { echo "[pi05-piper] policy uv environment is missing bin/python: ${POLICY_ENV}" >&2; exit 1; }
[[ -x "${EVAL_ENV}/bin/python" ]] || { echo "[pi05-piper] eval uv environment is missing bin/python: ${EVAL_ENV}" >&2; exit 1; }
if [[ -n "${ASSETS_ROOT}" ]]; then
  [[ -d "${ASSETS_ROOT}" ]] || { echo "[pi05-piper] assets root is missing: ${ASSETS_ROOT}" >&2; exit 1; }
  ASSETS_ROOT="$(cd "${ASSETS_ROOT}" && pwd)"
else
  [[ -d "${COMMON_ASSETS}" ]] || { echo "[pi05-piper] common assets root is missing: ${COMMON_ASSETS}" >&2; exit 1; }
fi

POLICY_DIR="$(cd "${POLICY_DIR}" && pwd)"
POLICY_REL="$(realpath --relative-to="${ROOT_DIR}/XPolicyLab" "${POLICY_DIR}")"
if [[ "${POLICY_REL}" == .. || "${POLICY_REL}" == ../* ]]; then
  echo "[pi05-piper] policy must be inside ${ROOT_DIR}/XPolicyLab: ${POLICY_DIR}" >&2
  exit 1
fi

if [[ "${ALLOW_DIRTY}" != "true" ]]; then
  if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=normal)" ]]; then
    echo "[pi05-piper] RoboDojo worktree is dirty; commit it or pass --allow-dirty" >&2
    exit 1
  fi
  if [[ -n "$(git -C "${ROOT_DIR}/XPolicyLab" status --porcelain --untracked-files=normal)" ]]; then
    echo "[pi05-piper] XPolicyLab worktree is dirty; commit it or pass --allow-dirty" >&2
    exit 1
  fi
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "[pi05-piper] nvidia-smi is required on the GPU worker" >&2; exit 1; }
  command -v ffmpeg >/dev/null 2>&1 || { echo "[pi05-piper] ffmpeg is required to validate output videos" >&2; exit 1; }
  readarray -t GPU_IDS < <(printf '%s\n' "${POLICY_GPU}" "${ENV_GPU}" | sort -nu)
  for gpu_id in "${GPU_IDS[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits)"
    if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
      echo "[pi05-piper] could not read free memory for GPU ${gpu_id}: ${free_mib}" >&2
      exit 1
    fi
    if (( free_mib < MIN_FREE_GPU_MIB )); then
      echo "[pi05-piper] GPU ${gpu_id} has ${free_mib} MiB free; require ${MIN_FREE_GPU_MIB} MiB" >&2
      exit 1
    fi
    echo "[pi05-piper] GPU ${gpu_id}: ${free_mib} MiB free"
  done
fi

echo "[pi05-piper] cgroup cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unavailable)"
echo "[pi05-piper] cgroup memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unavailable)"
echo "[pi05-piper] cgroup memory.current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo unavailable)"

RUNTIME_DIR="$(mktemp -d "${RUNTIME_BASE%/}/robodojo-pi05-piper.XXXXXX")"
cleanup() {
  if [[ "${KEEP_RUNTIME}" == "true" ]]; then
    echo "[pi05-piper] kept runtime: ${RUNTIME_DIR}"
  elif [[ -n "${RUNTIME_DIR}" && "${RUNTIME_DIR}" == "${RUNTIME_BASE%/}"/robodojo-pi05-piper.* ]]; then
    rm -rf -- "${RUNTIME_DIR}"
  fi
}
trap cleanup EXIT INT TERM

git -C "${ROOT_DIR}" archive HEAD | tar -x -C "${RUNTIME_DIR}"

link_submodule() {
  local relative_path="$1"
  local source_path="${ROOT_DIR}/${relative_path}"
  local target_path="${RUNTIME_DIR}/${relative_path}"
  [[ -d "${source_path}" ]] || { echo "[pi05-piper] submodule is missing: ${source_path}" >&2; exit 1; }
  mkdir -p "$(dirname "${target_path}")"
  if [[ -d "${target_path}" && ! -L "${target_path}" ]]; then
    rmdir "${target_path}" || {
      echo "[pi05-piper] expected an empty gitlink directory: ${target_path}" >&2
      exit 1
    }
  elif [[ -e "${target_path}" || -L "${target_path}" ]]; then
    echo "[pi05-piper] unexpected runtime path before submodule link: ${target_path}" >&2
    exit 1
  fi
  ln -s "${source_path}" "${target_path}"
}

link_submodule "XPolicyLab"
link_submodule "third_party/IsaacLab"
link_submodule "third_party/curobo"
ln -s "${ROOT_DIR}/eval_result" "${RUNTIME_DIR}/eval_result"

if [[ -n "${ASSETS_ROOT}" ]]; then
  ln -s "${ASSETS_ROOT}" "${RUNTIME_DIR}/Assets"
else
  LOCAL_ASSETS_ARCHIVE="${ASSETS_ARCHIVE}"
  if [[ "${ASSETS_ARCHIVE}" == hdfs://* ]]; then
    command -v hdfs >/dev/null 2>&1 || { echo "[pi05-piper] hdfs is required for ${ASSETS_ARCHIVE}" >&2; exit 1; }
    LOCAL_ASSETS_ARCHIVE="${RUNTIME_DIR}/hydrated-assets.tar"
    hdfs dfs -get "${ASSETS_ARCHIVE}" "${LOCAL_ASSETS_ARCHIVE}"
  fi
  [[ -f "${LOCAL_ASSETS_ARCHIVE}" ]] || { echo "[pi05-piper] asset archive is missing: ${LOCAL_ASSETS_ARCHIVE}" >&2; exit 1; }
  while IFS= read -r archive_path; do
    if [[ "${archive_path}" == /* || "${archive_path}" == ".." || "${archive_path}" == ../* || "${archive_path}" == */.. || "${archive_path}" == */../* ]]; then
      echo "[pi05-piper] unsafe path in asset archive: ${archive_path}" >&2
      exit 1
    fi
  done < <(tar -tf "${LOCAL_ASSETS_ARCHIVE}")
  tar -xf "${LOCAL_ASSETS_ARCHIVE}" -C "${RUNTIME_DIR}"
  [[ -d "${RUNTIME_DIR}/Assets" ]] || { echo "[pi05-piper] archive does not contain Assets/: ${LOCAL_ASSETS_ARCHIVE}" >&2; exit 1; }

  for common_dir in Background Material Room Sensor Traj; do
    if [[ ! -e "${RUNTIME_DIR}/Assets/${common_dir}" ]]; then
      [[ -d "${COMMON_ASSETS}/${common_dir}" ]] || {
        echo "[pi05-piper] common asset directory is missing: ${COMMON_ASSETS}/${common_dir}" >&2
        exit 1
      }
      ln -s "${COMMON_ASSETS}/${common_dir}" "${RUNTIME_DIR}/Assets/${common_dir}"
    fi
  done
fi

"${EVAL_ENV}/bin/python" "${ROOT_DIR}/scripts/internal/validate_pi05_piper_eval_assets.py" \
  --assets-root "${RUNTIME_DIR}/Assets" \
  --layout-ids "${LAYOUT_IDS}" \
  --seed "${SEED}"

IFS=',' read -r -a SELECTED_LAYOUTS <<< "${LAYOUT_IDS}"
EVAL_NUM="${#SELECTED_LAYOUTS[@]}"
mkdir -p "${ROOT_DIR}/eval_result/_launcher_logs"
LOG_PATH="${ROOT_DIR}/eval_result/_launcher_logs/${RUN_ID}.log"
mapfile -t EXISTING_RESULTS < <(find "${ROOT_DIR}/eval_result" -type f -path "*/${RUN_ID}/_result.json" -print)
if (( ${#EXISTING_RESULTS[@]} != 0 )); then
  echo "[pi05-piper] run id already has a result: ${RUN_ID}" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES
export ROBODOJO_ASSETS="${RUNTIME_DIR}/Assets"
export EVAL_LAYOUT_IDS="${LAYOUT_IDS}"
export EVAL_NUM
export ROBODOJO_RUN_ID="${RUN_ID}"
export ROBODOJO_MAX_BASH_RETRIES=1

echo "[pi05-piper] RoboDojo HEAD=$(git -C "${ROOT_DIR}" rev-parse HEAD)"
echo "[pi05-piper] XPolicyLab HEAD=$(git -C "${ROOT_DIR}/XPolicyLab" rev-parse HEAD)"
echo "[pi05-piper] run_id=${RUN_ID} layouts=${LAYOUT_IDS} eval_num=${EVAL_NUM}"
echo "[pi05-piper] runtime=${RUNTIME_DIR} log=${LOG_PATH} EULA=accepted"

EVAL_ARGS=(
  eval
  --policy-dir "${RUNTIME_DIR}/XPolicyLab/${POLICY_REL}"
  --task general_pickup_single
  --ckpt "${CHECKPOINT}"
  --env-cfg piper_single
  --action-type joint
  --seed "${SEED}"
  --policy-gpu "${POLICY_GPU}"
  --env-gpu "${ENV_GPU}"
  --policy-env "${POLICY_ENV}"
  --eval-env "${EVAL_ENV}"
  --eval-num "${EVAL_NUM}"
)
if [[ "${DRY_RUN}" == "true" ]]; then
  EVAL_ARGS+=(--dry-run)
fi
bash "${RUNTIME_DIR}/scripts/robodojo.sh" "${EVAL_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[pi05-piper] dry-run PASS"
  exit 0
fi

mapfile -t RESULT_FILES < <(find "${ROOT_DIR}/eval_result" -type f -path "*/${RUN_ID}/_result.json" -print)
if (( ${#RESULT_FILES[@]} != 1 )); then
  echo "[pi05-piper] expected exactly one result for ${RUN_ID}, found ${#RESULT_FILES[@]}" >&2
  printf '  %s\n' "${RESULT_FILES[@]}" >&2
  exit 1
fi
RESULT_FILE="${RESULT_FILES[0]}"
"${EVAL_ENV}/bin/python" -c '
import json
import sys

result_path, raw_layout_ids = sys.argv[1:]
expected = [int(value) for value in raw_layout_ids.split(",")]
with open(result_path, encoding="utf-8") as result_file:
    result = json.load(result_file)
details = result.get("details")
if not isinstance(details, dict):
    raise SystemExit(f"invalid result details: {result_path}")
actual = [int(detail["layout_id"]) for detail in details.values()]
if sorted(actual) != sorted(expected) or len(actual) != len(expected):
    raise SystemExit(f"layout mismatch: expected={expected} actual={actual}")
print(f"[pi05-piper] result PASS episodes={len(actual)} success_rate={result.get('"'"'success_rate'"'"')} file={result_path}")
' "${RESULT_FILE}" "${LAYOUT_IDS}"

RESULT_DIR="$(dirname "${RESULT_FILE}")"
mapfile -d '' -t VIDEO_FILES < <(find "${RESULT_DIR}" -maxdepth 1 -type f -name '*.mp4' -print0)
EXPECTED_VIDEOS=$((EVAL_NUM * 3))
if (( ${#VIDEO_FILES[@]} != EXPECTED_VIDEOS )); then
  echo "[pi05-piper] expected ${EXPECTED_VIDEOS} videos, found ${#VIDEO_FILES[@]} in ${RESULT_DIR}" >&2
  exit 1
fi
printf '%s\0' "${VIDEO_FILES[@]}" | xargs -0 -P 32 -n 1 bash -c 'ffmpeg -v error -i "$1" -f null -' _
echo "[pi05-piper] video decode PASS files=${#VIDEO_FILES[@]} parallel=32"

if [[ -n "${HDFS_OUTPUT}" ]]; then
  command -v hdfs >/dev/null 2>&1 || { echo "[pi05-piper] hdfs is required for --hdfs-output" >&2; exit 1; }
  RESULT_REL="$(realpath --relative-to="${ROOT_DIR}" "${RESULT_DIR}")"
  LOG_REL="$(realpath --relative-to="${ROOT_DIR}" "${LOG_PATH}")"
  RESULT_BUNDLE="${RUNTIME_DIR}/${RUN_ID}.tar"
  CHECKSUM_FILE="${RESULT_BUNDLE}.sha256"
  tar -cf "${RESULT_BUNDLE}" -C "${ROOT_DIR}" "${RESULT_REL}" "${LOG_REL}"
  (cd "${RUNTIME_DIR}" && sha256sum "$(basename "${RESULT_BUNDLE}")" > "$(basename "${CHECKSUM_FILE}")")
  hdfs dfs -mkdir -p "${HDFS_OUTPUT}"
  for local_file in "${RESULT_BUNDLE}" "${CHECKSUM_FILE}"; do
    remote_file="${HDFS_OUTPUT}/$(basename "${local_file}")"
    if hdfs dfs -test -e "${remote_file}"; then
      echo "[pi05-piper] refusing to overwrite HDFS output: ${remote_file}" >&2
      exit 1
    fi
    hdfs dfs -put "${local_file}" "${remote_file}"
  done
  echo "[pi05-piper] archived result to ${HDFS_OUTPUT}"
fi
