#!/usr/bin/env bash
# Validate the general_pickup_single (single-arm) task on the server.
#
# Usage:
#   bash validate_single_arm.sh <POLICY_DIR> <CKPT> <POLICY_ENV> [EVAL_NUM]
#
#   POLICY_DIR   e.g. XPolicyLab/policy/<POLICY_NAME>
#   CKPT         checkpoint name
#   POLICY_ENV   policy conda env name
#   EVAL_NUM     optional, default 1 (quick validation)
#
# Stages:
#   [1/4] static checks (no Isaac needed)
#   [2/4] doctor
#   [3/4] dry-run eval (no sim)
#   [4/4] runtime eval (Isaac + policy server; needs the single-arm policy to
#         emit unprefixed keys: arm_joint_state / ee_joint_state / ee_pose ...)
set -euo pipefail

POLICY_DIR="${1:?usage: bash validate_single_arm.sh <POLICY_DIR> <CKPT> <POLICY_ENV> [EVAL_NUM]}"
CKPT="${2:?missing CKPT}"
POLICY_ENV="${3:?missing POLICY_ENV}"
EVAL_NUM="${4:-1}"
TASK=general_pickup_single
ENV_CFG=arx_x5_single

cd "$(dirname "$0")"
echo "=== [1/4] static checks ==="
bash -n scripts/robodojo.sh scripts/eval_policy.sh
python scripts/internal/task_inventory.py --format json --check
bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy

echo "=== [2/4] inventory sanity (task must be listed) ==="
bash scripts/robodojo.sh tasks | grep -x "${TASK}"

echo "=== [3/4] dry-run eval ==="
bash scripts/robodojo.sh eval \
  --policy-dir "${POLICY_DIR}" \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --policy-env "${POLICY_ENV}" \
  --env-cfg "${ENV_CFG}" \
  --dry-run

echo "=== [4/4] runtime eval (EVAL_NUM=${EVAL_NUM}) ==="
bash scripts/robodojo.sh eval \
  --policy-dir "${POLICY_DIR}" \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --policy-env "${POLICY_ENV}" \
  --env-cfg "${ENV_CFG}" \
  --eval-num "${EVAL_NUM}"

echo "=== result check ==="
# Acceptance: _result.json exists and eval_time >= 1 (exit code alone is not enough)
RESULT=$(find eval_result -name "_result.json" -newermt "-1 hour" | head -1)
if [[ -z "${RESULT}" ]]; then
  echo "[FAIL] no _result.json written under eval_result/" >&2
  exit 1
fi
python - "${RESULT}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
eval_time = data.get("eval_time", 0)
assert eval_time >= 1, f"eval_time={eval_time} < 1"
print(f"[PASS] {sys.argv[1]}: eval_time={eval_time}")
print(json.dumps({k: data[k] for k in data if k in ("success", "fail", "success_rate", "eval_num")}, indent=2))
PY

echo ""
echo "ALL STAGES PASSED"
