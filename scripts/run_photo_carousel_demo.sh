#!/bin/bash
# Run the 5-photo carousel demo for general_pickup (RoboDojo).
set -euo pipefail

cd /home/ubuntu/RoboDojo/XPolicyLab/policy/demo_policy
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate RoboDojo
export EVAL_NUM=1

echo "Yes" | timeout 900 bash eval.sh RoboDojo general_pickup demo piper ee 0 0 0 xpl_policy RoboDojo