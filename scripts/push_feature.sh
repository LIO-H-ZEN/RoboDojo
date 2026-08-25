#!/bin/bash
# Push feature/table-display-playlist to GitHub (long timeout + retries).
set -uo pipefail

TOKEN="${GITHUB_TOKEN:?请先设置 GITHUB_TOKEN 环境变量}"
BRANCH="feature/table-display-playlist"

cd /home/ubuntu/RoboDojo

for attempt in 1 2 3 4; do
    echo "=== attempt ${attempt}/4 @ $(date +%H:%M:%S) ===" | tee -a /tmp/push_feature.log
    if timeout 280 git -c http.postBuffer=524288000 \
        -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=120 \
        push "https://${TOKEN}@github.com/LIO-H-ZEN/RoboDojo.git" "${BRANCH}" >> /tmp/push_feature.log 2>&1; then
        echo "PUSH OK on attempt ${attempt}" | tee -a /tmp/push_feature.log
        exit 0
    fi
    echo "push attempt ${attempt} failed (rc=$?), retrying in 8s..." | tee -a /tmp/push_feature.log
    sleep 8
done

echo "ALL PUSH ATTEMPTS FAILED" | tee -a /tmp/push_feature.log
exit 1