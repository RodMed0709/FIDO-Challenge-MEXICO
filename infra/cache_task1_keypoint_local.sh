#!/bin/bash
# Stage only the Task 1 files consumed by keypoint training on pod-local NVMe.
set -euo pipefail

SRC=${TASK1_SOURCE:-/workspace/data/Task1}
DST=${TASK1_CACHE:-/root/data_cache/Task1}
WORKERS=${CACHE_WORKERS:-10}

echo "=== TASK1 KEYPOINT CACHE START $(date --iso-8601=seconds) ==="
mkdir -p "$DST"

copy_scenario() {
  local scenario=$1
  local name
  name=$(basename "$scenario")
  mkdir -p "$DST/$name"
  (
    cd "$scenario"
    {
      printf '%s\0' Numerical
      find 'Stereo Left' -type f -name microscope.png -print0
    } | tar --null -T - -cf - 2>/dev/null
  ) | (cd "$DST/$name" && tar xf -)
  echo "[$(date +%H:%M:%S)] cached $name"
}
export -f copy_scenario
export DST

find "$SRC" -maxdepth 1 -mindepth 1 -type d -name 'Scenario_*' -print0 \
  | xargs -0 -P "$WORKERS" -I{} bash -c 'copy_scenario "$1"' _ '{}'

echo "=== TASK1 KEYPOINT CACHE DONE $(date --iso-8601=seconds) ==="
du -sh "$DST"
df -h /
