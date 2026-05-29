#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

cat >&2 <<'EOF'
Production pilot submission is intentionally disabled.

First run and review the O2 resource calibration workflow:

  bash scripts/o2/submit_resource_calibration.sh

Then review:

  results/o2_resource_calibration/<run_id>/summary.md
  results/o2_resource_calibration/<run_id>/summary.json

Fill production-pilot parameters only after calibration establishes safe memory,
wall-time, shard count, concurrency, and CPU/GPU backend choices.

Expected future inputs:

  PILOT_CONFIG=<reviewed config>
  PILOT_N_SHARDS=<reviewed shard count>
  PILOT_ARRAY_CONCURRENCY=<reviewed concurrency>
  PILOT_PARTITION=<reviewed partition>
  PILOT_TIME=<reviewed wall-time>
  PILOT_MEM=<reviewed memory>
  PILOT_CPUS=<reviewed cpus>
EOF

exit 2
