#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

mkdir -p results/o2_discovery
out="results/o2_discovery/cuda_discovery.txt"

exec > >(tee "$out") 2>&1

echo "Tailbin O2 CUDA discovery"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Repository: $repo_root"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo

if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: module command not found. Run this on O2 with environment modules available." >&2
  exit 2
fi

echo "module spider cuda"
module spider cuda || true
echo

echo "module purge"
module purge
echo

echo "module load gcc/14.2.0"
module load gcc/14.2.0
echo

echo "module avail cuda"
module avail cuda || true
echo

echo "module spider cupy"
module spider cupy || true
echo

echo "Loaded modules:"
module list 2>&1
