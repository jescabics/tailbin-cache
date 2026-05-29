#!/bin/bash
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_doublep
#SBATCH --mem=32G
#SBATCH -o o2_gpu_probe_%j.out
#SBATCH -e o2_gpu_probe_%j.err

module load gcc/9.2.0
module load cuda/11.7
/n/cluster/bin/job_gpu_monitor.sh &

python -m pip install --user -e .
# Install once in your environment if needed; comment out after first use.
# python -m pip install --user cupy-cuda11x

CONFIG=${CONFIG:-examples/kmax20_o2_gpu_probe.yaml}
OUT=${OUT:-o2_gpu_probe_${SLURM_JOB_ID}}

tailbin-cache plan --config "$CONFIG" --output-dir "$OUT/plan"
tailbin-cache build-hdf5 --config "$CONFIG" --output "$OUT/probe.h5" --limit-base-points 1
