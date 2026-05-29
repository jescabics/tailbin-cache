#!/bin/bash
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p short
#SBATCH --mem=16G
#SBATCH --array=0-99
#SBATCH -o o2_tailbin_cpu_%A_%a.out
#SBATCH -e o2_tailbin_cpu_%A_%a.err

module load gcc/9.2.0
python -m pip install --user -e .

CONFIG=${CONFIG:-config.yaml}
NSHARDS=${NSHARDS:-100}
OUTDIR=${OUTDIR:-tailbin_shards}
mkdir -p "$OUTDIR"

tailbin-cache build-hdf5 --config "$CONFIG" --output "$OUTDIR/shard_${SLURM_ARRAY_TASK_ID}.h5" --n-shards "$NSHARDS" --shard-index "$SLURM_ARRAY_TASK_ID"
