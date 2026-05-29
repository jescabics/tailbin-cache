# O2 Smoke/Audit Workflow

These scripts submit a first-pass Tailbin O2 smoke/audit run. They are not a production cache build.

## Intended O2 Location

Use this repository path on O2 unless the project location changes:

```bash
/n/data1/hms/sysbio/hormoz/users/javi/tailbin-cache
```

## Update Code On O2

From the repository root on O2:

```bash
cd /n/data1/hms/sysbio/hormoz/users/javi/tailbin-cache
git pull
git status
```

## Submit

Submit the CPU smoke job, GPU audit job, and dependent collector job:

```bash
bash scripts/o2/submit_smoke_audit.sh
```

The submitter prints the CPU job ID, GPU job ID, collector job ID, and monitoring commands.

## Monitor

```bash
squeue -u "$USER"
```

For a specific job printed by the submitter:

```bash
squeue -j <jobid>
O2_jobs_report -j <jobid>
```

## Outputs

Generated files stay in ignored project directories:

* `logs/`
* `outputs/o2_smoke/`
* `outputs/o2_gpu_audit/`
* `results/o2_smoke/`
* `results/o2_gpu_audit/`
* `results/tailbin_o2_smoke_audit_<timestamp>.tgz`

The collector bundles logs, selected outputs/results, O2 docs, example configs, git metadata, accounting output, and GPU monitor logs when present.

## Retrieve The Bundle

After the collector finishes, send back or download the newest bundle from:

```bash
results/tailbin_o2_smoke_audit_<timestamp>.tgz
```

For example, from your laptop you can use `scp` with your normal O2 access method, or attach/share the bundle through the agreed result-review workflow.

## Resource Overrides

Override modest defaults by setting environment variables before submission:

```bash
CPU_PARTITION=short CPU_TIME=1:00:00 CPU_MEM=2G CPU_CPUS=1 \
GPU_PARTITION=gpu_quad GPU_TIME=2:00:00 GPU_MEM=16G GPU_CPUS=4 GPU_GRES=gpu:1 GPU_CONSTRAINT=gpu_doublep \
COLLECT_PARTITION=short COLLECT_TIME=0:30:00 COLLECT_MEM=1G COLLECT_CPUS=1 \
bash scripts/o2/submit_smoke_audit.sh
```

You can omit `GPU_CONSTRAINT` by setting it to an empty string:

```bash
GPU_CONSTRAINT="" bash scripts/o2/submit_smoke_audit.sh
```

## Login Node Boundary

Do not run heavy Tailbin planners, HDF5 builds, GPU audits, or production jobs directly on login nodes. Use `bash scripts/o2/submit_smoke_audit.sh` to submit work through SLURM.
