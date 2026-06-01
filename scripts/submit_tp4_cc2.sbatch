#!/usr/bin/env bash
#SBATCH -A gen150
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J gaia-tp4-cc2
#SBATCH -o /lustre/orion/gen150/scratch/zye25/Agentic/logs/%x_%j.sbatch.log

set -euo pipefail

# Slurm executes a spooled copy of this file, so use a stable repo path.
GAIA_ROOT="${GAIA_ROOT:-/lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA}"
export CONFIG_FILE="${GAIA_ROOT}/config/tp4_concurrency.env"
export TP_LIST="4"
export CONCURRENCY_LIST="2"
export RUN_NAME="gaia_qwen35_tp4_cc2"
exec bash "${GAIA_ROOT}/scripts/run_tp4_concurrency.sh"
