#!/usr/bin/env bash
#SBATCH -A gen150
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J gaia-tp4-cc
#SBATCH -o /lustre/orion/gen150/scratch/zye25/Agentic/logs/%x_%j.sbatch.log

set -euo pipefail

# The submit file stays tiny on purpose. Tune the experiment in:
#   openclaw_GAIA/config/tp4_concurrency.env

# Slurm executes a spooled copy of this file, so use a stable repo path.
GAIA_ROOT="${GAIA_ROOT:-/lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA}"
exec bash "${GAIA_ROOT}/scripts/run_tp4_concurrency.sh"
