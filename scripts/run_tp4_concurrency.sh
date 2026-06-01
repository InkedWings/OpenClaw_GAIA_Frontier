#!/usr/bin/env bash
set -euo pipefail

# Run inside an existing Slurm allocation or from an sbatch job.
# Edit config/tp4_concurrency.env for model/backend/sweep settings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAIA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${GAIA_ROOT}/config/tp4_concurrency.env}"

# shellcheck source=../config/tp4_concurrency.env
source "${CONFIG_FILE}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[error] SLURM_JOB_ID is empty. Run this from salloc/sbatch." >&2
  exit 1
fi
if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  SLURM_JOB_NODELIST="$(squeue -h -j "${SLURM_JOB_ID}" -o %N | head -n 1)"
fi

NODE="${NODE:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${GAIA_ROOT}/runs/${RUN_NAME}_${SLURM_JOB_ID}_${RUN_STAMP}}"
MANAGE_VLLM_SCRIPT="${MANAGE_VLLM_SCRIPT:-${GAIA_ROOT}/scripts/manage_vllm.sh}"

# Keep web tools on the ORNL proxy, while local vLLM traffic bypasses it.
PROXY="${PROXY:-http://proxy.ccs.ornl.gov:3128}"
export http_proxy="${http_proxy:-$PROXY}"
export https_proxy="${https_proxy:-$PROXY}"
export HTTP_PROXY="${HTTP_PROXY:-$PROXY}"
export HTTPS_PROXY="${HTTPS_PROXY:-$PROXY}"
SLURM_HOSTS="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | paste -sd, -)"
SLURM_IPS="$(getent hosts $(scontrol show hostnames "${SLURM_JOB_NODELIST}") | awk '{print $1}' | paste -sd, -)"
export no_proxy="localhost,127.0.0.1,::1,.ornl.gov,.olcf.ornl.gov,.frontier.olcf.ornl.gov,${NODE},$(hostname -s),$(hostname -f),${SLURM_HOSTS},${SLURM_IPS}"
export NO_PROXY="${no_proxy}"

if [[ -z "${OPENCLAW_CONTEXT_WINDOW:-}" ]]; then
  OPENCLAW_CONTEXT_WINDOW="$((VLLM_CONTEXT_WINDOW - OPENCLAW_MAX_OUTPUT_TOKENS - OPENCLAW_CONTEXT_MARGIN))"
  if (( OPENCLAW_CONTEXT_WINDOW < 1 )); then
    OPENCLAW_CONTEXT_WINDOW=1
  fi
fi
BACKEND_READY_TIMEOUT_S="${BACKEND_READY_TIMEOUT_S:-1800}"

mkdir -p "${GAIA_ROOT}/runs" "${WORK}/logs"

echo "[info] gaia_root=${GAIA_ROOT}"
echo "[info] config=${CONFIG_FILE}"
echo "[info] job_id=${SLURM_JOB_ID} node=${NODE}"
echo "[info] out_dir=${OUT_DIR}"
echo "[info] tp_list=${TP_LIST} concurrency_list=${CONCURRENCY_LIST} rounds=${ROUNDS}"
echo "[info] rows=${ROWS_JSONL} idx_range=[${IDX_START},${IDX_END}) case_mode=${CASE_MODE}"
echo "[info] vllm_context_window=${VLLM_CONTEXT_WINDOW} openclaw_context_window=${OPENCLAW_CONTEXT_WINDOW} openclaw_max_output_tokens=${OPENCLAW_MAX_OUTPUT_TOKENS} margin=${OPENCLAW_CONTEXT_MARGIN}"
echo "[info] backend_ready_timeout_s=${BACKEND_READY_TIMEOUT_S}"

args=(
  "${GAIA_ROOT}/scripts/run_gaia_concurrency.py"
  --job-id "${SLURM_JOB_ID}"
  --node "${NODE}"
  --img "${IMG}"
  --model "${MODEL}"
  --rows-jsonl "${ROWS_JSONL}"
  --case-mode "${CASE_MODE}"
  --idx-start "${IDX_START}"
  --idx-end "${IDX_END}"
  --tp-list "${TP_LIST}"
  --concurrency-list "${CONCURRENCY_LIST}"
  --rounds "${ROUNDS}"
  --warmup-count "${WARMUP_COUNT}"
  --gpu-sample-sec "${GPU_SAMPLE_SEC}"
  --timeout "${TASK_TIMEOUT_S}"
  --backend-ready-timeout-s "${BACKEND_READY_TIMEOUT_S}"
  --port "${PORT}"
  --max-model-len "${VLLM_CONTEXT_WINDOW}"
  --openclaw-context-window "${OPENCLAW_CONTEXT_WINDOW}"
  --openclaw-context-margin "${OPENCLAW_CONTEXT_MARGIN}"
  --model-max-tokens "${OPENCLAW_MAX_OUTPUT_TOKENS}"
  --gpu-mem-util "${GPU_MEM_UTIL}"
  --vllm-api-key "${VLLM_API_KEY}"
  --vllm-entrypoint "${VLLM_ENTRYPOINT}"
  --vllm-generation-config "${VLLM_GENERATION_CONFIG}"
  --vllm-override-generation-config "${VLLM_OVERRIDE_GENERATION_CONFIG}"
  --python-bin "${PYTHON_BIN}"
  --openclaw-bin "${GAIA_ROOT}/.local/npm/lib/node_modules/openclaw/openclaw.mjs"
  --openclaw-config "${GAIA_ROOT}/config/openclaw.template.json"
  --node-bin-dir "${GAIA_ROOT}/.local/node/bin"
  --manage-backend-script "${MANAGE_VLLM_SCRIPT}"
  --out-dir "${OUT_DIR}"
)

[[ "${DISABLE_VLLM_SPECULATIVE}" == "1" ]] && args+=(--disable-vllm-speculative)
[[ "${RESUME}" == "1" ]] && args+=(--resume)
[[ "${SKIP_REPORT}" == "1" ]] && args+=(--skip-report)

exec "${PYTHON_BIN}" "${args[@]}"
