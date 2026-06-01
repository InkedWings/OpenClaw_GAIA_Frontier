#!/usr/bin/env bash
set -euo pipefail

# Standalone single-node vLLM launcher for Frontier allocations.
# The GAIA runner calls this script with ACTION=start/stop/status and passes
# model/container/TP settings through environment variables.

ACTION="${1:-start}"
SLURM_TARGET_JOB_ID="${SLURM_TARGET_JOB_ID:-${SLURM_JOB_ID:-}}"

if [[ -z "${SLURM_JOB_NODELIST:-}" && -n "${SLURM_TARGET_JOB_ID}" ]]; then
  SLURM_JOB_NODELIST="$(squeue -h -j "${SLURM_TARGET_JOB_ID}" -o %N | head -n 1)"
fi
if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "[error] SLURM_JOB_NODELIST is empty"
  exit 1
fi

WORK="${WORK:-$PWD}"
IMG="${IMG:-$WORK/containers/vllm0141rocm72.sif}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
VLLM_ENTRYPOINT="${VLLM_ENTRYPOINT:-api_server}"
VLLM_PYTHONPATH="${VLLM_PYTHONPATH:-}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
TOOL_PARSER_PLUGIN="${TOOL_PARSER_PLUGIN:-}"
GENERATION_CONFIG="${GENERATION_CONFIG:-}"
OVERRIDE_GENERATION_CONFIG="${OVERRIDE_GENERATION_CONFIG:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
WAIT_BACKENDS_READY="${WAIT_BACKENDS_READY:-1}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"
READY_POLL_INTERVAL_S="${READY_POLL_INTERVAL_S:-10}"
READY_CURL_TIMEOUT_S="${READY_CURL_TIMEOUT_S:-4}"

# Hard-disable eager mode for benchmark consistency.
if [[ "${ENFORCE_EAGER}" != "0" ]]; then
  echo "[warn] ENFORCE_EAGER is disabled by script policy; forcing ENFORCE_EAGER=0"
fi
ENFORCE_EAGER="0"

RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}_p${PORT}}"
STATE_DIR="${WORK}/logs/vllm_launcher"
LOG_DIR="${WORK}/logs"
PID_FILE="${STATE_DIR}/${RUN_TAG}.pid"
ENDPOINTS_FILE="${STATE_DIR}/${RUN_TAG}.endpoints.sh"
SRUN_LOG="${STATE_DIR}/${RUN_TAG}.srun.log"
mkdir -p "${STATE_DIR}" "${LOG_DIR}"

mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#HOSTS[@]}" -ne 1 ]]; then
  echo "[error] single-node launcher only, got nodes=${#HOSTS[@]}"
  exit 2
fi
HEAD_HOST="${HOSTS[0]}"
VLLM_LOG="${LOG_DIR}/vllm_${HEAD_HOST}_${PORT}.log"

if [[ "${MODEL}" == "Qwen/Qwen2.5-32B-Instruct" && -d "/lustre/orion/gen150/scratch/zye25/models/Qwen2.5-32B-Instruct" ]]; then
  MODEL="/lustre/orion/gen150/scratch/zye25/models/Qwen2.5-32B-Instruct"
fi

if [[ -z "${TOOL_CALL_PARSER}" ]]; then
  TOOL_CALL_PARSER="hermes"
fi

write_endpoints() {
  : > "${ENDPOINTS_FILE}"
  local url="http://${HEAD_HOST}:${PORT}/v1"
  printf 'export ENDPOINT_1=%q\n' "${url}" >> "${ENDPOINTS_FILE}"
  printf 'export ENDPOINT_A=%q\n' "${url}" >> "${ENDPOINTS_FILE}"
  printf 'export BASE_URLS=%q\n' "${url}" >> "${ENDPOINTS_FILE}"
}

is_ready() {
  srun --overlap --jobid "${SLURM_TARGET_JOB_ID}" -N1 -n1 --nodelist "${HEAD_HOST}" \
    bash -lc "curl -fsS --max-time ${READY_CURL_TIMEOUT_S} http://127.0.0.1:${PORT}/v1/models | grep -q '\"data\"'" >/dev/null 2>&1
}

cleanup_vllm_processes() {
  srun --overlap --jobid "${SLURM_TARGET_JOB_ID}" -N1 -n1 --nodelist "${HEAD_HOST}" bash -lc '
    pkill -f "VLLM::|vllm.entrypoints.openai.api_server|vllm serve|vllm.entrypoints" >/dev/null 2>&1 || true
    sleep 1
    pkill -9 -f "VLLM::|vllm.entrypoints.openai.api_server|vllm serve|vllm.entrypoints" >/dev/null 2>&1 || true
    if [[ -x /opt/rocm-default/bin/rocm-smi ]]; then
      /opt/rocm-default/bin/rocm-smi --showpidgpus --showpids --json 2>/dev/null | python3 - <<'"'"'PY'"'"'
import json
import os
import re
import signal
import sys

txt = sys.stdin.read().strip()
if not txt:
    raise SystemExit(0)
try:
    data = json.loads(txt)
except Exception:
    raise SystemExit(0)

pids = []
for key, val in (data.get("system") or {}).items():
    m = re.match(r"PID(\d+)", str(key))
    if not m:
        continue
    if "vllm" in str(val).lower():
        pids.append(int(m.group(1)))

for pid in sorted(set(pids)):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        pass
PY
      sleep 1
      /opt/rocm-default/bin/rocm-smi --showpidgpus --showpids --json 2>/dev/null | python3 - <<'"'"'PY'"'"'
import json
import os
import re
import signal
import sys

txt = sys.stdin.read().strip()
if not txt:
    raise SystemExit(0)
try:
    data = json.loads(txt)
except Exception:
    raise SystemExit(0)

for key, val in (data.get("system") or {}).items():
    m = re.match(r"PID(\d+)", str(key))
    if not m:
        continue
    if "vllm" not in str(val).lower():
        continue
    pid = int(m.group(1))
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        pass
PY
    fi
  ' >/dev/null 2>&1 || true
}

wait_ready() {
  [[ "${WAIT_BACKENDS_READY}" != "1" ]] && { echo "[info] skip readiness wait"; return 0; }
  local start now
  start=$(date +%s)
  while true; do
    if is_ready; then
      echo "[ok] ready: http://${HEAD_HOST}:${PORT}/v1/models"
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= READY_TIMEOUT_S )); then
      echo "[warn] timeout waiting readiness"
      echo "[hint] log: ${VLLM_LOG}"
      return 1
    fi
    echo "[wait] elapsed=$((now-start))s pending=http://${HEAD_HOST}:${PORT}/v1/models"
    sleep "${READY_POLL_INTERVAL_S}"
  done
}

start_backend() {
  rm -f "${PID_FILE}" "${VLLM_LOG}" "${SRUN_LOG}"
  : > "${VLLM_LOG}"
  cleanup_vllm_processes
  write_endpoints

  local serve_cmd
  if [[ "${VLLM_ENTRYPOINT}" == "serve" ]]; then
    serve_cmd="vllm serve $(printf '%q' "${MODEL}")"
  elif [[ "${VLLM_ENTRYPOINT}" == "api_server" ]]; then
    serve_cmd="python -m vllm.entrypoints.openai.api_server"
    serve_cmd+=" --model $(printf '%q' "${MODEL}")"
  else
    echo "[error] unsupported VLLM_ENTRYPOINT=${VLLM_ENTRYPOINT}; use api_server or serve"
    exit 2
  fi
  serve_cmd+=" --host 0.0.0.0 --port $(printf '%q' "${PORT}")"
  serve_cmd+=" --tensor-parallel-size $(printf '%q' "${TP_SIZE}")"
  serve_cmd+=" --gpu-memory-utilization $(printf '%q' "${GPU_MEMORY_UTILIZATION}")"
  serve_cmd+=" --max-model-len $(printf '%q' "${MAX_MODEL_LEN}")"
  if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    serve_cmd+=" --enforce-eager"
  fi
  if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
    serve_cmd+=" --enable-prefix-caching"
  fi
  if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
    serve_cmd+=" --enable-auto-tool-choice --tool-call-parser $(printf '%q' "${TOOL_CALL_PARSER}")"
    if [[ -n "${TOOL_PARSER_PLUGIN}" ]]; then
      serve_cmd+=" --tool-parser-plugin $(printf '%q' "${TOOL_PARSER_PLUGIN}")"
    fi
  fi
  if [[ -n "${GENERATION_CONFIG}" ]]; then
    serve_cmd+=" --generation-config $(printf '%q' "${GENERATION_CONFIG}")"
  fi
  if [[ -n "${OVERRIDE_GENERATION_CONFIG}" ]]; then
    serve_cmd+=" --override-generation-config $(printf '%q' "${OVERRIDE_GENERATION_CONFIG}")"
  fi
  if [[ -n "${REASONING_PARSER}" ]]; then
    serve_cmd+=" --reasoning-parser $(printf '%q' "${REASONING_PARSER}")"
  fi
  if [[ -n "${SPECULATIVE_CONFIG}" ]]; then
    serve_cmd+=" --speculative-config $(printf '%q' "${SPECULATIVE_CONFIG}")"
  fi
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    serve_cmd+=" ${VLLM_EXTRA_ARGS}"
  fi
  local env_prefix
  # Avoid leaking launcher-only VLLM_* variables into vLLM's environment scan.
  env_prefix="env -u VLLM_ENTRYPOINT -u VLLM_PYTHONPATH -u VLLM_EXTRA_ARGS OMP_NUM_THREADS=$(printf '%q' "${OMP_NUM_THREADS}")"
  if [[ -n "${VLLM_PYTHONPATH}" ]]; then
    env_prefix="PYTHONPATH=$(printf '%q' "${VLLM_PYTHONPATH}") ${env_prefix}"
  fi
  serve_cmd="${env_prefix} ${serve_cmd}"

  nohup srun --jobid "${SLURM_TARGET_JOB_ID}" -N1 -n1 --nodelist "${HEAD_HOST}" --overlap \
    apptainer exec --fakeroot --writable-tmpfs "${IMG}" bash -lc "${serve_cmd}" \
    > "${VLLM_LOG}" 2>&1 &
  local pid=$!
  echo "HEAD_PID=${pid}" > "${PID_FILE}"
  ln -sf "${VLLM_LOG}" "${SRUN_LOG}"

  echo "[ok] launcher started pid=${pid}"
  echo "[ok] vllm log: ${VLLM_LOG}"
  echo "[ok] endpoints: ${ENDPOINTS_FILE}"
  wait_ready
}

stop_backend() {
  local pid=""
  if [[ -f "${PID_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${PID_FILE}" || true
    pid="${HEAD_PID:-}"
  fi
  if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 1
    ps -p "${pid}" >/dev/null 2>&1 && kill -9 "${pid}" >/dev/null 2>&1 || true
  fi
  cleanup_vllm_processes
  rm -f "${PID_FILE}"
  echo "[ok] stopped"
}

status_backend() {
  if [[ -f "${PID_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${PID_FILE}" || true
    echo "HEAD_PID=${HEAD_PID:-}"
    if [[ -n "${HEAD_PID:-}" ]] && ps -p "${HEAD_PID}" >/dev/null 2>&1; then
      echo "[ok] launcher running"
    else
      echo "[warn] launcher not running"
    fi
  else
    echo "[info] no pid file"
  fi
  echo "[info] vllm log: ${VLLM_LOG}"
  echo "[info] endpoints: ${ENDPOINTS_FILE}"
  [[ -f "${ENDPOINTS_FILE}" ]] && sed -n '1,120p' "${ENDPOINTS_FILE}"
}

case "${ACTION}" in
  start) start_backend ;;
  stop) stop_backend ;;
  status) status_backend ;;
  *) echo "Usage: $0 {start|stop|status}"; exit 2 ;;
esac
