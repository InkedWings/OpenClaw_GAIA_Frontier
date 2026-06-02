#!/usr/bin/env bash
set -euo pipefail

# Launch one TP4 vLLM instance per Frontier node and run a GAIA concurrency
# sweep against all instances with fixed worker-to-backend round robin.
# This script intentionally avoids squeue. It assumes the nodes are already
# allocated and can be reached either through srun --overlap or SSH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAIA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NODES_CSV="${NODES_CSV:-frontier10362,frontier10363,frontier10365,frontier10366}"
IFS=',' read -r -a NODES <<< "${NODES_CSV}"

PORT="${PORT:-8011}"
WORK="${WORK:-/lustre/orion/gen150/scratch/zye25/Agentic}"
IMG="${IMG:-${WORK}/containers/vllm-openai-rocm-nightly.sif}"
MODEL="${MODEL:-${WORK}/hf/models/Qwen3.5-27B}"
PYTHON_BIN="${PYTHON_BIN:-/lustre/orion/gen150/scratch/zye25/conda-envs/AGAI/bin/python}"
ROWS_JSONL="${ROWS_JSONL:-${GAIA_ROOT}/data/gaia_2023_all_validation/rows.jsonl}"
LOCAL_CONFIG_FILE="${LOCAL_CONFIG_FILE:-${GAIA_ROOT}/config/local.env}"
SLURM_TARGET_JOB_ID="${SLURM_TARGET_JOB_ID:-${SLURM_JOB_ID:-0}}"
REMOTE_RUNNER="${REMOTE_RUNNER:-ssh}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout="${SSH_CONNECT_TIMEOUT_S:-10}"
  -o ServerAliveInterval="${SSH_SERVER_ALIVE_INTERVAL_S:-10}"
  -o ServerAliveCountMax="${SSH_SERVER_ALIVE_COUNT_MAX:-3}"
)

TP_LIST="${TP_LIST:-4}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-2,4,8}"
ROUNDS="${ROUNDS:-1}"
WARMUP_COUNT="${WARMUP_COUNT:-0}"
IDX_START="${IDX_START:-0}"
IDX_END="${IDX_END:--1}"
CASE_MODE="${CASE_MODE:-all}"
TASK_TIMEOUT_S="${TASK_TIMEOUT_S:-900}"
BACKEND_READY_TIMEOUT_S="${BACKEND_READY_TIMEOUT_S:-3600}"
GPU_SAMPLE_SEC="${GPU_SAMPLE_SEC:-1}"
VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
OPENCLAW_CONTEXT_WINDOW="${OPENCLAW_CONTEXT_WINDOW:-23552}"
OPENCLAW_CONTEXT_MARGIN="${OPENCLAW_CONTEXT_MARGIN:-1024}"
OPENCLAW_MAX_OUTPUT_TOKENS="${OPENCLAW_MAX_OUTPUT_TOKENS:-3584}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-gaia_4node_tp4_full_sweep}"
RUN_ROOT="${RUN_ROOT:-${GAIA_ROOT}/runs/${RUN_NAME}_${RUN_STAMP}}"
mkdir -p "${RUN_ROOT}" "${WORK}/logs" "${WORK}/logs/vllm_launcher"

if [[ -f "${LOCAL_CONFIG_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${LOCAL_CONFIG_FILE}"
  set +a
fi

join_by_comma() {
  local IFS=,
  echo "$*"
}

remote_bash_node() {
  local node
  node="$1"
  shift
  if [[ "${REMOTE_RUNNER}" == "srun" ]]; then
    if [[ "${SLURM_TARGET_JOB_ID}" == "0" || -z "${SLURM_TARGET_JOB_ID}" ]]; then
      echo "[error] REMOTE_RUNNER=srun requires SLURM_TARGET_JOB_ID or SLURM_JOB_ID" >&2
      return 2
    fi
    srun --overlap --jobid "${SLURM_TARGET_JOB_ID}" -N1 -n1 --nodelist "${node}" bash -s -- "$@"
  elif [[ "${REMOTE_RUNNER}" == "ssh" ]]; then
    ssh "${SSH_OPTS[@]}" "${node}" bash -s -- "$@"
  else
    echo "[error] unknown REMOTE_RUNNER=${REMOTE_RUNNER}; expected ssh or srun" >&2
    return 2
  fi
}

setup_proxy() {
  local proxy hosts
  proxy="${PROXY:-http://proxy.ccs.ornl.gov:3128}"
  hosts="$(join_by_comma "${NODES[@]}")"
  export http_proxy="${http_proxy:-${proxy}}"
  export https_proxy="${https_proxy:-${proxy}}"
  export HTTP_PROXY="${HTTP_PROXY:-${proxy}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${proxy}}"
  export no_proxy="localhost,127.0.0.1,::1,.ornl.gov,.olcf.ornl.gov,.frontier.olcf.ornl.gov,${hosts},$(hostname -s),$(hostname -f)"
  export NO_PROXY="${no_proxy}"
}

stop_backend_node() {
  local node
  node="$1"
  remote_bash_node "${node}" <<'REMOTE'
set -euo pipefail
python3 - <<'PY'
import os
import signal
import subprocess
import time

user = os.environ.get("USER")
patterns = ("VLLM::", "vllm.entrypoints", "vllm serve", "/usr/local/bin/vllm")
cmd = ["ps", "-u", user, "-o", "pid=,args="] if user else ["ps", "-eo", "pid=,args="]
out = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].decode("utf-8", "ignore")
pids = []
for line in out.splitlines():
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        continue
    pid_s, proc_cmd = parts
    try:
        pid = int(pid_s)
    except ValueError:
        continue
    if pid == os.getpid() or pid == os.getppid():
        continue
    if any(pattern in proc_cmd for pattern in patterns):
        pids.append(pid)

for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in sorted(set(pids)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except Exception:
            pass
    time.sleep(1)

if pids:
    print("stopped_vllm_pids=" + ",".join(str(pid) for pid in sorted(set(pids))))
PY
REMOTE
}

start_all_backends() {
  local node pids=() ready_files=()
  for node in "${NODES[@]}"; do
    local persist_after_ready ready_file
    persist_after_ready=0
    ready_file=""
    if [[ "${REMOTE_RUNNER}" == "srun" ]]; then
      persist_after_ready=1
      ready_file="${RUN_ROOT}/backend_ready_${node}.txt"
      rm -f "${ready_file}"
    fi
    (
      remote_bash_node "${node}" \
        "${RUN_NAME}_${RUN_STAMP}_${node}_tp4_p${PORT}" \
        "${WORK}/logs/vllm_${node}_${PORT}_${RUN_NAME}_${RUN_STAMP}.log" \
        "${WORK}/logs/vllm_launcher/${RUN_NAME}_${RUN_STAMP}_${node}_tp4_p${PORT}.pid" \
        "${MODEL}" "${IMG}" "${PORT}" "${VLLM_LOG_STATS_INTERVAL}" \
        "${BACKEND_READY_TIMEOUT_S}" "${GPU_MEM_UTIL}" "${MAX_MODEL_LEN}" \
        "${persist_after_ready}" "${ready_file}" <<'REMOTE'
set -euo pipefail
RUN_TAG="$1"
LOG="$2"
PID_FILE="$3"
MODEL="$4"
IMG="$5"
PORT="$6"
STATS_INTERVAL="$7"
READY_TIMEOUT_S="$8"
GPU_MEM_UTIL="$9"
MAX_MODEL_LEN="${10}"
PERSIST_AFTER_READY="${11}"
READY_FILE="${12}"
CACHE="/tmp/${USER}/openclaw_vllm_cache"
MIOPEN="/tmp/${USER}/openclaw_miopen_cache"
mkdir -p "$(dirname "${PID_FILE}")" "${CACHE}" "${CACHE}/torchinductor" /tmp/triton/cache "${MIOPEN}"

python3 - <<'PY'
import os
import signal
import subprocess
import time

user = os.environ.get("USER")
patterns = ("VLLM::", "vllm.entrypoints", "vllm serve", "/usr/local/bin/vllm")
cmd = ["ps", "-u", user, "-o", "pid=,args="] if user else ["ps", "-eo", "pid=,args="]
out = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].decode("utf-8", "ignore")
pids = []
for line in out.splitlines():
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        continue
    pid_s, proc_cmd = parts
    try:
        pid = int(pid_s)
    except ValueError:
        continue
    if any(pattern in proc_cmd for pattern in patterns):
        pids.append(pid)
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in sorted(set(pids)):
        try:
            os.kill(pid, sig)
        except Exception:
            pass
    time.sleep(1)
PY

: > "${LOG}"
args=(
  vllm serve "${MODEL}"
  --host 0.0.0.0
  --port "${PORT}"
  --tensor-parallel-size 4
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --generation-config vllm
  --override-generation-config '{"temperature":0.0,"top_p":1.0,"repetition_penalty":1.0}'
  --reasoning-parser qwen3
)

nohup env -u VLLM_CACHE_DIR \
  VLLM_LOG_STATS_INTERVAL="${STATS_INTERVAL}" \
  OMP_NUM_THREADS=1 \
  XDG_CACHE_HOME="${CACHE}" \
  TORCHINDUCTOR_CACHE_DIR="${CACHE}/torchinductor" \
  TRITON_CACHE_DIR=/tmp/triton/cache \
  MIOPEN_USER_DB_PATH="${MIOPEN}" \
  MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN}" \
  apptainer exec --fakeroot --writable-tmpfs "${IMG}" "${args[@]}" \
  > "${LOG}" 2>&1 &

echo "$!" > "${PID_FILE}"
echo "RUN_TAG=${RUN_TAG}"
echo "LOG=${LOG}"
echo "PID_FILE=${PID_FILE}"

start_ts="$(date +%s)"
deadline="$((start_ts + READY_TIMEOUT_S))"
while true; do
  if curl -fsS --max-time 4 "http://127.0.0.1:${PORT}/v1/models" | grep -q '"data"'; then
    now="$(date +%s)"
    ready_line="READY elapsed=$((now - start_ts))s"
    echo "${ready_line}"
    if [[ -n "${READY_FILE}" ]]; then
      echo "${ready_line}" > "${READY_FILE}"
    fi
    if [[ "${PERSIST_AFTER_READY}" == "1" ]]; then
      wait "$(cat "${PID_FILE}")"
      exit $?
    fi
    exit 0
  fi
  if ! kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "FAILED_READY backend process exited; log=${LOG}" >&2
    tail -n 120 "${LOG}" >&2 || true
    exit 1
  fi
  now="$(date +%s)"
  if (( now >= deadline )); then
    echo "FAILED_READY timeout; log=${LOG}" >&2
    tail -n 120 "${LOG}" >&2 || true
    exit 1
  fi
  if (( (now - start_ts) % 60 < 5 )); then
    echo "WAIT elapsed=$((now - start_ts))s"
    tail -n 8 "${LOG}" || true
  fi
sleep 5
done
REMOTE
    ) > "${RUN_ROOT}/backend_start_${node}.log" 2>&1 &
    pids+=("$!")
    ready_files+=("${ready_file}")
  done

  if [[ "${REMOTE_RUNNER}" == "srun" ]]; then
    local start_ts deadline now ready_count i
    start_ts="$(date +%s)"
    deadline="$((start_ts + BACKEND_READY_TIMEOUT_S))"
    while true; do
      ready_count=0
      for i in "${!NODES[@]}"; do
        if [[ -f "${ready_files[$i]}" ]]; then
          ready_count=$((ready_count + 1))
          continue
        fi
        if ! kill -0 "${pids[$i]}" >/dev/null 2>&1; then
          echo "[error] backend startup step exited before ready: node=${NODES[$i]}" >&2
          tail -n 120 "${RUN_ROOT}/backend_start_${NODES[$i]}.log" >&2 || true
          return 1
        fi
      done
      if [[ "${ready_count}" == "${#NODES[@]}" ]]; then
        break
      fi
      now="$(date +%s)"
      if (( now >= deadline )); then
        echo "[error] backend startup timeout after ${BACKEND_READY_TIMEOUT_S}s" >&2
        for node in "${NODES[@]}"; do
          echo "===== backend_start_${node}.log =====" >&2
          tail -n 80 "${RUN_ROOT}/backend_start_${node}.log" >&2 || true
        done
        return 1
      fi
      sleep 5
    done
  else
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  fi

  for node in "${NODES[@]}"; do
    echo "===== backend_start_${node}.log ====="
    tail -n 40 "${RUN_ROOT}/backend_start_${node}.log" || true
  done

  for node in "${NODES[@]}"; do
    echo "${WORK}/logs/vllm_${node}_${PORT}_${RUN_NAME}_${RUN_STAMP}.log" > "${RUN_ROOT}/vllm_log_${node}.txt"
  done
}

stop_all_backends() {
  local node pids=()
  for node in "${NODES[@]}"; do
    stop_backend_node "${node}" >/dev/null 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done
}

run_gaia_sweep() {
  local base_urls=() vllm_logs=() node base_urls_csv vllm_logs_csv nodes_csv
  for node in "${NODES[@]}"; do
    base_urls+=("http://${node}:${PORT}/v1")
    vllm_logs+=("$(cat "${RUN_ROOT}/vllm_log_${node}.txt")")
  done
  base_urls_csv="$(join_by_comma "${base_urls[@]}")"
  vllm_logs_csv="$(join_by_comma "${vllm_logs[@]}")"
  nodes_csv="$(join_by_comma "${NODES[@]}")"

  setup_proxy
  export VLLM_LOG_STATS_INTERVAL

  local runner_job_id
  runner_job_id=0
  if [[ "${REMOTE_RUNNER}" == "srun" ]]; then
    runner_job_id="${SLURM_TARGET_JOB_ID}"
  fi

  "${PYTHON_BIN}" "${GAIA_ROOT}/scripts/run_gaia_concurrency.py" \
    --job-id "${runner_job_id}" \
    --node "${nodes_csv}" \
    --remote-runner "${REMOTE_RUNNER}" \
    --skip-slurm-validation \
    --external-base-url "${base_urls_csv}" \
    --external-vllm-log "${vllm_logs_csv}" \
    --rows-jsonl "${ROWS_JSONL}" \
    --case-mode "${CASE_MODE}" \
    --idx-start "${IDX_START}" \
    --idx-end "${IDX_END}" \
    --tp-list "${TP_LIST}" \
    --concurrency-list "${CONCURRENCY_LIST}" \
    --rounds "${ROUNDS}" \
    --warmup-count "${WARMUP_COUNT}" \
    --gpu-sample-sec "${GPU_SAMPLE_SEC}" \
    --timeout "${TASK_TIMEOUT_S}" \
    --backend-ready-timeout-s "${BACKEND_READY_TIMEOUT_S}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --openclaw-context-window "${OPENCLAW_CONTEXT_WINDOW}" \
    --openclaw-context-margin "${OPENCLAW_CONTEXT_MARGIN}" \
    --model-max-tokens "${OPENCLAW_MAX_OUTPUT_TOKENS}" \
    --gpu-mem-util "${GPU_MEM_UTIL}" \
    --vllm-api-key "${VLLM_API_KEY:-vllm-local}" \
    --vllm-entrypoint serve \
    --vllm-generation-config vllm \
    --vllm-override-generation-config '{"temperature":0.0,"top_p":1.0,"repetition_penalty":1.0}' \
    --disable-vllm-speculative \
    --python-bin "${PYTHON_BIN}" \
    --openclaw-bin "${GAIA_ROOT}/.local/npm/lib/node_modules/openclaw/openclaw.mjs" \
    --openclaw-config "${GAIA_ROOT}/config/openclaw.template.json" \
    --node-bin-dir "${GAIA_ROOT}/.local/node/bin" \
    --out-dir "${RUN_ROOT}" \
    2>&1 | tee "${RUN_ROOT}/gaia_sweep.log"
}

cleanup() {
  if [[ "${KEEP_BACKENDS:-0}" != "1" ]]; then
    stop_all_backends || true
  fi
}
trap cleanup EXIT INT TERM

cat > "${RUN_ROOT}/run_config.txt" <<EOF
NODES_CSV=${NODES_CSV}
PORT=${PORT}
REMOTE_RUNNER=${REMOTE_RUNNER}
SLURM_TARGET_JOB_ID=${SLURM_TARGET_JOB_ID}
MODEL=${MODEL}
IMG=${IMG}
TP_LIST=${TP_LIST}
CONCURRENCY_LIST=${CONCURRENCY_LIST}
ROUNDS=${ROUNDS}
WARMUP_COUNT=${WARMUP_COUNT}
IDX_START=${IDX_START}
IDX_END=${IDX_END}
VLLM_LOG_STATS_INTERVAL=${VLLM_LOG_STATS_INTERVAL}
EOF

echo "[info] run_root=${RUN_ROOT}"
echo "[info] nodes=${NODES_CSV}"
echo "[info] tp_list=${TP_LIST} concurrency_list=${CONCURRENCY_LIST} rounds=${ROUNDS} idx=[${IDX_START},${IDX_END})"
echo "[info] starting ${#NODES[@]} TP4 vLLM backends"
start_all_backends
echo "[info] all backends ready; starting GAIA sweep"
run_gaia_sweep
echo "[ok] GAIA sweep done: ${RUN_ROOT}"
