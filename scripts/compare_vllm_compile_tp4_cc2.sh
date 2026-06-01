#!/usr/bin/env bash
set -euo pipefail

# Compare TP4/cc2 GAIA decode throughput with and without vLLM compile.
# This intentionally avoids Slurm/squeue and drives one already allocated node
# through SSH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAIA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NODE="${NODE:-frontier10362}"
PORT="${PORT:-8011}"
WORK="${WORK:-/lustre/orion/gen150/scratch/zye25/Agentic}"
IMG="${IMG:-${WORK}/containers/vllm-openai-rocm-nightly.sif}"
MODEL="${MODEL:-${WORK}/hf/models/Qwen3.5-27B}"
PYTHON_BIN="${PYTHON_BIN:-/lustre/orion/gen150/scratch/zye25/conda-envs/AGAI/bin/python}"
ROWS_JSONL="${ROWS_JSONL:-${GAIA_ROOT}/data/gaia_2023_all_validation/rows.jsonl}"
LOCAL_CONFIG_FILE="${LOCAL_CONFIG_FILE:-${GAIA_ROOT}/config/local.env}"

IDX_START="${IDX_START:-0}"
IDX_END="${IDX_END:-2}"
ROUNDS="${ROUNDS:-1}"
WARMUP_COUNT="${WARMUP_COUNT:-0}"
TASK_TIMEOUT_S="${TASK_TIMEOUT_S:-600}"
BACKEND_READY_TIMEOUT_S="${BACKEND_READY_TIMEOUT_S:-2400}"
GPU_SAMPLE_SEC="${GPU_SAMPLE_SEC:-1}"
VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
OPENCLAW_CONTEXT_WINDOW="${OPENCLAW_CONTEXT_WINDOW:-23552}"
OPENCLAW_CONTEXT_MARGIN="${OPENCLAW_CONTEXT_MARGIN:-1024}"
OPENCLAW_MAX_OUTPUT_TOKENS="${OPENCLAW_MAX_OUTPUT_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${GAIA_ROOT}/runs/vllm_compile_compare_tp4_cc2_${RUN_STAMP}}"
mkdir -p "${RUN_ROOT}" "${WORK}/logs" "${WORK}/logs/vllm_launcher"

if [[ -f "${LOCAL_CONFIG_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${LOCAL_CONFIG_FILE}"
  set +a
fi

setup_proxy() {
  local proxy
  proxy="${PROXY:-http://proxy.ccs.ornl.gov:3128}"
  export http_proxy="${http_proxy:-${proxy}}"
  export https_proxy="${https_proxy:-${proxy}}"
  export HTTP_PROXY="${HTTP_PROXY:-${proxy}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${proxy}}"
  export no_proxy="localhost,127.0.0.1,::1,.ornl.gov,.olcf.ornl.gov,.frontier.olcf.ornl.gov,${NODE},$(hostname -s),$(hostname -f)"
  export NO_PROXY="${no_proxy}"
}

stop_backend() {
  ssh "${NODE}" bash -s <<'REMOTE'
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
ps -u "${USER}" -o pid=,args= | awk '/VLLM::|vllm.entrypoints|vllm serve|\/usr\/local\/bin\/vllm/ && !/awk/ {print}'
REMOTE
}

start_backend() {
  local label eager run_tag vllm_log pid_file start_log ready_s
  label="$1"
  eager="$2"
  run_tag="qwen35_tp4_cc2_${label}_${RUN_STAMP}"
  vllm_log="${WORK}/logs/vllm_${NODE}_${PORT}_${run_tag}.log"
  pid_file="${WORK}/logs/vllm_launcher/${run_tag}.pid"
  start_log="${RUN_ROOT}/backend_start_${label}.log"

  stop_backend >/dev/null || true

  set +e
  ssh "${NODE}" bash -s -- \
    "${run_tag}" "${vllm_log}" "${pid_file}" "${eager}" "${MODEL}" "${IMG}" "${PORT}" \
    "${VLLM_LOG_STATS_INTERVAL}" "${BACKEND_READY_TIMEOUT_S}" "${GPU_MEM_UTIL}" "${MAX_MODEL_LEN}" \
    | tee "${start_log}"
  REMOTE_STATUS=${PIPESTATUS[0]}
  set -e
  if [[ "${REMOTE_STATUS}" != "0" ]]; then
    echo "[error] backend start failed for ${label}; see ${start_log}" >&2
    return "${REMOTE_STATUS}"
  fi

  ready_s="$(awk -F'=' '/^READY elapsed=/{gsub(/s/, "", $2); print $2}' "${start_log}" | tail -n 1)"
  echo "${ready_s:-}" > "${RUN_ROOT}/backend_ready_${label}.txt"
  echo "${vllm_log}" > "${RUN_ROOT}/vllm_log_${label}.txt"
}

run_remote_backend_script() {
  : # Documentation-only placeholder; the real remote script is below.
}

run_gaia() {
  local label eager out_dir vllm_log
  label="$1"
  eager="$2"
  out_dir="${RUN_ROOT}/${label}"
  vllm_log="$(cat "${RUN_ROOT}/vllm_log_${label}.txt")"

  setup_proxy
  export VLLM_LOG_STATS_INTERVAL

  local args=(
    "${GAIA_ROOT}/scripts/run_gaia_concurrency.py"
    --job-id 0
    --node "${NODE}"
    --remote-runner ssh
    --skip-slurm-validation
    --external-base-url "http://${NODE}:${PORT}/v1"
    --external-vllm-log "${vllm_log}"
    --rows-jsonl "${ROWS_JSONL}"
    --idx-start "${IDX_START}"
    --idx-end "${IDX_END}"
    --tp-list 4
    --concurrency-list 2
    --rounds "${ROUNDS}"
    --warmup-count "${WARMUP_COUNT}"
    --gpu-sample-sec "${GPU_SAMPLE_SEC}"
    --timeout "${TASK_TIMEOUT_S}"
    --max-model-len "${MAX_MODEL_LEN}"
    --openclaw-context-window "${OPENCLAW_CONTEXT_WINDOW}"
    --openclaw-context-margin "${OPENCLAW_CONTEXT_MARGIN}"
    --model-max-tokens "${OPENCLAW_MAX_OUTPUT_TOKENS}"
    --gpu-mem-util "${GPU_MEM_UTIL}"
    --vllm-entrypoint serve
    --vllm-generation-config vllm
    --vllm-override-generation-config '{"temperature":0.0,"top_p":1.0,"repetition_penalty":1.0}'
    --disable-vllm-speculative
    --python-bin "${PYTHON_BIN}"
    --openclaw-bin "${GAIA_ROOT}/.local/npm/lib/node_modules/openclaw/openclaw.mjs"
    --openclaw-config "${GAIA_ROOT}/config/openclaw.template.json"
    --node-bin-dir "${GAIA_ROOT}/.local/node/bin"
    --out-dir "${out_dir}"
  )
  if [[ "${eager}" == "1" ]]; then
    args+=(--enforce-eager)
  fi

  "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${RUN_ROOT}/gaia_${label}.log"
}

write_summary() {
  python3 - "${RUN_ROOT}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
labels = [("eager", "no_compile_enforce_eager"), ("compile", "compile_default")]

def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def mean(vals):
    return statistics.fmean(vals) if vals else 0.0

def q(vals, pct):
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = max(0, min(len(vals) - 1, int((len(vals) * pct + 0.999999) - 1)))
    return vals[idx]

rows = []
for label, desc in labels:
    metrics = run_root / label / "configs" / "tp4_cc2_r1" / "metrics"
    vllm_rows = read_jsonl(metrics / "vllm_timeseries.jsonl")
    task_rows = read_jsonl(metrics / "task_metrics.jsonl")
    decode_all = [float(r.get("decode_tps", 0.0)) for r in vllm_rows]
    decode_active = [
        float(r.get("decode_tps", 0.0))
        for r in vllm_rows
        if int(r.get("running_reqs", 0)) > 0 and float(r.get("decode_tps", 0.0)) > 0.0
    ]
    ready_file = run_root / f"backend_ready_{label}.txt"
    log_file = run_root / f"vllm_log_{label}.txt"
    ready_s = ready_file.read_text(encoding="utf-8").strip() if ready_file.exists() else ""
    vllm_log = log_file.read_text(encoding="utf-8").strip() if log_file.exists() else ""
    ok = sum(1 for r in task_rows if r.get("status") == "ok")
    exact = sum(1 for r in task_rows if r.get("exact_match") is True)
    lat = [float(r.get("task_latency_s", 0.0)) for r in task_rows if r.get("task_latency_s") is not None]
    rows.append({
        "label": label,
        "desc": desc,
        "ready_s": ready_s,
        "decode_mean_all": mean(decode_all),
        "decode_mean_active": mean(decode_active),
        "decode_p50_active": q(decode_active, 0.50),
        "decode_p95_active": q(decode_active, 0.95),
        "decode_max": max(decode_all) if decode_all else 0.0,
        "vllm_points": len(vllm_rows),
        "active_points": len(decode_active),
        "tasks": len(task_rows),
        "ok": ok,
        "exact": exact,
        "task_latency_mean": mean(lat),
        "out_dir": str(run_root / label),
        "vllm_log": vllm_log,
    })

winner = ""
if len(rows) == 2 and rows[0]["decode_mean_active"] and rows[1]["decode_mean_active"]:
    winner_row = max(rows, key=lambda r: r["decode_mean_active"])
    loser_row = min(rows, key=lambda r: r["decode_mean_active"])
    ratio = winner_row["decode_mean_active"] / loser_row["decode_mean_active"] if loser_row["decode_mean_active"] else 0.0
    winner = f"{winner_row['label']} higher by active decode mean, ratio={ratio:.3f}x"

lines = [
    "# vLLM Compile Decode Comparison",
    "",
    f"- run_root: `{run_root}`",
    f"- winner: `{winner or 'n/a'}`",
    "",
    "| label | mode | ready_s | decode_mean_all | decode_mean_active | decode_p50_active | decode_p95_active | decode_max | vllm_points | active_points | tasks_ok/exact/total | task_latency_mean_s |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['label']} | {r['desc']} | {r['ready_s'] or 'n/a'} | "
        f"{r['decode_mean_all']:.4f} | {r['decode_mean_active']:.4f} | "
        f"{r['decode_p50_active']:.4f} | {r['decode_p95_active']:.4f} | "
        f"{r['decode_max']:.4f} | {r['vllm_points']} | {r['active_points']} | "
        f"{r['ok']}/{r['exact']}/{r['tasks']} | {r['task_latency_mean']:.4f} |"
    )
lines.extend(["", "## Outputs"])
for r in rows:
    lines.append(f"- {r['label']} out_dir: `{r['out_dir']}`")
    lines.append(f"- {r['label']} vllm_log: `{r['vllm_log']}`")

(run_root / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(run_root / "comparison_summary.md")
PY
}

run_one_mode() {
  local label eager
  label="$1"
  eager="$2"
  echo "[info] starting backend label=${label} eager=${eager}"
  start_backend "${label}" "${eager}" <<'REMOTE'
set -euo pipefail
RUN_TAG="$1"
LOG="$2"
PID_FILE="$3"
EAGER="$4"
MODEL="$5"
IMG="$6"
PORT="$7"
STATS_INTERVAL="$8"
READY_TIMEOUT_S="$9"
GPU_MEM_UTIL="${10}"
MAX_MODEL_LEN="${11}"
CACHE="/tmp/${USER}/openclaw_vllm_cache"
MIOPEN="/tmp/${USER}/openclaw_miopen_cache"
mkdir -p "$(dirname "${PID_FILE}")" "${CACHE}" "${CACHE}/torchinductor" /tmp/triton/cache "${MIOPEN}"
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
if [[ "${EAGER}" == "1" ]]; then
  args+=(--enforce-eager)
fi

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
echo "EAGER=${EAGER}"

start_ts="$(date +%s)"
deadline="$((start_ts + READY_TIMEOUT_S))"
while true; do
  if curl -fsS --max-time 4 "http://127.0.0.1:${PORT}/v1/models" | grep -q '"data"'; then
    now="$(date +%s)"
    echo "READY elapsed=$((now - start_ts))s"
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
  echo "[info] running GAIA label=${label}"
  run_gaia "${label}" "${eager}"
  if [[ "${KEEP_BACKEND:-0}" != "1" ]]; then
    stop_backend >/dev/null || true
  fi
}

cleanup() {
  if [[ "${KEEP_BACKEND:-0}" != "1" ]]; then
    stop_backend >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

cat > "${RUN_ROOT}/run_config.txt" <<EOF
NODE=${NODE}
PORT=${PORT}
MODEL=${MODEL}
IMG=${IMG}
IDX_START=${IDX_START}
IDX_END=${IDX_END}
ROUNDS=${ROUNDS}
WARMUP_COUNT=${WARMUP_COUNT}
VLLM_LOG_STATS_INTERVAL=${VLLM_LOG_STATS_INTERVAL}
EOF

echo "[info] run_root=${RUN_ROOT}"
echo "[info] node=${NODE} port=${PORT} tp=4 cc=2 idx=[${IDX_START},${IDX_END})"

run_one_mode "eager" "1"
run_one_mode "compile" "0"

summary_path="$(write_summary)"
echo "[ok] comparison summary: ${summary_path}"
