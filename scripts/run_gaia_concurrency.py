#!/usr/bin/env python3
"""Run GAIA cases at multiple concurrency levels against one local vLLM backend.

Most knobs are passed by scripts/run_tp4_concurrency.sh from
config/tp4_concurrency.env. This Python file owns the mechanics:
start backend, generate per-worker OpenClaw configs, run GAIA rows, collect
session/latency/backend metrics, and build the aggregate report.
"""

import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from run_gaia_manual_question import (
    build_prompt,
    extract_primary_answer,
    normalize_text,
    run_openclaw_agent,
)

REQUEST_COUNTER_CANDIDATES = [
    "vllm:request_success_total",
    "vllm:request_completed_total",
    "vllm:requests_total",
    "vllm:num_requests_total",
    "vllm_request_success_total",
    "vllm_request_completed_total",
]


@dataclass
class ConfigPoint:
    config_id: str
    tp: int
    concurrency: int
    round_id: int


def parse_args() -> argparse.Namespace:
    openclaw_root = Path(__file__).resolve().parents[1]
    work_root = openclaw_root.parent
    parser = argparse.ArgumentParser(description="Run GAIA concurrency matrix benchmark.")
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--node", required=True)
    parser.add_argument(
        "--rows-jsonl",
        default=str(openclaw_root / "data" / "gaia_top20_bundle_20260414_052552" / "rows_textonly10.jsonl"),
    )
    parser.add_argument("--case-mode", choices=["all", "first8"], default="all")
    parser.add_argument("--idx-start", type=int, default=0, help="Inclusive row idx lower bound after loading rows.")
    parser.add_argument("--idx-end", type=int, default=-1, help="Exclusive row idx upper bound. -1 means no upper bound.")
    parser.add_argument("--tp-list", default="2,4")
    parser.add_argument("--concurrency-list", default="1,2,4,6,8")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--gpu-sample-sec", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--backend-ready-timeout-s",
        type=int,
        default=1800,
        help="Seconds to wait for vLLM /v1/models readiness before failing backend startup.",
    )
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument(
        "--model",
        default="/lustre/orion/gen150/scratch/zye25/Agentic/hf/models/Qwen3.5-27B",
    )
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument(
        "--openclaw-context-window",
        type=int,
        default=0,
        help="contextWindow written into OpenClaw worker configs. Defaults to max_model_len - model_max_tokens - openclaw_context_margin.",
    )
    parser.add_argument(
        "--openclaw-context-margin",
        type=int,
        default=1024,
        help="Safety margin subtracted from the OpenClaw contextWindow auto-default to cover chat template/tool rendering overhead.",
    )
    parser.add_argument("--gpu-mem-util", type=float, default=0.95)
    parser.add_argument("--model-max-tokens", type=int, default=8192)
    parser.add_argument("--vllm-api-key", default="vllm-local")
    parser.add_argument(
        "--vllm-entrypoint",
        choices=["api_server", "serve"],
        default="serve",
        help="Use vLLM's official 'vllm serve' entrypoint or the legacy api_server module.",
    )
    parser.add_argument(
        "--vllm-pythonpath",
        default="",
        help="Optional PYTHONPATH prepended inside the vLLM container.",
    )
    parser.add_argument(
        "--vllm-generation-config",
        default="vllm",
        help="vLLM --generation-config value. Use 'vllm' to ignore model HF sampling defaults.",
    )
    parser.add_argument(
        "--vllm-override-generation-config",
        default='{"temperature":0.0,"top_p":1.0,"repetition_penalty":1.0}',
        help="JSON string passed to vLLM --override-generation-config.",
    )
    parser.add_argument(
        "--vllm-reasoning-parser",
        default="",
        help="vLLM --reasoning-parser value. Empty auto-selects qwen3 for Qwen3.5 models.",
    )
    parser.add_argument(
        "--vllm-speculative-config",
        default="",
        help="JSON string passed to vLLM --speculative-config. Empty enables Qwen3.5 MTP auto-default.",
    )
    parser.add_argument(
        "--disable-vllm-speculative",
        action="store_true",
        help="Do not pass --speculative-config to vLLM, including Qwen3.5's auto MTP default.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(openclaw_root / "runs" / f"gaia_concurrency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument("--quantile-method", default="nearest-rank")
    parser.add_argument("--session-store", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python-bin", default="/lustre/orion/gen150/scratch/zye25/conda-envs/AGAI/bin/python")
    parser.add_argument("--openclaw-bin", default=str(openclaw_root / ".local" / "npm" / "lib" / "node_modules" / "openclaw" / "openclaw.mjs"))
    parser.add_argument("--openclaw-config", default=str(openclaw_root / "state" / "openclaw.json"))
    parser.add_argument("--node-bin-dir", default=str(openclaw_root / ".local" / "node" / "bin"))
    parser.add_argument("--manage-backend-script", default=str(openclaw_root / "scripts" / "manage_vllm.sh"))
    parser.add_argument("--img", default=str(work_root / "containers" / "vllm-openai-rocm-nightly.sif"))
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seed_minimal_workspace(workspace: Path) -> None:
    ensure_dir(workspace)
    # Pre-create empty instruction files so OpenClaw does not inject its
    # heavyweight default workspace templates into every worker prompt.
    for name in (
        "AGENTS.md",
        "SOUL.md",
        "TOOLS.md",
        "IDENTITY.md",
        "USER.md",
        "HEARTBEAT.md",
        "BOOTSTRAP.md",
    ):
        p = workspace / name
        if not p.exists():
            p.write_text("", encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_cmd(cmd: List[str], timeout: Optional[int] = None, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, env=env, cwd=str(cwd) if cwd else None)


def run_srun(job_id: int, node: str, script: str, timeout: int = 20) -> subprocess.CompletedProcess:
    cmd = [
        "srun",
        "--overlap",
        "--jobid",
        str(job_id),
        "-N1",
        "-n1",
        "--nodelist",
        node,
        "bash",
        "-lc",
        script,
    ]
    return run_cmd(cmd, timeout=timeout)


def parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def iso_or_empty(dt: Optional[datetime]) -> str:
    return dt.isoformat() if dt else ""


def read_rows(rows_path: Path, case_mode: str, idx_start: int = 0, idx_end: int = -1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in rows_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        rows.append(obj)

    if idx_start < 0:
        raise ValueError("--idx-start must be >= 0")
    if idx_end >= 0 and idx_end < idx_start:
        raise ValueError("--idx-end must be >= --idx-start, or -1")
    if idx_start or idx_end >= 0:
        rows = [
            row for row in rows
            if int(row.get("idx", -1)) >= idx_start and (idx_end < 0 or int(row.get("idx", -1)) < idx_end)
        ]

    if case_mode == "first8":
        return rows[:8]
    return rows


def parse_list_int(spec: str) -> List[int]:
    values: List[int] = []
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(int(x))
    return values


def is_qwen35_model(model: str) -> bool:
    return re.search(r"qwen3[._-]?5", str(model).lower()) is not None


def qwen35_speculative_config() -> str:
    return '{"method":"mtp","num_speculative_tokens":2}'


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    write_json(path, manifest)


def set_manifest_status(manifest: Dict[str, Any], config_id: str, status: str, note: str = "") -> None:
    for c in manifest.get("configs", []):
        if c.get("config_id") == config_id:
            c["status"] = status
            c["note"] = note
            c["updated_at"] = datetime.now().isoformat()
            return


def parse_endpoints_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for key in ("ENDPOINT_A", "ENDPOINT_1", "BASE_URLS"):
        m = re.search(rf"export\s+{key}=(.+)", text)
        if m:
            v = m.group(1).strip().strip("\"").strip("'")
            if v:
                return v
    raise RuntimeError(f"unable to parse endpoint from {path}")


def start_backend(args: argparse.Namespace, tp: int, run_tag: str, work_root: Path) -> Tuple[str, Path]:
    tool_call_parser = "hermes"
    reasoning_parser = str(args.vllm_reasoning_parser or "")
    speculative_config = str(args.vllm_speculative_config or "")
    vllm_pythonpath = str(args.vllm_pythonpath or "")
    if is_qwen35_model(args.model):
        tool_call_parser = "qwen3_coder"
        if not reasoning_parser:
            reasoning_parser = "qwen3"
        if not speculative_config and not args.disable_vllm_speculative:
            speculative_config = qwen35_speculative_config()
        if not vllm_pythonpath:
            overlay = work_root / "openclaw_GAIA" / ".pydeps-vllm-qwen35"
            if overlay.exists():
                vllm_pythonpath = str(overlay)
    if args.vllm_override_generation_config:
        json.loads(args.vllm_override_generation_config)
    if speculative_config:
        json.loads(speculative_config)
    env = os.environ.copy()
    env.update(
        {
            "SLURM_TARGET_JOB_ID": str(args.job_id),
            "SLURM_JOB_NODELIST": str(args.node),
            "WORK": str(work_root),
            "IMG": str(args.img),
            "MODEL": str(args.model),
            "PORT": str(args.port),
            "TP_SIZE": str(tp),
            "GPU_MEMORY_UTILIZATION": str(args.gpu_mem_util),
            "MAX_MODEL_LEN": str(args.max_model_len),
            "RUN_TAG": run_tag,
            "READY_TIMEOUT_S": str(args.backend_ready_timeout_s),
            "WAIT_BACKENDS_READY": "1",
            "ENFORCE_EAGER": "1" if args.enforce_eager else "0",
            "VLLM_ENTRYPOINT": str(args.vllm_entrypoint),
            "VLLM_PYTHONPATH": vllm_pythonpath,
            "ENABLE_AUTO_TOOL_CHOICE": "1",
            "TOOL_CALL_PARSER": tool_call_parser,
            "GENERATION_CONFIG": str(args.vllm_generation_config or ""),
            "OVERRIDE_GENERATION_CONFIG": str(args.vllm_override_generation_config or ""),
            "REASONING_PARSER": reasoning_parser,
            "SPECULATIVE_CONFIG": speculative_config,
        }
    )
    log(f"starting backend TP={tp} run_tag={run_tag}")
    proc_timeout_s = max(2200, int(args.backend_ready_timeout_s) + 400)
    proc = run_cmd(["bash", str(args.manage_backend_script), "start"], timeout=proc_timeout_s, env=env)
    if proc.returncode != 0:
        vllm_log = work_root / "logs" / f"vllm_{args.node}_{args.port}.log"
        log_tail = ""
        if vllm_log.exists():
            lines = vllm_log.read_text(encoding="utf-8", errors="ignore").splitlines()
            log_tail = "\n".join(lines[-80:])
        raise RuntimeError(
            "backend start failed:\n"
            f"manage stderr:\n{proc.stderr}\n"
            f"manage stdout:\n{proc.stdout}\n"
            f"vLLM log: {vllm_log}\n"
            f"vLLM log tail:\n{log_tail}"
        )
    endpoints = work_root / "logs" / "vllm_launcher" / f"{run_tag}.endpoints.sh"
    if not endpoints.exists():
        candidates = sorted((work_root / "logs" / "vllm_launcher").glob(f"*_p{args.port}.endpoints.sh"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError("backend started but endpoints file missing")
        endpoints = candidates[0]
    base_url = parse_endpoints_file(endpoints)
    return base_url, endpoints


def stop_backend(args: argparse.Namespace, run_tag: str, work_root: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "SLURM_TARGET_JOB_ID": str(args.job_id),
            "SLURM_JOB_NODELIST": str(args.node),
            "WORK": str(work_root),
            "PORT": str(args.port),
            "RUN_TAG": run_tag,
        }
    )
    run_cmd(["bash", str(args.manage_backend_script), "stop"], timeout=180, env=env)


def check_backend_health(args: argparse.Namespace) -> Tuple[bool, str]:
    proc = run_srun(
        args.job_id,
        args.node,
        f"curl -fsS --max-time 4 http://127.0.0.1:{args.port}/v1/models | grep -q '\"data\"' && echo ok",
        timeout=25,
    )
    ok = proc.returncode == 0 and "ok" in (proc.stdout or "")
    detail = (proc.stderr or proc.stdout or "").strip()
    return ok, detail[:300]


def extract_payload_answer(parsed: Dict[str, Any]) -> str:
    payloads = parsed.get("payloads") if isinstance(parsed, dict) else None
    if not payloads:
        return ""
    parts: List[str] = []
    for item in payloads:
        if isinstance(item, dict):
            t = (item.get("text") or "").strip()
            if t:
                parts.append(t)
    return "\n".join(parts)


def parse_session_detailed(session_file: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "assistant_turns": 0,
        "tool_calls": 0,
        "tool_success_calls": 0,
        "tool_success_rate": 0.0,
        "last_stop_reason": None,
        "final_assistant_text": "",
        "final_answer_primary": "",
        "step_rows": [],
        "inference_rows": [],
        "tool_rows": [],
    }
    if not session_file.exists():
        return out

    events: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for line in session_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        ts = parse_iso(str(obj.get("timestamp", "")))
        event = {
            "id": obj.get("id"),
            "type": obj.get("type"),
            "parentId": obj.get("parentId"),
            "ts": ts,
            "obj": obj,
        }
        events.append(event)
        if event["id"]:
            by_id[str(event["id"])] = event

    pending_tool: Dict[str, Dict[str, Any]] = {}
    step_no = 0
    final_text = ""

    for ev in events:
        if ev["type"] != "message":
            continue
        obj = ev["obj"]
        msg = obj.get("message") or {}
        role = msg.get("role")

        if role == "assistant":
            step_no += 1
            out["assistant_turns"] += 1

            parent_id = str(ev.get("parentId") or "")
            parent_ev = by_id.get(parent_id)
            parent_ts = parent_ev.get("ts") if parent_ev else None
            parent_type = parent_ev.get("type") if parent_ev else None
            assistant_ts = ev.get("ts")
            step_latency_s = None
            if assistant_ts and parent_ts:
                step_latency_s = (assistant_ts - parent_ts).total_seconds()

            usage = msg.get("usage") or {}
            input_tokens = usage.get("input")
            output_tokens = usage.get("output")
            total_tokens = usage.get("totalTokens")
            stop_reason = msg.get("stopReason")
            if stop_reason:
                out["last_stop_reason"] = stop_reason

            step_row = {
                "step_no": step_no,
                "assistant_msg_id": str(ev.get("id") or ""),
                "parent_event_id": parent_id,
                "step_start_ts": iso_or_empty(parent_ts),
                "step_end_ts": iso_or_empty(assistant_ts),
                "step_latency_s": step_latency_s,
                "stop_reason": stop_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            out["step_rows"].append(step_row)

            inf_row = {
                "assistant_msg_id": str(ev.get("id") or ""),
                "parent_type": parent_type,
                "parent_ts": iso_or_empty(parent_ts),
                "assistant_ts": iso_or_empty(assistant_ts),
                "inference_latency_s": step_latency_s,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            out["inference_rows"].append(inf_row)

            text_parts: List[str] = []
            for block in msg.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "toolCall":
                    call_id = str(block.get("id") or "")
                    if call_id:
                        pending_tool[call_id] = {
                            "tool_call_id": call_id,
                            "tool_name": block.get("name") or "",
                            "tool_start_ts": iso_or_empty(assistant_ts),
                            "tool_start_dt": assistant_ts,
                            "assistant_msg_id": str(ev.get("id") or ""),
                        }
                elif btype == "text":
                    t = (block.get("text") or "").strip()
                    if t:
                        text_parts.append(t)

            if stop_reason == "stop" and text_parts:
                final_text = "\n".join(text_parts).strip()

        elif role == "toolResult":
            call_id = str(msg.get("toolCallId") or "")
            details = msg.get("details") if isinstance(msg.get("details"), dict) else {}
            entry = pending_tool.pop(call_id, {
                "tool_call_id": call_id,
                "tool_name": msg.get("toolName") or "",
                "tool_start_ts": "",
                "tool_start_dt": None,
            })
            end_dt = ev.get("ts")
            took_ms_raw = details.get("tookMs") if details else None
            took_ms: Optional[float] = None
            if took_ms_raw is not None:
                try:
                    took_ms = float(took_ms_raw)
                except Exception:
                    took_ms = None

            latency_s: Optional[float] = None
            start_dt = entry.get("tool_start_dt")
            if took_ms is not None:
                latency_s = took_ms / 1000.0
            elif start_dt and end_dt:
                latency_s = (end_dt - start_dt).total_seconds()

            status_raw = str(details.get("status", "")).lower() if details else ""
            is_error = bool(msg.get("isError")) or status_raw == "error"
            tool_status = "error" if is_error else "ok"
            error_text = ""
            if is_error:
                if isinstance(details, dict):
                    error_text = str(details.get("error") or details.get("message") or "")
                if not error_text:
                    content = msg.get("content") or []
                    if content and isinstance(content[0], dict):
                        error_text = str(content[0].get("text") or "")[:400]

            row = {
                "tool_call_id": entry.get("tool_call_id") or call_id,
                "tool_name": entry.get("tool_name") or msg.get("toolName") or "",
                "tool_start_ts": entry.get("tool_start_ts") or "",
                "tool_end_ts": iso_or_empty(end_dt),
                "tool_latency_s": latency_s,
                "tool_status": tool_status,
                "is_error": is_error,
                "error_text": error_text,
            }
            out["tool_rows"].append(row)

    out["tool_calls"] = len(out["tool_rows"])
    out["tool_success_calls"] = sum(1 for r in out["tool_rows"] if not r.get("is_error"))
    out["tool_success_rate"] = (
        out["tool_success_calls"] / out["tool_calls"] if out["tool_calls"] > 0 else 1.0
    )
    out["final_assistant_text"] = final_text
    out["final_answer_primary"] = extract_primary_answer(final_text)
    return out


def wait_for_session_file(session_store: Path, session_id: str, wait_s: int = 12) -> Path:
    target = session_store / f"{session_id}.jsonl"
    t0 = time.time()
    while time.time() - t0 <= wait_s:
        if target.exists():
            return target
        time.sleep(0.5)
    return target


def resolve_session_file(runtime_home: Path, session_store: Path, session_id: str) -> Path:
    candidate_stores = [
        runtime_home / ".openclaw" / "agents" / "main" / "sessions",
        session_store,
    ]
    for idx, store in enumerate(candidate_stores):
        wait_s = 12 if idx == 0 else 2
        found = wait_for_session_file(store, session_id, wait_s=wait_s)
        if found.exists():
            return found
    return candidate_stores[0] / f"{session_id}.jsonl"


def create_worker_config(template_cfg: Path, out_cfg: Path, workspace: Path, base_url: str, model_id: str, api_key: str, context_window: int, model_max_tokens: int) -> None:
    cfg = json.loads(template_cfg.read_text(encoding="utf-8"))
    cfg.setdefault("models", {}).setdefault("providers", {})
    cfg["models"]["providers"]["vllm"] = {
        "baseUrl": base_url,
        "apiKey": api_key,
        "api": "openai-completions",
        "models": [
            {
                "id": model_id,
                "name": f"vLLM {model_id}",
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": context_window,
                "maxTokens": model_max_tokens,
                "compat": {
                    "supportsStore": False,
                    "supportsDeveloperRole": False,
                    "supportsStrictMode": False,
                },
            }
        ],
    }
    cfg.setdefault("agents", {}).setdefault("defaults", {})
    cfg["agents"]["defaults"].setdefault("model", {})
    cfg["agents"]["defaults"]["model"]["primary"] = f"vllm/{model_id}"
    cfg["agents"]["defaults"]["workspace"] = str(workspace)

    seed_minimal_workspace(workspace)
    ensure_dir(out_cfg.parent)
    write_json(out_cfg, cfg)


def parse_metric_text(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "{" in s:
            continue
        parts = s.split()
        if len(parts) != 2:
            continue
        name, value = parts[0], parts[1]
        try:
            out[name] = float(value)
        except Exception:
            continue
    return out


class GPUSampler(threading.Thread):
    def __init__(self, job_id: int, node: str, interval_s: float):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.node = node
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.samples: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def _parse_float(self, x: Any) -> Optional[float]:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.upper().startswith("N/A"):
            return None
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def run(self) -> None:
        while not self.stop_event.is_set():
            t0 = time.time()
            ts = datetime.now().isoformat()
            proc = run_srun(
                self.job_id,
                self.node,
                "/opt/rocm-default/bin/rocm-smi --showuse --showmemuse --showpower --showtemp --json 2>/dev/null",
                timeout=20,
            )
            if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
                try:
                    data = json.loads(proc.stdout)
                    for card, vals in data.items():
                        m = re.search(r"(\d+)$", card)
                        gpu_id = int(m.group(1)) if m else -1
                        sample = {
                            "ts": ts,
                            "gpu_id": gpu_id,
                            "gpu_use_pct": self._parse_float(vals.get("GPU use (%)")),
                            "vram_pct": self._parse_float(vals.get("GPU Memory Allocated (VRAM%)")),
                            "power_w": self._parse_float(vals.get("Average Graphics Package Power (W)")),
                            "temp_edge_c": self._parse_float(vals.get("Temperature (Sensor edge) (C)")),
                            "temp_mem_c": self._parse_float(vals.get("Temperature (Sensor memory) (C)")),
                        }
                        self.samples.append(sample)
                except Exception as e:
                    self.errors.append(f"gpu parse error: {e}")
            else:
                err = proc.stderr.strip() or proc.stdout.strip()
                if err:
                    self.errors.append(err[:240])
            dt = time.time() - t0
            sleep_s = self.interval_s - dt
            if sleep_s > 0:
                self.stop_event.wait(timeout=sleep_s)

    def stop(self) -> None:
        self.stop_event.set()


class BackendSampler(threading.Thread):
    def __init__(self, job_id: int, node: str, port: int, interval_s: float):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.node = node
        self.port = port
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.samples: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            t0 = time.time()
            ts = datetime.now().isoformat()
            proc = run_srun(
                self.job_id,
                self.node,
                f"curl -fsS --max-time 3 http://127.0.0.1:{self.port}/metrics",
                timeout=20,
            )
            row: Dict[str, Any] = {"ts": ts}
            if proc.returncode == 0:
                metrics = parse_metric_text(proc.stdout)
                for name in REQUEST_COUNTER_CANDIDATES:
                    if name in metrics:
                        row[name] = metrics[name]
                self.samples.append(row)
            else:
                err = proc.stderr.strip() or proc.stdout.strip()
                if err:
                    self.errors.append(err[:240])
            dt = time.time() - t0
            sleep_s = self.interval_s - dt
            if sleep_s > 0:
                self.stop_event.wait(timeout=sleep_s)

    def stop(self) -> None:
        self.stop_event.set()


def parse_vllm_timeseries(log_path: Path, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not log_path.exists():
        return rows
    year = start_dt.year
    rx = re.compile(
        r"INFO\s+(\d{2})-(\d{2})\s+(\d{2}:\d{2}:\d{2}).*Avg prompt throughput:\s*([0-9.]+)\s*tokens/s,\s*Avg generation throughput:\s*([0-9.]+)\s*tokens/s,\s*Running:\s*(\d+)\s*reqs,\s*(?:Pending|Waiting):\s*(\d+)\s*reqs,.*GPU KV cache usage:\s*([0-9.]+)%"
    )
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = rx.search(line)
        if not m:
            continue
        month, day = int(m.group(1)), int(m.group(2))
        hhmmss = m.group(3)
        ts = datetime.strptime(f"{year}-{month:02d}-{day:02d} {hhmmss}", "%Y-%m-%d %H:%M:%S")
        if ts < start_dt or ts > end_dt:
            continue
        rows.append(
            {
                "ts": ts.isoformat(sep=" "),
                "prefill_tps": float(m.group(4)),
                "decode_tps": float(m.group(5)),
                "running_reqs": int(m.group(6)),
                "pending_reqs": int(m.group(7)),
                "gpu_kv_cache_pct": float(m.group(8)),
            }
        )
    return rows


def build_request_throughput_series(samples: List[Dict[str, Any]], task_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if len(samples) >= 2:
        samples_sorted = sorted(samples, key=lambda x: x.get("ts", ""))
        chosen_metric = ""
        for name in REQUEST_COUNTER_CANDIDATES:
            vals = [s.get(name) for s in samples_sorted if s.get(name) is not None]
            if len(vals) >= 2 and (max(vals) - min(vals)) > 0:
                chosen_metric = name
                break
        if chosen_metric:
            for i in range(1, len(samples_sorted)):
                a = samples_sorted[i - 1]
                b = samples_sorted[i]
                va = a.get(chosen_metric)
                vb = b.get(chosen_metric)
                ta = parse_iso(str(a.get("ts", "")))
                tb = parse_iso(str(b.get("ts", "")))
                if va is None or vb is None or ta is None or tb is None:
                    continue
                dt = (tb - ta).total_seconds()
                if dt <= 0:
                    continue
                delta = vb - va
                if delta < 0:
                    continue
                rows.append(
                    {
                        "window_start": ta.isoformat(),
                        "window_end": tb.isoformat(),
                        "completed_requests": delta,
                        "request_tps": delta / dt,
                        "source": chosen_metric,
                    }
                )
            if rows:
                return rows

    # fallback by task completion events
    done = []
    for r in task_rows:
        t = parse_iso(str(r.get("end_ts", "")))
        if t is not None:
            done.append(t)
    done.sort()
    if len(done) >= 2:
        for i in range(1, len(done)):
            dt = (done[i] - done[i - 1]).total_seconds()
            if dt <= 0:
                continue
            rows.append(
                {
                    "window_start": done[i - 1].isoformat(),
                    "window_end": done[i].isoformat(),
                    "completed_requests": 1,
                    "request_tps": 1.0 / dt,
                    "source": "task_completion_fallback",
                }
            )
    elif len(done) == 1 and task_rows:
        latency = float(task_rows[0].get("task_latency_s") or 0)
        if latency > 0:
            rows.append(
                {
                    "window_start": done[0].isoformat(),
                    "window_end": done[0].isoformat(),
                    "completed_requests": 1,
                    "request_tps": 1.0 / latency,
                    "source": "task_completion_fallback",
                }
            )
    return rows


def compute_energy(gpu_rows: List[Dict[str, Any]], formal_task_count: int) -> List[Dict[str, Any]]:
    by_gpu: Dict[int, List[Tuple[datetime, float]]] = defaultdict(list)
    for row in gpu_rows:
        gpu_id = int(row.get("gpu_id", -1))
        ts = parse_iso(str(row.get("ts", "")))
        pw = row.get("power_w")
        if ts is None or pw is None:
            continue
        try:
            p = float(pw)
        except Exception:
            continue
        by_gpu[gpu_id].append((ts, p))

    out: List[Dict[str, Any]] = []
    total_j = 0.0
    for gpu_id in sorted(by_gpu.keys()):
        pairs = sorted(by_gpu[gpu_id], key=lambda x: x[0])
        if len(pairs) < 2:
            energy_j = 0.0
        else:
            energy_j = 0.0
            for i in range(1, len(pairs)):
                t0, p0 = pairs[i - 1]
                t1, p1 = pairs[i]
                dt = (t1 - t0).total_seconds()
                if dt <= 0:
                    continue
                energy_j += (p0 + p1) * 0.5 * dt
        total_j += energy_j
        out.append(
            {
                "gpu_id": gpu_id,
                "energy_j": energy_j,
                "energy_wh": energy_j / 3600.0,
                "total_energy_wh": "",
                "energy_per_task_wh": "",
            }
        )

    total_wh = total_j / 3600.0
    per_task = total_wh / formal_task_count if formal_task_count > 0 else 0.0
    out.append(
        {
            "gpu_id": "TOTAL",
            "energy_j": total_j,
            "energy_wh": total_wh,
            "total_energy_wh": total_wh,
            "energy_per_task_wh": per_task,
        }
    )
    return out


def run_one_task(
    args: argparse.Namespace,
    config_point: ConfigPoint,
    row: Dict[str, Any],
    worker_id: int,
    worker_cfg: Path,
    worker_workspace: Path,
    config_raw_dir: Path,
    openclaw_root: Path,
    runtime_home: Path,
    session_store: Path,
    prefix_tag: str = "",
) -> Dict[str, Any]:
    case_idx = int(row.get("idx", -1))
    task_id = str(row.get("task_id", ""))
    question = (row.get("question") or row.get("Question") or "").strip()
    expected_answer = (row.get("final_answer") or row.get("Final answer") or "").strip()

    attachment_text = (row.get("local_attachment") or "").strip()
    attachment = Path(attachment_text) if attachment_text else None
    prompt = build_prompt(question, attachment, args.python_bin)
    task_prefix = f"{prefix_tag}case{case_idx}_w{worker_id}"
    prompt_path = config_raw_dir / f"{task_prefix}.prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    session_id = f"gaia_cc_{config_point.config_id}_idx{case_idx}_{uuid.uuid4().hex[:10]}"
    start_dt = datetime.now()
    run_res = run_openclaw_agent(
        node_bin_dir=Path(args.node_bin_dir),
        openclaw_bin=Path(args.openclaw_bin),
        openclaw_config=worker_cfg,
        session_id=session_id,
        prompt=prompt,
        timeout_s=args.timeout,
        cwd=openclaw_root,
        extra_env={
            "HOME": str(runtime_home),
            "XDG_CACHE_HOME": str(runtime_home / ".cache"),
        },
    )
    end_dt = datetime.now()

    stdout_path = config_raw_dir / f"{task_prefix}.stdout.log"
    stderr_path = config_raw_dir / f"{task_prefix}.stderr.log"
    parsed_path = config_raw_dir / f"{task_prefix}.parsed.json"
    stdout_path.write_text(run_res.get("stdout", ""), encoding="utf-8")
    stderr_path.write_text(run_res.get("stderr", ""), encoding="utf-8")
    write_json(parsed_path, run_res.get("parsed") or {})

    session_src = resolve_session_file(runtime_home, session_store, session_id)
    session_copy = config_raw_dir / f"{task_prefix}.session.jsonl"
    if session_src.exists():
        shutil.copy2(session_src, session_copy)

    session_metrics = parse_session_detailed(session_copy if session_copy.exists() else session_src)

    parsed = run_res.get("parsed") or {}
    payload_answer = extract_payload_answer(parsed)
    payload_stop = None
    if isinstance(parsed, dict):
        meta = parsed.get("meta") or {}
        payload_stop = meta.get("stopReason")

    primary = session_metrics.get("final_answer_primary") or extract_primary_answer(payload_answer)
    stop_reason = payload_stop or session_metrics.get("last_stop_reason")
    status = "ok"
    if run_res.get("returncode") != 0:
        status = "error"
    if str(stop_reason or "").lower() == "error":
        status = "error"
    if "network connection error" in payload_answer.lower():
        status = "error"

    exact_match: Optional[bool] = None
    if expected_answer and status == "ok":
        exact_match = normalize_text(primary) == normalize_text(expected_answer)

    task_row = {
        "tp": config_point.tp,
        "concurrency": config_point.concurrency,
        "round": config_point.round_id,
        "case_idx": case_idx,
        "task_id": task_id,
        "start_ts": start_dt.isoformat(),
        "end_ts": end_dt.isoformat(),
        "task_latency_s": float(run_res.get("elapsed_s") or 0.0),
        "status": status,
        "exact_match": exact_match,
        "step_count": session_metrics.get("assistant_turns", 0),
        "tool_calls": session_metrics.get("tool_calls", 0),
        "tool_success_calls": session_metrics.get("tool_success_calls", 0),
        "tool_success_rate": session_metrics.get("tool_success_rate", 1.0),
        "session_id": session_id,
        "stop_reason": stop_reason,
        "expected_answer": expected_answer,
        "answer_primary": primary,
        "worker_id": worker_id,
        "workspace": str(worker_workspace),
        "prompt_file": str(prompt_path),
        "stdout_file": str(stdout_path),
        "stderr_file": str(stderr_path),
        "parsed_file": str(parsed_path),
        "session_file": str(session_copy if session_copy.exists() else session_src),
    }

    step_rows: List[Dict[str, Any]] = []
    for row_step in session_metrics.get("step_rows", []):
        out_row = {
            "tp": config_point.tp,
            "concurrency": config_point.concurrency,
            "round": config_point.round_id,
            "case_idx": case_idx,
        }
        out_row.update(row_step)
        step_rows.append(out_row)

    inf_rows: List[Dict[str, Any]] = []
    for row_inf in session_metrics.get("inference_rows", []):
        out_row = {
            "tp": config_point.tp,
            "concurrency": config_point.concurrency,
            "round": config_point.round_id,
            "case_idx": case_idx,
        }
        out_row.update(row_inf)
        inf_rows.append(out_row)

    tool_rows: List[Dict[str, Any]] = []
    for row_tool in session_metrics.get("tool_rows", []):
        out_row = {
            "tp": config_point.tp,
            "concurrency": config_point.concurrency,
            "round": config_point.round_id,
            "case_idx": case_idx,
        }
        out_row.update(row_tool)
        tool_rows.append(out_row)

    return {
        "task_row": task_row,
        "step_rows": step_rows,
        "inference_rows": inf_rows,
        "tool_rows": tool_rows,
    }


def run_config_point(
    args: argparse.Namespace,
    config_point: ConfigPoint,
    rows: List[Dict[str, Any]],
    base_url: str,
    run_root: Path,
    openclaw_root: Path,
    template_cfg: Path,
    vllm_log_path: Path,
) -> Dict[str, Any]:
    cfg_dir = run_root / "configs" / f"tp{config_point.tp}_cc{config_point.concurrency}_r{config_point.round_id}"
    raw_dir = cfg_dir / "raw"
    metrics_dir = cfg_dir / "metrics"
    workers_dir = cfg_dir / "workers"
    ensure_dir(raw_dir)
    ensure_dir(metrics_dir)
    ensure_dir(workers_dir)

    worker_cfgs: Dict[int, Path] = {}
    worker_workspaces: Dict[int, Path] = {}
    runtime_home = cfg_dir / "runtime_home"
    session_store = Path(args.session_store)
    ensure_dir(runtime_home)
    ensure_dir(runtime_home / ".cache")
    ensure_dir(session_store)
    for wid in range(config_point.concurrency):
        wdir = workers_dir / f"worker_{wid}"
        cfg_path = wdir / "openclaw.worker.json"
        workspace = wdir / "workspace"
        create_worker_config(
            template_cfg=template_cfg,
            out_cfg=cfg_path,
            workspace=workspace,
            base_url=base_url,
            model_id=args.model,
            api_key=args.vllm_api_key,
            context_window=args.openclaw_context_window,
            model_max_tokens=args.model_max_tokens,
        )
        worker_cfgs[wid] = cfg_path
        worker_workspaces[wid] = workspace

    warmup_metrics: List[Dict[str, Any]] = []
    warmup_rows = rows[: max(0, args.warmup_count)]
    for i, warm in enumerate(warmup_rows, start=1):
        log(f"[{config_point.config_id}] warmup {i}/{len(warmup_rows)}")
        warm_out = run_one_task(
            args=args,
            config_point=config_point,
            row=warm,
            worker_id=0,
            worker_cfg=worker_cfgs[0],
            worker_workspace=worker_workspaces[0],
            config_raw_dir=raw_dir,
            openclaw_root=openclaw_root,
            runtime_home=runtime_home,
            session_store=session_store,
            prefix_tag=f"warmup{i}_",
        )
        warm_task = dict(warm_out.get("task_row", {}))
        warm_task["warmup_index"] = i
        warmup_metrics.append(warm_task)

    sampler_gpu = GPUSampler(job_id=args.job_id, node=args.node, interval_s=args.gpu_sample_sec)
    sampler_backend = BackendSampler(job_id=args.job_id, node=args.node, port=args.port, interval_s=2.0)

    start_dt = datetime.now()
    sampler_gpu.start()
    sampler_backend.start()

    task_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    inference_rows: List[Dict[str, Any]] = []
    tool_rows: List[Dict[str, Any]] = []

    worker_queue: queue.Queue[int] = queue.Queue()
    for wid in range(config_point.concurrency):
        worker_queue.put(wid)

    def _job(row_obj: Dict[str, Any]) -> Dict[str, Any]:
        wid = worker_queue.get()
        try:
            try:
                return run_one_task(
                    args=args,
                    config_point=config_point,
                    row=row_obj,
                    worker_id=wid,
                    worker_cfg=worker_cfgs[wid],
                    worker_workspace=worker_workspaces[wid],
                    config_raw_dir=raw_dir,
                    openclaw_root=openclaw_root,
                    runtime_home=runtime_home,
                    session_store=session_store,
                )
            except Exception as e:
                case_idx = int(row_obj.get("idx", -1))
                task_id = str(row_obj.get("task_id", ""))
                err_ts = datetime.now().isoformat()
                return {
                    "task_row": {
                        "tp": config_point.tp,
                        "concurrency": config_point.concurrency,
                        "round": config_point.round_id,
                        "case_idx": case_idx,
                        "task_id": task_id,
                        "start_ts": err_ts,
                        "end_ts": err_ts,
                        "task_latency_s": 0.0,
                        "status": "error",
                        "exact_match": None,
                        "step_count": 0,
                        "tool_calls": 0,
                        "tool_success_calls": 0,
                        "tool_success_rate": 1.0,
                        "session_id": "",
                        "stop_reason": "worker_exception",
                        "expected_answer": str(row_obj.get("final_answer", "")),
                        "answer_primary": "",
                        "worker_id": wid,
                        "workspace": str(worker_workspaces[wid]),
                        "prompt_file": "",
                        "stdout_file": "",
                        "stderr_file": "",
                        "parsed_file": "",
                        "session_file": "",
                        "error_text": str(e)[:500],
                    },
                    "step_rows": [],
                    "inference_rows": [],
                    "tool_rows": [],
                }
        finally:
            worker_queue.put(wid)

    with ThreadPoolExecutor(max_workers=config_point.concurrency) as ex:
        futures = [ex.submit(_job, r) for r in rows]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                err_ts = datetime.now().isoformat()
                task_rows.append(
                    {
                        "tp": config_point.tp,
                        "concurrency": config_point.concurrency,
                        "round": config_point.round_id,
                        "case_idx": -1,
                        "task_id": "",
                        "start_ts": err_ts,
                        "end_ts": err_ts,
                        "task_latency_s": 0.0,
                        "status": "error",
                        "exact_match": None,
                        "step_count": 0,
                        "tool_calls": 0,
                        "tool_success_calls": 0,
                        "tool_success_rate": 1.0,
                        "session_id": "",
                        "stop_reason": "future_exception",
                        "expected_answer": "",
                        "answer_primary": "",
                        "worker_id": -1,
                        "workspace": "",
                        "prompt_file": "",
                        "stdout_file": "",
                        "stderr_file": "",
                        "parsed_file": "",
                        "session_file": "",
                        "error_text": str(e)[:500],
                    }
                )
                continue
            task_rows.append(result["task_row"])
            step_rows.extend(result["step_rows"])
            inference_rows.extend(result["inference_rows"])
            tool_rows.extend(result["tool_rows"])

    end_dt = datetime.now()
    sampler_gpu.stop()
    sampler_backend.stop()
    sampler_gpu.join(timeout=20)
    sampler_backend.join(timeout=20)

    task_rows = sorted(task_rows, key=lambda r: int(r.get("case_idx", -1)))

    for row in sampler_gpu.samples:
        row.update(
            {
                "tp": config_point.tp,
                "concurrency": config_point.concurrency,
                "round": config_point.round_id,
            }
        )
    for row in sampler_backend.samples:
        row.update(
            {
                "tp": config_point.tp,
                "concurrency": config_point.concurrency,
                "round": config_point.round_id,
            }
        )

    vllm_rows = parse_vllm_timeseries(vllm_log_path, start_dt, end_dt)
    for row in vllm_rows:
        row.update(
            {
                "tp": config_point.tp,
                "concurrency": config_point.concurrency,
                "round": config_point.round_id,
            }
        )

    req_tps_rows = build_request_throughput_series(sampler_backend.samples, task_rows)
    for row in req_tps_rows:
        row.update(
            {
                "tp": config_point.tp,
                "concurrency": config_point.concurrency,
                "round": config_point.round_id,
            }
        )

    energy_rows = compute_energy(sampler_gpu.samples, formal_task_count=len(task_rows))
    for row in energy_rows:
        row.update(
            {
                "tp": config_point.tp,
                "concurrency": config_point.concurrency,
                "round": config_point.round_id,
            }
        )

    write_jsonl(metrics_dir / "task_metrics.jsonl", task_rows)
    write_jsonl(metrics_dir / "warmup_task_metrics.jsonl", warmup_metrics)
    write_jsonl(metrics_dir / "step_metrics.jsonl", step_rows)
    write_jsonl(metrics_dir / "inference_metrics.jsonl", inference_rows)
    write_jsonl(metrics_dir / "tool_metrics.jsonl", tool_rows)
    write_jsonl(metrics_dir / "gpu_samples.jsonl", sampler_gpu.samples)
    write_jsonl(metrics_dir / "vllm_timeseries.jsonl", vllm_rows)
    write_jsonl(metrics_dir / "request_throughput_timeseries.jsonl", req_tps_rows)
    write_csv(metrics_dir / "energy_summary.csv", energy_rows)

    summary = {
        "config_id": config_point.config_id,
        "tp": config_point.tp,
        "concurrency": config_point.concurrency,
        "round": config_point.round_id,
        "start_ts": start_dt.isoformat(),
        "end_ts": end_dt.isoformat(),
        "formal_task_count": len(task_rows),
        "warmup_task_count": len(warmup_metrics),
        "ok_tasks": sum(1 for r in task_rows if r.get("status") == "ok"),
        "error_tasks": sum(1 for r in task_rows if r.get("status") == "error"),
        "exact_match_true": sum(1 for r in task_rows if r.get("exact_match") is True),
        "tool_calls": sum(int(r.get("tool_calls", 0)) for r in task_rows),
        "gpu_sample_count": len(sampler_gpu.samples),
        "backend_sample_count": len(sampler_backend.samples),
        "vllm_points": len(vllm_rows),
        "request_tps_points": len(req_tps_rows),
        "gpu_sampler_errors": sampler_gpu.errors,
        "backend_sampler_errors": sampler_backend.errors,
        "vllm_log": str(vllm_log_path),
    }
    write_json(metrics_dir / "config_summary.json", summary)
    return summary


def build_config_points(tp_list: List[int], conc_list: List[int], rounds: int) -> List[ConfigPoint]:
    points: List[ConfigPoint] = []
    for tp in tp_list:
        for rnd in range(1, rounds + 1):
            for cc in conc_list:
                cid = f"tp{tp}_cc{cc}_r{rnd}"
                points.append(ConfigPoint(config_id=cid, tp=tp, concurrency=cc, round_id=rnd))
    return points


def validate_environment(args: argparse.Namespace, run_root: Path) -> None:
    if not Path(args.rows_jsonl).exists():
        raise FileNotFoundError(f"rows-jsonl not found: {args.rows_jsonl}")
    if not Path(args.openclaw_bin).exists():
        raise FileNotFoundError(f"openclaw bin not found: {args.openclaw_bin}")
    if not Path(args.openclaw_config).exists():
        raise FileNotFoundError(f"openclaw config not found: {args.openclaw_config}")
    if not Path(args.manage_backend_script).exists():
        raise FileNotFoundError(f"manage backend script not found: {args.manage_backend_script}")

    sq = run_cmd(["squeue", "-j", str(args.job_id), "-o", "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"], timeout=30)
    if sq.returncode != 0:
        raise RuntimeError(f"squeue failed: {sq.stderr}")
    if str(args.job_id) not in sq.stdout:
        raise RuntimeError(f"job {args.job_id} not found in squeue output")
    if args.node not in sq.stdout:
        log(f"warning: node {args.node} not shown in squeue line, continue anyway")

    roc = run_srun(args.job_id, args.node, "test -x /opt/rocm-default/bin/rocm-smi && echo ok", timeout=30)
    if roc.returncode != 0 or "ok" not in roc.stdout:
        raise RuntimeError("/opt/rocm-default/bin/rocm-smi not executable on target node")

    ensure_dir(run_root)


def infer_vllm_log(work_root: Path, base_url: str, port: int) -> Path:
    m = re.match(r"https?://([^:/]+):(\d+)", base_url)
    if m:
        host = m.group(1)
        p = m.group(2)
        pth = work_root / "logs" / f"vllm_{host}_{p}.log"
        if pth.exists():
            return pth
    candidates = sorted((work_root / "logs").glob(f"vllm_*_{port}.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return work_root / "logs" / f"vllm_unknown_{port}.log"


def main() -> int:
    args = parse_args()
    if args.enforce_eager:
        log("warning: --enforce-eager is ignored (hard-disabled in launcher policy)")
    tp_list = parse_list_int(args.tp_list)
    conc_list = parse_list_int(args.concurrency_list)

    run_root = Path(args.out_dir)
    if not args.session_store:
        args.session_store = str(run_root / ".openclaw_runtime" / "agents" / "main" / "sessions")
    manifest_path = run_root / "manifest.json"

    openclaw_root = Path(__file__).resolve().parents[1]
    work_root = openclaw_root.parent

    validate_environment(args, run_root)
    if args.openclaw_context_margin < 0:
        raise ValueError("--openclaw-context-margin must be >= 0")
    if args.openclaw_context_window <= 0:
        args.openclaw_context_window = max(1, args.max_model_len - args.model_max_tokens - args.openclaw_context_margin)
    if args.openclaw_context_window + args.model_max_tokens > args.max_model_len:
        log(
            "warning: OpenClaw contextWindow + maxTokens exceeds vLLM max_model_len "
            f"({args.openclaw_context_window} + {args.model_max_tokens} > {args.max_model_len})"
        )
    rows = read_rows(Path(args.rows_jsonl), args.case_mode, args.idx_start, args.idx_end)
    if len(rows) < 1:
        raise RuntimeError("selected rows count is 0")

    points = build_config_points(tp_list, conc_list, args.rounds)
    ensure_dir(run_root / "configs")
    ensure_dir(run_root / "aggregate")
    ensure_dir(run_root / "plots")

    manifest = load_manifest(manifest_path)
    if not manifest or not args.resume:
        manifest = {
            "run_id": run_root.name,
            "created_at": datetime.now().isoformat(),
            "job_id": args.job_id,
            "node": args.node,
            "rows_jsonl": str(args.rows_jsonl),
            "case_mode": args.case_mode,
            "idx_start": args.idx_start,
            "idx_end": args.idx_end,
            "tp_list": tp_list,
            "concurrency_list": conc_list,
            "rounds": args.rounds,
            "warmup_count": args.warmup_count,
            "gpu_sample_sec": args.gpu_sample_sec,
            "max_model_len": args.max_model_len,
            "openclaw_context_window": args.openclaw_context_window,
            "openclaw_context_margin": args.openclaw_context_margin,
            "model_max_tokens": args.model_max_tokens,
            "configs": [
                {
                    "config_id": p.config_id,
                    "tp": p.tp,
                    "concurrency": p.concurrency,
                    "round": p.round_id,
                    "status": "pending",
                    "note": "",
                    "updated_at": datetime.now().isoformat(),
                }
                for p in points
            ],
        }
        save_manifest(manifest_path, manifest)

    selected_rows_path = run_root / "selected_rows.jsonl"
    write_jsonl(selected_rows_path, rows)

    template_cfg = Path(args.openclaw_config)

    try:
        for tp in tp_list:
            run_tag = f"{run_root.name}_tp{tp}_p{args.port}"
            base_url, endpoints = start_backend(args, tp=tp, run_tag=run_tag, work_root=work_root)
            vllm_log_path = infer_vllm_log(work_root, base_url, args.port)
            log(f"backend ready tp={tp} base_url={base_url} endpoints={endpoints}")

            tp_points = [p for p in points if p.tp == tp]
            for cp in tp_points:
                # skip finished in resume mode
                existing = next((c for c in manifest.get("configs", []) if c.get("config_id") == cp.config_id), None)
                if existing and existing.get("status") == "done" and args.resume:
                    log(f"skip completed config {cp.config_id}")
                    continue

                set_manifest_status(manifest, cp.config_id, "running")
                save_manifest(manifest_path, manifest)
                log(f"run config {cp.config_id}")
                config_done = False
                last_error = ""
                for attempt in (1, 2):
                    healthy, detail = check_backend_health(args)
                    if not healthy:
                        if attempt == 1:
                            log(f"backend unhealthy before {cp.config_id}; restarting once. detail={detail}")
                            stop_backend(args, run_tag=run_tag, work_root=work_root)
                            base_url, endpoints = start_backend(args, tp=tp, run_tag=run_tag, work_root=work_root)
                            vllm_log_path = infer_vllm_log(work_root, base_url, args.port)
                            log(f"backend re-ready tp={tp} base_url={base_url} endpoints={endpoints}")
                            continue
                        raise RuntimeError(f"backend unhealthy before config start: {detail}")

                    try:
                        summary = run_config_point(
                            args=args,
                            config_point=cp,
                            rows=rows,
                            base_url=base_url,
                            run_root=run_root,
                            openclaw_root=openclaw_root,
                            template_cfg=template_cfg,
                            vllm_log_path=vllm_log_path,
                        )
                        cfg_dir = run_root / "configs" / f"tp{cp.tp}_cc{cp.concurrency}_r{cp.round_id}"
                        write_json(cfg_dir / "metrics" / "config_summary.json", summary)
                        set_manifest_status(manifest, cp.config_id, "done")
                        save_manifest(manifest_path, manifest)
                        log(f"config done {cp.config_id}")
                        config_done = True
                        break
                    except Exception as e:
                        last_error = str(e)
                        if attempt == 1:
                            healthy_after, detail_after = check_backend_health(args)
                            if not healthy_after:
                                log(f"{cp.config_id} failed with unhealthy backend; restarting once. detail={detail_after}")
                                stop_backend(args, run_tag=run_tag, work_root=work_root)
                                base_url, endpoints = start_backend(args, tp=tp, run_tag=run_tag, work_root=work_root)
                                vllm_log_path = infer_vllm_log(work_root, base_url, args.port)
                                log(f"backend re-ready tp={tp} base_url={base_url} endpoints={endpoints}")
                                continue
                        last_error = str(e)
                        break

                if not config_done:
                    set_manifest_status(manifest, cp.config_id, "failed", note=last_error)
                    save_manifest(manifest_path, manifest)
                    log(f"config failed {cp.config_id}: {last_error}")

            stop_backend(args, run_tag=run_tag, work_root=work_root)
            log(f"backend stopped tp={tp}")
    finally:
        # best-effort cleanup for both TP tags
        for tp in tp_list:
            run_tag = f"{run_root.name}_tp{tp}_p{args.port}"
            try:
                stop_backend(args, run_tag=run_tag, work_root=work_root)
            except Exception:
                pass

    if not args.skip_report:
        report_cmd = [
            args.python_bin,
            str(openclaw_root / "scripts" / "build_gaia_concurrency_report.py"),
            "--run-root",
            str(run_root),
            "--quantile-method",
            args.quantile_method,
            "--report-md",
            "report.md",
        ]
        proc = run_cmd(report_cmd, timeout=600)
        if proc.returncode != 0:
            log(f"report generation failed: {proc.stderr or proc.stdout}")
        else:
            log("report generation done")

    print(str(run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
