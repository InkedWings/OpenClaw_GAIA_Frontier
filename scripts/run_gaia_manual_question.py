#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    openclaw_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run one manual GAIA-style question on OpenClaw and save full process artifacts."
    )
    parser.add_argument("--question", default="A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?")
    parser.add_argument("--question-file", default="", help="Path to a text file containing one question.")
    parser.add_argument("--attachment", default="", help="Optional local attachment path.")
    parser.add_argument("--expected-answer", default="1", help="Optional reference answer for exact-match check.")
    parser.add_argument("--run-id", default=f"gaia_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--openclaw-bin", default=str(openclaw_root / ".local" / "npm" / "lib" / "node_modules" / "openclaw" / "openclaw.mjs"))
    parser.add_argument("--openclaw-config", default=str(openclaw_root / "config" / "openclaw.template.json"))
    parser.add_argument("--node-bin-dir", default=str(openclaw_root / ".local" / "node" / "bin"))
    parser.add_argument("--output-root", default=str(openclaw_root / "runs"))
    parser.add_argument("--session-store", default=str(Path.home() / ".openclaw" / "agents" / "main" / "sessions"))
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN_HINT", "/lustre/orion/gen150/scratch/zye25/conda-envs/AGAI/bin/python"),
        help="Path hint inserted into prompt helper text (kept aligned with GAIA runner).",
    )
    return parser.parse_args()


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_progress(progress_file: Path, message: str) -> None:
    line = f"[{now_ts()}] {message}"
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_primary_answer(answer_text: str) -> str:
    text = (answer_text or "").strip()
    if not text:
        return ""
    matches = list(re.finditer(r"FINAL(?:_|\s+)ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL))
    if matches:
        ans = matches[-1].group(1).strip().splitlines()[0].strip()
        if ans.startswith("[") and ans.endswith("]"):
            ans = ans[1:-1].strip()
        return ans
    return text.splitlines()[0].strip()


def build_prompt(question: str, attachment: Optional[Path], python_bin: str) -> str:
    base = (
        "You are a general AI assistant. I will ask you a question. Work out the answer privately. "
        "Do not reveal your reasoning process, "
        "and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. "
        "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated "
        "list of numbers and/or strings. If you are asked for a number, don't use comma to write "
        "your number neither use units such as $ or percent sign unless specified otherwise. "
        "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), "
        "and write the digits in plain text unless specified otherwise. If you are asked for a comma "
        "separated list, apply the above rules depending of whether the element to be put in the "
        "list is a number or a string."
    )
    attach_part = f"\n\nAttachment local path: {attachment}" if attachment else ""
    _ = python_bin
    return f"{base}\n\nQuestion:\n{question}{attach_part}"


def extract_json_from_text(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    last_obj: Dict[str, Any] = {}
    last_payload_obj: Dict[str, Any] = {}
    if not text:
        return last_obj

    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                last_obj = obj
                if "payloads" in obj or "meta" in obj:
                    last_payload_obj = obj
        except Exception:
            continue
    return last_payload_obj or last_obj


def run_openclaw_agent(
    node_bin_dir: Path,
    openclaw_bin: Path,
    openclaw_config: Path,
    session_id: str,
    prompt: str,
    timeout_s: int,
    cwd: Path,
    extra_env: Optional[Dict[str, str]] = None,
    launcher_prefix: Optional[List[str]] = None,
) -> Dict[str, Any]:
    node_exe = node_bin_dir / "node"
    env = os.environ.copy()
    env["OPENCLAW_CONFIG_PATH"] = str(openclaw_config)
    agai_bin = "/lustre/orion/gen150/scratch/zye25/conda-envs/AGAI/bin"
    env["PATH"] = f"{agai_bin}:{node_bin_dir}:{env.get('PATH', '')}"
    env["PYTHON"] = f"{agai_bin}/python"
    env["PYTHON_BIN_HINT"] = f"{agai_bin}/python"
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    # Run the real OpenClaw node entrypoint directly. PATH includes node_bin_dir
    # so the shebang resolves to the copied Node install in this repo.
    _ = node_exe
    cmd = [
        str(openclaw_bin),
        "agent",
        "--local",
        "--session-id",
        session_id,
        "--message",
        prompt,
        "--thinking",
        "off",
        "--timeout",
        str(timeout_s),
        "--json",
    ]
    if launcher_prefix:
        cmd = [str(x) for x in launcher_prefix] + cmd

    t0 = time.time()
    hard_timeout_s = max(timeout_s + 30, timeout_s)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=hard_timeout_s,
        )
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        stdout_raw = e.stdout or ""
        stderr_raw = e.stderr or ""
        if isinstance(stdout_raw, bytes):
            stdout = stdout_raw.decode("utf-8", errors="ignore")
        else:
            stdout = str(stdout_raw)
        if isinstance(stderr_raw, bytes):
            stderr = stderr_raw.decode("utf-8", errors="ignore")
        else:
            stderr = str(stderr_raw)
        stderr = stderr + f"\n[runner] hard timeout reached ({hard_timeout_s}s)"
        return {
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_s": round(elapsed, 2),
            "parsed": {},
            "cmd": cmd,
        }

    parsed: Dict[str, Any] = {}
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = extract_json_from_text(stdout)
    if (not parsed or "payloads" not in parsed) and stderr:
        parsed = extract_json_from_text(stderr)

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_s": round(elapsed, 2),
        "parsed": parsed,
        "cmd": cmd,
    }


def parse_session_jsonl(session_file: Path) -> Dict[str, Any]:
    metrics = {
        "session_file": str(session_file),
        "session_exists": session_file.exists(),
        "event_count": 0,
        "assistant_turns": 0,
        "tool_calls_count": 0,
        "tool_names": [],
        "tool_errors_count": 0,
        "last_stop_reason": None,
        "final_assistant_text": "",
        "final_answer_primary": "",
    }
    if not session_file.exists():
        return metrics

    tool_names: List[str] = []
    for line in session_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        metrics["event_count"] += 1
        if obj.get("type") == "message":
            msg = obj.get("message", {})
            role = msg.get("role")
            if role == "assistant":
                metrics["assistant_turns"] += 1
                if msg.get("stopReason"):
                    metrics["last_stop_reason"] = msg.get("stopReason")
                text_blocks: List[str] = []
                for block in msg.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "toolCall":
                        metrics["tool_calls_count"] += 1
                        if block.get("name"):
                            tool_names.append(block["name"])
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_blocks.append((block.get("text") or "").strip())
                if msg.get("stopReason") == "stop" and text_blocks:
                    metrics["final_assistant_text"] = "\n".join(text_blocks).strip()
            elif role == "toolResult":
                details = msg.get("details", {})
                if isinstance(details, dict) and details.get("status") == "error":
                    metrics["tool_errors_count"] += 1

    metrics["tool_names"] = sorted(set(tool_names))
    metrics["final_answer_primary"] = extract_primary_answer(metrics["final_assistant_text"])
    return metrics


def write_markdown_report(path: Path, result: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# GAIA Manual Single Question Report\n\n")
        f.write(f"- run_id: `{result['run_id']}`\n")
        f.write(f"- session_id: `{result['session_id']}`\n")
        f.write(f"- status: `{result['status']}`\n")
        f.write(f"- duration_s: `{result['duration_s']}`\n")
        f.write(f"- returncode: `{result['returncode']}`\n")
        f.write(f"- stop_reason: `{result.get('stop_reason')}`\n")
        f.write(f"- tool_calls_count: `{result.get('tool_calls_count')}`\n")
        f.write(f"- tool_errors_count: `{result.get('tool_errors_count')}`\n")
        f.write(f"- exact_match: `{result.get('exact_match')}`\n\n")
        f.write("## Final Answer (Primary)\n\n")
        f.write(f"{result.get('answer_primary', '')}\n\n")
        f.write("## Expected Answer\n\n")
        f.write(f"{result.get('expected_answer', '')}\n\n")
        f.write("## Artifacts\n\n")
        f.write(f"- prompt: `{result['prompt_path']}`\n")
        f.write(f"- stdout: `{result['stdout_path']}`\n")
        f.write(f"- stderr: `{result['stderr_path']}`\n")
        f.write(f"- parsed: `{result['parsed_path']}`\n")
        f.write(f"- session: `{result['session_copy_path']}`\n")
        f.write(f"- result_json: `{result['result_json_path']}`\n")


def load_question(args: argparse.Namespace) -> str:
    if args.question.strip():
        return args.question.strip()
    if args.question_file:
        p = Path(args.question_file)
        if not p.exists():
            raise FileNotFoundError(f"question file not found: {p}")
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def main() -> int:
    args = parse_args()
    try:
        question = load_question(args)
    except Exception as e:
        print(f"[error] failed to load question: {e}", file=sys.stderr)
        return 2
    if not question:
        print("[error] question is empty. Pass --question or --question-file.", file=sys.stderr)
        return 2

    openclaw_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.output_root) / args.run_id
    raw_dir = run_dir / "raw"
    sessions_dir = run_dir / "sessions"
    ensure_dir(run_dir)
    ensure_dir(raw_dir)
    ensure_dir(sessions_dir)

    progress_file = run_dir / "progress.log"
    progress_file.write_text("", encoding="utf-8")
    append_progress(progress_file, f"run_id={args.run_id} start")

    attachment_local: Optional[Path] = None
    if args.attachment:
        p = Path(args.attachment)
        if not p.exists():
            append_progress(progress_file, f"attachment not found, ignored: {p}")
        else:
            attachment_local = p
            append_progress(progress_file, f"attachment: {attachment_local}")

    prompt = build_prompt(question, attachment_local, args.python_bin)
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    (run_dir / "question.txt").write_text(question, encoding="utf-8")
    append_progress(progress_file, "prompt written")

    session_id = f"gaia_manual_{args.run_id}_{int(time.time())}"
    run_res = run_openclaw_agent(
        node_bin_dir=Path(args.node_bin_dir),
        openclaw_bin=Path(args.openclaw_bin),
        openclaw_config=Path(args.openclaw_config),
        session_id=session_id,
        prompt=prompt,
        timeout_s=args.timeout,
        cwd=openclaw_root,
    )
    append_progress(progress_file, f"openclaw finished returncode={run_res['returncode']}")

    stdout_path = raw_dir / "stdout.log"
    stderr_path = raw_dir / "stderr.log"
    parsed_path = raw_dir / "parsed.json"
    stdout_path.write_text(run_res["stdout"], encoding="utf-8")
    stderr_path.write_text(run_res["stderr"], encoding="utf-8")
    parsed_path.write_text(json.dumps(run_res.get("parsed", {}), ensure_ascii=False, indent=2), encoding="utf-8")

    payload_answer = ""
    stop_reason = None
    parsed = run_res.get("parsed", {}) or {}
    meta = parsed.get("meta", {}) if isinstance(parsed, dict) else {}
    if isinstance(parsed, dict):
        payloads = parsed.get("payloads") or []
        text_parts: List[str] = []
        for p in payloads:
            if isinstance(p, dict):
                t = (p.get("text") or "").strip()
                if t:
                    text_parts.append(t)
        if text_parts:
            payload_answer = "\n".join(text_parts)
        stop_reason = meta.get("stopReason")

    session_file = Path(args.session_store) / f"{session_id}.jsonl"
    session_copy_path = sessions_dir / f"{session_id}.jsonl"
    if session_file.exists():
        session_copy_path.write_text(session_file.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    sess_metrics = parse_session_jsonl(session_file)

    answer_primary = sess_metrics.get("final_answer_primary") or extract_primary_answer(payload_answer)
    expected = args.expected_answer.strip()
    is_error = (
        run_res["returncode"] != 0
        or (str(stop_reason).lower() == "error")
        or (str(sess_metrics.get("last_stop_reason", "")).lower() == "error")
        or ("network connection error" in payload_answer.lower())
    )
    exact_match: Optional[bool]
    if expected and not is_error:
        exact_match = normalize_text(answer_primary) == normalize_text(expected)
    else:
        exact_match = None

    result = {
        "run_id": args.run_id,
        "session_id": session_id,
        "status": "error" if is_error else "ok",
        "duration_s": run_res["elapsed_s"],
        "returncode": run_res["returncode"],
        "stop_reason": stop_reason or sess_metrics.get("last_stop_reason"),
        "tool_calls_count": sess_metrics.get("tool_calls_count", 0),
        "tool_names": sess_metrics.get("tool_names", []),
        "tool_errors_count": sess_metrics.get("tool_errors_count", 0),
        "assistant_turns": sess_metrics.get("assistant_turns", 0),
        "answer_from_session": sess_metrics.get("final_assistant_text", ""),
        "answer_from_payload": payload_answer,
        "answer_primary": answer_primary,
        "expected_answer": expected,
        "exact_match": exact_match,
        "question_file": args.question_file,
        "attachment": str(attachment_local) if attachment_local else "",
        "prompt_path": str(prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "parsed_path": str(parsed_path),
        "session_copy_path": str(session_copy_path),
    }

    result_json_path = run_dir / "result.json"
    result["result_json_path"] = str(result_json_path)
    result_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = run_dir / "report.md"
    write_markdown_report(report_path, result)
    append_progress(progress_file, f"artifacts saved under {run_dir}")

    print(json.dumps({"run_dir": str(run_dir), "report": str(report_path), "result": str(result_json_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
