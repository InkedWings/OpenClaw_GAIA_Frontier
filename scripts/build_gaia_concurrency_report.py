#!/usr/bin/env python3

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build GAIA concurrency benchmark report.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--quantile-method", default="nearest-rank", choices=["nearest-rank"])
    parser.add_argument("--report-md", default="report.md")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--aggregate-dir", default="aggregate")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--root-hint", default=str(root))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def quantile_nearest_rank(values: List[float], q: float) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    clean.sort()
    n = len(clean)
    rank = max(1, math.ceil(q * n))
    idx = min(n - 1, rank - 1)
    return clean[idx]


def fmt(v: Optional[float], n: int = 4) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), n)


def collect_config_dirs(run_root: Path) -> List[Path]:
    cfg_root = run_root / "configs"
    if not cfg_root.exists():
        return []
    dirs = [p for p in cfg_root.iterdir() if p.is_dir()]
    dirs.sort()
    return dirs


def parse_config_key(name: str) -> Tuple[int, int, int]:
    # tp2_cc4_r1
    tp, cc, rr = 0, 0, 0
    for part in name.split("_"):
        if part.startswith("tp"):
            tp = int(part[2:])
        elif part.startswith("cc"):
            cc = int(part[2:])
        elif part.startswith("r"):
            rr = int(part[1:])
    return tp, cc, rr


def aggregate_one_config(cfg_dir: Path) -> Dict[str, Any]:
    metrics = cfg_dir / "metrics"
    task_rows = read_jsonl(metrics / "task_metrics.jsonl")
    step_rows = read_jsonl(metrics / "step_metrics.jsonl")
    inf_rows = read_jsonl(metrics / "inference_metrics.jsonl")
    tool_rows = read_jsonl(metrics / "tool_metrics.jsonl")
    vllm_rows = read_jsonl(metrics / "vllm_timeseries.jsonl")
    req_rows = read_jsonl(metrics / "request_throughput_timeseries.jsonl")
    gpu_rows = read_jsonl(metrics / "gpu_samples.jsonl")
    energy_csv = metrics / "energy_summary.csv"

    tp, cc, rr = parse_config_key(cfg_dir.name)

    def lat_vals(rows: List[Dict[str, Any]], key: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            v = r.get(key)
            if v is None or v == "":
                continue
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    task_lat = lat_vals(task_rows, "task_latency_s")
    step_lat = lat_vals(step_rows, "step_latency_s")
    tool_lat = lat_vals(tool_rows, "tool_latency_s")
    inf_lat = lat_vals(inf_rows, "inference_latency_s")

    latency_summary = {
        "config_id": cfg_dir.name,
        "tp": tp,
        "concurrency": cc,
        "round": rr,
        "task_count": len(task_lat),
        "step_count": len(step_lat),
        "tool_count": len(tool_lat),
        "inference_count": len(inf_lat),
        "task_p50": fmt(quantile_nearest_rank(task_lat, 0.50)),
        "task_p95": fmt(quantile_nearest_rank(task_lat, 0.95)),
        "task_p99": fmt(quantile_nearest_rank(task_lat, 0.99)),
        "step_p50": fmt(quantile_nearest_rank(step_lat, 0.50)),
        "step_p95": fmt(quantile_nearest_rank(step_lat, 0.95)),
        "step_p99": fmt(quantile_nearest_rank(step_lat, 0.99)),
        "tool_p50": fmt(quantile_nearest_rank(tool_lat, 0.50)),
        "tool_p95": fmt(quantile_nearest_rank(tool_lat, 0.95)),
        "tool_p99": fmt(quantile_nearest_rank(tool_lat, 0.99)),
        "inference_p50": fmt(quantile_nearest_rank(inf_lat, 0.50)),
        "inference_p95": fmt(quantile_nearest_rank(inf_lat, 0.95)),
        "inference_p99": fmt(quantile_nearest_rank(inf_lat, 0.99)),
    }

    ok_tasks = sum(1 for r in task_rows if str(r.get("status", "")) == "ok")
    exact_true = sum(1 for r in task_rows if r.get("exact_match") is True)
    tool_total = len(tool_rows)
    tool_ok = sum(1 for r in tool_rows if not bool(r.get("is_error")))

    success_summary = {
        "config_id": cfg_dir.name,
        "tp": tp,
        "concurrency": cc,
        "round": rr,
        "task_total": len(task_rows),
        "task_ok": ok_tasks,
        "task_success_rate": fmt(ok_tasks / len(task_rows) if task_rows else 0.0),
        "exact_match_true": exact_true,
        "exact_match_rate": fmt(exact_true / len(task_rows) if task_rows else 0.0),
        "tool_calls": tool_total,
        "tool_success_calls": tool_ok,
        "tool_success_rate": fmt(tool_ok / tool_total if tool_total else 1.0),
        "avg_step_count": fmt(mean([float(r.get("step_count", 0)) for r in task_rows]) if task_rows else 0.0),
        "avg_tool_calls_per_task": fmt(mean([float(r.get("tool_calls", 0)) for r in task_rows]) if task_rows else 0.0),
    }

    prefill = lat_vals(vllm_rows, "prefill_tps")
    decode = lat_vals(vllm_rows, "decode_tps")
    req_tps = lat_vals(req_rows, "request_tps")
    throughput_summary = {
        "config_id": cfg_dir.name,
        "tp": tp,
        "concurrency": cc,
        "round": rr,
        "prefill_tps_mean": fmt(mean(prefill) if prefill else 0.0),
        "decode_tps_mean": fmt(mean(decode) if decode else 0.0),
        "request_tps_mean": fmt(mean(req_tps) if req_tps else 0.0),
        "request_tps_p95": fmt(quantile_nearest_rank(req_tps, 0.95)),
        "vllm_points": len(vllm_rows),
        "request_points": len(req_rows),
    }

    gpu_use = lat_vals(gpu_rows, "gpu_use_pct")
    vram = lat_vals(gpu_rows, "vram_pct")
    power = lat_vals(gpu_rows, "power_w")
    kv = lat_vals(vllm_rows, "gpu_kv_cache_pct")
    total_wh = 0.0
    energy_per_task_wh = 0.0
    if energy_csv.exists():
        try:
            with energy_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("gpu_id")) == "TOTAL":
                        total_wh = float(row.get("total_energy_wh") or 0.0)
                        energy_per_task_wh = float(row.get("energy_per_task_wh") or 0.0)
                        break
        except Exception:
            pass

    resource_summary = {
        "config_id": cfg_dir.name,
        "tp": tp,
        "concurrency": cc,
        "round": rr,
        "gpu_use_mean": fmt(mean(gpu_use) if gpu_use else 0.0),
        "vram_pct_mean": fmt(mean(vram) if vram else 0.0),
        "kv_cache_pct_mean": fmt(mean(kv) if kv else 0.0),
        "power_w_mean": fmt(mean(power) if power else 0.0),
        "total_energy_wh": fmt(total_wh),
        "energy_per_task_wh": fmt(energy_per_task_wh),
        "gpu_sample_count": len(gpu_rows),
    }

    return {
        "config_id": cfg_dir.name,
        "tp": tp,
        "concurrency": cc,
        "round": rr,
        "task_rows": task_rows,
        "step_rows": step_rows,
        "tool_rows": tool_rows,
        "inference_rows": inf_rows,
        "vllm_rows": vllm_rows,
        "req_rows": req_rows,
        "gpu_rows": gpu_rows,
        "latency_summary": latency_summary,
        "success_summary": success_summary,
        "throughput_summary": throughput_summary,
        "resource_summary": resource_summary,
    }


def mean_by_tp_cc(rows: List[Dict[str, Any]], key: str) -> Dict[Tuple[int, int], float]:
    groups: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for row in rows:
        v = row.get(key)
        if v is None or v == "":
            continue
        try:
            groups[(int(row["tp"]), int(row["concurrency"]))].append(float(v))
        except Exception:
            continue
    return {k: mean(vs) for k, vs in groups.items() if vs}


def maybe_plot(aggregate: Dict[str, Any], plots_dir: Path) -> List[str]:
    outputs: List[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return outputs

    lat_rows = aggregate["latency_summary_rows"]
    succ_rows = aggregate["success_summary_rows"]
    thr_rows = aggregate["throughput_summary_rows"]
    res_rows = aggregate["resource_summary_rows"]

    tps = sorted({int(r["tp"]) for r in lat_rows})
    ccs = sorted({int(r["concurrency"]) for r in lat_rows})

    # 1) Task latency p50/p95/p99 line chart by TP
    fig, axes = plt.subplots(1, max(1, len(tps)), figsize=(6 * max(1, len(tps)), 4), squeeze=False)
    for i, tp in enumerate(tps):
        ax = axes[0][i]
        sub = [r for r in lat_rows if int(r["tp"]) == tp]
        by_cc = defaultdict(list)
        for r in sub:
            by_cc[int(r["concurrency"])].append(r)
        xs = sorted(by_cc.keys())
        p50 = [mean([float(x["task_p50"]) for x in by_cc[c] if x.get("task_p50") is not None]) for c in xs]
        p95 = [mean([float(x["task_p95"]) for x in by_cc[c] if x.get("task_p95") is not None]) for c in xs]
        p99 = [mean([float(x["task_p99"]) for x in by_cc[c] if x.get("task_p99") is not None]) for c in xs]
        ax.plot(xs, p50, marker="o", label="P50")
        ax.plot(xs, p95, marker="o", label="P95")
        ax.plot(xs, p99, marker="o", label="P99")
        ax.set_title(f"Task Latency TP={tp}")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Latency (s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    p = plots_dir / "task_latency_p50_p95_p99.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    outputs.append(str(p))

    # 2) Heatmaps for step/tool/inference p95
    metrics = [("step_p95", "step_latency_p95_heatmap.png"), ("tool_p95", "tool_latency_p95_heatmap.png"), ("inference_p95", "inference_latency_p95_heatmap.png")]
    for key, name in metrics:
        matrix = []
        for tp in tps:
            rowv = []
            for cc in ccs:
                vals = [float(r[key]) for r in lat_rows if int(r["tp"]) == tp and int(r["concurrency"]) == cc and r.get(key) is not None]
                rowv.append(mean(vals) if vals else 0.0)
            matrix.append(rowv)
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        im = ax.imshow(matrix, aspect="auto")
        ax.set_yticks(range(len(tps)))
        ax.set_yticklabels([str(tp) for tp in tps])
        ax.set_xticks(range(len(ccs)))
        ax.set_xticklabels([str(cc) for cc in ccs])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("TP")
        ax.set_title(key)
        fig.colorbar(im, ax=ax)
        out_path = plots_dir / name
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(str(out_path))

    # 3) Success rate bars
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    labels = []
    task_rates = []
    tool_rates = []
    for tp in tps:
        for cc in ccs:
            sub = [r for r in succ_rows if int(r["tp"]) == tp and int(r["concurrency"]) == cc]
            if not sub:
                continue
            labels.append(f"TP{tp}-C{cc}")
            task_rates.append(mean([float(r["task_success_rate"]) for r in sub]))
            tool_rates.append(mean([float(r["tool_success_rate"]) for r in sub]))
    xs = list(range(len(labels)))
    w = 0.4
    ax.bar([x - w / 2 for x in xs], task_rates, width=w, label="task_success_rate")
    ax.bar([x + w / 2 for x in xs], tool_rates, width=w, label="tool_success_rate")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Task/Tool Success Rates")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    out_path = plots_dir / "success_rates.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    outputs.append(str(out_path))

    # 4) Throughput lines
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for tp in tps:
        sub = [r for r in thr_rows if int(r["tp"]) == tp]
        by_cc = defaultdict(list)
        for r in sub:
            by_cc[int(r["concurrency"])].append(r)
        xs = sorted(by_cc.keys())
        prefill = [mean([float(x["prefill_tps_mean"]) for x in by_cc[c]]) for c in xs]
        decode = [mean([float(x["decode_tps_mean"]) for x in by_cc[c]]) for c in xs]
        req = [mean([float(x["request_tps_mean"]) for x in by_cc[c]]) for c in xs]
        axes[0].plot(xs, prefill, marker="o", label=f"TP{tp}")
        axes[1].plot(xs, decode, marker="o", label=f"TP{tp}")
        axes[2].plot(xs, req, marker="o", label=f"TP{tp}")
    axes[0].set_title("Prefill TPS")
    axes[1].set_title("Decode TPS")
    axes[2].set_title("Request TPS")
    for ax in axes:
        ax.set_xlabel("Concurrency")
        ax.grid(True, alpha=0.3)
        ax.legend()
    out_path = plots_dir / "throughput_lines.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    outputs.append(str(out_path))

    # 5) Resource bars (energy per task and power)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = []
    energy_vals = []
    power_vals = []
    for tp in tps:
        for cc in ccs:
            sub = [r for r in res_rows if int(r["tp"]) == tp and int(r["concurrency"]) == cc]
            if not sub:
                continue
            labels.append(f"TP{tp}-C{cc}")
            energy_vals.append(mean([float(r["energy_per_task_wh"]) for r in sub]))
            power_vals.append(mean([float(r["power_w_mean"]) for r in sub]))
    xs = list(range(len(labels)))
    axes[0].bar(xs, energy_vals)
    axes[0].set_title("Energy per Task (Wh)")
    axes[1].bar(xs, power_vals)
    axes[1].set_title("Mean GPU Power (W)")
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    out_path = plots_dir / "resource_energy_power.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    outputs.append(str(out_path))

    # 6) Time-series (GPU use/VRAM/KV/Power) for one representative config per TP
    for tp in tps:
        # pick cc max, round 1 if possible
        candidates = [r for r in res_rows if int(r["tp"]) == tp]
        if not candidates:
            continue
        max_cc = max(int(r["concurrency"]) for r in candidates)
        cfg_id = ""
        for r in candidates:
            if int(r["concurrency"]) == max_cc and int(r["round"]) == 1:
                cfg_id = r["config_id"]
                break
        if not cfg_id:
            cfg_id = candidates[0]["config_id"]

        cfg_gpu = aggregate["gpu_by_config"].get(cfg_id, [])
        cfg_vllm = aggregate["vllm_by_config"].get(cfg_id, [])
        if not cfg_gpu or not cfg_vllm:
            continue

        # use average over cards for each ts
        gpu_group = defaultdict(lambda: {"gpu_use": [], "vram": [], "power": []})
        for row in cfg_gpu:
            ts = row.get("ts")
            if not ts:
                continue
            if row.get("gpu_use_pct") is not None:
                gpu_group[ts]["gpu_use"].append(float(row["gpu_use_pct"]))
            if row.get("vram_pct") is not None:
                gpu_group[ts]["vram"].append(float(row["vram_pct"]))
            if row.get("power_w") is not None:
                gpu_group[ts]["power"].append(float(row["power_w"]))

        ts_sorted = sorted(gpu_group.keys())
        x = list(range(len(ts_sorted)))
        gpu_use = [mean(gpu_group[t]["gpu_use"]) if gpu_group[t]["gpu_use"] else 0.0 for t in ts_sorted]
        vram = [mean(gpu_group[t]["vram"]) if gpu_group[t]["vram"] else 0.0 for t in ts_sorted]
        power = [mean(gpu_group[t]["power"]) if gpu_group[t]["power"] else 0.0 for t in ts_sorted]

        kv_ts = [float(r.get("gpu_kv_cache_pct") or 0.0) for r in cfg_vllm]
        kv_x = list(range(len(kv_ts)))

        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
        axes[0].plot(x, gpu_use)
        axes[0].set_title(f"TP{tp} representative config {cfg_id}: GPU use %")
        axes[1].plot(x, vram)
        axes[1].set_title("VRAM %")
        axes[2].plot(kv_x, kv_ts)
        axes[2].set_title("KV cache %")
        axes[3].plot(x, power)
        axes[3].set_title("Power W")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        out_path = plots_dir / f"timeseries_tp{tp}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(str(out_path))

    return outputs


def build_report_text(run_root: Path, aggregate: Dict[str, Any], plot_paths: List[str]) -> str:
    lat_rows = aggregate["latency_summary_rows"]
    succ_rows = aggregate["success_summary_rows"]
    thr_rows = aggregate["throughput_summary_rows"]
    res_rows = aggregate["resource_summary_rows"]

    cfg_total = len(lat_rows)
    task_total = sum(int(r.get("task_total", 0)) for r in succ_rows)
    task_ok = sum(int(r.get("task_ok", 0)) for r in succ_rows)

    lines: List[str] = []
    lines.append("# GAIA Concurrency Benchmark Report")
    lines.append("")
    lines.append(f"- run_root: `{run_root}`")
    lines.append(f"- config_points: `{cfg_total}`")
    lines.append(f"- formal_tasks: `{task_total}`")
    lines.append(f"- task_ok: `{task_ok}`")
    lines.append("")

    lines.append("## Key Aggregate Files")
    lines.append("")
    lines.append(f"- `aggregate/latency_summary_by_config.csv`")
    lines.append(f"- `aggregate/success_summary_by_config.csv`")
    lines.append(f"- `aggregate/throughput_summary_by_config.csv`")
    lines.append(f"- `aggregate/resource_summary_by_config.csv`")
    lines.append(f"- `aggregate/global_summary.json`")
    lines.append("")

    # compact KPI table by TP/Concurrency averaging rounds
    lines.append("## KPI (Averaged Over Rounds)")
    lines.append("")
    lines.append("| TP | Concurrency | Task P50(s) | Task P95(s) | Task P99(s) | Task Success | Tool Success | Decode TPS | Req TPS | Energy/Task(Wh) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    by_key = defaultdict(lambda: {"lat": [], "succ": [], "thr": [], "res": []})
    for r in lat_rows:
        by_key[(int(r["tp"]), int(r["concurrency"]))]["lat"].append(r)
    for r in succ_rows:
        by_key[(int(r["tp"]), int(r["concurrency"]))]["succ"].append(r)
    for r in thr_rows:
        by_key[(int(r["tp"]), int(r["concurrency"]))]["thr"].append(r)
    for r in res_rows:
        by_key[(int(r["tp"]), int(r["concurrency"]))]["res"].append(r)

    for (tp, cc) in sorted(by_key.keys()):
        group = by_key[(tp, cc)]

        def gmean(items: List[Dict[str, Any]], key: str) -> float:
            vals = []
            for x in items:
                v = x.get(key)
                if v is None or v == "":
                    continue
                vals.append(float(v))
            return mean(vals) if vals else 0.0

        lines.append(
            f"| {tp} | {cc} | {gmean(group['lat'], 'task_p50'):.4f} | {gmean(group['lat'], 'task_p95'):.4f} | {gmean(group['lat'], 'task_p99'):.4f} | {gmean(group['succ'], 'task_success_rate'):.4f} | {gmean(group['succ'], 'tool_success_rate'):.4f} | {gmean(group['thr'], 'decode_tps_mean'):.4f} | {gmean(group['thr'], 'request_tps_mean'):.4f} | {gmean(group['res'], 'energy_per_task_wh'):.6f} |"
        )

    lines.append("")
    lines.append("## Plots")
    lines.append("")
    if plot_paths:
        for p in plot_paths:
            rel = Path(p).relative_to(run_root)
            lines.append(f"- `{rel}`")
    else:
        lines.append("- (no plots generated)")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    if not run_root.exists():
        print(f"[error] run-root not found: {run_root}", file=sys.stderr)
        return 2

    aggregate_dir = run_root / args.aggregate_dir
    plots_dir = run_root / args.plots_dir
    ensure_dir(aggregate_dir)
    ensure_dir(plots_dir)

    cfg_dirs = collect_config_dirs(run_root)
    if not cfg_dirs:
        print("[error] no config directories found", file=sys.stderr)
        return 2

    all_task_rows: List[Dict[str, Any]] = []
    all_step_rows: List[Dict[str, Any]] = []
    all_tool_rows: List[Dict[str, Any]] = []
    all_inf_rows: List[Dict[str, Any]] = []
    all_vllm_rows: List[Dict[str, Any]] = []
    all_req_rows: List[Dict[str, Any]] = []
    all_gpu_rows: List[Dict[str, Any]] = []

    latency_summary_rows: List[Dict[str, Any]] = []
    success_summary_rows: List[Dict[str, Any]] = []
    throughput_summary_rows: List[Dict[str, Any]] = []
    resource_summary_rows: List[Dict[str, Any]] = []

    gpu_by_config: Dict[str, List[Dict[str, Any]]] = {}
    vllm_by_config: Dict[str, List[Dict[str, Any]]] = {}

    for cfg in cfg_dirs:
        one = aggregate_one_config(cfg)

        all_task_rows.extend(one["task_rows"])
        all_step_rows.extend(one["step_rows"])
        all_tool_rows.extend(one["tool_rows"])
        all_inf_rows.extend(one["inference_rows"])
        all_vllm_rows.extend(one["vllm_rows"])
        all_req_rows.extend(one["req_rows"])
        all_gpu_rows.extend(one["gpu_rows"])

        latency_summary_rows.append(one["latency_summary"])
        success_summary_rows.append(one["success_summary"])
        throughput_summary_rows.append(one["throughput_summary"])
        resource_summary_rows.append(one["resource_summary"])

        gpu_by_config[one["config_id"]] = one["gpu_rows"]
        vllm_by_config[one["config_id"]] = one["vllm_rows"]

    write_jsonl(aggregate_dir / "all_task_metrics.jsonl", all_task_rows)
    write_jsonl(aggregate_dir / "all_step_metrics.jsonl", all_step_rows)
    write_jsonl(aggregate_dir / "all_tool_metrics.jsonl", all_tool_rows)
    write_jsonl(aggregate_dir / "all_inference_metrics.jsonl", all_inf_rows)
    write_jsonl(aggregate_dir / "all_vllm_timeseries.jsonl", all_vllm_rows)
    write_jsonl(aggregate_dir / "all_request_throughput_timeseries.jsonl", all_req_rows)
    write_jsonl(aggregate_dir / "all_gpu_samples.jsonl", all_gpu_rows)

    write_csv(aggregate_dir / "latency_summary_by_config.csv", latency_summary_rows)
    write_csv(aggregate_dir / "success_summary_by_config.csv", success_summary_rows)
    write_csv(aggregate_dir / "throughput_summary_by_config.csv", throughput_summary_rows)
    write_csv(aggregate_dir / "resource_summary_by_config.csv", resource_summary_rows)

    global_summary = {
        "config_count": len(latency_summary_rows),
        "formal_task_count": len(all_task_rows),
        "task_ok": sum(1 for r in all_task_rows if str(r.get("status", "")) == "ok"),
        "task_error": sum(1 for r in all_task_rows if str(r.get("status", "")) == "error"),
        "exact_match_true": sum(1 for r in all_task_rows if r.get("exact_match") is True),
        "exact_match_false": sum(1 for r in all_task_rows if r.get("exact_match") is False),
        "exact_match_none": sum(1 for r in all_task_rows if r.get("exact_match") is None),
        "task_latency_p50": quantile_nearest_rank([float(r.get("task_latency_s") or 0.0) for r in all_task_rows], 0.50),
        "task_latency_p95": quantile_nearest_rank([float(r.get("task_latency_s") or 0.0) for r in all_task_rows], 0.95),
        "task_latency_p99": quantile_nearest_rank([float(r.get("task_latency_s") or 0.0) for r in all_task_rows], 0.99),
    }
    write_json(aggregate_dir / "global_summary.json", global_summary)

    aggregate_pack = {
        "latency_summary_rows": latency_summary_rows,
        "success_summary_rows": success_summary_rows,
        "throughput_summary_rows": throughput_summary_rows,
        "resource_summary_rows": resource_summary_rows,
        "gpu_by_config": gpu_by_config,
        "vllm_by_config": vllm_by_config,
    }

    plot_paths: List[str] = []
    if not args.no_plots:
        plot_paths = maybe_plot(aggregate_pack, plots_dir)

    if not args.no_report:
        report_text = build_report_text(run_root, aggregate_pack, plot_paths)
        report_path = run_root / args.report_md
        report_path.write_text(report_text, encoding="utf-8")

    print(str(run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
