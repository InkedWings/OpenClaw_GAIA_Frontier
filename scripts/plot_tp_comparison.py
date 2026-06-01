#!/usr/bin/env python3

import argparse
import csv
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUMMARY_FILES = {
    "latency": "latency_summary_by_config.csv",
    "success": "success_summary_by_config.csv",
    "throughput": "throughput_summary_by_config.csv",
    "resource": "resource_summary_by_config.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cross-run TP/concurrency comparison plots from GAIA aggregate summaries."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="Run directory containing aggregate/*.csv. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where merged CSV, report, and plots will be written.",
    )
    parser.add_argument(
        "--title",
        default="GAIA Qwen35 TP/Concurrency Comparison",
        help="Title used in the generated report.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4) -> str:
    v = to_float(value)
    if math.isnan(v):
        return ""
    return f"{v:.{digits}f}"


def metric(row: Dict[str, Any], key: str) -> float:
    return to_float(row.get(key))


def mean_clean(values: Iterable[Any]) -> float:
    clean = [to_float(v) for v in values]
    clean = [v for v in clean if not math.isnan(v)]
    return mean(clean) if clean else math.nan


def merge_run(run_root: Path) -> List[Dict[str, Any]]:
    aggregate_dir = run_root / "aggregate"
    tables = {
        name: read_csv(aggregate_dir / filename)
        for name, filename in SUMMARY_FILES.items()
    }
    if not any(tables.values()):
        raise FileNotFoundError(f"No aggregate summary CSVs found under {aggregate_dir}")

    merged: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
    for table_name, rows in tables.items():
        for row in rows:
            config_id = row.get("config_id", "")
            tp = to_int(row.get("tp"))
            cc = to_int(row.get("concurrency"))
            rr = to_int(row.get("round"))
            key = (config_id, tp, cc, rr)
            out = merged.setdefault(
                key,
                {
                    "source_run": run_root.name,
                    "source_path": str(run_root),
                    "config_id": config_id,
                    "tp": tp,
                    "concurrency": cc,
                    "round": rr,
                },
            )
            if table_name == "latency":
                out.update(
                    {
                        "task_count": to_int(row.get("task_count")),
                        "step_count": to_int(row.get("step_count")),
                        "tool_count": to_int(row.get("tool_count")),
                        "inference_count": to_int(row.get("inference_count")),
                        "task_p50_s": to_float(row.get("task_p50")),
                        "task_p95_s": to_float(row.get("task_p95")),
                        "task_p99_s": to_float(row.get("task_p99")),
                        "step_p95_s": to_float(row.get("step_p95")),
                        "tool_p95_s": to_float(row.get("tool_p95")),
                        "inference_p95_s": to_float(row.get("inference_p95")),
                    }
                )
            elif table_name == "success":
                out.update(
                    {
                        "task_total": to_int(row.get("task_total")),
                        "task_ok": to_int(row.get("task_ok")),
                        "task_success_rate": to_float(row.get("task_success_rate")),
                        "exact_match_true": to_int(row.get("exact_match_true")),
                        "exact_match_rate": to_float(row.get("exact_match_rate")),
                        "tool_calls": to_int(row.get("tool_calls")),
                        "tool_success_rate": to_float(row.get("tool_success_rate")),
                        "avg_step_count": to_float(row.get("avg_step_count")),
                        "avg_tool_calls_per_task": to_float(row.get("avg_tool_calls_per_task")),
                    }
                )
            elif table_name == "throughput":
                out.update(
                    {
                        "prefill_tps_mean": to_float(row.get("prefill_tps_mean")),
                        "decode_tps_mean": to_float(row.get("decode_tps_mean")),
                        "request_tps_mean": to_float(row.get("request_tps_mean")),
                        "request_tps_p95": to_float(row.get("request_tps_p95")),
                    }
                )
            elif table_name == "resource":
                out.update(
                    {
                        "gpu_use_mean": to_float(row.get("gpu_use_mean")),
                        "vram_pct_mean": to_float(row.get("vram_pct_mean")),
                        "kv_cache_pct_mean": to_float(row.get("kv_cache_pct_mean")),
                        "power_w_mean": to_float(row.get("power_w_mean")),
                        "total_energy_wh": to_float(row.get("total_energy_wh")),
                        "energy_per_task_wh": to_float(row.get("energy_per_task_wh")),
                    }
                )

    return sorted(merged.values(), key=lambda r: (r["tp"], r["concurrency"], r["round"], r["source_run"]))


def aggregate_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(to_int(row.get("tp")), to_int(row.get("concurrency")))].append(row)

    out: List[Dict[str, Any]] = []
    for (tp, cc), group in sorted(groups.items()):
        item: Dict[str, Any] = {
            "tp": tp,
            "concurrency": cc,
            "runs": ";".join(sorted({str(r.get("source_run", "")) for r in group})),
            "configs": ";".join(sorted({str(r.get("config_id", "")) for r in group})),
            "replicates": len(group),
        }
        sum_keys = ["task_total", "task_ok", "exact_match_true", "tool_calls"]
        for key in sum_keys:
            item[key] = sum(to_int(r.get(key)) for r in group)
        mean_keys = [
            "task_success_rate",
            "exact_match_rate",
            "tool_success_rate",
            "avg_step_count",
            "avg_tool_calls_per_task",
            "task_p50_s",
            "task_p95_s",
            "task_p99_s",
            "step_p95_s",
            "tool_p95_s",
            "inference_p95_s",
            "prefill_tps_mean",
            "decode_tps_mean",
            "request_tps_mean",
            "request_tps_p95",
            "gpu_use_mean",
            "vram_pct_mean",
            "kv_cache_pct_mean",
            "power_w_mean",
            "total_energy_wh",
            "energy_per_task_wh",
        ]
        for key in mean_keys:
            item[key] = mean_clean(r.get(key) for r in group)
        out.append(item)
    return out


def setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    return plt


def color_for_tp(tp: int) -> str:
    return {
        2: "#4C78A8",
        4: "#F58518",
        8: "#54A24B",
    }.get(tp, "#B279A2")


def plot_line_panels(
    plt: Any,
    rows: List[Dict[str, Any]],
    panels: List[Tuple[str, str, str]],
    output_path: Path,
    title: str,
) -> None:
    tps = sorted({to_int(r.get("tp")) for r in rows})
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4), squeeze=False)
    for ax, (key, panel_title, y_label) in zip(axes[0], panels):
        for tp in tps:
            sub = sorted([r for r in rows if to_int(r.get("tp")) == tp], key=lambda r: to_int(r.get("concurrency")))
            xs = [to_int(r.get("concurrency")) for r in sub if not math.isnan(metric(r, key))]
            ys = [metric(r, key) for r in sub if not math.isnan(metric(r, key))]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", linewidth=2, label=f"TP{tp}", color=color_for_tp(tp))
            for x, y in zip(xs, ys):
                ax.annotate(fmt(y, 2), (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(panel_title)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(y_label)
        ax.set_xticks(sorted({to_int(r.get("concurrency")) for r in rows}))
        ax.legend()
    fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_rates(plt: Any, rows: List[Dict[str, Any]], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda r: (to_int(r.get("tp")), to_int(r.get("concurrency"))))
    labels = [f"TP{to_int(r.get('tp'))}\nCC{to_int(r.get('concurrency'))}" for r in ordered]
    series = [
        ("task_success_rate", "Task success", "#4C78A8"),
        ("exact_match_rate", "Exact match", "#F58518"),
        ("tool_success_rate", "Tool success", "#54A24B"),
    ]
    x = list(range(len(ordered)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for i, (key, label, color) in enumerate(series):
        offset = (i - 1) * width
        vals = [metric(r, key) for r in ordered]
        ax.bar([p + offset for p in x], vals, width=width, label=label, color=color)
        for p, v in zip(x, vals):
            if not math.isnan(v):
                ax.text(p + offset, min(1.04, v + 0.015), fmt(v, 2), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate")
    ax.set_title("Quality and Success Rates")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(plt: Any, rows: List[Dict[str, Any]], output_path: Path) -> None:
    import matplotlib.cm as cm

    tps = sorted({to_int(r.get("tp")) for r in rows})
    ccs = sorted({to_int(r.get("concurrency")) for r in rows})
    by_key = {(to_int(r.get("tp")), to_int(r.get("concurrency"))): r for r in rows}
    panels = [
        ("task_p95_s", "Task P95 latency (s)", "viridis"),
        ("decode_tps_mean", "Decode TPS", "magma"),
        ("exact_match_rate", "Exact match rate", "cividis"),
        ("energy_per_task_wh", "Energy per task (Wh)", "plasma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (key, title, cmap_name) in zip(axes.ravel(), panels):
        matrix: List[List[float]] = []
        for tp in tps:
            matrix.append([metric(by_key.get((tp, cc), {}), key) for cc in ccs])
        cmap = cm.get_cmap(cmap_name).copy()
        cmap.set_bad(color="#EEEEEE")
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(ccs)))
        ax.set_xticklabels([str(cc) for cc in ccs])
        ax.set_yticks(range(len(tps)))
        ax.set_yticklabels([str(tp) for tp in tps])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("TP")
        for i, tp in enumerate(tps):
            for j, cc in enumerate(ccs):
                value = metric(by_key.get((tp, cc), {}), key)
                if not math.isnan(value):
                    digits = 2 if key != "exact_match_rate" else 3
                    ax.text(j, i, fmt(value, digits), ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.84)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(plt: Any, rows: List[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for row in sorted(rows, key=lambda r: (to_int(r.get("tp")), to_int(r.get("concurrency")))):
        tp = to_int(row.get("tp"))
        cc = to_int(row.get("concurrency"))
        x = metric(row, "task_p95_s")
        y = metric(row, "exact_match_rate")
        decode = metric(row, "decode_tps_mean")
        if math.isnan(x) or math.isnan(y):
            continue
        size = 80 if math.isnan(decode) else max(80, min(420, 55 + decode * 8))
        ax.scatter(x, y, s=size, color=color_for_tp(tp), alpha=0.78, edgecolor="white", linewidth=1.0)
        ax.annotate(f"TP{tp}/CC{cc}", (x, y), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Task P95 latency (s), lower is better")
    ax.set_ylabel("Exact match rate, higher is better")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Accuracy vs Latency Tradeoff")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def quote_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "\\'")


def quote_gnuplot(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def write_gnuplot_script(path: Path, script: str) -> None:
    path.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        ["gnuplot", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gnuplot failed for {path}:\n{proc.stderr}")


def write_line_data(data_dir: Path, rows: List[Dict[str, Any]], key: str) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for tp in sorted({to_int(r.get("tp")) for r in rows}):
        path = data_dir / f"{key}_tp{tp}.dat"
        with path.open("w", encoding="utf-8") as f:
            for row in sorted([r for r in rows if to_int(r.get("tp")) == tp], key=lambda r: to_int(r.get("concurrency"))):
                value = metric(row, key)
                if not math.isnan(value):
                    f.write(f"{to_int(row.get('concurrency'))} {value}\n")
        out[tp] = path
    return out


def gnuplot_line_panels(
    rows: List[Dict[str, Any]],
    panels: List[Tuple[str, str, str]],
    output_path: Path,
    title: str,
    data_dir: Path,
) -> None:
    ccs = sorted({to_int(r.get("concurrency")) for r in rows})
    xtics = ", ".join([f"'{cc}' {cc}" for cc in ccs])
    script_lines = [
        "set terminal pngcairo size 1800,560 enhanced font 'Arial,10'",
        f"set output '{quote_path(output_path)}'",
        "set datafile missing 'nan'",
        "set key outside top horizontal",
        f"set multiplot layout 1,{len(panels)} title '{quote_gnuplot(title)}' font ',14'",
    ]
    for key, panel_title, y_label in panels:
        files = write_line_data(data_dir, rows, key)
        plot_parts = []
        for tp, path in sorted(files.items()):
            plot_parts.append(
                f"'{quote_path(path)}' using 1:2 with linespoints lw 2 pt 7 lc rgb '{color_for_tp(tp)}' title 'TP{tp}'"
            )
        script_lines.extend(
            [
                f"set title '{quote_gnuplot(panel_title)}'",
                "set xlabel 'Concurrency'",
                f"set ylabel '{quote_gnuplot(y_label)}'",
                f"set xtics ({xtics})",
                "set yrange [0:*]",
                "set grid",
                "plot " + ", ".join(plot_parts),
            ]
        )
    script_lines.extend(["unset multiplot", "set output"])
    write_gnuplot_script(data_dir / (output_path.stem + ".gp"), "\n".join(script_lines) + "\n")


def gnuplot_rates(rows: List[Dict[str, Any]], output_path: Path, data_dir: Path) -> None:
    data_path = data_dir / "quality_success_rates.dat"
    with data_path.open("w", encoding="utf-8") as f:
        f.write("label task_success exact_match tool_success\n")
        for row in sorted(rows, key=lambda r: (to_int(r.get("tp")), to_int(r.get("concurrency")))):
            f.write(
                f"TP{to_int(row.get('tp'))}-CC{to_int(row.get('concurrency'))} "
                f"{metric(row, 'task_success_rate')} {metric(row, 'exact_match_rate')} {metric(row, 'tool_success_rate')}\n"
            )

    script = f"""
set terminal pngcairo size 1500,640 enhanced font 'Arial,10'
set output '{quote_path(output_path)}'
set style data histogram
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.9
set yrange [0:1.12]
set ylabel 'Rate'
set title 'Quality and Success Rates' font ',14'
set xtics rotate by -45
set grid ytics
set key outside top horizontal
plot '{quote_path(data_path)}' using 2:xtic(1) title 'Task success' lc rgb '#4C78A8', \\
     '' using 3 title 'Exact match' lc rgb '#F58518', \\
     '' using 4 title 'Tool success' lc rgb '#54A24B'
set output
"""
    write_gnuplot_script(data_dir / "quality_success_rates.gp", script)


def write_heatmap_data(
    data_dir: Path,
    rows: List[Dict[str, Any]],
    key: str,
    tps: List[int],
    ccs: List[int],
) -> Tuple[Path, Path]:
    by_key = {(to_int(r.get("tp")), to_int(r.get("concurrency"))): r for r in rows}
    data_path = data_dir / f"heatmap_{key}.dat"
    labels_path = data_dir / f"heatmap_{key}_labels.dat"
    with data_path.open("w", encoding="utf-8") as data_f, labels_path.open("w", encoding="utf-8") as labels_f:
        for y, tp in enumerate(tps):
            for x, cc in enumerate(ccs):
                value = metric(by_key.get((tp, cc), {}), key)
                if math.isnan(value):
                    data_f.write(f"{x} {y} NaN\n")
                else:
                    data_f.write(f"{x} {y} {value}\n")
                    digits = 3 if key == "exact_match_rate" else 2
                    labels_f.write(f"{x} {y} {fmt(value, digits)}\n")
            data_f.write("\n")
    return data_path, labels_path


def gnuplot_heatmaps(rows: List[Dict[str, Any]], output_path: Path, data_dir: Path) -> None:
    tps = sorted({to_int(r.get("tp")) for r in rows})
    ccs = sorted({to_int(r.get("concurrency")) for r in rows})
    xtics = ", ".join([f"'{cc}' {idx}" for idx, cc in enumerate(ccs)])
    ytics = ", ".join([f"'{tp}' {idx}" for idx, tp in enumerate(tps)])
    panels = [
        ("task_p95_s", "Task P95 latency (s)", "rgbformulae 33,13,10"),
        ("decode_tps_mean", "Decode TPS", "rgbformulae 7,5,15"),
        ("exact_match_rate", "Exact match rate", "rgbformulae 22,13,-31"),
        ("energy_per_task_wh", "Energy per task (Wh)", "rgbformulae 30,31,32"),
    ]
    script_lines = [
        "set terminal pngcairo size 1400,920 enhanced font 'Arial,10'",
        f"set output '{quote_path(output_path)}'",
        "set multiplot layout 2,2 title 'TP/Concurrency Heatmaps' font ',14'",
        "unset key",
        "set view map",
        f"set xrange [-0.5:{len(ccs) - 0.5}]",
        f"set yrange [-0.5:{len(tps) - 0.5}]",
        f"set xtics ({xtics})",
        f"set ytics ({ytics})",
        "set xlabel 'Concurrency'",
        "set ylabel 'TP'",
    ]
    for key, title, palette in panels:
        data_path, labels_path = write_heatmap_data(data_dir, rows, key, tps, ccs)
        script_lines.extend(
            [
                f"set title '{quote_gnuplot(title)}'",
                f"set palette {palette}",
                f"plot '{quote_path(data_path)}' using 1:2:3 with image, "
                f"'{quote_path(labels_path)}' using 1:2:3 with labels tc rgb 'white' notitle",
            ]
        )
    script_lines.extend(["unset multiplot", "set output"])
    write_gnuplot_script(data_dir / "tp_cc_heatmaps.gp", "\n".join(script_lines) + "\n")


def gnuplot_tradeoff(rows: List[Dict[str, Any]], output_path: Path, data_dir: Path) -> None:
    files: Dict[int, Path] = {}
    for tp in sorted({to_int(r.get("tp")) for r in rows}):
        path = data_dir / f"tradeoff_tp{tp}.dat"
        with path.open("w", encoding="utf-8") as f:
            for row in sorted([r for r in rows if to_int(r.get("tp")) == tp], key=lambda r: to_int(r.get("concurrency"))):
                x = metric(row, "task_p95_s")
                y = metric(row, "exact_match_rate")
                if math.isnan(x) or math.isnan(y):
                    continue
                f.write(f"{x} {y} TP{tp}/CC{to_int(row.get('concurrency'))}\n")
        files[tp] = path

    plot_parts = []
    for tp, path in sorted(files.items()):
        plot_parts.append(
            f"'{quote_path(path)}' using 1:2 with points pt 7 ps 1.4 lc rgb '{color_for_tp(tp)}' title 'TP{tp}'"
        )
        plot_parts.append(
            f"'{quote_path(path)}' using 1:2:3 with labels offset char 1,1 tc rgb '{color_for_tp(tp)}' notitle"
        )
    script = f"""
set terminal pngcairo size 1050,720 enhanced font 'Arial,10'
set output '{quote_path(output_path)}'
set title 'Accuracy vs Latency Tradeoff' font ',14'
set xlabel 'Task P95 latency (s), lower is better'
set ylabel 'Exact match rate, higher is better'
set xrange [0:*]
set yrange [0:1.05]
set grid
set key outside top horizontal
plot {", ".join(plot_parts)}
set output
"""
    write_gnuplot_script(data_dir / "accuracy_latency_tradeoff.gp", script)


def plot_with_gnuplot(rows: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    if shutil.which("gnuplot") is None:
        raise RuntimeError("matplotlib is unavailable and gnuplot was not found on PATH")

    data_dir = output_dir / "_plot_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        output_dir / "latency_scaling.png",
        output_dir / "throughput_scaling.png",
        output_dir / "quality_success_rates.png",
        output_dir / "resource_scaling.png",
        output_dir / "tp_cc_heatmaps.png",
        output_dir / "accuracy_latency_tradeoff.png",
    ]
    gnuplot_line_panels(
        rows,
        [
            ("task_p50_s", "Task P50", "Seconds"),
            ("task_p95_s", "Task P95", "Seconds"),
            ("task_p99_s", "Task P99", "Seconds"),
        ],
        plots[0],
        "Task Latency Scaling",
        data_dir,
    )
    gnuplot_line_panels(
        rows,
        [
            ("prefill_tps_mean", "Prefill TPS", "Tokens/s"),
            ("decode_tps_mean", "Decode TPS", "Tokens/s"),
            ("request_tps_mean", "Request TPS", "Requests/s"),
        ],
        plots[1],
        "Throughput Scaling",
        data_dir,
    )
    gnuplot_rates(rows, plots[2], data_dir)
    gnuplot_line_panels(
        rows,
        [
            ("gpu_use_mean", "Mean GPU Use", "%"),
            ("power_w_mean", "Mean GPU Power", "W"),
            ("energy_per_task_wh", "Energy per Task", "Wh"),
            ("kv_cache_pct_mean", "Mean KV Cache", "%"),
        ],
        plots[3],
        "Resource Scaling",
        data_dir,
    )
    gnuplot_heatmaps(rows, plots[4], data_dir)
    gnuplot_tradeoff(rows, plots[5], data_dir)
    return plots


def write_report(path: Path, title: str, run_roots: List[Path], rows: List[Dict[str, Any]], plots: List[Path]) -> None:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Input Runs")
    lines.append("")
    for run_root in run_roots:
        lines.append(f"- `{run_root}`")
    lines.append("")
    lines.append("## KPI Table")
    lines.append("")
    lines.append(
        "| TP | CC | Tasks | Task OK | Exact Match | Task P50(s) | Task P95(s) | Decode TPS | Req TPS | GPU Use % | Power W | Energy/Task Wh |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (to_int(x.get("tp")), to_int(x.get("concurrency")))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(to_int(r.get("tp"))),
                    str(to_int(r.get("concurrency"))),
                    str(to_int(r.get("task_total"))),
                    str(to_int(r.get("task_ok"))),
                    fmt(r.get("exact_match_rate"), 3),
                    fmt(r.get("task_p50_s"), 2),
                    fmt(r.get("task_p95_s"), 2),
                    fmt(r.get("decode_tps_mean"), 2),
                    fmt(r.get("request_tps_mean"), 4),
                    fmt(r.get("gpu_use_mean"), 2),
                    fmt(r.get("power_w_mean"), 2),
                    fmt(r.get("energy_per_task_wh"), 4),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for plot in plots:
        lines.append(f"- `{plot.name}`")
    lines.append("- `combined_tp_cc_metrics.csv`")
    lines.append("- `combined_tp_cc_metrics_by_run.csv`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_roots = [Path(p).resolve() for p in args.run_root]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_run: List[Dict[str, Any]] = []
    for run_root in run_roots:
        rows_by_run.extend(merge_run(run_root))

    rows = aggregate_points(rows_by_run)

    columns = [
        "tp",
        "concurrency",
        "runs",
        "configs",
        "replicates",
        "task_total",
        "task_ok",
        "task_success_rate",
        "exact_match_true",
        "exact_match_rate",
        "tool_calls",
        "tool_success_rate",
        "avg_step_count",
        "avg_tool_calls_per_task",
        "task_p50_s",
        "task_p95_s",
        "task_p99_s",
        "step_p95_s",
        "tool_p95_s",
        "inference_p95_s",
        "prefill_tps_mean",
        "decode_tps_mean",
        "request_tps_mean",
        "request_tps_p95",
        "gpu_use_mean",
        "vram_pct_mean",
        "kv_cache_pct_mean",
        "power_w_mean",
        "total_energy_wh",
        "energy_per_task_wh",
    ]
    by_run_columns = [
        "source_run",
        "source_path",
        "config_id",
        "tp",
        "concurrency",
        "round",
        "task_total",
        "task_ok",
        "task_success_rate",
        "exact_match_true",
        "exact_match_rate",
        "tool_calls",
        "tool_success_rate",
        "task_p50_s",
        "task_p95_s",
        "task_p99_s",
        "prefill_tps_mean",
        "decode_tps_mean",
        "request_tps_mean",
        "gpu_use_mean",
        "power_w_mean",
        "total_energy_wh",
        "energy_per_task_wh",
    ]
    write_csv(output_dir / "combined_tp_cc_metrics.csv", rows, columns)
    write_csv(output_dir / "combined_tp_cc_metrics_by_run.csv", rows_by_run, by_run_columns)

    plots = [
        output_dir / "latency_scaling.png",
        output_dir / "throughput_scaling.png",
        output_dir / "quality_success_rates.png",
        output_dir / "resource_scaling.png",
        output_dir / "tp_cc_heatmaps.png",
        output_dir / "accuracy_latency_tradeoff.png",
    ]

    try:
        plt = setup_matplotlib()
    except ImportError:
        plots = plot_with_gnuplot(rows, output_dir)
    else:
        plot_line_panels(
            plt,
            rows,
            [
                ("task_p50_s", "Task P50", "Seconds"),
                ("task_p95_s", "Task P95", "Seconds"),
                ("task_p99_s", "Task P99", "Seconds"),
            ],
            plots[0],
            "Task Latency Scaling",
        )
        plot_line_panels(
            plt,
            rows,
            [
                ("prefill_tps_mean", "Prefill TPS", "Tokens/s"),
                ("decode_tps_mean", "Decode TPS", "Tokens/s"),
                ("request_tps_mean", "Request TPS", "Requests/s"),
            ],
            plots[1],
            "Throughput Scaling",
        )
        plot_rates(plt, rows, plots[2])
        plot_line_panels(
            plt,
            rows,
            [
                ("gpu_use_mean", "Mean GPU Use", "%"),
                ("power_w_mean", "Mean GPU Power", "W"),
                ("energy_per_task_wh", "Energy per Task", "Wh"),
                ("kv_cache_pct_mean", "Mean KV Cache", "%"),
            ],
            plots[3],
            "Resource Scaling",
        )
        plot_heatmaps(plt, rows, plots[4])
        plot_tradeoff(plt, rows, plots[5])
    write_report(output_dir / "report.md", args.title, run_roots, rows, plots)

    print(f"Wrote {len(rows)} TP/CC points to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
