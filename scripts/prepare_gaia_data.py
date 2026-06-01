#!/usr/bin/env python3
"""Prepare GAIA rows in the compact jsonl format used by the runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download/normalize GAIA validation rows.")
    parser.add_argument("--dataset", default="gaia-benchmark/GAIA")
    parser.add_argument("--config", default="2023_all")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--idx-start", type=int, default=0, help="Inclusive dataset index.")
    parser.add_argument("--idx-end", type=int, default=-1, help="Exclusive dataset index. -1 means dataset end.")
    parser.add_argument("--download-attachments", action="store_true", help="Download local files for file-based GAIA tasks.")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--output-dir", default=str(root / "data" / "gaia_2023_all_validation"))
    return parser.parse_args()


def token_from_env(args: argparse.Namespace) -> str:
    token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    )
    if not token:
        raise RuntimeError("Missing Hugging Face token. Set HF_TOKEN or pass --hf-token.")
    return token


def maybe_int(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return value


def main() -> int:
    args = parse_args()
    token = token_from_env(args)
    out_dir = Path(args.output_dir).resolve()
    attachments_root = out_dir / "attachments"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.download_attachments:
        attachments_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.config, split=args.split, token=token)
    idx_end = len(ds) if args.idx_end < 0 else min(args.idx_end, len(ds))
    if args.idx_start < 0 or args.idx_start > idx_end:
        raise ValueError(f"bad idx range: idx_start={args.idx_start}, idx_end={idx_end}")

    rows: List[Dict[str, Any]] = []
    for idx in range(args.idx_start, idx_end):
        row = ds[idx]
        file_path = (row.get("file_path") or "").strip()
        item: Dict[str, Any] = {
            "idx": idx,
            "task_id": row.get("task_id", ""),
            "question": (row.get("Question") or "").strip(),
            "final_answer": (row.get("Final answer") or "").strip(),
            "level": maybe_int(row.get("Level")),
            "file_name": (row.get("file_name") or "").strip(),
            "file_path": file_path,
            "local_attachment": "",
        }
        if args.download_attachments and file_path:
            local_path = hf_hub_download(
                repo_id=args.dataset,
                repo_type="dataset",
                filename=file_path,
                token=token,
                local_dir=str(attachments_root / f"idx{idx}"),
            )
            item["local_attachment"] = str(Path(local_path).resolve())
        rows.append(item)

    rows_file = out_dir / "rows.jsonl"
    rows_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "config": args.config,
                "split": args.split,
                "idx_start": args.idx_start,
                "idx_end": idx_end,
                "count": len(rows),
                "download_attachments": bool(args.download_attachments),
                "rows_file": str(rows_file),
                "attachments_dir": str(attachments_root) if args.download_attachments else "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
