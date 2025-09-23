#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dedup_experiment.config import load_config
from dedup_experiment.dedup_stream import exact_dedup, near_dedup, write_drop_ids
from tools._cli import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deduplication on chunk metadata")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--chunks", required=True, help="Directory containing manifest.json")
    parser.add_argument("--output", required=True, help="Path to drop list JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    chunks_dir = resolve_path(args.chunks)
    manifest_path = chunks_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    chunk_meta_path = manifest["chunk_metadata_path"]
    shingle_path = manifest.get("shingle_path")

    drop_exact, stats_exact = exact_dedup(chunk_meta_path)
    drop_near, stats_near = near_dedup(shingle_path, cfg)
    drop_ids = drop_exact | drop_near
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_drop_ids(drop_ids, output_path)

    summary = {
        "exact": stats_exact.__dict__,
        "near": stats_near.__dict__,
        "total_drop": len(drop_ids),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
