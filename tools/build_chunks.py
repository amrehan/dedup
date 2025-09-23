#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dedup_experiment.config import load_config
from dedup_experiment.chunker import stream_and_chunk
from tools._cli import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream dataset and build chunk shards")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--output", required=True, help="Directory to write chunk shards")
    parser.add_argument("--shard_size", type=int, default=10000, help="Number of chunks per shard file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = stream_and_chunk(cfg, str(output_dir), shard_size=args.shard_size)
    summary_path = output_dir / "build_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote shards to {args.output}")
    print(f"Shard count: {len(manifest['shards'])}")
    print(f"Total tokens: {manifest['total_tokens']}")


if __name__ == "__main__":
    main()
