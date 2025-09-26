#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dedup_experiment.config import load_config
from dedup_experiment.data import prefetch_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch and cache a dataset split")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--split",
        required=False,
        help="Override the split expression to prefetch (defaults to config prefetched split or dataset split)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading even if the cache already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    prefetch_dataset(cfg.dataset, split_override=args.split, force=args.force)
    resolved_split = args.split or cfg.dataset.prefetch.split or cfg.dataset.split
    print(f"Prefetch complete for {cfg.dataset.name} ({cfg.dataset.subset}) split '{resolved_split}'")


if __name__ == "__main__":
    main()
