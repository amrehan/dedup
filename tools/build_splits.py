#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from tools.utils import stratified_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratify dataset into train/val/test splits")
    parser.add_argument("--input", required=True, help="Path to dataset metadata CSV/Parquet")
    parser.add_argument("--output", required=True, help="Directory to write split files")
    parser.add_argument(
        "--fields",
        required=False,
        default="configs/dataset_fields.yaml",
        help="YAML mapping of stratification field names",
    )
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    return parser.parse_args()


def load_field_map(path: Path) -> List[str]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)
    return list(mapping.values())


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    fields = load_field_map(Path(args.fields))
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(f"Missing stratification fields: {missing}")
    train_idx, val_idx, test_idx = stratified_sample(df, args.train_frac, args.val_frac, args.test_frac, fields)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_ids.json").write_text(json.dumps(list(map(int, train_idx))), encoding="utf-8")
    (out_dir / "val_ids.json").write_text(json.dumps(list(map(int, val_idx))), encoding="utf-8")
    (out_dir / "test_ids.json").write_text(json.dumps(list(map(int, test_idx))), encoding="utf-8")


if __name__ == "__main__":
    main()
