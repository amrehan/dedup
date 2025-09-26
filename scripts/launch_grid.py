#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch experiment grid from CSV")
    parser.add_argument("--grid", required=True, help="CSV with columns name,dedup,tokens,model,seed")
    parser.add_argument("--base", default=str(ROOT / "configs/base_gpt2.yaml"))
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def build_command(base_cfg: Path, row: dict) -> List[str]:
    overrides = [
        ROOT / "configs/overrides/model" / f"{row['model']}.yaml",
        ROOT / "configs/overrides/dedup" / f"{row['dedup']}.yaml",
        ROOT / "configs/overrides/tokens" / f"{row['tokens']}.yaml",
    ]
    out_cfg = ROOT / ".tmp" / f"{row['name']}.yaml"
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    compose = [
        "python",
        str(ROOT / "tools/compose_config.py"),
        "--base",
        str(base_cfg),
        "--out",
        str(out_cfg),
        "--overrides",
    ] + [str(path) for path in overrides]
    train = [
        "python",
        str(ROOT / "run_experiment.py"),
        "--config",
        str(out_cfg),
    ]
    return compose, train


def main() -> None:
    args = parse_args()
    base_cfg = Path(args.base)
    with open(args.grid, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            compose_cmd, train_cmd = build_command(base_cfg, row)
            if args.dry_run:
                print(" ".join(compose_cmd))
                print(" ".join(train_cmd))
                continue
            subprocess.run(compose_cmd, check=True)
            env = dict(**dict(), seed=row.get("seed", ""))
            subprocess.run(train_cmd, check=True, env=env if env["seed"] else None)


if __name__ == "__main__":
    main()
