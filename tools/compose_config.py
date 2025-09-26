#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in b.items():
        if key in a and isinstance(a[key], dict) and isinstance(value, dict):
            merge(a[key], value)
        else:
            a[key] = copy.deepcopy(value)
    return a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose YAML configs via deep merge")
    parser.add_argument("--base", required=True, help="Base YAML config")
    parser.add_argument("--overrides", nargs="*", default=[], help="Override YAML files applied in order")
    parser.add_argument("--out", required=True, help="Destination YAML path")
    parser.add_argument("--dump", action="store_true", help="Print merged config to stdout as JSON for inspection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged: Dict[str, Any] = yaml.safe_load(Path(args.base).read_text())
    for override in args.overrides:
        override_dict = yaml.safe_load(Path(override).read_text())
        if override_dict:
            merge(merged, override_dict)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    if args.dump:
        print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
