#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Set

from tools.utils import hash_text, normalize_text, manifest_hash


def load_chunks(path: Path) -> Set[str]:
    hashes = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            if data.get("kept"):
                hashes.add(data["hash"])
    return hashes


def load_corpus(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            yield data["text"]


def build_drop_list(train_hashes: Set[str], corpus: Iterable[str]) -> List[str]:
    drops = []
    for text in corpus:
        digest = hash_text(normalize_text(text))
        if digest in train_hashes:
            drops.append(digest)
    return drops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-deduplicate val/test against train manifest")
    parser.add_argument("--manifest", required=True, help="Path to train manifest.jsonl")
    parser.add_argument("--val", required=True, help="Val corpus JSONL")
    parser.add_argument("--test", required=True, help="Test corpus JSONL")
    parser.add_argument("--out", required=True, help="Directory to write drop lists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_hashes = load_chunks(Path(args.manifest))
    drops = {
        "val": build_drop_list(train_hashes, load_corpus(Path(args.val))),
        "test": build_drop_list(train_hashes, load_corpus(Path(args.test))),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, hashes in drops.items():
        (out_dir / f"{split}_drop.json").write_text(json.dumps(hashes), encoding="utf-8")
    (out_dir / "crossdedup.hash").write_text(manifest_hash(Path(args.manifest)), encoding="utf-8")


if __name__ == "__main__":
    main()
