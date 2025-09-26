#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from tools.utils import ManifestEntry, hash_text, normalize_text, write_manifest, manifest_hash


def dedup_documents(docs: Iterable[tuple[str, str]], prefix: str = "doc") -> tuple[list[ManifestEntry], list[ManifestEntry]]:
    seen = set()
    kept: list[ManifestEntry] = []
    dropped: list[ManifestEntry] = []
    for doc_id, text in docs:
        norm = normalize_text(text)
        digest = hash_text(norm)
        entry = ManifestEntry(id=f"{prefix}:{doc_id}", hash=digest, kept=False)
        if digest in seen:
            entry.reason = "duplicate"
            dropped.append(entry)
        else:
            seen.add(digest)
            entry.kept = True
            kept.append(entry)
    return kept, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run doc-level deduplication over a preloaded corpus")
    parser.add_argument("--input", required=True, help="JSONL with {id, text}")
    parser.add_argument("--out", required=True, help="Directory to write manifest + hashes")
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[tuple[str, str]]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            yield str(data["id"]), data["text"]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept, dropped = dedup_documents(load_jsonl(input_path))
    write_manifest(kept, out_dir / "manifest.jsonl")
    write_manifest(dropped, out_dir / "dropped.jsonl")
    (out_dir / "manifest.hash").write_text(manifest_hash(out_dir / "manifest.jsonl"), encoding="utf-8")


if __name__ == "__main__":
    main()
