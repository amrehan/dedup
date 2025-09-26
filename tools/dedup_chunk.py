#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from transformers import AutoTokenizer

from tools.utils import ManifestEntry, chunk_tokens, hash_text, hash_int, normalize_text, write_manifest, manifest_hash


def load_docs(path: Path) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            yield str(data["id"]), data["text"]


def chunk_exact(docs: Iterable[tuple[str, str]], tokenizer, size: int) -> tuple[List[ManifestEntry], List[ManifestEntry]]:
    seen = set()
    kept: List[ManifestEntry] = []
    dropped: List[ManifestEntry] = []
    for doc_id, text in docs:
        tokens = tokenizer.encode(normalize_text(text))
        for idx, chunk in enumerate(chunk_tokens(tokens, size=size, stride=size)):
            chunk_text = tokenizer.decode(chunk)
            digest = hash_text(chunk_text)
            entry = ManifestEntry(id=f"chunk:{doc_id}:{idx}", hash=digest, kept=False)
            if digest in seen:
                entry.reason = "duplicate"
                dropped.append(entry)
            else:
                seen.add(digest)
                entry.kept = True
                kept.append(entry)
    return kept, dropped


def chunk_near(docs: Iterable[tuple[str, str]], tokenizer, size: int, threshold: float, num_perm: int) -> tuple[List[ManifestEntry], List[ManifestEntry]]:
    shingles: List[Tuple[str, int, np.ndarray]] = []
    kept: List[ManifestEntry] = []
    dropped: List[ManifestEntry] = []
    rng = np.random.default_rng(seed=1337)
    a = rng.integers(1, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    prime = (1 << 61) - 1

    def signature(shingle_set: set[int]) -> np.ndarray:
        if not shingle_set:
            return np.full(num_perm, prime, dtype=np.uint64)
        shingles_np = np.fromiter(shingle_set, dtype=np.uint64)
        return ((a[:, None] * shingles_np[None, :] + b[:, None]) % prime).min(axis=1)

    chunks: List[tuple[str, int, str]] = []
    for doc_id, text in docs:
        tokens = tokenizer.encode(normalize_text(text))
        for idx, chunk in enumerate(chunk_tokens(tokens, size=size, stride=size)):
            chunk_text = tokenizer.decode(chunk)
            chunk_id = f"chunk:{doc_id}:{idx}"
            hash_digest = hash_text(chunk_text)
            chunks.append((chunk_id, hash_digest, chunk_text))
            shingle_set = {hash_int(chunk_text[i : i + 32]) for i in range(0, len(chunk_text) - 32, 4)}
            shingles.append((chunk_id, hash_digest, signature(shingle_set)))

    seen = {}
    for chunk_id, digest, chunk_text in chunks:
        if digest in seen:
            dropped.append(ManifestEntry(id=chunk_id, hash=digest, kept=False, reason="duplicate"))
        else:
            seen[digest] = chunk_id
            kept.append(ManifestEntry(id=chunk_id, hash=digest, kept=True))

    bands = 4
    rows = num_perm // bands
    buckets: dict[tuple[int, tuple[int, ...]], List[int]] = {}
    for idx, (_, _, sig) in enumerate(shingles):
        for band in range(bands):
            key = (band, tuple(sig[band * rows : (band + 1) * rows]))
            buckets.setdefault(key, []).append(idx)

    to_drop = set()
    for bucket in buckets.values():
        if len(bucket) <= 1:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a_idx, b_idx = bucket[i], bucket[j]
                id_a, hash_a, sig_a = shingles[a_idx]
                id_b, hash_b, sig_b = shingles[b_idx]
                if hash_a == hash_b:
                    continue
                jac = (sig_a == sig_b).sum() / num_perm
                if jac >= threshold:
                    to_drop.add(hash_b)

    for entry in kept[:]:
        if entry.hash in to_drop and entry.kept:
            entry.kept = False
            entry.reason = "near_duplicate"
            dropped.append(entry)
            kept.remove(entry)

    return kept, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk-level deduplication")
    parser.add_argument("--input", required=True, help="JSONL with {id, text}")
    parser.add_argument("--out", required=True, help="Directory for manifests")
    parser.add_argument("--mode", choices=["exact", "near"], default="exact")
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--num_perm", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = list(load_docs(Path(args.input)))
    if args.mode == "exact":
        kept, dropped = chunk_exact(docs, tokenizer, size=args.chunk)
    else:
        kept, dropped = chunk_near(docs, tokenizer, size=args.chunk, threshold=args.threshold, num_perm=args.num_perm)
    write_manifest(kept, out_dir / "manifest.jsonl")
    write_manifest(dropped, out_dir / "dropped.jsonl")
    (out_dir / "manifest.hash").write_text(manifest_hash(out_dir / "manifest.jsonl"), encoding="utf-8")


if __name__ == "__main__":
    main()
