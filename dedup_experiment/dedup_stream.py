from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

from .config import ExperimentConfig
from .dedup import DedupStats


@dataclass
class ShingleEntry:
    shard_id: int
    local_index: int
    shingles: List[int]


def load_chunk_metadata(path: str | Path) -> Iterator[Tuple[int, int, int]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            meta = json.loads(line)
            yield meta["shard_id"], meta["local_index"], meta["exact_hash"]


def load_shingles(path: str | Path) -> Iterator[ShingleEntry]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            yield ShingleEntry(
                shard_id=data["shard_id"],
                local_index=data["local_index"],
                shingles=data.get("shingles", []),
            )


def exact_dedup(metadata_path: str | Path) -> Tuple[Set[Tuple[int, int]], DedupStats]:
    hash_to_chunks: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for shard_id, local_idx, hash_value in load_chunk_metadata(metadata_path):
        hash_to_chunks[hash_value].append((shard_id, local_idx))
    drop_ids: Set[Tuple[int, int]] = set()
    stats = DedupStats()
    for chunk_list in hash_to_chunks.values():
        if len(chunk_list) <= 1:
            continue
        stats.exact_duplicates += len(chunk_list) - 1
        drop_ids.update(chunk_list[1:])
    return drop_ids, stats


def compute_minhash_matrix(entries: List[ShingleEntry], cfg: ExperimentConfig) -> np.ndarray:
    num_perm = cfg.dedup.near.num_permutations
    rng = np.random.default_rng(seed=cfg.seed)
    a = rng.integers(1, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    prime = (1 << 61) - 1
    signatures = []
    for entry in entries:
        shingles = entry.shingles
        if not shingles:
            signatures.append(np.full(num_perm, prime, dtype=np.uint64))
            continue
        shingles_np = np.array(shingles, dtype=np.uint64)
        sig = ((a[:, None] * shingles_np[None, :] + b[:, None]) % prime).min(axis=1)
        signatures.append(sig)
    return np.stack(signatures, axis=0)


def lsh_candidates(signatures: np.ndarray, cfg: ExperimentConfig, indices: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    num_perm = cfg.dedup.near.num_permutations
    band_size = cfg.dedup.near.band_size
    bands = num_perm // band_size
    candidates: Set[Tuple[int, int]] = set()
    for band in range(bands):
        start = band * band_size
        end = start + band_size
        buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        band_slice = signatures[:, start:end]
        for idx, row in enumerate(band_slice):
            buckets[tuple(row.tolist())].append(idx)
        for bucket in buckets.values():
            if len(bucket) <= 1:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a_idx = bucket[i]
                    b_idx = bucket[j]
                    chunk_a = indices[a_idx]
                    chunk_b = indices[b_idx]
                    if chunk_a == chunk_b:
                        continue
                    candidates.add(tuple(sorted([chunk_a, chunk_b])))
    return candidates


def near_dedup(shingle_path: str | Path, cfg: ExperimentConfig) -> Tuple[Set[Tuple[int, int]], DedupStats]:
    entries = list(load_shingles(shingle_path))
    indices = [(e.shard_id, e.local_index) for e in entries]
    signatures = compute_minhash_matrix(entries, cfg)
    candidates = lsh_candidates(signatures, cfg, indices)
    drop_ids: Set[Tuple[int, int]] = set()
    stats = DedupStats()
    for (a, b) in candidates:
        entry_a = entries[indices.index(a)]
        entry_b = entries[indices.index(b)]
        set_a = set(entry_a.shingles)
        set_b = set(entry_b.shingles)
        if not set_a or not set_b:
            continue
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jac = intersection / union if union else 0.0
        stats.near_candidates += 1
        if jac >= cfg.dedup.near.threshold:
            stats.near_duplicates += 1
            # drop longer chunk arbitrarily keep first
            drop_ids.add(b)
    return drop_ids, stats


def write_drop_ids(drop_ids: Set[Tuple[int, int]], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for shard_id, local_index in sorted(drop_ids):
            handle.write(json.dumps({"shard_id": shard_id, "local_index": local_index}) + "\n")
