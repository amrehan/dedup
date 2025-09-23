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


def minhash_signature(shingles: Set[int], a: np.ndarray, b: np.ndarray, prime: int) -> np.ndarray:
    num_perm = a.shape[0]
    if not shingles:
        return np.full(num_perm, prime, dtype=np.uint64)
    shingles_np = np.fromiter((s & ((1 << 61) - 1) for s in shingles), dtype=np.uint64)
    sig = ((a[:, None] * shingles_np[None, :] + b[:, None]) % prime).min(axis=1)
    return sig


def lsh_stream(signatures: Iterator[np.ndarray], cfg: ExperimentConfig, chunk_ids: Iterator[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    num_perm = cfg.dedup.near.num_permutations
    band_size = cfg.dedup.near.band_size
    bands = num_perm // band_size
    bucket_maps: List[Dict[Tuple[int, ...], List[int]]] = [defaultdict(list) for _ in range(bands)]
    chunk_list: List[Tuple[int, int]] = []
    for idx, (signature, chunk_id) in enumerate(zip(signatures, chunk_ids)):
        chunk_list.append(chunk_id)
        for band in range(bands):
            start = band * band_size
            end = start + band_size
            key = tuple(signature[start:end].tolist())
            bucket_maps[band][key].append(idx)
    candidates: Set[Tuple[int, int]] = set()
    for band_dict in bucket_maps:
        for bucket in band_dict.values():
            if len(bucket) <= 1:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a_idx = bucket[i]
                    b_idx = bucket[j]
                    chunk_a = chunk_list[a_idx]
                    chunk_b = chunk_list[b_idx]
                    if chunk_a == chunk_b:
                        continue
                    candidates.add(tuple(sorted([chunk_a, chunk_b])))
    return candidates


def near_dedup(shingle_path: str | Path, cfg: ExperimentConfig) -> Tuple[Set[Tuple[int, int]], DedupStats]:
    entries_iter = load_shingles(shingle_path)
    num_perm = cfg.dedup.near.num_permutations
    rng = np.random.default_rng(seed=cfg.seed)
    a = rng.integers(1, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    prime = (1 << 61) - 1

    chunk_ids: List[Tuple[int, int]] = []
    shingle_sets: List[Set[int]] = []
    signatures: List[np.ndarray] = []
    for entry in entries_iter:
        chunk_id = (entry.shard_id, entry.local_index)
        shingle_set = set(entry.shingles)
        chunk_ids.append(chunk_id)
        shingle_sets.append(shingle_set)
        signatures.append(minhash_signature(shingle_set, a, b, prime))

    if not signatures:
        return set(), DedupStats()

    candidates = lsh_stream(iter(signatures), cfg, iter(chunk_ids))
    drop_ids: Set[Tuple[int, int]] = set()
    stats = DedupStats()
    index_map = {cid: i for i, cid in enumerate(chunk_ids)}
    for (chunk_a, chunk_b) in candidates:
        idx_a = index_map[chunk_a]
        idx_b = index_map[chunk_b]
        set_a = shingle_sets[idx_a]
        set_b = shingle_sets[idx_b]
        if not set_a or not set_b:
            continue
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jac = intersection / union if union else 0.0
        stats.near_candidates += 1
        if jac >= cfg.dedup.near.threshold:
            stats.near_duplicates += 1
            drop_ids.add(chunk_b)
    return drop_ids, stats


def write_drop_ids(drop_ids: Set[Tuple[int, int]], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for shard_id, local_index in sorted(drop_ids):
            handle.write(json.dumps({"shard_id": shard_id, "local_index": local_index}) + "\n")
