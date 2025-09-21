from __future__ import annotations

import dataclasses
import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import DedupConfig, ExperimentConfig, NearDedupConfig, RunConfig
from .data import ChunkRecord, SplitChunks

logger = logging.getLogger(__name__)

_MAX_HASH = (1 << 61) - 1
_PRIME = (1 << 61) - 1


@dataclass
class DedupStats:
    exact_duplicates: int = 0
    exact_candidates: int = 0
    near_duplicates: int = 0
    near_candidates: int = 0
    cross_split_exact: int = 0
    cross_split_near: int = 0


@dataclass
class DedupResult:
    kept: List[ChunkRecord]
    dropped: List[ChunkRecord]
    stats: DedupStats


def _hash_text64(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _MAX_HASH


def _shingle(text: str, size: int) -> Set[str]:
    tokens = text.split()
    if not tokens or len(tokens) < size:
        return set(tokens) if tokens else set()
    shingles = set()
    for i in range(len(tokens) - size + 1):
        shingles.add(" ".join(tokens[i : i + size]))
    return shingles


class MinHasher:
    def __init__(self, num_perm: int, seed: int = 0):
        if num_perm % 4 != 0:
            raise ValueError("num_perm should be divisible by 4 for LSH banding")
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, _MAX_HASH, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _MAX_HASH, size=num_perm, dtype=np.uint64)

    def signature(self, shingles: Set[int]) -> np.ndarray:
        if not shingles:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        values = np.array(list(shingles), dtype=np.uint64)
        sig = ((self.a[:, None] * values[None, :] + self.b[:, None]) % _PRIME).min(axis=1)
        return sig.astype(np.uint64)


def _prepare_shingles(chunks: Sequence[ChunkRecord], cfg: NearDedupConfig) -> Tuple[List[Set[int]], List[bool]]:
    shingle_sets: List[Set[int]] = []
    allow_near: List[bool] = []
    for chunk in chunks:
        shingles = _shingle(chunk.normalized_text, cfg.shingle_size)
        hashed = {_hash_text64(s) for s in shingles}
        shingle_sets.append(hashed)
        allow_near.append(chunk.length >= cfg.skip_short_tokens)
    return shingle_sets, allow_near


def _near_duplicates(
    chunks: Sequence[ChunkRecord],
    shingle_sets: Sequence[Set[int]],
    allow_flags: Sequence[bool],
    cfg: NearDedupConfig,
    threshold: float,
    seed: int,
) -> Tuple[Set[int], int, int]:
    if not chunks:
        return set(), 0, 0

    num_perm = cfg.num_permutations
    band_rows = cfg.band_size
    if num_perm % band_rows != 0:
        raise ValueError("num_permutations must be divisible by band_size")

    bands = num_perm // band_rows
    minhasher = MinHasher(num_perm=num_perm, seed=seed)
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    candidate_pairs: Set[Tuple[int, int]] = set()

    signatures: List[np.ndarray] = []
    for idx, (chunk, shingles, allow) in enumerate(zip(chunks, shingle_sets, allow_flags)):
        sig = minhasher.signature(shingles)
        signatures.append(sig)
        if not allow:
            continue
        for band in range(bands):
            start = band * band_rows
            band_key = (band, tuple(sig[start : start + band_rows]))
            bucket = buckets.setdefault(band_key, [])
            for other_idx in bucket:
                if other_idx == idx:
                    continue
                pair = (other_idx, idx) if other_idx < idx else (idx, other_idx)
                candidate_pairs.add(pair)
            bucket.append(idx)

    to_drop: Set[int] = set()
    near_candidate_count = 0
    for i, j in sorted(candidate_pairs):
        if allow_flags[i] is False and allow_flags[j] is False:
            continue
        set_i = shingle_sets[i]
        set_j = shingle_sets[j]
        if not set_i or not set_j:
            continue
        intersection = len(set_i & set_j)
        union = len(set_i | set_j)
        if union == 0:
            continue
        jaccard = intersection / union
        near_candidate_count += 1
        if jaccard >= threshold:
            # Prefer keeping the longer chunk, tie-break by lower index (keeps earlier occurrence).
            keep_idx, drop_idx = (i, j)
            if len(chunks[j].tokens) > len(chunks[i].tokens):
                keep_idx, drop_idx = j, i
            elif len(chunks[j].tokens) == len(chunks[i].tokens) and j < i:
                keep_idx, drop_idx = j, i
            if drop_idx not in to_drop and drop_idx != keep_idx:
                to_drop.add(drop_idx)
    return to_drop, near_candidate_count, len(candidate_pairs)


def run_dedup(chunks: List[ChunkRecord], run: RunConfig, cfg: DedupConfig, seed: int) -> DedupResult:
    stats = DedupStats()
    if not chunks:
        return DedupResult(kept=[], dropped=[], stats=stats)

    kept: List[ChunkRecord] = []
    dropped: List[ChunkRecord] = []

    seen_hashes: Dict[int, str] = {}
    if run.apply_exact:
        for chunk in chunks:
            digest = _hash_text64(chunk.normalized_text)
            if digest in seen_hashes:
                stats.exact_duplicates += 1
                dropped.append(chunk)
            else:
                seen_hashes[digest] = chunk.chunk_id
                kept.append(chunk)
    else:
        kept = list(chunks)

    if run.apply_near:
        threshold = run.near_threshold or cfg.near.threshold
        shingle_sets, allow_flags = _prepare_shingles(kept, cfg.near)
        drop_indices, examined_pairs, candidate_pairs = _near_duplicates(
            kept,
            shingle_sets,
            allow_flags,
            cfg.near,
            threshold=threshold,
            seed=seed,
        )
        stats.near_candidates = candidate_pairs
        stats.near_duplicates = len(drop_indices)
        new_kept: List[ChunkRecord] = []
        for idx, chunk in enumerate(kept):
            if idx in drop_indices:
                dropped.append(chunk)
            else:
                new_kept.append(chunk)
        kept = new_kept
    return DedupResult(kept=kept, dropped=dropped, stats=stats)


def cross_split_dedup(train: List[ChunkRecord], other: List[ChunkRecord], cfg: DedupConfig) -> Tuple[List[ChunkRecord], DedupStats]:
    stats = DedupStats()
    if not train or not other:
        return train, stats
    train_kept: List[ChunkRecord] = []
    val_hashes = {_hash_text64(chunk.normalized_text) for chunk in other}
    drop_indices: Set[int] = set()
    for idx, chunk in enumerate(train):
        digest = _hash_text64(chunk.normalized_text)
        if digest in val_hashes:
            stats.cross_split_exact += 1
            drop_indices.add(idx)
    if cfg.near.enabled and other:
        shingle_sets_other, _ = _prepare_shingles(other, cfg.near)
        shingle_sets_train, allow_flags = _prepare_shingles(train, cfg.near)
        minhasher = MinHasher(num_perm=cfg.near.num_permutations, seed=7)
        band_rows = cfg.near.band_size
        candidate_pairs: Set[int] = set()
        signatures_other = [minhasher.signature(s) for s in shingle_sets_other]
        signatures_train = [minhasher.signature(s) for s in shingle_sets_train]
        bands = cfg.near.num_permutations // band_rows
        buckets: Dict[Tuple[int, Tuple[int, ...]], List[Tuple[str, int]]] = {}
        for other_idx, sig in enumerate(signatures_other):
            for band in range(bands):
                start = band * band_rows
                key = (band, tuple(sig[start : start + band_rows]))
                buckets.setdefault(key, []).append(("other", other_idx))
        for train_idx, sig in enumerate(signatures_train):
            if not allow_flags[train_idx]:
                continue
            for band in range(bands):
                start = band * band_rows
                key = (band, tuple(sig[start : start + band_rows]))
                bucket = buckets.get(key)
                if not bucket:
                    continue
                for source, idx_other in bucket:
                    if source != "other":
                        continue
                    # compute actual similarity
                    set_train = shingle_sets_train[train_idx]
                    set_other = shingle_sets_other[idx_other]
                    if not set_train or not set_other:
                        continue
                    jac = len(set_train & set_other) / len(set_train | set_other)
                    if jac >= cfg.near.threshold:
                        drop_indices.add(train_idx)
                        stats.cross_split_near += 1
                        break
                if train_idx in drop_indices:
                    break
        stats.near_candidates = 0
    for idx, chunk in enumerate(train):
        if idx in drop_indices:
            continue
        train_kept.append(chunk)
    return train_kept, stats


def prepare_run_chunks(
    split: SplitChunks,
    cfg: ExperimentConfig,
    run: RunConfig,
) -> Tuple[SplitChunks, DedupStats]:
    train_result = run_dedup(list(split.train), run, cfg.dedup, seed=cfg.seed)
    val_result = run_dedup(list(split.val), run, cfg.dedup, seed=cfg.seed + 1)
    test_result = run_dedup(list(split.test), run, cfg.dedup, seed=cfg.seed + 2)

    stats = DedupStats(
        exact_duplicates=train_result.stats.exact_duplicates,
        near_duplicates=train_result.stats.near_duplicates,
        near_candidates=train_result.stats.near_candidates,
    )

    if cfg.dedup.cross_split:
        adjusted_train, cs_stats_val = cross_split_dedup(train_result.kept, val_result.kept, cfg.dedup)
        adjusted_train, cs_stats_test = cross_split_dedup(adjusted_train, test_result.kept, cfg.dedup)
        stats.cross_split_exact = cs_stats_val.cross_split_exact + cs_stats_test.cross_split_exact
        stats.cross_split_near = cs_stats_val.cross_split_near + cs_stats_test.cross_split_near
        train_result = DedupResult(kept=adjusted_train, dropped=train_result.dropped, stats=train_result.stats)

    return SplitChunks(train=train_result.kept, val=val_result.kept, test=test_result.kept), stats
