#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dedup_experiment.config import load_config
from dedup_experiment.dedup_stream import minhash_signature


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    k = max(0, min(len(values) - 1, int(math.ceil(pct / 100.0 * len(values)) - 1)))
    return sorted(values)[k]


def generate_synthetic_chunks(root: Path, num_chunks: int = 5000, *, seed: int = 13) -> Dict[str, str]:
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "chunks.jsonl"
    shingles_path = root / "shingles.jsonl"
    hashes: List[int] = []
    base_shingles: List[List[int]] = []
    for _ in range(max(1, num_chunks // 100)):
        base = {rng.getrandbits(32) for _ in range(80)}
        base_shingles.append(list(base))
    with meta_path.open("w", encoding="utf-8") as meta_file, shingles_path.open("w", encoding="utf-8") as shingle_file:
        for idx in range(num_chunks):
            shard_id = idx // 500
            local_index = idx % 500
            if hashes and rng.random() < 0.15:
                exact_hash = rng.choice(hashes)
            else:
                exact_hash = rng.getrandbits(61)
                hashes.append(exact_hash)
            base = rng.choice(base_shingles)
            shingles = set(base)
            noise = {rng.getrandbits(32) for _ in range(10)}
            if rng.random() < 0.4:
                shingles.update(noise)
            if rng.random() < 0.35 and len(shingles) > 20:
                remove_count = rng.randint(1, 10)
                for _ in range(remove_count):
                    shingles.pop()
            shingles_list = list(shingles)
            meta = {"shard_id": shard_id, "local_index": local_index, "exact_hash": exact_hash}
            shingle_entry = {"shard_id": shard_id, "local_index": local_index, "shingles": shingles_list}
            meta_file.write(json.dumps(meta) + "\n")
            shingle_file.write(json.dumps(shingle_entry) + "\n")
    manifest = {
        "chunk_metadata_path": str(meta_path),
        "shingle_path": str(shingles_path),
        "shards": [],
        "total_tokens": 0,
    }
    manifest_path = root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    return {"manifest": str(manifest_path), "chunk_metadata": str(meta_path), "shingles": str(shingles_path)}


def profile_exact(metadata_path: Path) -> Dict[str, float]:
    parse_times: List[float] = []
    hash_to_chunks: Dict[int, List[Tuple[int, int]]] = {}
    read_start = time.perf_counter()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            t0 = time.perf_counter()
            meta = json.loads(line)
            t1 = time.perf_counter()
            parse_times.append(t1 - t0)
            key = meta["exact_hash"]
            hash_to_chunks.setdefault(key, []).append((meta["shard_id"], meta["local_index"]))
    read_time = time.perf_counter() - read_start
    dedup_start = time.perf_counter()
    duplicates = 0
    for chunk_list in hash_to_chunks.values():
        if len(chunk_list) > 1:
            duplicates += len(chunk_list) - 1
    dedup_time = time.perf_counter() - dedup_start
    return {
        "read_time_s": read_time,
        "dedup_time_s": dedup_time,
        "total_chunks": sum(len(v) for v in hash_to_chunks.values()),
        "exact_duplicates": duplicates,
        "parse_p90_ms": percentile(parse_times, 90) * 1000.0,
        "parse_mean_ms": (statistics.mean(parse_times) * 1000.0) if parse_times else 0.0,
    }


def profile_lsh(signatures: Sequence[np.ndarray], chunk_ids: Sequence[Tuple[int, int]], num_perm: int, band_size: int) -> Tuple[set[Tuple[int, int]], Dict[str, float]]:
    bands = num_perm // band_size
    bucket_maps: List[Dict[Tuple[int, ...], List[int]]] = [dict() for _ in range(bands)]
    bucket_build_times: List[float] = []
    build_start = time.perf_counter()
    for idx, signature in enumerate(signatures):
        for band in range(bands):
            start = band * band_size
            key = tuple(signature[start : start + band_size])
            t0 = time.perf_counter()
            bucket = bucket_maps[band].setdefault(key, [])
            bucket.append(idx)
            bucket_build_times.append(time.perf_counter() - t0)
    build_time = time.perf_counter() - build_start
    candidate_pairs: set[Tuple[int, int]] = set()
    pair_times: List[float] = []
    collect_start = time.perf_counter()
    for band_map in bucket_maps:
        for bucket in band_map.values():
            if len(bucket) <= 1:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    t0 = time.perf_counter()
                    a_idx = bucket[i]
                    b_idx = bucket[j]
                    pair = tuple(sorted((chunk_ids[a_idx], chunk_ids[b_idx])))
                    candidate_pairs.add(pair)
                    pair_times.append(time.perf_counter() - t0)
    collect_time = time.perf_counter() - collect_start
    metrics = {
        "bucket_build_time_s": build_time,
        "bucket_build_p90_ms": percentile(bucket_build_times, 90) * 1000.0,
        "candidate_collect_time_s": collect_time,
        "candidate_collect_p90_ms": percentile(pair_times, 90) * 1000.0,
        "bucket_ops": len(bucket_build_times),
        "candidate_pairs": len(candidate_pairs),
    }
    return candidate_pairs, metrics


def profile_near(shingle_path: Path, cfg) -> Dict[str, float]:
    parse_times: List[float] = []
    signature_times: List[float] = []
    verify_times: List[float] = []
    chunk_ids: List[Tuple[int, int]] = []
    shingle_sets: List[set[int]] = []
    with shingle_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            t0 = time.perf_counter()
            data = json.loads(line)
            t1 = time.perf_counter()
            parse_times.append(t1 - t0)
            chunk_ids.append((data["shard_id"], data["local_index"]))
            shingle_sets.append(set(data.get("shingles", [])))
    rng = np.random.default_rng(seed=cfg.seed)
    num_perm = cfg.dedup.near.num_permutations
    a = rng.integers(1, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    prime = (1 << 61) - 1
    signatures: List[np.ndarray] = []
    for shingle_set in shingle_sets:
        t0 = time.perf_counter()
        sig = minhash_signature(shingle_set, a, b, prime)
        t1 = time.perf_counter()
        signatures.append(sig)
        signature_times.append(t1 - t0)
    candidates, lsh_metrics = profile_lsh(
        signatures,
        chunk_ids,
        num_perm=num_perm,
        band_size=cfg.dedup.near.band_size,
    )
    drop_ids: set[Tuple[int, int]] = set()
    index_map = {cid: idx for idx, cid in enumerate(chunk_ids)}
    for pair in candidates:
        (chunk_a, chunk_b) = pair
        idx_a = index_map[chunk_a]
        idx_b = index_map[chunk_b]
        set_a = shingle_sets[idx_a]
        set_b = shingle_sets[idx_b]
        if not set_a or not set_b:
            continue
        t0 = time.perf_counter()
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = intersection / union if union else 0.0
        if jaccard >= cfg.dedup.near.threshold:
            drop_ids.add(chunk_b)
        verify_times.append(time.perf_counter() - t0)
    metrics = {
        "parse_p90_ms": percentile(parse_times, 90) * 1000.0,
        "parse_mean_ms": statistics.mean(parse_times) * 1000.0 if parse_times else 0.0,
        "signature_p90_ms": percentile(signature_times, 90) * 1000.0,
        "signature_mean_ms": statistics.mean(signature_times) * 1000.0 if signature_times else 0.0,
        "verify_p90_ms": percentile(verify_times, 90) * 1000.0,
        "verify_mean_ms": statistics.mean(verify_times) * 1000.0 if verify_times else 0.0,
        "candidates": len(candidates),
        "drop_ids": len(drop_ids),
    }
    metrics.update(lsh_metrics)
    metrics["parse_ops"] = len(parse_times)
    metrics["signature_ops"] = len(signature_times)
    metrics["verify_ops"] = len(verify_times)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile run_dedup pipeline with synthetic data")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--workdir", default="profiling_smoketest", help="Directory for synthetic data")
    parser.add_argument("--chunks", type=int, default=5000, help="Number of synthetic chunks to generate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    workdir = Path(args.workdir)
    generated = generate_synthetic_chunks(workdir, num_chunks=args.chunks)

    exact_metrics = profile_exact(Path(generated["chunk_metadata"]))
    near_metrics = profile_near(Path(generated["shingles"]), cfg)

    summary = {
        "exact": exact_metrics,
        "near": near_metrics,
        "total_chunks": args.chunks,
        "config": args.config,
    }
    report_path = workdir / "dedup_profile.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
