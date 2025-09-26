#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dedup_experiment.config import load_config
from dedup_experiment.chunker import stream_and_chunk
from tools._cli import resolve_path


def _run_worker(
    config_path: str,
    output_dir: str,
    shard_size: int,
    worker_index: int,
    num_workers: int,
    tokenizer_batch: int | None,
) -> str:
    cfg = load_config(config_path)
    manifest = stream_and_chunk(
        cfg,
        output_dir,
        shard_size=shard_size,
        worker_index=worker_index,
        num_workers=num_workers,
        tokenizer_batch_size=tokenizer_batch,
    )
    manifest_path = Path(output_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(manifest_path)


def _merge_worker_outputs(worker_dirs: List[Path], final_dir: Path) -> dict:
    final_dir.mkdir(parents=True, exist_ok=True)
    chunk_meta_path = final_dir / "chunks.jsonl"
    shingle_path = final_dir / "shingles.jsonl"

    manifest_shards: List[dict] = []
    total_tokens = 0
    chunk_tokens_total = 0
    next_shard_id = 0

    with chunk_meta_path.open("w", encoding="utf-8") as chunk_out, shingle_path.open("w", encoding="utf-8") as shingle_out:
        for worker_dir in sorted(worker_dirs):
            manifest_path = worker_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shard_id_map = {}
            for shard in manifest.get("shards", []):
                old_id = shard["shard_id"]
                new_id = next_shard_id
                shard_id_map[old_id] = new_id
                src = Path(shard["path"]).resolve()
                dst = final_dir / f"shard_{new_id:06d}.pt"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dst)
                manifest_shards.append(
                    {
                        "shard_id": new_id,
                        "path": str(dst),
                        "chunk_count": shard["chunk_count"],
                        "token_count": shard["token_count"],
                    }
                )
                next_shard_id += 1
            worker_chunk_path = Path(manifest["chunk_metadata_path"]).resolve()
            worker_shingle_path = Path(manifest["shingle_path"]).resolve()
            with worker_chunk_path.open("r", encoding="utf-8") as src_chunk:
                for line in src_chunk:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    data["shard_id"] = shard_id_map[data["shard_id"]]
                    chunk_out.write(json.dumps(data) + "\n")
            with worker_shingle_path.open("r", encoding="utf-8") as src_shingle:
                for line in src_shingle:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    data["shard_id"] = shard_id_map[data["shard_id"]]
                    shingle_out.write(json.dumps(data) + "\n")
            total_tokens += manifest.get("total_tokens", 0)
            chunk_tokens_total += manifest.get("chunk_tokens", 0)

    final_manifest = {
        "shards": manifest_shards,
        "chunk_metadata_path": str(chunk_meta_path),
        "shingle_path": str(shingle_path),
        "total_tokens": total_tokens,
        "chunk_tokens": chunk_tokens_total,
    }
    manifest_path = final_dir / "manifest.json"
    manifest_path.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    summary_path = final_dir / "build_summary.json"
    summary_path.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    return final_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel chunk builder")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Directory to write combined chunks")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker processes")
    parser.add_argument("--shard_size", type=int, default=10000, help="Chunks per shard file")
    parser.add_argument("--tokenizer_batch", type=int, default=None, help="Docs per tokenizer batch")
    parser.add_argument("--keep_worker_dirs", action="store_true", help="Do not delete worker directories after merge")
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("workers must be >= 1")

    cfg_path = args.config
    final_output = resolve_path(args.output)
    final_output.mkdir(parents=True, exist_ok=True)
    workers_root = (final_output.parent / f".{final_output.name}_workers").resolve()
    workers_root.mkdir(parents=True, exist_ok=True)

    worker_dirs: List[Path] = []
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for worker_idx in range(args.workers):
            worker_dir = workers_root / f"worker_{worker_idx:02d}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            worker_dirs.append(worker_dir)
            futures.append(
                executor.submit(
                    _run_worker,
                    cfg_path,
                    str(worker_dir),
                    args.shard_size,
                    worker_idx,
                    args.workers,
                    args.tokenizer_batch,
                )
            )
        for fut in as_completed(futures):
            fut.result()

    manifest = _merge_worker_outputs(worker_dirs, final_output)

    if not args.keep_worker_dirs:
        shutil.rmtree(workers_root, ignore_errors=True)

    print(json.dumps({"shards": len(manifest["shards"]), "total_tokens": manifest["total_tokens"]}, indent=2))


if __name__ == "__main__":
    main()
