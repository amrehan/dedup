from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import prefetch_dataset
from .utils import normalize_text, chunk_tokens, shingle_text


@dataclass
class ShardMetadata:
    shard_id: int
    path: str
    chunk_count: int
    token_count: int


@dataclass
class ChunkMetadata:
    shard_id: int
    local_index: int
    length: int
    exact_hash: int


class ChunkShardWriter:
    def __init__(self, output_dir: Path, shard_size: int = 10000) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._buffer_tokens: List[List[int]] = []
        self._buffer_hashes: List[int] = []
        self._buffer_lengths: List[int] = []
        self._buffer_doc_ids: List[int] = []
        self._buffer_shingles: List[List[int]] = []
        self._shard_metadatas: List[ShardMetadata] = []
        self._total_tokens = 0
        self._shard_id = 0
        self._chunk_meta_path = self.output_dir / "chunks.jsonl"
        self._chunk_meta_file = self._chunk_meta_path.open("w", encoding="utf-8")
        self._shingle_path = self.output_dir / "shingles.jsonl"
        self._shingle_file = self._shingle_path.open("w", encoding="utf-8")

    @staticmethod
    def _hash_text(text: str) -> int:
        digest = hashlib.sha1(text.encode("utf-8")).digest()[:8]
        value = int.from_bytes(digest, "big", signed=False)
        if value >= 2**63:
            value -= 2**64
        return value

    def add_chunk(self, tokens: List[int], doc_id: int, normalized_text: str, shingles: Optional[List[int]] = None) -> None:
        self._buffer_tokens.append(tokens)
        self._buffer_lengths.append(len(tokens))
        self._buffer_doc_ids.append(doc_id)
        self._buffer_hashes.append(self._hash_text(normalized_text))
        self._buffer_shingles.append(shingles or [])
        if len(self._buffer_tokens) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer_tokens:
            return
        shard_path = self.output_dir / f"shard_{self._shard_id:06d}.pt"
        tensor_tokens = [torch.tensor(t, dtype=torch.int32) for t in self._buffer_tokens]
        torch.save(
            {
                "tokens": tensor_tokens,
                "hashes": torch.tensor(self._buffer_hashes, dtype=torch.int64),
            "lengths": torch.tensor(self._buffer_lengths, dtype=torch.int32),
            "doc_ids": torch.tensor(self._buffer_doc_ids, dtype=torch.int64),
        },
        shard_path,
    )
        chunk_count = len(self._buffer_tokens)
        token_count = sum(self._buffer_lengths)
        shard_meta = ShardMetadata(
            shard_id=self._shard_id,
            path=str(shard_path),
            chunk_count=chunk_count,
            token_count=token_count,
        )
        self._shard_metadatas.append(shard_meta)
        for local_idx, (length, h, shingles) in enumerate(
            zip(self._buffer_lengths, self._buffer_hashes, self._buffer_shingles)
        ):
            meta = ChunkMetadata(
                shard_id=self._shard_id,
                local_index=local_idx,
                length=length,
                exact_hash=h,
            )
            self._chunk_meta_file.write(json.dumps(asdict(meta)) + "\n")
            self._shingle_file.write(
                json.dumps({
                    "shard_id": self._shard_id,
                    "local_index": local_idx,
                    "shingles": shingles,
                })
                + "\n"
            )
        self._total_tokens += token_count
        self._buffer_tokens.clear()
        self._buffer_hashes.clear()
        self._buffer_lengths.clear()
        self._buffer_doc_ids.clear()
        self._buffer_shingles.clear()
        self._shard_id += 1

    def finalize(self) -> Dict[str, List[Dict]]:
        self._flush()
        self._chunk_meta_file.flush()
        self._chunk_meta_file.close()
        self._shingle_file.flush()
        self._shingle_file.close()
        manifest = {
            "shards": [asdict(meta) for meta in self._shard_metadatas],
            "chunk_metadata_path": str(self._chunk_meta_path),
            "shingle_path": str(self._shingle_path),
            "total_tokens": self._total_tokens,
        }
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest


def stream_and_chunk(cfg: ExperimentConfig, output_dir: str, shard_size: int = 10000) -> Dict:
    out_dir = Path(output_dir)
    writer = ChunkShardWriter(out_dir, shard_size=shard_size)
    prefetch_dataset(cfg.dataset)
    dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subset,
        split=cfg.dataset.split,
        streaming=cfg.dataset.streaming,
        use_auth_token=os.environ.get("HF_TOKEN"),
    )
    if cfg.dataset.streaming:
        dataset = dataset.shuffle(seed=cfg.dataset.shuffle_seed, buffer_size=cfg.dataset.shuffle_buffer)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    chunk_tokens_count = 0
    document_iter = enumerate(dataset)
    max_docs = cfg.dataset.max_documents
    doc_total = max_docs if (max_docs is not None) else math.inf
    for idx, item in tqdm(document_iter, desc="chunking", unit="doc"):
        if idx >= doc_total:
            break
        text = item.get(cfg.dataset.text_field, "")
        if not text:
            text = item.get("raw_content", "")
        if not text:
            continue
        normalized_doc = normalize_text(text)
        token_ids = tokenizer.encode(normalized_doc)
        shingles_doc = shingle_text(normalized_doc, cfg.dedup.near.shingle_size)
        for chunk in chunk_tokens(token_ids, cfg.dedup.chunk_tokens, cfg.dedup.stride_tokens):
            if len(chunk) < cfg.dedup.min_chunk_tokens:
                continue
            normalized_chunk = normalize_text(tokenizer.decode(chunk))
            shingles = shingle_text(normalized_chunk, cfg.dedup.near.shingle_size)
            writer.add_chunk(chunk, doc_id=idx, normalized_text=normalized_chunk, shingles=shingles)
            chunk_tokens_count += len(chunk)
    manifest = writer.finalize()
    manifest["chunk_tokens"] = chunk_tokens_count
    return manifest
