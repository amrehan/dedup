from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from datasets import IterableDataset, Dataset, load_dataset
from transformers import AutoTokenizer

from .config import DatasetConfig, DedupConfig, ExperimentConfig, RunConfig
from .utils import chunk_tokens as iter_token_chunks, normalize_text

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: int
    position: int
    tokens: List[int]
    normalized_text: str
    length: int


@dataclass
class SplitChunks:
    train: List[ChunkRecord]
    val: List[ChunkRecord]
    test: List[ChunkRecord]


def get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_documents(cfg: DatasetConfig) -> List[str]:
    token = get_hf_token()
    logger.info("Loading %s (subset=%s, split=%s)", cfg.name, cfg.subset, cfg.split)
    kwargs = {"use_auth_token": token} if token else {}
    if cfg.local_cache_dir:
        kwargs["cache_dir"] = cfg.local_cache_dir
    dataset = load_dataset(cfg.name, cfg.subset, split=cfg.split, streaming=cfg.streaming, **kwargs)
    if cfg.streaming:
        if not isinstance(dataset, IterableDataset):
            raise ValueError("Expected IterableDataset when streaming=True")
        dataset = dataset.shuffle(seed=cfg.shuffle_seed, buffer_size=cfg.shuffle_buffer)
    else:
        if not isinstance(dataset, Dataset):
            raise ValueError("Expected Dataset when streaming=False")
        dataset = dataset.shuffle(seed=cfg.shuffle_seed)
    docs: List[str] = []
    for item in dataset:
        text = item.get(cfg.text_field)
        if not text:
            continue
        docs.append(text)
        if cfg.max_documents and len(docs) >= cfg.max_documents:
            break
    if not docs:
        raise RuntimeError("Failed to load any documents from dataset")
    logger.info("Loaded %d documents", len(docs))
    return docs


def split_documents(docs: Sequence[str], cfg: DatasetConfig, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    docs_shuffled = [docs[i] for i in indices]
    total = len(docs_shuffled)
    train_count = max(1, int(total * cfg.train_fraction))
    val_count = max(1, int(total * cfg.val_fraction))
    if train_count + val_count >= total:
        train_count = max(1, total - 2)
        val_count = 1
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
    train_docs = docs_shuffled[:train_count]
    val_docs = docs_shuffled[train_count : train_count + val_count]
    test_docs = docs_shuffled[train_count + val_count :]
    return train_docs, val_docs, test_docs


def build_chunks(docs: Sequence[str], tokenizer: AutoTokenizer, cfg: DedupConfig, prefix: str) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []
    for doc_id, raw_text in enumerate(docs):
        normalized_doc = normalize_text(
            raw_text,
            lowercase=cfg.lowercase,
            strip_html=cfg.strip_html,
            collapse_whitespace=cfg.collapse_whitespace,
        )
        token_ids = tokenizer.encode(normalized_doc)
        for idx, tokens_chunk in enumerate(
            iter_token_chunks(token_ids, cfg.chunk_tokens, cfg.stride_tokens)
        ):
            if len(tokens_chunk) < cfg.min_chunk_tokens:
                continue
            normalized_chunk = normalize_text(
                tokenizer.decode(tokens_chunk),
                lowercase=cfg.lowercase,
                strip_html=False,
                collapse_whitespace=cfg.collapse_whitespace,
            )
            chunk_id = f"{prefix}-doc{doc_id:05d}-chunk{idx:04d}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    position=idx,
                    tokens=list(tokens_chunk),
                    normalized_text=normalized_chunk,
                    length=len(tokens_chunk),
                )
            )
    return chunks


def load_and_chunk(cfg: ExperimentConfig):
    docs = load_documents(cfg.dataset)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = cfg.dedup.chunk_tokens
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    train_docs, val_docs, test_docs = split_documents(docs, cfg.dataset, cfg.seed)
    train_chunks = build_chunks(train_docs, tokenizer, cfg.dedup, prefix="train")
    val_chunks = build_chunks(val_docs, tokenizer, cfg.dedup, prefix="val")
    test_chunks = build_chunks(test_docs, tokenizer, cfg.dedup, prefix="test")
    logger.info("Prepared %d train chunks, %d val chunks, %d test chunks", len(train_chunks), len(val_chunks), len(test_chunks))
    return SplitChunks(train=train_chunks, val=val_chunks, test=test_chunks), tokenizer


def chunk_stats(chunks: Sequence[ChunkRecord]) -> Dict[str, float]:
    if not chunks:
        return {"count": 0, "avg_len": 0.0, "tokens": 0}
    total_tokens = sum(chunk.length for chunk in chunks)
    avg_len = total_tokens / len(chunks)
    return {"count": len(chunks), "avg_len": avg_len, "tokens": total_tokens}
