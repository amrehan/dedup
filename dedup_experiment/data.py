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
from datasets import Dataset, DownloadMode, IterableDataset, load_dataset
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


def prefetch_dataset(dataset_cfg: DatasetConfig, *, split_override: Optional[str] = None, force: Optional[bool] = None) -> None:
    """Ensure the requested dataset split is fully downloaded to the local cache."""

    if not dataset_cfg.prefetch.enabled and split_override is None and force is None:
        return

    split = split_override or dataset_cfg.prefetch.split or dataset_cfg.split
    if not split:
        raise ValueError("Dataset split must be provided for prefetching")

    token = get_hf_token()
    kwargs = {
        "streaming": False,
        "split": split,
        "trust_remote_code": True,
    }
    if token:
        kwargs["token"] = token
    if dataset_cfg.local_cache_dir:
        kwargs["cache_dir"] = dataset_cfg.local_cache_dir
    if dataset_cfg.data_files is not None:
        kwargs["data_files"] = dataset_cfg.data_files
    if dataset_cfg.prefetch.revision:
        kwargs["revision"] = dataset_cfg.prefetch.revision

    force_download = dataset_cfg.prefetch.force_download if force is None else force
    download_mode = DownloadMode.FORCE_REDOWNLOAD if force_download else DownloadMode.REUSE_CACHE_IF_EXISTS

    logger.info(
        "Prefetching %s (subset=%s, split=%s) with download_mode=%s",
        dataset_cfg.name,
        dataset_cfg.subset,
        split,
        download_mode.name,
    )
    try:
        load_dataset(
            dataset_cfg.name,
            dataset_cfg.subset,
            download_mode=download_mode,
            **kwargs,
        )
    except TypeError:
        logger.warning("datasets version does not accept trust_remote_code; retrying without it")
        kwargs.pop("trust_remote_code", None)
        load_dataset(
            dataset_cfg.name,
            dataset_cfg.subset,
            download_mode=download_mode,
            **kwargs,
        )
    logger.info("Prefetch complete for %s split %s", dataset_cfg.name, split)


def load_documents(cfg: DatasetConfig) -> List[str]:
    token = get_hf_token()
    logger.info("Loading %s (subset=%s, split=%s)", cfg.name, cfg.subset, cfg.split)
    kwargs = {}
    if token:
        kwargs["token"] = token
    if cfg.local_cache_dir:
        kwargs["cache_dir"] = cfg.local_cache_dir
    if cfg.data_files is not None:
        kwargs["data_files"] = cfg.data_files
    try:
        dataset = load_dataset(
            cfg.name,
            cfg.subset,
            split=cfg.split,
            streaming=cfg.streaming,
            trust_remote_code=True,
            **kwargs,
        )
    except TypeError:
        logger.warning("datasets version does not accept trust_remote_code; retrying without it")
        dataset = load_dataset(
            cfg.name,
            cfg.subset,
            split=cfg.split,
            streaming=cfg.streaming,
            **kwargs,
        )
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
        text = item.get(cfg.text_field) if cfg.text_field else None
        if not text:
            # Try common fallbacks used by public corpora.
            for candidate in [field for field in (cfg.text_field, "raw_content", "text") if field]:
                candidate_text = item.get(candidate)
                if isinstance(candidate_text, str) and candidate_text.strip():
                    text = candidate_text
                    break
        if not text:
            # Last resort: pick the first non-empty string field if present.
            for value in item.values():
                if isinstance(value, str) and value.strip():
                    text = value
                    break
        if not text:
            continue
        docs.append(text)
        if cfg.max_documents and len(docs) >= cfg.max_documents:
            break
    if not docs:
        hint = (
            "No documents were yielded. Check that you exported HF_TOKEN and have "
            "accepted any gating on togethercomputer/RedPajama-Data-V2, or point "
            "dataset.name at an accessible subset."
        )
        raise RuntimeError(f"Failed to load any documents from dataset. {hint}")
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
