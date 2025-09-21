from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import List, Tuple

from transformers import AutoTokenizer

from .config import ExperimentConfig
from .data import ChunkRecord
from .utils import normalize_text


@dataclass
class CanaryStore:
    phrases: List[str]
    inserted_chunk_ids: List[str]


def generate_canaries(cfg: ExperimentConfig, seed: int) -> List[str]:
    rng = random.Random(seed)
    canaries: List[str] = []
    alphabet = string.ascii_lowercase + string.digits
    for idx in range(cfg.num_canaries):
        body = "".join(rng.choices(alphabet, k=cfg.canary_length))
        canaries.append(f"{cfg.canary_prefix}-{idx:03d}:{body}")
    return canaries


def insert_canaries(
    train_chunks: List[ChunkRecord],
    canaries: List[str],
    tokenizer: AutoTokenizer,
    cfg: ExperimentConfig,
) -> Tuple[List[ChunkRecord], CanaryStore]:
    augmented = list(train_chunks)
    inserted_ids: List[str] = []
    for idx, phrase in enumerate(canaries):
        tokens = tokenizer.encode(phrase)
        if len(tokens) >= cfg.dedup.chunk_tokens:
            tokens = tokens[: cfg.dedup.chunk_tokens - 1]
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)
        chunk_id = f"canary-{idx:03d}"
        record = ChunkRecord(
            chunk_id=chunk_id,
            doc_id=-1,
            position=idx,
            tokens=tokens,
            normalized_text=normalize_text(
                tokenizer.decode(tokens),
                lowercase=cfg.dedup.lowercase,
                strip_html=False,
                collapse_whitespace=cfg.dedup.collapse_whitespace,
            ),
            length=len(tokens),
        )
        augmented.append(record)
        inserted_ids.append(chunk_id)
    return augmented, CanaryStore(phrases=canaries, inserted_chunk_ids=inserted_ids)
