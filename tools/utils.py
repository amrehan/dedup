from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence

import xxhash

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text before hashing so doc-level dedup is stable."""

    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip().lower()


def hash_text(text: str) -> str:
    return xxhash.xxh64(text.encode("utf-8")).hexdigest()


def hash_int(text: str) -> int:
    return xxhash.xxh64_intdigest(text.encode("utf-8"))


@dataclass
class ManifestEntry:
    id: str
    hash: str
    kept: bool
    reason: str | None = None
    metadata: dict | None = None

    def to_json(self) -> str:
        data = {"id": self.id, "hash": self.hash, "kept": self.kept}
        if self.reason:
            data["reason"] = self.reason
        if self.metadata:
            data["meta"] = self.metadata
        return json.dumps(data, ensure_ascii=False)


def write_manifest(entries: Iterable[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.to_json() + "\n")


def manifest_hash(path: Path) -> str:
    hasher = xxhash.xxh64()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def chunk_tokens(tokens: Sequence[int], size: int, stride: int) -> Iterator[Sequence[int]]:
    if size <= 0 or stride <= 0:
        raise ValueError("chunk size and stride must be positive")
    n = len(tokens)
    for start in range(0, n, stride):
        chunk = tokens[start : start + size]
        if len(chunk) < size:
            break
        yield chunk


def batched(iterable: Iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def stratified_sample(df, train_frac: float, val_frac: float, test_frac: float, by: Sequence[str]):  # pragma: no cover - wrapper validated in tests
    if pd is None:
        raise RuntimeError("pandas is required for stratified sampling")
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-3):
        raise ValueError("fractions must sum to 1.0")
    grouped = df.groupby(list(by))
    train_indices = []
    val_indices = []
    test_indices = []
    for _, group in grouped:
        n = len(group)
        if n == 0:
            continue
        train_n = max(1, int(round(n * train_frac)))
        val_n = max(1, int(round(n * val_frac)))
        if train_n + val_n >= n:
            train_n = max(1, n - 2)
            val_n = 1
        test_n = n - train_n - val_n
        if test_n <= 0:
            test_n = 1
            if train_n > 1:
                train_n -= 1
        shuffled = group.sample(frac=1.0, random_state=0)
        train_indices.extend(shuffled.iloc[:train_n].index)
        val_indices.extend(shuffled.iloc[train_n : train_n + val_n].index)
        test_indices.extend(shuffled.iloc[train_n + val_n : train_n + val_n + test_n].index)
    return train_indices, val_indices, test_indices
