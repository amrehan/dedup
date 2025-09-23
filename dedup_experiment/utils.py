from __future__ import annotations

import html
import random
import re
import unicodedata
from typing import Iterable, Iterator, List


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, lowercase: bool = True, strip_html: bool = True, collapse_whitespace: bool = True) -> str:
    if strip_html:
        text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def chunk_tokens(tokens: List[int], chunk_size: int, stride: int) -> Iterator[List[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    n = len(tokens)
    if n == 0:
        return
    for start in range(0, n, stride):
        chunk = tokens[start : start + chunk_size]
        if not chunk:
            break
        yield chunk


def batched(iterable: Iterable, batch_size: int) -> Iterator[List]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def pick_random_indices(n: int, k: int, rng: random.Random) -> List[int]:
    if k >= n:
        return list(range(n))
    return rng.sample(range(n), k)


def ensure_min_length(chunks: List[List[int]], min_tokens: int) -> List[List[int]]:
    return [chunk for chunk in chunks if len(chunk) >= min_tokens]


def shingle_text(text: str, size: int) -> List[int]:
    if size <= 0:
        return []
    tokens = text.split()
    if len(tokens) < size:
        return []
    shingles = [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]
    return [hash(sh) for sh in shingles]
