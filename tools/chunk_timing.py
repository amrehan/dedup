from __future__ import annotations

import atexit
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

_CHUNK_TIMING_ENV = os.getenv("CHUNK_TIMING", "")
_ENABLED = _CHUNK_TIMING_ENV.lower() not in {"", "0", "false", "no"}
_SAMPLE_SIZE_DEFAULT = 200_000


def _parse_int(env_var: str, default: int) -> int:
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(0, parsed)


_SAMPLE_SIZE = _parse_int("CHUNK_TIMING_SAMPLE_SIZE", _SAMPLE_SIZE_DEFAULT)
_OUTPUT_PATH = Path(os.getenv("CHUNK_TIMING_OUTPUT", "outputs/chunk_timing.json"))
_RANDOM = random.Random(0)
_STAT_LOCK = threading.Lock()


@dataclass
class _StageStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    samples: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value
        if _SAMPLE_SIZE == 0:
            return
        if len(self.samples) < _SAMPLE_SIZE:
            self.samples.append(value)
            return
        # Reservoir sampling to keep samples representative.
        idx = _RANDOM.randint(0, self.count - 1)
        if idx < _SAMPLE_SIZE:
            self.samples[idx] = value


_STAGES: Dict[str, _StageStats] = {}


def _record(name: str, value: float) -> None:
    with _STAT_LOCK:
        stats = _STAGES.get(name)
        if stats is None:
            stats = _StageStats()
            _STAGES[name] = stats
        stats.add(value)


class _Timer:
    __slots__ = ("_name", "_start")

    def __init__(self, name: str) -> None:
        self._name = name
        self._start = 0.0

    def __enter__(self) -> None:
        self._start = time.perf_counter()
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        duration = time.perf_counter() - self._start
        _record(self._name, duration)
        return False


class _NullTimer:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return False


_NULL_TIMER = _NullTimer()


def stage(name: str):
    if not _ENABLED:
        return _NULL_TIMER
    return _Timer(name)


def _percentile(sorted_samples: List[float], pct: float) -> float:
    if not sorted_samples:
        return 0.0
    rank = pct / 100.0 * (len(sorted_samples) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_samples[int(rank)]
    weight = rank - lower
    return (1.0 - weight) * sorted_samples[lower] + weight * sorted_samples[upper]


def _emit_report() -> None:
    if not _ENABLED:
        return
    if not _STAGES:
        return
    data = {
        "meta": {
            "enabled": True,
            "sample_size": _SAMPLE_SIZE,
            "env": _CHUNK_TIMING_ENV,
        },
        "stages": {},
    }

    with _STAT_LOCK:
        for name, stats in _STAGES.items():
            if stats.count == 0:
                continue
            mean = stats.total / stats.count
            variance = max(0.0, (stats.total_sq / stats.count) - (mean * mean))
            stddev = math.sqrt(variance)
            samples_sorted = sorted(stats.samples)
            data["stages"][name] = {
                "count": stats.count,
                "mean_ms": mean * 1000.0,
                "stddev_ms": stddev * 1000.0,
                "min_ms": (stats.minimum * 1000.0) if stats.minimum != math.inf else 0.0,
                "max_ms": (stats.maximum * 1000.0) if stats.maximum != -math.inf else 0.0,
                "p50_ms": _percentile(samples_sorted, 50.0) * 1000.0,
                "p90_ms": _percentile(samples_sorted, 90.0) * 1000.0,
                "p95_ms": _percentile(samples_sorted, 95.0) * 1000.0,
                "p99_ms": _percentile(samples_sorted, 99.0) * 1000.0,
                "sample_size": len(samples_sorted),
            }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


atexit.register(_emit_report)

def record(name: str, value: float) -> None:
    if not _ENABLED:
        return
    _record(name, value)


def record_many(name: str, values) -> None:
    if not _ENABLED:
        return
    for value in values:
        _record(name, value)


__all__ = ["stage", "record", "record_many"]
