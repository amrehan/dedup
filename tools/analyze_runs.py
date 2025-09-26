#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import math


def load_summary(path: Path) -> List[Dict[str, str]]:
    lines = path.read_text().strip().splitlines()
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        rows.append(dict(zip(headers, values)))
    return rows


def group_by_run(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["run"], []).append(row)
    return grouped


def mean_and_ci(values: Iterable[float], confidence: float = 0.95) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("nan")
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std_err = math.sqrt(variance) / math.sqrt(len(values))
    # approximate using normal quantile (1.96 for 95%)
    ci = 1.96 * std_err
    return mean, ci


def data_equivalent_factor(baseline: List[Tuple[float, float]], dedup: List[Tuple[float, float]], target_metric: float) -> float:
    """Linear interpolation to find tokens baseline needs to match dedup metric."""
    base_tokens = interpolate_tokens(baseline, target_metric)
    dedup_tokens = interpolate_tokens(dedup, target_metric)
    if base_tokens is None or dedup_tokens is None or dedup_tokens == 0:
        return float("nan")
    return base_tokens / dedup_tokens


def interpolate_tokens(curve: List[Tuple[float, float]], target: float) -> float | None:
    sorted_curve = sorted(curve, key=lambda x: x[0])
    for i in range(len(sorted_curve) - 1):
        (t0, m0), (t1, m1) = sorted_curve[i], sorted_curve[i + 1]
        if (m0 >= target and m1 <= target) or (m0 <= target and m1 >= target):
            if m0 == m1:
                return t0
            ratio = (target - m0) / (m1 - m0)
            return t0 + ratio * (t1 - t0)
    return None


def analyze(run_dirs: List[Path]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.tsv"
        if not summary_path.exists():
            continue
        rows = load_summary(summary_path)
        metrics = [float(row.get("val_perplexity", "nan")) for row in rows]
        mean, ci = mean_and_ci(metrics)
        summary[run_dir.name] = {"val_ppl_mean": mean, "val_ppl_ci95": ci}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate run summaries and compute simple stats")
    parser.add_argument("--runs", nargs="+", help="Run directories containing summary.tsv")
    parser.add_argument("--out", required=True, help="Destination JSON report")
    parser.add_argument("--baseline_curve", nargs="*", help="Pairs tokens:metric for baseline e.g. 3e9:50 6e9:40")
    parser.add_argument("--dedup_curve", nargs="*", help="Pairs tokens:metric for dedup")
    parser.add_argument("--target_metric", type=float, help="Target metric for data-equivalent factor")
    return parser.parse_args()


def parse_curve(pairs: List[str]) -> List[Tuple[float, float]]:
    curve = []
    for pair in pairs:
        tokens_str, metric_str = pair.split(":")
        curve.append((float(tokens_str), float(metric_str)))
    return curve


def main() -> None:
    args = parse_args()
    run_dirs = [Path(p) for p in args.runs]
    stats = analyze(run_dirs)
    if args.target_metric is not None and args.baseline_curve and args.dedup_curve:
        base_curve = parse_curve(args.baseline_curve)
        dedup_curve = parse_curve(args.dedup_curve)
        stats["data_equivalent_factor"] = {
            "target": args.target_metric,
            "value": data_equivalent_factor(base_curve, dedup_curve, args.target_metric),
        }
    Path(args.out).write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
