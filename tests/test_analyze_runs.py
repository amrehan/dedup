import json
from pathlib import Path

from tools.analyze_runs import analyze, data_equivalent_factor


def test_analyze_reads_summary(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.tsv").write_text("run\tval_perplexity\nfoo\t10\nfoo\t8\n", encoding="utf-8")
    stats = analyze([run_dir])
    assert "run" in stats
    assert abs(stats["run"]["val_ppl_mean"] - 9.0) < 1e-6


def test_data_equivalent_factor_simple():
    baseline = [(3e9, 50.0), (6e9, 40.0)]
    dedup = [(3e9, 48.0), (6e9, 38.0)]
    factor = data_equivalent_factor(baseline, dedup, target_metric=40.0)
    assert factor > 1.0
