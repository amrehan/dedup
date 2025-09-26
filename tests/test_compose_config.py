from pathlib import Path

from tools.compose_config import merge


def test_merge_nested_dicts(tmp_path: Path):
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"y": 3, "z": 4}, "c": 5}
    merged = merge(base, override)
    assert merged["b"]["y"] == 3
    assert merged["b"]["z"] == 4
    assert merged["c"] == 5
