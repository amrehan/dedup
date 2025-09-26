import json
from pathlib import Path

from tools.cross_dedup import build_drop_list
from tools.utils import hash_text


def test_build_drop_list_matches_hashes(tmp_path: Path):
    train_hashes = {hash_text("hello"), hash_text("world")}
    corpus = ["hello", "foo", "bar", "world"]
    drops = build_drop_list(train_hashes, corpus)
    dropped_texts = {hash_text("hello"), hash_text("world")}
    assert set(drops) == dropped_texts
