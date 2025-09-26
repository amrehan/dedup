import math
from pathlib import Path

import pandas as pd

from tools import utils


def test_normalize_and_hash_idempotent():
    text_a = " Hello\nWorld  "
    text_b = "hello world"
    assert utils.normalize_text(text_a) == utils.normalize_text(text_b)
    assert utils.hash_text(utils.normalize_text(text_a)) == utils.hash_text(utils.normalize_text(text_b))
    assert utils.hash_int(utils.normalize_text(text_a)) == utils.hash_int(utils.normalize_text(text_b))


def test_chunk_tokens_stride():
    tokens = list(range(10))
    chunks = list(utils.chunk_tokens(tokens, size=4, stride=2))
    assert chunks[0] == [0, 1, 2, 3]
    assert chunks[1] == [2, 3, 4, 5]


def test_stratified_sample_balances(tmp_path: Path):
    df = pd.DataFrame({
        "id": range(100),
        "url_host": ["a.com"] * 50 + ["b.com"] * 50,
        "language": ["en"] * 60 + ["es"] * 40,
        "quality_bin": [i % 5 for i in range(100)],
        "num_tokens": [100 + i for i in range(100)],
    })
    train_idx, val_idx, test_idx = utils.stratified_sample(df, 0.8, 0.1, 0.1, ["url_host", "language"])
    total = len(train_idx) + len(val_idx) + len(test_idx)
    assert total == len(df)
    assert math.isclose(len(val_idx) / len(df), 0.1, rel_tol=0.3)
    assert math.isclose(len(test_idx) / len(df), 0.1, rel_tol=0.3)
