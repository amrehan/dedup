from tools.dedup_chunk import chunk_exact, chunk_near


class DummyTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def test_chunk_exact_drops_duplicates():
    tokenizer = DummyTokenizer()
    docs = [("1", "abcdabcd"), ("2", "abcdabcd")]
    kept, dropped = chunk_exact(docs, tokenizer, size=4)
    assert len(kept) == 1  # repeated chunks collapse to one entry
    assert any(entry.id.startswith("chunk:2") for entry in dropped)


def test_chunk_near_merges_similar():
    tokenizer = DummyTokenizer()
    docs = [("1", "a" * 64), ("2", "a" * 60 + "b" * 4)]
    kept, dropped = chunk_near(docs, tokenizer, size=16, threshold=0.8, num_perm=32)
    assert len(kept) < 4  # near-duplicate chunks should be removed
    assert len(dropped) > 0
