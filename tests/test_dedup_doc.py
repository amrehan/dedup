from tools.dedup_doc import dedup_documents


def test_dedup_documents_simple():
    docs = [("1", "hello"), ("2", "hello"), ("3", "world")]
    kept, dropped = dedup_documents(docs)
    kept_ids = [entry.id for entry in kept]
    dropped_ids = [entry.id for entry in dropped]
    assert kept_ids == ["doc:1", "doc:3"]
    assert dropped_ids == ["doc:2"]
