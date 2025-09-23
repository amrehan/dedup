from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import IterableDataset


class ChunkShardDataset(IterableDataset):
    """Iterable dataset that streams chunk shards produced by chunker."""

    def __init__(
        self,
        manifest_path: str | Path,
        block_size: int,
        drop_ids: Optional[set[tuple[int, int]]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if self.manifest_path.is_dir():
            self.manifest_path = self.manifest_path / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.shards: List[dict] = manifest["shards"]
        self.block_size = block_size
        self.drop_ids = drop_ids or set()

    def __iter__(self) -> Iterator[torch.Tensor]:
        for shard_meta in self.shards:
            shard_path = Path(shard_meta["path"])
            if not shard_path.exists():
                continue
            data = torch.load(shard_path)
            token_list: Sequence[torch.Tensor] = data["tokens"]
            for local_idx, token_tensor in enumerate(token_list):
                chunk_id = (shard_meta["shard_id"], local_idx)
                if chunk_id in self.drop_ids:
                    continue
                tokens = token_tensor.long()
                length = tokens.size(0)
                if length < self.block_size:
                    continue
                if length > self.block_size:
                    tokens = tokens[: self.block_size]
                yield tokens


class CyclingDataLoader:
    """Simple wrapper to cycle through a DataLoader indefinitely."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
