#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dedup_experiment.config import load_config
from dedup_experiment.dataset import ChunkShardDataset, CyclingDataLoader
from dedup_experiment.train import train_language_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model from chunk shards")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--chunks", required=True, help="Directory containing shard manifest")
    parser.add_argument("--output", required=False, help="Override output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output:
        cfg.output_dir = args.output
    dataset = ChunkShardDataset(args.chunks, block_size=cfg.training.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        drop_last=True,
        collate_fn=lambda batch: torch.stack(batch, dim=0),
    )
    cycling_loader = CyclingDataLoader(dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_language_model(
        run_name="streaming",
        train_chunks=None,
        cfg=cfg,
        tokenizer_vocab_size=len(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
        output_dir=Path(cfg.output_dir),
        cycling_loader=cycling_loader,
        device=device,
    )

if __name__ == "__main__":
    main()
