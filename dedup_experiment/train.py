from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .config import ExperimentConfig, TrainingConfig
from .data import ChunkRecord
from .models import SimpleGPT, create_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistoryEntry:
    step: int
    loss: float
    lr: float
    tokens_processed: int
    elapsed: float


@dataclass
class TrainingOutputs:
    model: SimpleGPT
    history: List[TrainingHistoryEntry]
    train_tokens: int
    final_loss: float
    checkpoint_path: Optional[Path]


def _select_device(preference: str = "auto") -> torch.device:
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device preference: {preference}")
    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_tensor(chunks: List[ChunkRecord], eos_token_id: Optional[int], limit: Optional[int] = None) -> torch.Tensor:
    tokens: List[int] = []
    for chunk in chunks:
        tokens.extend(chunk.tokens)
        if eos_token_id is not None:
            tokens.append(eos_token_id)
    if not tokens:
        raise ValueError("No tokens available for dataset")
    if limit is not None and limit > 0:
        tokens = tokens[:limit]
    return torch.tensor(tokens, dtype=torch.long)


def _get_batch(tokens: torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= block_size:
        raise ValueError("Not enough tokens to sample a batch")
    max_start = len(tokens) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,), device=device)
    x = torch.stack([tokens[i : i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


def _cosine_lr(step: int, cfg: TrainingConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_language_model(
    run_name: str,
    train_chunks: List[ChunkRecord],
    cfg: ExperimentConfig,
    tokenizer_vocab_size: int,
    eos_token_id: Optional[int],
    output_dir: Path,
) -> TrainingOutputs:
    device = _select_device(cfg.training.device)
    logger.info("Training %s on %s", run_name, device)

    tokens_tensor = _make_tensor(train_chunks, eos_token_id, cfg.training.max_train_tokens).to(device)
    train_tokens = len(tokens_tensor)
    model = create_model(cfg.training, vocab_size=tokenizer_vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=tuple(cfg.training.betas),
        weight_decay=cfg.training.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.training.precision in {"fp16", "bf16"}))

    history: List[TrainingHistoryEntry] = []
    total_tokens_processed = 0
    start_time = time.time()

    autocast_dtype = None
    if device.type == "cuda":
        if cfg.training.precision == "bf16" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        elif cfg.training.precision == "fp16":
            autocast_dtype = torch.float16

    for step in range(cfg.training.max_steps):
        lr = _cosine_lr(step, cfg.training)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype):
            x, y = _get_batch(tokens_tensor, cfg.training.block_size, cfg.training.batch_size, device)
            logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Loss should not be None during training")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.training.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

        total_tokens_processed += cfg.training.batch_size * cfg.training.block_size

        if (step + 1) % cfg.training.log_interval == 0 or step == 0:
            elapsed = time.time() - start_time
            history.append(
                TrainingHistoryEntry(
                    step=step + 1,
                    loss=float(loss.item()),
                    lr=lr,
                    tokens_processed=total_tokens_processed,
                    elapsed=elapsed,
                )
            )
            logger.info(
                "%s step %d | loss %.4f | lr %.3e | tokens %d | tok/s %.1f",
                run_name,
                step + 1,
                loss.item(),
                lr,
                total_tokens_processed,
                total_tokens_processed / max(1.0, elapsed),
            )

    checkpoint_path: Optional[Path] = None
    if cfg.training.save_checkpoints:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"{run_name}_final.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)

    final_loss = history[-1].loss if history else float("nan")

    return TrainingOutputs(
        model=model,
        history=history,
        train_tokens=train_tokens,
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
    )
