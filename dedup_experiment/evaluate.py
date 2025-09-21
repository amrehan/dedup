from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset

from .canary import CanaryStore
from .config import ExperimentConfig
from .data import ChunkRecord

logger = logging.getLogger(__name__)


@dataclass
class EvalResults:
    val_perplexity: float
    downstream: Dict[str, float]
    canary_recall: float
    canary_examples: List[Tuple[str, str]]


def _prepare_tokens(chunks: List[ChunkRecord], eos_token_id: Optional[int], limit: Optional[int] = None) -> torch.Tensor:
    tokens: List[int] = []
    for chunk in chunks:
        tokens.extend(chunk.tokens)
        if eos_token_id is not None:
            tokens.append(eos_token_id)
    if not tokens:
        raise ValueError("Empty validation tokens")
    if limit is not None and limit > 0:
        tokens = tokens[:limit]
    return torch.tensor(tokens, dtype=torch.long)


def _iter_batches(tensor: torch.Tensor, block_size: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    total = len(tensor) - block_size - 1
    if total <= 0:
        raise ValueError("Not enough tokens for evaluation")
    for start in range(0, total, batch_size):
        batch_inputs = []
        batch_targets = []
        for i in range(start, min(total, start + batch_size)):
            x = tensor[i : i + block_size]
            y = tensor[i + 1 : i + 1 + block_size]
            batch_inputs.append(x)
            batch_targets.append(y)
        yield torch.stack(batch_inputs), torch.stack(batch_targets)


def evaluate_perplexity(model, cfg: ExperimentConfig, val_chunks: List[ChunkRecord], eos_token_id: Optional[int]) -> float:
    device = next(model.parameters()).device
    tokens = _prepare_tokens(val_chunks, eos_token_id, cfg.evaluation.max_val_tokens).to(device)
    block_size = cfg.training.block_size
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for x, y in _iter_batches(tokens, block_size, cfg.evaluation.val_batch_size):
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Loss should not be None during evaluation")
            batch_tokens = y.numel()
            total_loss += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens
    if total_tokens == 0:
        raise ValueError("No tokens evaluated for perplexity")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def _load_dataset(name: str, *args, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    try:
        return load_dataset(name, *args, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return load_dataset(name, *args, **kwargs)


def _token_log_prob(model, tokenizer, context: str, continuation: str, device: torch.device, block_size: int) -> float:
    prompt = context
    if continuation:
        prompt = context + continuation
    ids = tokenizer.encode(prompt)
    if len(ids) < 2:
        return float("-inf")
    if len(ids) > block_size + 1:
        ids = ids[-(block_size + 1) :]
    input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    target_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids)
    log_probs = torch.log_softmax(logits, dim=-1)
    score = 0.0
    for t in range(target_ids.size(1)):
        token_id = target_ids[0, t]
        score += float(log_probs[0, t, token_id].item())
    return score


def evaluate_lambada(model, tokenizer, cfg: ExperimentConfig, max_samples: int) -> float:
    device = next(model.parameters()).device
    dataset = _load_dataset("EleutherAI/lambada_openai", split="test")
    limit = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)
    correct = 0
    evaluated = 0
    for example in dataset:
        if evaluated >= limit:
            break
        text = example["text"].strip()
        if " " not in text:
            continue
        context, target = text.rsplit(" ", 1)
        completion_ids = tokenizer.encode(target)
        if not completion_ids:
            continue
        with torch.no_grad():
            prompt_ids = tokenizer.encode(context + " ")
            if len(prompt_ids) > cfg.training.block_size:
                prompt_ids = prompt_ids[-cfg.training.block_size :]
            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            last_logits = logits[0, -1]
            predicted_token = torch.argmax(last_logits).item()
        evaluated += 1
        if completion_ids[0] == predicted_token:
            correct += 1
    return correct / max(1, evaluated)


def evaluate_piqa(model, tokenizer, cfg: ExperimentConfig, max_samples: int) -> float:
    device = next(model.parameters()).device
    ds = _load_dataset("piqa", split="validation")
    total = min(max_samples, len(ds))
    correct = 0
    for idx in range(total):
        row = ds[idx]
        prompt = row["goal"].strip()
        sol1 = row["sol1"].strip()
        sol2 = row["sol2"].strip()
        score1 = _token_log_prob(model, tokenizer, prompt + " ", " " + sol1, device, cfg.training.block_size)
        score2 = _token_log_prob(model, tokenizer, prompt + " ", " " + sol2, device, cfg.training.block_size)
        pred = 0 if score1 >= score2 else 1
        if pred == row["label"]:
            correct += 1
    return correct / max(1, total)


def evaluate_hellaswag(model, tokenizer, cfg: ExperimentConfig, max_samples: int) -> float:
    device = next(model.parameters()).device
    ds = _load_dataset("hellaswag", split="validation")
    total = min(max_samples, len(ds))
    correct = 0
    for idx in range(total):
        row = ds[idx]
        prompt = (row["ctx_a"] + " " + row["ctx_b"]).strip()
        options = row["endings"]
        best_score = -float("inf")
        pred = 0
        for option_idx, option in enumerate(options):
            score = _token_log_prob(model, tokenizer, prompt + " ", " " + option.strip(), device, cfg.training.block_size)
            if score > best_score:
                best_score = score
                pred = option_idx
        if pred == row["label"]:
            correct += 1
    return correct / max(1, total)


def evaluate_downstream(model, tokenizer, cfg: ExperimentConfig) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    tasks = cfg.evaluation.downstream_tasks
    if "lambada_openai" in tasks:
        metrics["lambada_openai"] = evaluate_lambada(model, tokenizer, cfg, tasks["lambada_openai"])
    if "piqa" in tasks:
        metrics["piqa"] = evaluate_piqa(model, tokenizer, cfg, tasks["piqa"])
    if "hellaswag" in tasks:
        metrics["hellaswag"] = evaluate_hellaswag(model, tokenizer, cfg, tasks["hellaswag"])
    return metrics


def _generate_greedy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    device: torch.device,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    ids = ids[-block_size:]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
            if x.size(1) > block_size:
                x = x[:, -block_size:]
    return tokenizer.decode(x[0].tolist())


def evaluate_canaries(
    model,
    tokenizer,
    cfg: ExperimentConfig,
    canaries: CanaryStore,
) -> Tuple[float, List[Tuple[str, str]]]:
    device = next(model.parameters()).device
    recovered = 0
    examples: List[Tuple[str, str]] = []
    for idx, phrase in enumerate(canaries.phrases):
        prefix = phrase.split(":")[0] + ":"
        generated = _generate_greedy(
            model,
            tokenizer,
            prefix,
            max_new_tokens=cfg.evaluation.canary_decode_max_tokens,
            block_size=cfg.training.block_size,
            device=device,
        )
        if phrase in generated:
            recovered += 1
            if len(examples) < 5:
                examples.append((prefix, generated))
    recall = recovered / max(1, len(canaries.phrases))
    return recall, examples


def run_evaluation(
    model,
    tokenizer,
    cfg: ExperimentConfig,
    val_chunks: List[ChunkRecord],
    canaries: CanaryStore,
    eos_token_id: Optional[int],
) -> EvalResults:
    ppl = evaluate_perplexity(model, cfg, val_chunks, eos_token_id)
    downstream = evaluate_downstream(model, tokenizer, cfg)
    canary_recall, examples = evaluate_canaries(model, tokenizer, cfg, canaries)
    return EvalResults(
        val_perplexity=ppl,
        downstream=downstream,
        canary_recall=canary_recall,
        canary_examples=examples,
    )
