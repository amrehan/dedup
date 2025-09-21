from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from dedup_experiment.canary import CanaryStore, generate_canaries, insert_canaries
from dedup_experiment.config import ExperimentConfig, RunConfig, load_config
from dedup_experiment.data import SplitChunks, chunk_stats, load_and_chunk
from dedup_experiment.dedup import prepare_run_chunks
from dedup_experiment.evaluate import EvalResults, run_evaluation
from dedup_experiment.report import write_json, write_tsv
from dedup_experiment.train import TrainingOutputs, train_language_model


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
)
logger = logging.getLogger("dedup_experiment")


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_output_dirs(cfg: ExperimentConfig):
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)


def _summarize_evaluations(results: List[Dict]) -> Dict:
    summary = {}
    for result in results:
        run = result["run"]
        summary[run] = result
    return summary


def run_single_experiment(cfg: ExperimentConfig, run: RunConfig, base_split: SplitChunks, tokenizer, canaries: CanaryStore, output_dir: Path):
    run_split, dedup_stats = prepare_run_chunks(base_split, cfg, run)
    logger.info(
        "Run %s: %d train chunks (%d tokens)",
        run.name,
        len(run_split.train),
        sum(chunk.length for chunk in run_split.train),
    )
    train_chunks_with_canaries, run_canaries = insert_canaries(run_split.train, canaries.phrases, tokenizer, cfg)

    training_outputs = train_language_model(
        run.name,
        train_chunks_with_canaries,
        cfg,
        tokenizer_vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        output_dir=output_dir,
    )

    eval_results = run_evaluation(
        training_outputs.model,
        tokenizer,
        cfg,
        val_chunks=run_split.val,
        canaries=run_canaries,
        eos_token_id=tokenizer.eos_token_id,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result_record = {
        "run": run.name,
        "description": run.description or "",
        "train_tokens": training_outputs.train_tokens,
        "train_chunks": len(train_chunks_with_canaries),
        "val_chunks": len(run_split.val),
        "test_chunks": len(run_split.test),
        "val_perplexity": eval_results.val_perplexity,
        "canary_recall": eval_results.canary_recall,
        "downstream": eval_results.downstream,
        "dedup_exact_removed": dedup_stats.exact_duplicates,
        "dedup_near_removed": dedup_stats.near_duplicates,
        "cross_split_exact": dedup_stats.cross_split_exact,
        "cross_split_near": dedup_stats.cross_split_near,
        "training_history": [
            {
                "step": entry.step,
                "loss": entry.loss,
                "lr": entry.lr,
                "tokens_processed": entry.tokens_processed,
                "elapsed": entry.elapsed,
            }
            for entry in training_outputs.history
        ],
        "canary_examples": eval_results.canary_examples,
    }
    return result_record


def main():
    parser = argparse.ArgumentParser(description="Run deduplication experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML file")
    parser.add_argument("--output", default=None, help="Optional override for output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output is not None:
        cfg.output_dir = args.output
        cfg.report_file = str(Path(cfg.output_dir) / "report.json")
        cfg.table_file = str(Path(cfg.output_dir) / "summary.tsv")
    _ensure_output_dirs(cfg)
    _seed_everything(cfg.seed)

    logger.info("Loading dataset and preparing chunks ...")
    split_chunks, tokenizer = load_and_chunk(cfg)
    base_stats = {
        "train": chunk_stats(split_chunks.train),
        "val": chunk_stats(split_chunks.val),
        "test": chunk_stats(split_chunks.test),
    }
    logger.info("Base stats: %s", json.dumps(base_stats, indent=2))

    canary_phrases = generate_canaries(cfg, seed=cfg.seed + 99)
    canary_store = CanaryStore(phrases=canary_phrases, inserted_chunk_ids=[])

    run_results: List[Dict] = []
    for run in cfg.runs:
        logger.info("===== Starting run: %s =====", run.name)
        result = run_single_experiment(cfg, run, split_chunks, tokenizer, canary_store, Path(cfg.output_dir))
        run_results.append(result)

    summary = _summarize_evaluations(run_results)
    write_json(Path(cfg.report_file), {"config": dataclasses.asdict(cfg), "results": summary})

    headers = [
        "run",
        "val_perplexity",
        "canary_recall",
        "train_tokens",
        "dedup_exact_removed",
        "dedup_near_removed",
        "cross_split_exact",
        "cross_split_near",
    ]
    rows = []
    for result in run_results:
        row = {key: result.get(key, "") for key in headers}
        rows.append(row)
    write_tsv(Path(cfg.table_file), rows, headers)

    logger.info("Finished. Summary saved to %s and %s", cfg.report_file, cfg.table_file)


if __name__ == "__main__":
    main()
