# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Nothing yet.

### Changed
- Nothing yet.

### Fixed
- Nothing yet.

## [0.4.0] - 2025-09-26
### Added
- Streaming chunk builder (`dedup_experiment/chunker.py`) and CLI helpers (`tools/build_chunks.py`, `tools/train_from_chunks.py`) to write chunk shards to disk and train via an iterable dataset.
- `ChunkShardDataset` iterable loader and optional streaming support in the trainer so we can iterate over shard files without materialising entire corpora.
- Disk-backed dedup pipeline (`dedup_experiment/dedup_stream.py`) and `tools/run_dedup.py` for generating drop lists from chunk metadata.
- CLI path resolver that honours a `DEDUP_WORKDIR` environment variable so tooling paths relax to Colab Drive, Lambda Workspaces, RunPod mounts, etc.
- Dataset prefetch support: `DatasetPrefetchConfig` and `prefetch_dataset()` with optional forced download, invoked automatically by the chunk builder; plus `tools/download_dataset.py` helper.
- Layered config scaffold: `configs/base_gpt2.yaml`, `configs/dataset_fields.yaml`, overrides under `configs/overrides/{dedup,model,tokens}`, and smoke configs (`smoke_lorem.yaml`, `smoke_wikitext.yaml`, `test_stream_{2k,20k}.yaml`).
- Orchestration and analysis utilities in `tools/`: `analyze_runs.py`, `build_splits.py`, `dedup_doc.py`, `dedup_chunk.py`, `cross_dedup.py`, `compose_config.py`, `build_chunks_mp.py`, `chunk_timing.py`, `profile_run_dedup.py`.
- Launcher scripts under `scripts/`: `run_pilot.sh`, `launch_grid.py`, `cpu_pipeline.sh`.
- Smoke tests covering config composition, dedup manifests, analysis helpers, and shared utils.

### Changed
- Default `streaming: false` in `configs/test_stream.yaml` to make local smoke runs deterministic against the cache.
- README documentation expanded to describe the experiment contract, layered configs, helper tooling, and dataset prefetch workflow.

### Fixed
- Ensure streaming chunker records shingles per chunk and flushes metadata correctly so disk-backed dedup no longer re-materialises documents.
- Stream MinHash signatures during near-dedup to cut peak memory when generating drop lists.

## [0.3.1] - 2025-09-22
### Added
- `configs/full_sample10b.yaml` for scaling experiments on the RedPajama sample-10B split, including larger chunking, multiple MinHash thresholds, and checkpointed 8-layer GPT training. (#0e8a227)
- README instructions for launching and tuning the full run. (#0e8a227)

## [0.3.0] - 2025-09-22
### Added
- Automatically opt into Hugging Face remote-code execution inside `run_experiment.py` so datasets that ship custom loaders work without prompts. (#08847a6)
- Explicit PIQA fallback transformer that parses the hosted Parquet dump, allowing evaluation to proceed even when the upstream script is unavailable. (#745a4bc, #d2b6315, #7b09def)

### Changed
- Tuned the smoke-test training recipe (lower learning rate, smaller batch, longer warm-up) to prevent `nan` spikes on the 200k-token run. (#745a4bc)
- Updated README guidance to reflect the new dataset cap and PIQA fallback source. (#745a4bc, #d2b6315, #7b09def)

### Fixed
- Ensured PIQA no longer attempts to download non-existent JSONL files by sourcing the data directly from `ivanpanshin/piqa_qa_formatted`. (#4cc46f5, #d2b6315, #7b09def)
- Added missing `pyopenssl` requirement to avoid TLS-related import errors when Colab installs dependencies fresh. (#6770947)

## [0.2.0] - 2025-09-21
### Added
- Broadened the default dataset slice (200k docs, wider val/test split) to make perplexity stable while keeping runs manageable. (#1c8854e, #36c2c87)
- Dedicated fallback and graceful skipping for downstream evaluations when datasets cannot be fetched in sandboxed environments. (#1b3a894, #c6bd45c, #d643787)

### Changed
- Migrated the training loop to the `torch.amp` APIs and added fp32 defaults for improved compatibility. (#bf3dd8a, #84f98ec)
- Switched GEMU activations to functional implementations to match broader torch builds. (#80b6f6a)

### Fixed
- Pinned `datasets<3` to prevent breaking changes in script execution, and auto-approved remote dataset code when available. (#5f93807, #c4f2366)
- Corrected dataset field fallback logic so raw content is preserved when pre-normalised text fields are missing. (#ebe56da)
- Pointed LAMBADA evaluation at the actively maintained `EleutherAI/lambada_openai` dataset. (#54648bf)

## [0.1.0] - 2025-09-20
### Added
- Initial end-to-end experiment scaffold: utilities, dataset loader, dedup pipeline, GPT model, trainer, evaluation suite, reporting utilities, and CLI entrypoint. (#a62ebcc through #f2739af)
- Baseline configuration, dependency list, and project documentation to support smoke tests out of the box. (#647907f, #0126eae, #eb55e7f)

### Fixed
- Reverted the early attempt to force remote-code execution while the pinning strategy was finalised. (#f0dd4a9)
