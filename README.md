# Dedup Experiment Runner

End-to-end, reproducible experiment that compares language-model training with three dedup recipes:

1. **`no_dedup`** – raw chunks (baseline)
2. **`exact`** – exact hash deduplication
3. **`near`** – exact hash + MinHash/LSH (Jaccard ≥ 0.85)

The project streams a slice of RedPajama-v2 (or any Hugging Face dataset you point it at), chunks it into fixed windows, applies dedup strategies, injects canary strings, trains a compact GPT-style model, and evaluates perplexity, tiny downstream tasks, and memorization leakage.

Everything is orchestrated from a single entrypoint so you can clone → set a token → run one command.

## Quickstart

```bash
# 1. Clone the repo
$ git clone <your-repo-url> dedup-experiment && cd dedup-experiment

# 2. (Optional) create a clean environment
your_env$ python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
your_env$ pip install -r requirements.txt

# 4. Authenticate with Hugging Face (dataset + tokenizer downloads)
your_env$ export HF_TOKEN="<your_hf_token>"

# 5. (Optional) Prefetch the dataset split locally. This ensures chunking runs
#    against a fully-downloaded cache (helpful for large subsets like sample-10B).
your_env$ python tools/download_dataset.py --config configs/default.yaml

# 6. Launch the experiment (uses configs/default.yaml by default)
your_env$ python run_experiment.py
```

Outputs land under `outputs/`:

- `outputs/report.json` – rich JSON dump (config + per-run metrics)
- `outputs/summary.tsv` – compact table for quick comparisons
- Optional checkpoints if `training.save_checkpoints` is enabled

## What the pipeline does

1. **Ingest & chunk** – streams documents (default: RedPajama-v2 sample), normalizes text, tokenizes with GPT-2 BPE, slices into 512-token windows.
2. **Deduplicate** – applies exact hashing and optional MinHash/LSH (configurable shingles, bands, thresholds). Cross-split checks keep val/test clean.
3. **Canary injection** – adds 100 unique strings to the train split for memorization detection.
4. **Training** – runs a 4-layer GPT (~150M params) with cosine LR, AdamW, gradient clipping. Token budget capped by `training.max_train_tokens`.
5. **Evaluation** – reports val perplexity, micro-evals on LAMBADA-open, PIQA, HellaSwag (sampled subsets), and canary recall via greedy decoding.
6. **Reporting** – records dedup statistics (# exact/near drops, cross-split removals), training curves, evaluation metrics, and example canary generations.

## Configuration

All knobs live in [`configs/default.yaml`](configs/default.yaml). Key sections:

- `dataset`: Hugging Face dataset path, subset, split, number of documents to sample, and stream/buffer options. Defaults stream up to 200k docs from the `sample` split and allocate 10% of docs to val/test combined so perplexity has enough tokens to be well-behaved.
- `dataset.prefetch`: toggle to download a split before chunking (`enabled`, optional `split`, `force_download`, `revision`). Works with `tools/download_dataset.py` and is automatically invoked by chunk building when enabled.
- `dedup`: chunk length/stride, normalization, MinHash parameters, and whether to cross-dedup train vs val/test.
- `training`: model depth/width, learning-rate schedule, precision, batch size, max steps, and optional token budget cap. Defaults now run in fp32 with a 5e-5 peak LR and smaller batch size so the demo stays numerically stable.
- `evaluation`: batch sizes, max val tokens, downstream task sample counts, and canary decoding length.
- `runs`: the dedup recipes to execute (baseline/exact/near by default). Add/edit blocks here to try different Jaccard thresholds.

### Experiment contract & layered configs

The full research protocol lives in [`EXPERIMENT.md`](EXPERIMENT.md). For large-scale runs we layer configs under `configs/`:

- `base_gpt2.yaml` captures frozen knobs (optimizer, tokenizer, eval cadence, etc.).
- `configs/overrides/dedup/*.yaml` select doc-level vs chunk-level dedup manifests.
- `configs/overrides/model/*.yaml` swap GPT-2 small/medium hyperparameters.
- `configs/overrides/tokens/*.yaml` set the token budget per learning-curve point.
- `configs/dataset_fields.yaml` documents which metadata columns are used for stratified sampling checks.

Launcher scripts (added in later phases) will compose these YAML fragments so every experiment adheres to the contract without manual editing.

### Helper tooling

- `tools/build_splits.py` – stratify metadata into train/val/test ID lists.
- `tools/dedup_doc.py` / `tools/dedup_chunk.py` – produce doc- and chunk-level manifests (exact/near).
- `tools/cross_dedup.py` – drop val/test examples that collide with a train manifest.
- `tools/compose_config.py` – merge base + override YAMLs; used by `scripts/run_pilot.sh` and `scripts/launch_grid.py`.
- `tools/analyze_runs.py` – aggregate `summary.tsv` outputs and compute simple CIs/data-equivalent factors.

Scripts under `scripts/` show the expected orchestration (`run_pilot.sh` for the quick gate, `launch_grid.py` for the full sweep).

Edit the YAML and rerun `python run_experiment.py --config path/to/your.yaml` to launch variants. Every run is deterministic w.r.t `seed`.

## Hardware & runtime

The default config is tuned for a single GPU (e.g., 24 GB). For CPU-only debugging, shrink:

```yaml
training:
  block_size: 128
  n_layer: 2
  n_head: 2
  n_embd: 128
  batch_size: 4
  max_steps: 50
training:
  max_train_tokens: 50000
```

You can also set `training.device: cpu` for force-CPU smoke tests.

### Scaling up

Larger runs stream chunks to disk and train from them. The workflow is:

```bash
# 0) (Optional but recommended) prefetch the dataset split once
python tools/download_dataset.py --config configs/full_sample10b.yaml --split train --force

# 1) build chunk shards
python tools/build_chunks.py --config configs/full_sample10b.yaml --output chunk_shards

# 2) run dedup to generate a drop list
python tools/run_dedup.py --config configs/full_sample10b.yaml --chunks chunk_shards --output chunk_shards/drop.jsonl

# 3) train using the streamed dataset
python tools/train_from_chunks.py --config configs/full_sample10b.yaml --chunks chunk_shards --drop chunk_shards/drop.jsonl --output outputs/full_run
```

Tip: set `DEDUP_WORKDIR` to aim all relative CLI paths at a mounted volume (e.g.,
Colab Drive, Lambda Workspaces, RunPod). With the environment variable exported,
the commands above can be simplified:

```bash
export DEDUP_WORKDIR=/workspace/dedup
python tools/build_chunks.py --config configs/full_sample10b.yaml --output chunks
python tools/run_dedup.py --config configs/full_sample10b.yaml --chunks chunks --output drop.jsonl
python tools/train_from_chunks.py --config configs/full_sample10b.yaml --chunks chunks --drop drop.jsonl --output outputs/full_run
```

Key differences vs. the smoke test:

- Streams the `sample-10B` subset (≈10B tokens) with 96/2/2 train/val/test split.
- Uses 1,024-token chunking and ablates MinHash thresholds of 0.80 / 0.85 / 0.90.
- Trains an 8-layer GPT (dmodel=512, bf16) for 16k steps at batch size 64 (≈500M tokens).
- Expands evaluation to 1k PIQA/HellaSwag prompts and 500 Lambada samples.

Adjust `max_train_tokens`, `batch_size`, or `max_steps` to fit your compute budget. The chunk/dedup steps are embarrassingly parallel; expect to need multiple A100-class GPUs for the training phase.

## Notes & tips

- **HF auth**: the runner reads `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`). Make sure it has dataset access.
- **HF remote code**: the script automatically sets `HF_DATASETS_TRUST_REMOTE_CODE=1` so dataset builders that rely on custom scripts load without prompting. PIQA is always sourced from the Parquet dump at `ivanpanshin/piqa_qa_formatted`, since the original loader is brittle on Colab.
- **Custom datasets**: point `dataset.name/subset/text_field` at any Hugging Face text set. Streaming keeps memory usage small.
- **Logging**: monitor training in real time via the CLI logs. Each run reports loss, LR, and tokens/sec every `log_interval` steps.
- **Extending**: add more dedup recipes by appending to `runs` or tweak evaluation by editing the `evaluation.downstream_tasks` map.
- **Resume/CI**: the entrypoint is idempotent; wrap it in Modal/RunPod/SageMaker jobs if you want managed execution.

Happy dedupping!
