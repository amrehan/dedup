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

# 5. Launch the experiment (uses configs/default.yaml by default)
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
- `dedup`: chunk length/stride, normalization, MinHash parameters, and whether to cross-dedup train vs val/test.
- `training`: model depth/width, learning-rate schedule, precision, batch size, max steps, and optional token budget cap. Defaults now run in fp32 with a 5e-5 peak LR and smaller batch size so the demo stays numerically stable.
- `evaluation`: batch sizes, max val tokens, downstream task sample counts, and canary decoding length.
- `runs`: the dedup recipes to execute (baseline/exact/near by default). Add/edit blocks here to try different Jaccard thresholds.

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

## Notes & tips

- **HF auth**: the runner reads `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`). Make sure it has dataset access.
- **HF remote code**: the script automatically sets `HF_DATASETS_TRUST_REMOTE_CODE=1` so dataset builders that rely on custom scripts load without prompting. PIQA also has an explicit fallback that streams the validation JSONL directly if the Hub loader misbehaves on Colab.
- **Custom datasets**: point `dataset.name/subset/text_field` at any Hugging Face text set. Streaming keeps memory usage small.
- **Logging**: monitor training in real time via the CLI logs. Each run reports loss, LR, and tokens/sec every `log_interval` steps.
- **Extending**: add more dedup recipes by appending to `runs` or tweak evaluation by editing the `evaluation.downstream_tasks` map.
- **Resume/CI**: the entrypoint is idempotent; wrap it in Modal/RunPod/SageMaker jobs if you want managed execution.

Happy dedupping!
