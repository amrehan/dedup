# Data-Efficiency via Chunk-Level Dedup — Experiment Contract

## Goal & Hypothesis
- **Hypothesis:** 1024-token chunk dedup (exact and near) matches or beats doc-level dedup quality with roughly 15 % fewer training tokens.
- **Claim format:** “At matched quality, the doc-level baseline needs ≥ 1.15× the tokens of chunk-level dedup.”

## Metrics
- **Primary:** Validation perplexity on a holdout that has been cross-deduplicated against each training manifest (exact + near).
- **Secondary:** HellaSwag-small, PIQA, planted-canary recall.
- **Audits:** Exact and near duplicate counts dropped train↔val/test.

## Dataset & Splits
- **Source:** RedPajama-v2 (web); target up to ~10 B train tokens.
- **Sampling:** Stratify by host, language, quality decile, and doc length using the field map in `configs/dataset_fields.yaml`.
- **Prefetch:** Download once with `tools/download_dataset.py` (or equivalent) so all runs read from the same local cache; no streaming-time sampling.
- **Doc-level dedup (definition):** Normalize raw document text, compute `xxhash64`, keep the first occurrence before any chunking. Persist kept/dropped IDs and manifest hash.
- **Splits:** Materialize train/val/test lists first; cross-dedup val/test against every train manifest before training. Save manifest hashes for traceability.

## Experimental Factors (only knobs that vary)
1. **Baseline:** Doc-level exact dedup.
2. **Chunk-Exact:** 1024-token non-overlapping windows deduped by exact hash.
3. **Chunk-Near:** 1024-token windows deduped by MinHash/LSH with Jaccard ≥ 0.85.

## Learning-Curve Grid
- **Token budgets:** {3 B, 6 B, 8.5 B, 10 B} tokens per run.
- **Models:** GPT-2 Small (≈124 M) and GPT-2 Medium (≈355 M) from random init.
- **Seeds:** 3 seeds for GPT-2 Small, 2 seeds for GPT-2 Medium, paired across conditions.
- **Total runs:** 3 conditions × 4 budgets × (3 + 2 seeds) = 60 runs.

## Invariants (frozen knobs for fairness)
- Tokenizer: GPT-2 BPE (50257).
- Sequence length: 1024.
- Optimizer: AdamW with shared LR schedule, warmup, weight decay, gradient clip.
- Batch tokens, data ordering, logging cadence, evaluation cadence, and deterministic seeds per run are fixed.
- No early stopping; every run consumes its full token budget.

## Compute & I/O Plan
- **CPU jobs (dedup/signatures):** Hetzner CPX/CCX nodes with ≥32 GB RAM, local NVMe scratch; archive manifests to object storage.
- **GPU jobs (train/eval):** 4090-class or A100-80G; checkpoints/logs pushed to object storage.

## Decision Rules
- **Primary win:** For both model sizes, the chunk-near 0.85 condition at 0.85× tokens has a validation perplexity 95 % CI that overlaps or improves on the doc-baseline at 1.0× tokens (paired seeds).
- **Data-equivalent factor:** Fit simple power-law curves and report `tokens_baseline / tokens_dedup` needed to achieve equal perplexity. Declare success if ≥ 1.15 without secondary-metric regressions.

## Pilot Gate (cheap screening)
- Train GPT-2 Small for 0.5–1.0 B tokens on Baseline@1.0× vs Chunk-Exact@0.85× with identical seed.
- Proceed to the full grid if chunk-exact shows ≥ 0.5–1.0 % perplexity gain; otherwise adjust window size (512/2048) or near-dup threshold (0.80/0.90) and re-run the pilot.

## Logging & Repro
- Track git commit, merged config YAML, train/val/test manifest hashes, seeds, tokens processed, throughput, losses, val metrics, downstream scores, and canary stats.
- Prefer W&B or MLflow; otherwise emit structured JSON plus TSV summaries under `runs/`.

## Risks & Mitigations
- **Small effect size:** Use paired seeds + confidence intervals; involve secondary metrics.
- **Distribution drift after dedup:** Compare host/lang/quality/length histograms pre/post dedup; reweight or adjust sampling if needed.
- **Leakage:** Enforce cross-dedup for val/test; audit duplicate counts each run.

---

A runnable skeleton (configs, scripts, tests) will live alongside this contract so every experiment adheres to the agreement above.
