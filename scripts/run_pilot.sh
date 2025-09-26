#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
TMP_DIR="$ROOT/.tmp"
mkdir -p "$TMP_DIR"

DATA_DIR="$ROOT/data"
TRAIN_JSONL="$DATA_DIR/train.jsonl"
VAL_JSONL="$DATA_DIR/val.jsonl"
TEST_JSONL="$DATA_DIR/test.jsonl"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Missing $TRAIN_JSONL. Generate JSONL corpora before running pilot." >&2
  exit 1
fi

# 1) Doc-level dedup manifest
python "$ROOT/tools/dedup_doc.py" --input "$TRAIN_JSONL" --out "$ROOT/runs/doc_exact"

# 2) Chunk-level exact manifest
python "$ROOT/tools/dedup_chunk.py" --mode exact --chunk 1024 --input "$TRAIN_JSONL" --out "$ROOT/runs/chunk_exact"

# 3) Cross-dedup val/test against each train manifest
python "$ROOT/tools/cross_dedup.py" \
  --manifest "$ROOT/runs/doc_exact/manifest.jsonl" \
  --val "$VAL_JSONL" \
  --test "$TEST_JSONL" \
  --out "$ROOT/runs/doc_exact"

python "$ROOT/tools/cross_dedup.py" \
  --manifest "$ROOT/runs/chunk_exact/manifest.jsonl" \
  --val "$VAL_JSONL" \
  --test "$TEST_JSONL" \
  --out "$ROOT/runs/chunk_exact"

# 4) Compose configs
python "$ROOT/tools/compose_config.py" \
  --base "$ROOT/configs/base_gpt2.yaml" \
  --overrides "$ROOT/configs/overrides/model/gpt2s.yaml" "$ROOT/configs/overrides/dedup/doc_exact.yaml" "$ROOT/configs/overrides/tokens/3b.yaml" \
  --out "$TMP_DIR/pilot_doc_exact.yaml"

python "$ROOT/tools/compose_config.py" \
  --base "$ROOT/configs/base_gpt2.yaml" \
  --overrides "$ROOT/configs/overrides/model/gpt2s.yaml" "$ROOT/configs/overrides/dedup/chunk_exact.yaml" \
  --out "$TMP_DIR/pilot_chunk_exact.yaml"

# 5) Run experiments (override CFG via env if desired)
DOC_CFG=${DOC_CFG:-$TMP_DIR/pilot_doc_exact.yaml}
CHUNK_CFG=${CHUNK_CFG:-$TMP_DIR/pilot_chunk_exact.yaml}

python "$ROOT/run_experiment.py" --config "$DOC_CFG"
python "$ROOT/run_experiment.py" --config "$CHUNK_CFG"

# 6) Aggregate results
python "$ROOT/tools/analyze_runs.py" \
  --runs "$ROOT/runs/doc_exact" "$ROOT/runs/chunk_exact" \
  --out "$ROOT/runs/pilot_report.json"

echo "Pilot complete. Report: $ROOT/runs/pilot_report.json"
