#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
PYTHON=${CPU_PYTHON:-$ROOT/.venv/bin/python}
CFG=${1:-$ROOT/configs/full_sample10b.yaml}
WORKDIR=${CPU_WORKDIR:-$ROOT/runs/cpu_pipeline}
CHUNKS_DIR=${CPU_CHUNKS_DIR:-$WORKDIR/chunks}
DROP_PATH=${CPU_DROP_PATH:-$WORKDIR/drop.jsonl}

if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter $PYTHON not found; set CPU_PYTHON to your environment." >&2
  exit 1
fi

mkdir -p "$WORKDIR"

log() { printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }

log "Prefetching dataset (config: $CFG)"
PYTHONPATH=$ROOT "$PYTHON" "$ROOT/tools/download_dataset.py" --config "$CFG"

log "Building chunk shards into $CHUNKS_DIR"
PYTHONPATH=$ROOT "$PYTHON" "$ROOT/tools/build_chunks.py" --config "$CFG" --output "$CHUNKS_DIR"

log "Running deduplication to produce $DROP_PATH"
PYTHONPATH=$ROOT "$PYTHON" "$ROOT/tools/run_dedup.py" --config "$CFG" --chunks "$CHUNKS_DIR" --output "$DROP_PATH"

log "CPU pipeline finished. Outputs:"
log "  Chunks: $CHUNKS_DIR"
log "  Drop list: $DROP_PATH"
