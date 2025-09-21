from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_tsv(path: Path, rows: List[Dict[str, str]], headers: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(headers) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")
