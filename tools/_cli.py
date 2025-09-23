"""Common CLI helpers for dedup tools."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT_ENV = "DEDUP_WORKDIR"


def resolve_path(path_str: str, *, root_env: str = DEFAULT_ROOT_ENV) -> Path:
    """Resolve a user-provided path for CLI scripts.

    - Expands environment variables and ``~``.
    - If the path is relative and ``root_env`` is set, prefix the path with that
      directory. This makes it easy to aim all outputs at a mounted volume (e.g.
      Lambda ``/workspace`` or Colab ``/content/drive``) without rewriting
      configs.
    - Otherwise, resolve against the current working directory.
    """

    expanded = Path(os.path.expandvars(os.path.expanduser(path_str)))
    if expanded.is_absolute():
        return expanded

    root_value = os.environ.get(root_env)
    if root_value:
        root = Path(os.path.expandvars(os.path.expanduser(root_value)))
        return root / expanded

    return Path.cwd() / expanded

