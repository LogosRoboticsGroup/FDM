"""Publish complete JSON files for readers on shared training storage."""

import json
import os
import tempfile
from pathlib import Path


def atomic_dump_json(data, path):
    """Replace a JSON file only after a unique sibling temporary file is closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary_path = stream.name
            json.dump(data, stream, indent=2)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
