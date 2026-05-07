from __future__ import annotations

from pathlib import Path


def prune_empty_dirs(root: Path) -> int:
    """Remove empty directories below root, deepest first."""
    if not root.exists():
        return 0
    removed = 0
    dirs = sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
    for path in dirs:
        try:
            path.rmdir()
        except OSError:
            continue
        removed += 1
    return removed
