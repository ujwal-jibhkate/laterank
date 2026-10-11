"""Making renames durable: fsync a folder, so a rename inside it survives a power cut."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_folder(folder: Path) -> None:
    """fsync `folder` itself. POSIX only: Windows can't open folders, and NTFS journals renames."""
    if os.name == "nt":
        return
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
