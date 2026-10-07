"""Write the app's memory use to the log, and hand freed memory back.

Standalone module with no Streamlit dependency.

Streamlit Community Cloud restricts an app that uses too much memory, and its
log says nothing about memory: when that happened on 4 October 2026, the log
held no figure to diagnose it with.  log_memory() writes one after each piece
of heavy work, so that the next log does.

return_freed_memory() asks the C library to give back what Python has freed.
Rendering a PDF page allocates tens of MB that are freed straight away, but
glibc may keep freed memory for reuse instead of returning it, which leaves
the process near the size of its busiest moment.  log_memory() reports the
size before and after, so the log also shows what returning it was worth.

Both need Linux with glibc, which is what Community Cloud runs.  Elsewhere
the size is unknown, nothing is logged and nothing is returned.
"""

import contextlib
import ctypes
import gc
import logging
import os
from pathlib import Path

_STATM = Path("/proc/self/statm")


def _logger() -> logging.Logger:
    logger = logging.getLogger("plotpick")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOG = _logger()


def resident_mb(statm: Path = _STATM) -> float | None:
    """Memory the process holds right now, in MB; None where it cannot be read."""
    try:
        pages = int(statm.read_text().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6


def return_freed_memory() -> None:
    """Collect garbage and ask glibc to return freed memory to the system."""
    gc.collect()
    # Without glibc there is no libc.so.6, or it has no malloc_trim.
    with contextlib.suppress(OSError, AttributeError):
        trim = ctypes.CDLL("libc.so.6").malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim(0)


def log_memory(after: str, detail: str = "") -> None:
    """Return freed memory and log the process size before and after."""
    before = resident_mb()
    return_freed_memory()
    now = resident_mb()
    if before is None or now is None:
        return
    LOG.info(
        "memory after %s: %.0f MB, %.0f MB once freed memory was returned%s",
        after, before, now, f"; {detail}" if detail else "",
    )
