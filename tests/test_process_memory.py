"""Tests for process_memory.py: reading the process size and logging it."""

import logging
import os
import sys

import pytest

from plotpick import process_memory
from plotpick.process_memory import log_memory, resident_mb, return_freed_memory


@pytest.fixture
def logged():
    """The lines the module writes to its own logger."""
    lines: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    handler = Collect()
    process_memory.LOG.addHandler(handler)
    yield lines
    process_memory.LOG.removeHandler(handler)


def test_resident_size_is_the_second_field_of_statm(tmp_path):
    statm = tmp_path / "statm"
    statm.write_text("9000 2000 300 1 0 4000 0\n")
    assert resident_mb(statm) == 2000 * os.sysconf("SC_PAGE_SIZE") / 1e6


@pytest.mark.parametrize("content", [None, "", "9000", "9000 many 300"])
def test_unreadable_size_is_none_not_an_error(tmp_path, content):
    statm = tmp_path / "statm"
    if content is not None:
        statm.write_text(content)
    assert resident_mb(statm) is None


def test_size_is_known_on_linux_where_the_app_is_hosted():
    size = resident_mb()
    if sys.platform == "linux":
        assert size is not None and size > 0
    else:
        assert size is None


def test_returning_freed_memory_never_raises():
    """On Linux this calls glibc; anywhere else it must quietly do nothing."""
    return_freed_memory()


def test_log_line_gives_the_size_before_and_after(monkeypatch, logged):
    sizes = iter([812.4, 431.6])
    monkeypatch.setattr(process_memory, "resident_mb", lambda: next(sizes))
    log_memory("reading 2 upload(s)", "all sessions hold 14 figure(s), 12 MB")
    assert logged == [
        "memory after reading 2 upload(s): 812 MB, 432 MB once freed memory "
        "was returned; all sessions hold 14 figure(s), 12 MB"
    ]


def test_nothing_is_logged_where_the_size_is_unknown(monkeypatch, logged):
    monkeypatch.setattr(process_memory, "resident_mb", lambda: None)
    log_memory("reading 2 upload(s)")
    assert logged == []
