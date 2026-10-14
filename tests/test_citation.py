"""Tests for citation.py: the reference, the version and a run's provenance.

The reference is written once, in citation.py; the drift tests check that the
README and CITATION.cff still name the same work.
"""

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from plotpick import __version__, citation

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def fresh_build() -> Iterator[None]:
    """build() is cached; each test starts and ends without a cached value."""
    citation.build.cache_clear()
    yield
    citation.build.cache_clear()


def result(model: str | None, at: str | None) -> dict[str, Any]:
    return {"data": [], "model": model, "extracted_at": at}


def test_reference_names_the_work_and_says_it_is_a_preprint() -> None:
    assert citation.TITLE in citation.REFERENCE
    assert citation.DOI_URL in citation.REFERENCE
    assert "preprint, not peer reviewed" in citation.REFERENCE


def test_bibtex_is_balanced_and_carries_the_identifiers() -> None:
    assert citation.BIBTEX.count("{") == citation.BIBTEX.count("}")
    assert citation.BIBTEX.startswith("@misc{carstensen2026plotpick,")
    for value in (citation.DOI, citation.ARXIV_ID, citation.DOI_URL):
        assert value in citation.BIBTEX


@pytest.mark.parametrize("name", ["README.md", "CITATION.cff"])
def test_readme_and_cff_cite_the_same_work(name: str) -> None:
    text = (REPO / name).read_text(encoding="utf-8")
    assert citation.DOI in text
    assert citation.TITLE in text
    assert citation.SOFTWARE_DOI in text


def test_build_starts_with_the_package_version() -> None:
    assert citation.build().startswith(__version__)


def test_build_without_git_is_the_version(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("git")
    monkeypatch.setattr(citation.subprocess, "run", no_git)
    assert citation.build() == __version__


def test_build_ignores_another_projects_checkout(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def other_repo(args: list[str], **_kwargs: object) -> Any:
        return subprocess.CompletedProcess(args, 0, "/elsewhere\nabc1234\n")
    monkeypatch.setattr(citation.subprocess, "run", other_repo)
    assert citation.build() == __version__


def test_provenance_collects_models_and_dates() -> None:
    about = citation.provenance({
        "b": result("claude-sonnet-5-5", "2026-10-13T09:00:00+00:00"),
        "a": result("claude-haiku-4-5", "2026-10-12T23:59:00+00:00"),
        "c": result("claude-sonnet-5-5", "2026-10-13T10:00:00+00:00"),
    })
    assert about["tool"] == "PlotPick"
    assert about["models"] == ["claude-haiku-4-5", "claude-sonnet-5-5"]
    assert about["extraction_dates_utc"] == ["2026-10-12", "2026-10-13"]
    assert about["doi"] == citation.DOI
    assert about["cite"] == citation.REFERENCE


def test_provenance_of_nothing_says_so() -> None:
    about = citation.provenance({"x": result(None, None)})
    assert about["models"] == []
    assert about["extraction_dates_utc"] == []
    lines = citation.about_lines(about)
    assert "not recorded" in lines[1]
    assert lines[2] == f"Please cite: {citation.REFERENCE}"


def test_methods_sentence_for_one_day() -> None:
    about = citation.provenance(
        {"x": result("claude-sonnet-5-5", "2026-10-13T09:00:00+00:00")})
    sentence = citation.methods_sentence(about)
    assert f"PlotPick version {about['version']}" in sentence
    assert "using claude-sonnet-5-5 on 2026-10-13;" in sentence
    assert citation.IN_TEXT in sentence
    assert citation.SOFTWARE_DOI_URL in sentence


def test_methods_sentence_for_several_days_and_models() -> None:
    about = citation.provenance({
        "a": result("claude-haiku-4-5", "2026-10-01T09:00:00+00:00"),
        "b": result("claude-sonnet-5-5", "2026-10-13T09:00:00+00:00"),
    })
    sentence = citation.methods_sentence(about)
    assert "claude-haiku-4-5 and claude-sonnet-5-5" in sentence
    assert "between 2026-10-01 and 2026-10-13" in sentence


def test_methods_sentence_leaves_the_checking_to_the_user() -> None:
    """It must not claim the values were checked: only the user knows."""
    sentence = citation.methods_sentence(citation.provenance({}))
    assert "[model] [date]" in sentence
    assert "[say who checked every value" in sentence
