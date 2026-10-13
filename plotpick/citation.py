"""How to cite PlotPick, and what a review reports about a run of it.

The reference lives here and only here: the app's panels and every export take
it from this module, and tests check that the README and CITATION.cff carry the
same DOI and title. When the JOSS paper replaces the preprint as the work to
cite, this module and CITATION.cff are the two places to change.

A systematic review reports the tools used to collect data (PRISMA 2020 item
9), with their version and the date of use; provenance() gathers those facts
from a session's results, and methods_sentence() writes them as a sentence the
user completes and pastes into a methods section.
"""

import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from plotpick import __version__

TITLE = (
    "PlotPick: AI-powered batch extraction of numerical data from scientific "
    "figures"
)
ARXIV_ID = "2605.06021"
DOI = "10.48550/arXiv.2605.06021"
DOI_URL = f"https://doi.org/{DOI}"
REPO_URL = "https://github.com/tommycarstensen/plotpick"

REFERENCE = (
    f"Carstensen T. {TITLE}. arXiv:{ARXIV_ID} [preprint, not peer reviewed]. "
    f"2026. {DOI_URL}"
)
BIBTEX = (
    "@misc{carstensen2026plotpick,\n"
    "  author = {Carstensen, Tommy},\n"
    "  title = {{PlotPick}: {AI}-powered batch extraction of numerical"
    " data from scientific figures},\n"
    "  year = {2026},\n"
    f"  eprint = {{{ARXIV_ID}}},\n"
    "  archivePrefix = {arXiv},\n"
    f"  doi = {{{DOI}}},\n"
    f"  url = {{{DOI_URL}}},\n"
    "}"
)
IN_TEXT = f"Carstensen, 2026, arXiv:{ARXIV_ID}"


@cache
def build() -> str:
    """The PlotPick version, with the git commit when run from a checkout.

    The hosted app runs from a clone of the repository, so the commit says
    exactly which code read the figures.  An installed copy has no commit.
    """
    root = Path(__file__).resolve().parent.parent
    try:
        lines = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return __version__
    # Only this repository's commit: a copy installed inside some other
    # project's checkout would otherwise report that project's commit.
    if len(lines) == 2 and Path(lines[0]).resolve() == root:
        return f"{__version__} (commit {lines[1]})"
    return __version__


def provenance(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """What a review reports about this run: tool, version, models and dates."""
    models = sorted({str(r["model"]) for r in results.values() if r.get("model")})
    dates = sorted({
        str(r["extracted_at"])[:10]
        for r in results.values() if r.get("extracted_at")
    })
    return {
        "tool": "PlotPick",
        "version": build(),
        "url": REPO_URL,
        "models": models,
        "extraction_dates_utc": dates,
        "cite": REFERENCE,
        "doi": DOI,
    }


def _dates_text(dates: list[str]) -> str:
    if not dates:
        return "[date]"
    if len(dates) == 1:
        return f"on {dates[0]}"
    return f"between {dates[0]} and {dates[-1]}"


def about_lines(about: Mapping[str, Any]) -> list[str]:
    """The provenance as short lines, for export headers and the About sheet."""
    models = ", ".join(about["models"]) or "not recorded"
    dates = ", ".join(about["extraction_dates_utc"]) or "not recorded"
    return [
        f"Extracted with PlotPick {about['version']}, {about['url']}",
        f"Model: {models}; extracted (UTC): {dates}",
        f"Please cite: {about['cite']}",
        "Every value is a model's reading of a figure: check it before use.",
    ]


def methods_sentence(about: Mapping[str, Any]) -> str:
    """A methods sentence with this run's facts and a gap the user fills in."""
    models = " and ".join(about["models"]) or "[model]"
    return (
        f"Numerical data were extracted from figures with PlotPick version "
        f"{about['version']} ({IN_TEXT}), using {models} "
        f"{_dates_text(about['extraction_dates_utc'])}; [say who checked every "
        f"value against its figure, and how disagreements were resolved]."
    )
