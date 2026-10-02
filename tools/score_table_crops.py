"""Score each backend's table crops against the tables PMC publishes.

TEMPORARY: part of the PyMuPDF -> pypdfium2 evaluation on the pdfium-eval
branch; delete together with tools/compare_pdf_backends.py after the port.

Agreement between the backends says nothing about which one is right when
they differ.  validation/tables holds the cell contents of every table of
the validation papers, taken from the PMC XML, so a table crop can be scored
on its own: of the table's cells that are printed on the page, what share
lies inside the crop?

    python tools/score_table_crops.py [--report DIR] [--tables DIR]

Reads the per-PDF results that compare_pdf_backends.py cached in DIR/pages.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("PLOTPICK_DATA", Path.home() / "plotpick_data"))
DEFAULT_REPORT = DATA / "pdf_backend_eval" / "report"
DEFAULT_TABLES = REPO.parent / "validation" / "tables"

BACKENDS = ("pymupdf", "pdfium")
MIN_CELLS = 5      # table cells that must be on the page for a crop to be scored
MARGIN = 0.05      # recall difference that counts as one backend doing better
DASHES = str.maketrans(dict.fromkeys(map(chr, (*range(0x2010, 0x2015), 0x2212)), "-"))


def normalise(text: str) -> str:
    """Text reduced to what survives any layout: no spaces, plain dashes."""
    text = unicodedata.normalize("NFKC", text).translate(DASHES)
    return re.sub(r"\s+", "", text).lower()


def table_cells(tables: list[dict[str, Any]]) -> dict[int, set[str]]:
    """Distinct cell strings of each numbered table, normalised."""
    cells: dict[int, set[str]] = {}
    for table in tables:
        number = re.search(r"\d+", table.get("label", ""))
        if not number:
            continue
        found = {
            normalise(cell)
            for row in [table.get("headers", []), *table.get("rows", [])]
            for cell in row if isinstance(cell, str)
        }
        cells.setdefault(int(number.group()), set()).update(
            c for c in found if len(c) >= 3
        )
    return cells


def score_pdf(res: dict[str, Any], cells: dict[int, set[str]]) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    doc = pdfium.PdfDocument(res["path"])
    for row in res["pages"]:
        tables = {
            b: [e for e in row[b] if e["label"].startswith("Table_")]
            for b in BACKENDS
        }
        if row["rotation"] or not any(tables.values()):
            continue
        page = doc[row["page"] - 1]
        left, _, _, top = page.get_bbox()
        textpage = page.get_textpage()
        whole = normalise(textpage.get_text_bounded())
        for backend in BACKENDS:
            for index, element in enumerate(tables[backend]):
                wanted = cells.get(int(element["label"].split("_")[1]), set())
                on_page = {c for c in wanted if c in whole}
                if len(on_page) < MIN_CELLS:
                    continue
                x0, y0, x1, y1 = element["rect"]
                inside = ""
                if x1 > x0 and y1 > y0:
                    inside = normalise(textpage.get_text_bounded(
                        left + x0, top - y1, left + x1, top - y0,
                    ))
                scores.append({
                    "pdf": res["pdf"], "page": row["page"], "backend": backend,
                    "label": element["label"], "index": index,
                    "cells": len(on_page),
                    "recall": sum(c in inside for c in on_page) / len(on_page),
                    "area": max(x1 - x0, 0) * max(y1 - y0, 0),
                })
        textpage.close()
        page.close()
    doc.close()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score each backend's table crops against PMC's tables."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--list", action="store_true",
                        help="print every crop where the backends differ")
    args = parser.parse_args()

    scores: list[dict[str, Any]] = []
    known = dict.fromkeys(BACKENDS, 0)
    published = 0
    for cached in sorted((args.report / "pages").glob("*.json")):
        res = json.loads(cached.read_text(encoding="utf-8"))
        source = args.tables / f"{Path(res['pdf']).stem}.json"
        if not source.exists() or res["open_error"]:
            continue
        cells = table_cells(json.loads(source.read_text(encoding="utf-8")))
        scores.extend(score_pdf(res, cells))
        # Which of the published tables got a caption detected at all?
        published += len(cells)
        for backend in BACKENDS:
            labels = {e["label"] for row in res["pages"] for e in row[backend]}
            known[backend] += sum(f"Table_{n}" in labels for n in cells)

    print(f"published tables: {published}; caption found: {known}")
    for backend in BACKENDS:
        mine = [s["recall"] for s in scores if s["backend"] == backend]
        if not mine:
            continue
        print(f"{backend}: {len(mine)} table crops scored, "
              f"mean cell recall {sum(mine) / len(mine):.3f}, "
              f"complete (>= 0.95) {sum(r >= 0.95 for r in mine)}, "
              f"mostly missing (< 0.5) {sum(r < 0.5 for r in mine)}")

    by_key: dict[tuple[str, int, str, int], dict[str, dict[str, Any]]] = {}
    for s in scores:
        key = (s["pdf"], s["page"], s["label"], s["index"])
        by_key.setdefault(key, {})[s["backend"]] = s
    pairs = [v for v in by_key.values() if len(v) == 2]
    better = [p for p in pairs
              if p["pdfium"]["recall"] > p["pymupdf"]["recall"] + MARGIN]
    worse = [p for p in pairs
             if p["pdfium"]["recall"] < p["pymupdf"]["recall"] - MARGIN]
    print(f"crops scored for both backends: {len(pairs)}; pdfium holds more of "
          f"the table in {len(better)}, less in {len(worse)}, "
          f"the same in {len(pairs) - len(better) - len(worse)}")
    only = {b: sum(len(v) == 1 and b in v for v in by_key.values())
            for b in BACKENDS}
    print(f"crops only one backend produced: {only}")
    if args.list:
        for tag, group in (("pdfium better", better), ("pdfium worse", worse)):
            for p in group:
                a, b = p["pymupdf"], p["pdfium"]
                growth = b["area"] / max(a["area"], 1)
                print(f"  {tag}: {a['pdf']} p{a['page']} {a['label']} "
                      f"recall {a['recall']:.2f} -> {b['recall']:.2f} "
                      f"({a['cells']} cells), area x{growth:.2f}")


if __name__ == "__main__":
    sys.exit(main())
