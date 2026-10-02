"""Compare the PyMuPDF and pypdfium2 backends of pdf_figures on real PDFs.

TEMPORARY: part of the PyMuPDF -> pypdfium2 evaluation on the pdfium-eval
branch; delete together with pdf_backend_pymupdf.py after the port.

For every page of every PDF it records, per backend, the figures and tables
found, and for each one found by both: how far the crop rectangles agree and
how far the rendered pixels agree.  It also checks that the refactored
PyMuPDF path still equals pdf_figures.py as it is on main.

    python tools/compare_pdf_backends.py PDF_OR_DIR [...] [--out DIR]

Results are cached per PDF in OUT/pages, so an interrupted run resumes.
OUT/review.html shows every disagreement side by side.
"""

import argparse
import html
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pdf_figures import PdfPage, Rect, find_figures, open_pdf  # noqa: E402

DATA = Path(os.environ.get("PLOTPICK_DATA", Path.home() / "plotpick_data"))
DEFAULT_OUT = DATA / "pdf_backend_eval" / "report"
DEFAULT_PAIRS = REPO.parent / "validation" / "pairs.json"

BACKENDS = ("pymupdf", "pdfium")
# Region Hovedstaden navy for PyMuPDF; the orange of the project's plots
# for pdfium, as the two rectangles are drawn on top of each other.
COLOURS = {"pymupdf": "#002555", "pdfium": "#ff7f00"}
AGREE_IOU = 0.95
RENDER_DPI = 100
# Mean grey-level difference (0-1) of the two renderings, after 4x
# downscaling, above which a crop is listed for review.  Ordinary pages sit
# at 0.02-0.07: the engines anti-alias differently and PDFium draws hairlines
# a full pixel wide.
RENDER_REVIEW = 0.08
# ... and so is a crop where one engine left much less ink than the other.
INK_RATIO = 0.7


def load_original() -> Any | None:
    """pdf_figures.py as committed on main, to check the refactor against."""
    shown = subprocess.run(
        ["git", "-C", str(REPO), "show", "main:pdf_figures.py"],
        capture_output=True, text=True, check=False,
    )
    if shown.returncode != 0 or "page.get_drawings()" not in shown.stdout:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(shown.stdout)
    spec = importlib.util.spec_from_file_location("pdf_figures_main", handle.name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Path(handle.name).unlink()
    return module


def iou(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def degenerate(rect: list[float]) -> bool:
    return rect[2] - rect[0] < 1 or rect[3] - rect[1] < 1


def render_difference(a: Image.Image, b: Image.Image) -> dict[str, float]:
    """How far two renderings of one region differ, and how much ink each has."""
    size = (max(a.width // 4, 1), max(a.height // 4, 1))
    small_a, small_b = (
        im.convert("L").resize(size, Image.Resampling.BOX) for im in (a, b)
    )
    count = size[0] * size[1]
    difference = ImageStat.Stat(ImageChops.difference(small_a, small_b)).mean[0]
    return {
        "diff": difference / 255,
        "ink_pymupdf": sum(small_a.histogram()[:250]) / count,
        "ink_pdfium": sum(small_b.histogram()[:250]) / count,
        "dw": b.width - a.width,
        "dh": b.height - a.height,
    }


def render_suspect(render: dict[str, float]) -> bool:
    """Whether two renderings differ by more than the engines' usual noise."""
    low = min(render["ink_pymupdf"], render["ink_pdfium"])
    high = max(render["ink_pymupdf"], render["ink_pdfium"])
    return render["diff"] > RENDER_REVIEW or low < INK_RATIO * high - 0.01


def detect(page: PdfPage) -> tuple[list[dict[str, Any]], float, str | None]:
    """Run the detector; return its elements, the time taken and any error."""
    start = time.perf_counter()
    try:
        elements = find_figures(page)
        error = None
    except Exception as exc:  # noqa: BLE001 -- record whatever a backend raises
        elements = []
        error = f"{type(exc).__name__}: {exc}"
    return elements, time.perf_counter() - start, error


def analyse_pdf(path: Path, original: Any | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pdf": path.name, "path": str(path), "pages": [], "open_error": {},
        "n_pages": 0,
    }
    docs: dict[str, Any] = {}
    for backend in BACKENDS:
        try:
            docs[backend] = open_pdf(path, backend)
        except Exception as exc:  # noqa: BLE001 -- record whatever a backend raises
            result["open_error"][backend] = f"{type(exc).__name__}: {exc}"
    if len(docs) < 2:
        for doc in docs.values():
            doc.close()
        return result

    raw = pymupdf.open(str(path))
    result["n_pages"] = len(docs["pymupdf"])
    if len(docs["pymupdf"]) != len(docs["pdfium"]):
        result["open_error"]["page_count"] = (
            f"pymupdf {len(docs['pymupdf'])} vs pdfium {len(docs['pdfium'])}"
        )
    for index, (mpage, ppage) in enumerate(
        zip(docs["pymupdf"], docs["pdfium"], strict=False)
    ):
        row = analyse_page(index, mpage, ppage, raw[index], original)
        result["pages"].append(row)
    raw.close()
    for doc in docs.values():
        doc.close()
    return result


def analyse_page(
    index: int, mpage: PdfPage, ppage: PdfPage, raw_page: Any, original: Any | None,
) -> dict[str, Any]:
    ref, t_ref, e_ref = detect(mpage)
    new, t_new, e_new = detect(ppage)
    crop_origin = raw_page.cropbox.tl - raw_page.mediabox.tl
    row: dict[str, Any] = {
        "page": index + 1,
        "rotation": raw_page.rotation,
        "cropbox_shift": abs(crop_origin.x) > 0.01 or abs(crop_origin.y) > 0.01,
        "size_ok": (abs(mpage.width - ppage.width) < 0.5
                    and abs(mpage.height - ppage.height) < 0.5),
        "time": {"pymupdf": t_ref, "pdfium": t_new},
        "error": {"pymupdf": e_ref, "pdfium": e_new},
        "pymupdf": [
            {"label": e["label"], "caption": e["caption"],
             "rect": [round(v, 2) for v in e["crop_rect"]]} for e in ref
        ],
        "pdfium": [
            {"label": e["label"], "caption": e["caption"],
             "rect": [round(v, 2) for v in e["crop_rect"]]} for e in new
        ],
    }
    if original is not None and e_ref is None:
        try:
            before = original.find_figures_on_page(raw_page)
            row["refactor_ok"] = (
                [(e["label"], e["caption"], tuple(e["crop_rect"])) for e in before]
                == [(e["label"], e["caption"], tuple(e["crop_rect"])) for e in ref]
            )
        except Exception as exc:  # noqa: BLE001 -- record whatever main raises
            row["refactor_ok"] = False
            row["error"]["original"] = f"{type(exc).__name__}: {exc}"

    row["labels_equal"] = (
        [e["label"] for e in ref] == [e["label"] for e in new]
    )
    row["matched"] = []
    if row["labels_equal"]:
        for r, n in zip(row["pymupdf"], row["pdfium"], strict=True):
            item: dict[str, Any] = {
                "label": r["label"],
                "iou": round(iou(r["rect"], n["rect"]), 4),
                "edge": round(max(
                    abs(a - b) for a, b in zip(r["rect"], n["rect"], strict=True)
                ), 2),
            }
            # Same rectangle through both renderers isolates the rasteriser.
            if (item["iou"] >= AGREE_IOU and row["rotation"] == 0
                    and not degenerate(r["rect"])):
                try:
                    clip = Rect(*r["rect"])
                    item["render"] = render_difference(
                        mpage.render(clip, RENDER_DPI),
                        ppage.render(clip, RENDER_DPI),
                    )
                except Exception as exc:  # noqa: BLE001 -- record, keep going
                    item["render_error"] = f"{type(exc).__name__}: {exc}"
            row["matched"].append(item)
    return row


# ---------------------------------------------------------------------------
# Review page
# ---------------------------------------------------------------------------

def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    if scale == 1.0:
        return img
    size = (max(int(img.width * scale), 1), max(int(img.height * scale), 1))
    return img.resize(size, Image.Resampling.LANCZOS)


def make_panel(
    path: Path, row: dict[str, Any], label: str | None, target: Path,
) -> None:
    """Page with both backends' rectangles, and (for one label) both crops."""
    thumb_dpi = 60
    scale = thumb_dpi / 72
    tiles: list[tuple[str, Image.Image]] = []
    docs = {b: open_pdf(path, b) for b in BACKENDS}
    iterators = {b: iter(doc) for b, doc in docs.items()}
    pages: dict[str, PdfPage] = {}
    for backend, iterator in iterators.items():
        for _ in range(row["page"]):
            pages[backend] = next(iterator)

    overview = pages["pdfium"].render(None, thumb_dpi)
    draw = ImageDraw.Draw(overview)
    for backend in BACKENDS:
        for element in row[backend]:
            if label is not None and element["label"] != label:
                continue
            x0, y0, x1, y1 = (v * scale for v in element["rect"])
            if x1 > x0 and y1 > y0:
                draw.rectangle((x0, y0, x1, y1), outline=COLOURS[backend], width=3)
            # PyMuPDF's label above its rectangle, pdfium's below, so that
            # two rectangles in the same place keep both labels readable.
            above = backend == "pymupdf"
            draw.text(
                (min(x0, x1), min(y0, y1) - 16 if above else max(y0, y1) + 2),
                f"{element['label']} ({backend})",
                fill=COLOURS[backend], font=_font(13),
            )
    tiles.append(("page (pdfium render)", overview))

    if label is not None:
        for backend in BACKENDS:
            element = next(
                (e for e in row[backend] if e["label"] == label), None
            )
            if element is None or degenerate(element["rect"]):
                continue
            try:
                crop = pages[backend].render(Rect(*element["rect"]), 80)
            except Exception:  # noqa: BLE001 -- a panel may lack a tile
                continue
            tiles.append((f"{backend} crop", crop))

    tiles = [(name, _fit(img, 620, 760)) for name, img in tiles]
    gap, head = 16, 26
    width = sum(img.width for _, img in tiles) + gap * (len(tiles) + 1)
    height = max(img.height for _, img in tiles) + head + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = gap
    for name, img in tiles:
        colour = COLOURS.get(name.split()[0], "#002555")
        draw.text((x, 4), name, fill=colour, font=_font(15))
        canvas.paste(img, (x, head))
        draw.rectangle((x - 1, head - 1, x + img.width, head + img.height),
                       outline="#ccd3dd")
        x += img.width + gap
    canvas.save(target)
    del pages, iterators
    for doc in docs.values():
        doc.close()


def build_review(results: list[dict[str, Any]], out: Path, sample: int) -> None:
    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    sections: dict[str, list[tuple[str, str]]] = {
        "Labels differ": [], "Crop differs (IoU below 0.95)": [],
        "Rendering differs": [], "Rotated pages": [],
        "Sample of agreeing crops": [],
    }
    agreeing: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    def add(section: str, res: dict[str, Any], row: dict[str, Any],
            label: str | None, note: str) -> None:
        name = f"{Path(res['pdf']).stem}_p{row['page']}_{label or 'page'}.png"
        target = panels / name
        if not target.exists():
            try:
                make_panel(Path(res["path"]), row, label, target)
            except Exception as exc:  # noqa: BLE001 -- report, keep going
                note += f" [panel failed: {type(exc).__name__}: {exc}]"
        title = f"{res['pdf']} p.{row['page']} {label or ''} -- {note}"
        sections[section].append((title, f"panels/{name}"))

    for res in results:
        for row in res["pages"]:
            if not (row["pymupdf"] or row["pdfium"]):
                continue
            if row["rotation"]:
                add("Rotated pages", res, row, None,
                    f"/Rotate {row['rotation']}")
                continue
            if not row["labels_equal"]:
                add("Labels differ", res, row, None,
                    "pymupdf " + ", ".join(e["label"] for e in row["pymupdf"])
                    + " | pdfium " + ", ".join(e["label"] for e in row["pdfium"]))
                continue
            for item in row["matched"]:
                if item["iou"] < AGREE_IOU:
                    add("Crop differs (IoU below 0.95)", res, row, item["label"],
                        f"IoU {item['iou']}, largest edge offset {item['edge']} pt")
                elif "render_error" in item:
                    add("Rendering differs", res, row, item["label"],
                        item["render_error"])
                elif "render" in item and render_suspect(item["render"]):
                    add("Rendering differs", res, row, item["label"],
                        "mean grey difference {diff:.3f}, ink pymupdf "
                        "{ink_pymupdf:.3f} vs pdfium {ink_pdfium:.3f}"
                        .format(**item["render"]))
                else:
                    agreeing.append((res, row, item))

    for res, row, item in random.Random(0).sample(
        agreeing, min(sample, len(agreeing))
    ):
        add("Sample of agreeing crops", res, row, item["label"],
            f"IoU {item['iou']}")

    parts = [
        "<!doctype html><meta charset='utf-8'><title>PDF backend review</title>",
        "<style>body{font-family:sans-serif;margin:24px;color:#002555}"
        "h2{border-bottom:2px solid #007dbb;padding-bottom:4px}"
        "figure{margin:0 0 28px}figcaption{font-size:14px;margin-bottom:6px}"
        "img{max-width:100%;border:1px solid #ccd3dd}"
        ".pymupdf{color:#002555;font-weight:bold}"
        ".pdfium{color:#ff7f00;font-weight:bold}</style>",
        "<h1>PyMuPDF vs pypdfium2: crops to review</h1>",
        "<p>Rectangles: <span class='pymupdf'>navy = PyMuPDF</span>, "
        "<span class='pdfium'>orange = pypdfium2</span>.</p>",
    ]
    for section, items in sections.items():
        parts.append(f"<h2>{html.escape(section)} ({len(items)})</h2>")
        for title, src in items:
            parts.append(
                f"<figure><figcaption>{html.escape(title)}</figcaption>"
                f"<img loading='lazy' src='{html.escape(src)}'></figure>"
            )
    (out / "review.html").write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def summarise(results: list[dict[str, Any]], pairs: Path) -> dict[str, Any]:
    pages = [(res, row) for res in results for row in res["pages"]]
    detected = [(res, row) for res, row in pages
                if row["pymupdf"] or row["pdfium"]]
    upright = [(res, row) for res, row in detected if row["rotation"] == 0]
    matched = [item for _, row in upright for item in row["matched"]]
    renders = [item["render"] for item in matched if "render" in item]
    checked = [row for _, row in pages if "refactor_ok" in row]

    summary: dict[str, Any] = {
        "pdfs": len(results),
        "pdfs_not_opened": {
            b: sum(b in res["open_error"] for res in results) for b in BACKENDS
        },
        "page_count_mismatch": sum(
            "page_count" in res["open_error"] for res in results
        ),
        "pages": len(pages),
        "page_size_mismatch": sum(not row["size_ok"] for _, row in pages),
        "pages_rotated": sum(row["rotation"] != 0 for _, row in pages),
        "pages_cropbox_shift": sum(row["cropbox_shift"] for _, row in pages),
        "page_errors": {
            b: sum(row["error"][b] is not None for _, row in pages)
            for b in BACKENDS
        },
        "refactor_check": {
            "pages_checked": len(checked),
            "pages_identical_to_main": sum(row["refactor_ok"] for row in checked),
        },
        "pages_with_detections": len(detected),
        "rotated_pages_with_detections": len(detected) - len(upright),
        "upright_pages_labels_equal": sum(
            row["labels_equal"] for _, row in upright
        ),
        "upright_pages_labels_differ": sum(
            not row["labels_equal"] for _, row in upright
        ),
        "elements": {
            b: sum(len(row[b]) for _, row in detected) for b in BACKENDS
        },
        "degenerate_rects": {
            b: sum(degenerate(e["rect"]) for _, row in detected for e in row[b])
            for b in BACKENDS
        },
        "matched_elements": len(matched),
        "matched_iou_ge_0.99": sum(i["iou"] >= 0.99 for i in matched),
        "matched_iou_ge_0.95": sum(i["iou"] >= AGREE_IOU for i in matched),
        "matched_iou_lt_0.95": sum(i["iou"] < AGREE_IOU for i in matched),
        "matched_edge_le_1pt": sum(i["edge"] <= 1 for i in matched),
        "matched_edge_le_3pt": sum(i["edge"] <= 3 for i in matched),
        "render_errors": sum("render_error" in i for i in matched),
        "seconds": {
            b: round(sum(row["time"][b] for _, row in pages), 1) for b in BACKENDS
        },
        "slowest_page_seconds": {
            b: round(max((row["time"][b] for _, row in pages), default=0), 2)
            for b in BACKENDS
        },
    }
    if renders:
        diffs = [r["diff"] for r in renders]
        summary["render"] = {
            "compared": len(renders),
            "diff_median": round(quantile(diffs, 0.5), 4),
            "diff_p95": round(quantile(diffs, 0.95), 4),
            "diff_max": round(max(diffs), 4),
            "listed_for_review": sum(render_suspect(r) for r in renders),
            "pdfium_much_less_ink": sum(
                r["ink_pdfium"] < INK_RATIO * r["ink_pymupdf"] - 0.01
                for r in renders
            ),
            "pdfium_much_more_ink": sum(
                r["ink_pymupdf"] < INK_RATIO * r["ink_pdfium"] - 0.01
                for r in renders
            ),
            "size_within_2px": sum(
                abs(r["dw"]) <= 2 and abs(r["dh"]) <= 2 for r in renders
            ),
        }
    if pairs.exists():
        wanted = {p["pmcid"]: f"Fig_{p['figure_num']}"
                  for p in json.loads(pairs.read_text(encoding="utf-8"))}
        found = dict.fromkeys(BACKENDS, 0)
        both = total = 0
        for res in results:
            target = wanted.get(Path(res["pdf"]).stem)
            if target is None or res["open_error"]:
                continue
            total += 1
            hit = {
                b: any(e["label"] == target for row in res["pages"] for e in row[b])
                for b in BACKENDS
            }
            for backend in BACKENDS:
                found[backend] += hit[backend]
            both += all(hit.values())
        summary["validation_target_figure_found"] = {
            "papers": total, **found, "both": both,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the PyMuPDF and pypdfium2 backends of pdf_figures."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many PDFs not yet cached")
    parser.add_argument("--sample", type=int, default=40,
                        help="agreeing crops to include in the review page")
    parser.add_argument("--force", action="store_true",
                        help="recompute PDFs that are already cached")
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a threshold of the pdfium backend, "
                             "e.g. INDENT_DIST=inf, to see what it is worth")
    args = parser.parse_args()
    for override in args.set:
        name, value = override.split("=", 1)
        import pdf_backend_pdfium
        if not hasattr(pdf_backend_pdfium, name):
            parser.error(f"pdf_backend_pdfium has no constant {name}")
        setattr(pdf_backend_pdfium, name, float(value))
        print(f"override: {name} = {float(value)}")

    pdfs: list[Path] = []
    for path in args.paths:
        pdfs.extend(sorted(path.glob("*.pdf")) if path.is_dir() else [path])
    cache = args.out / "pages"
    cache.mkdir(parents=True, exist_ok=True)
    original = load_original()
    if original is None:
        print("note: main:pdf_figures.py is not the pre-split version; "
              "the refactor check is skipped")

    results: list[dict[str, Any]] = []
    fresh = 0
    for number, path in enumerate(pdfs, 1):
        cached = cache / f"{path.stem}.json"
        if cached.exists() and not args.force:
            results.append(json.loads(cached.read_text(encoding="utf-8")))
            continue
        if args.limit and fresh >= args.limit:
            continue
        start = time.perf_counter()
        res = analyse_pdf(path, original)
        cached.write_text(json.dumps(res), encoding="utf-8")
        results.append(res)
        fresh += 1
        print(f"[{number}/{len(pdfs)}] {path.name}: {res['n_pages']} pages, "
              f"{time.perf_counter() - start:.1f}s", flush=True)

    summary = summarise(results, args.pairs)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not args.no_review:
        build_review(results, args.out, args.sample)
        print(f"review page: {args.out / 'review.html'}")


if __name__ == "__main__":
    main()
