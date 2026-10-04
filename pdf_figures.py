"""Detect and extract figures/tables from PDF pages using caption heuristics.

Standalone module with no Streamlit dependency -- usable from both the app
and from batch scripts.

The detection works on plain page facts (text blocks, image boxes, drawing
boxes), which pdf_backend_pdfium reads from the PDF with pypdfium2.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

CAPTION_RE = re.compile(
    r"^(Supplementary\s+)?Fig(ure|\.)\s*\d"
    r"|^(Supplementary\s+)?Table\s*\d"
    r"|^Suppl\.?\s+Fig",
    re.IGNORECASE,
)
MIN_IMG_DIM = 50
H_MARGIN = 20
V_MARGIN = 6


class PdfError(Exception):
    """A PDF, or one page or region of it, that cannot be read or rendered."""


@dataclass
class Rect:
    """A box in page points, origin top-left, as the page is displayed."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.x0, self.y0, self.x1, self.y1))


class PdfPage(Protocol):
    """What the backend exposes for one page."""

    width: float
    height: float

    def text_blocks(self) -> list[dict[str, Any]]:
        """Paragraph-level blocks, each {"text": str, "bbox": Rect}."""
        ...

    def image_rects(self) -> list[Rect]: ...

    def drawing_rects(self) -> list[Rect]: ...

    def render(self, clip: Rect | None, dpi: int) -> Image.Image:
        """Rasterise the page, or the clip region of it, as an RGB image.

        A region too large to hold at dpi is rendered at a lower resolution.
        Raises PdfError when there is nothing to render: a clip or a page
        that covers no pixel.
        """
        ...


class PdfDocument(Protocol):
    """What the backend exposes for one open PDF."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[PdfPage]:
        """Yield the pages in order.

        A page is only good until the next one is requested: use it inside
        the loop, do not collect pages in a list.
        """
        ...

    def __enter__(self) -> "PdfDocument": ...

    def __exit__(self, *exc: object) -> None: ...

    def close(self) -> None: ...


def open_pdf(source: bytes | str | Path) -> PdfDocument:
    """Open a PDF, given as bytes or a path.

    Raises PdfError for a file that is damaged, password-protected or not a
    PDF; so do the document and its pages for a part they cannot read.
    """
    # Imported here because the backend imports Rect from this module.
    from pdf_backend_pdfium import Document
    return Document(source)


def label_from_caption(text: str) -> str | None:
    m = re.match(
        r"(Supplementary\s+|Suppl\.?\s+)?(Fig(?:ure|\.)?|Table)\s*(\d+)",
        text.strip(), re.IGNORECASE,
    )
    if not m:
        return None
    prefix = "Suppl_" if m.group(1) else ""
    kind = "Fig" if "fig" in m.group(2).lower() else "Table"
    return f"{prefix}{kind}_{m.group(3)}"


def _is_two_column(text_blocks: list[dict], pw: float) -> bool:
    left = right = 0
    for tb in text_blocks:
        w = tb["bbox"].x1 - tb["bbox"].x0
        if w > pw * 0.55 or w < 30 or len(tb["text"]) < 10:
            continue
        mid_x = (tb["bbox"].x0 + tb["bbox"].x1) / 2
        if mid_x < pw * 0.35:
            left += 1
        elif mid_x > pw * 0.65:
            right += 1
    return left >= 3 and right >= 3


def _column_bounds(
    cap_x0: float, cap_x1: float, pw: float, two_col: bool,
) -> tuple[float, float]:
    if not two_col:
        return 0, pw
    mid = (cap_x0 + cap_x1) / 2
    center = pw * 0.5
    if mid < pw * 0.4:
        return 0, center + 4
    if mid > pw * 0.6:
        return center - 4, pw
    return 0, pw


def _padded_rect(
    x0: float, y0: float, x1: float, y1: float, pw: float, ph: float,
) -> Rect:
    return Rect(
        max(0, x0 - H_MARGIN), max(0, y0 - V_MARGIN),
        min(pw, x1 + H_MARGIN), min(ph, y1 + V_MARGIN),
    )


def find_figures(page: PdfPage) -> list[dict[str, Any]]:
    """Detect figures/tables on a PDF page via caption text.

    Returns list of dicts with keys: label, caption, crop_rect.
    """
    pw, ph = page.width, page.height
    text_blocks = page.text_blocks()

    captions: list[dict] = []
    for tb in text_blocks:
        if not CAPTION_RE.match(tb["text"]):
            continue
        label = label_from_caption(tb["text"])
        if not label:
            continue
        captions.append({
            "label": label,
            "type": "table" if "table" in label.lower() else "figure",
            "text": tb["text"][:80],
            "bbox": tb["bbox"],
        })
    captions.sort(key=lambda c: c["bbox"].y0)

    if not captions:
        return []

    img_rects = [
        r for r in page.image_rects()
        if r.x1 - r.x0 > MIN_IMG_DIM and r.y1 - r.y0 > MIN_IMG_DIM
    ]
    drawings = page.drawing_rects()
    two_col = _is_two_column(text_blocks, pw)

    def same_column(a: dict, b: dict) -> bool:
        if not two_col:
            return True
        a_mid = (a["bbox"].x0 + a["bbox"].x1) / 2
        b_mid = (b["bbox"].x0 + b["bbox"].x1) / 2
        return (a_mid < pw * 0.5) == (b_mid < pw * 0.5)

    elements: list[dict] = []
    for i, cap in enumerate(captions):
        prev_y = 0
        for j in range(i - 1, -1, -1):
            if same_column(cap, captions[j]):
                prev_y = captions[j]["bbox"].y1
                break
        next_y = ph
        for j in range(i + 1, len(captions)):
            if same_column(cap, captions[j]):
                next_y = captions[j]["bbox"].y0
                break

        cap_x0, cap_x1 = cap["bbox"].x0, cap["bbox"].x1
        cap_y0, cap_y1 = cap["bbox"].y0, cap["bbox"].y1
        col_left, col_right = _column_bounds(cap_x0, cap_x1, pw, two_col)

        if cap["type"] == "figure":
            associated = [
                ir for ir in img_rects
                if prev_y - 20 <= (ir.y0 + ir.y1) / 2 <= next_y + 20
            ]
            if associated:
                x0 = min(ir.x0 for ir in associated)
                y0 = min(ir.y0 for ir in associated)
                x1 = max(ir.x1 for ir in associated)
                y1 = max(ir.y1 for ir in associated)
            else:
                nearby = [
                    d for d in drawings
                    if (d.y0 >= prev_y - 10
                        and d.y1 <= cap_y1 + 10
                        and d.x0 >= cap_x0 - 60
                        and d.x1 <= cap_x1 + 60)
                ]
                if nearby:
                    x0 = min(d.x0 for d in nearby)
                    y0 = min(d.y0 for d in nearby)
                    x1 = max(d.x1 for d in nearby)
                    y1 = max(d.y1 for d in nearby)
                else:
                    x0, y0 = cap_x0, prev_y
                    x1, y1 = cap_x1, cap_y0

            x0, y0 = min(x0, cap_x0), min(y0, cap_y0)
            x1, y1 = max(x1, cap_x1), max(y1, cap_y1)
            y0 = max(y0, prev_y)

            crop = _padded_rect(x0, y0, x1, y1, pw, ph)
            crop.x0 = max(crop.x0, col_left)
            crop.x1 = min(crop.x1, col_right, x1 + 4)
        else:
            cap_mid_x = (cap_x0 + cap_x1) / 2
            cap_half_w = (cap_x1 - cap_x0) / 2 + 15

            def _scan(
                candidates: list[dict],
                *,
                cap_y1: float = cap_y1,
                cap_x0: float = cap_x0,
                cap_x1: float = cap_x1,
            ) -> tuple[float, float, float]:
                bottom = cap_y1
                lx0, lx1 = cap_x0, cap_x1
                max_gap = None
                for tb in candidates:
                    gap = max(0, tb["bbox"].y0 - bottom)
                    if max_gap is not None and gap > max_gap + 2:
                        break
                    max_gap = gap if max_gap is None else max(max_gap, gap)
                    bottom = max(bottom, tb["bbox"].y1)
                    lx0 = min(lx0, tb["bbox"].x0)
                    lx1 = max(lx1, tb["bbox"].x1)
                return bottom, lx0, lx1

            below = sorted(
                [tb for tb in text_blocks
                 if tb["bbox"].y0 >= cap_y1 - 2
                 and tb["bbox"].y0 < next_y
                 and abs((tb["bbox"].x0 + tb["bbox"].x1) / 2 - cap_mid_x) < cap_half_w],
                key=lambda tb: tb["bbox"].y0,
            )
            table_bottom, tx0, tx1 = _scan(below)
            if table_bottom - cap_y1 < 20:
                below_all = sorted(
                    [tb for tb in text_blocks
                     if tb["bbox"].y0 >= cap_y1 - 2 and tb["bbox"].y0 < next_y],
                    key=lambda tb: tb["bbox"].y0,
                )
                table_bottom, tx0, tx1 = _scan(below_all)

            crop = _padded_rect(tx0, cap_y0, tx1, table_bottom, pw, ph)
            if (tx1 - tx0) < pw * 0.55:
                crop.x0 = max(crop.x0, col_left)
                crop.x1 = min(crop.x1, col_right)

        elements.append(
            {"label": cap["label"], "caption": cap["text"][:80], "crop_rect": crop}
        )
    return elements
