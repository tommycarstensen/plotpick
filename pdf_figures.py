"""Detect and extract figures/tables from PDF pages using caption heuristics.

Standalone module with no Streamlit dependency -- usable from both the app
and from batch scripts.

The detection works on plain page facts (text blocks, image boxes, drawing
boxes), which pdf_backend_pdfium reads from the PDF with pypdfium2.
"""

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

# "Fig 1" without a full stop is PLOS's style; "Figure S1" and "Table S2"
# number supplementary items.
CAPTION_RE = re.compile(
    r"^(Supplementary\s+)?Fig(ure|\.)?\s*S?\d"
    r"|^(Supplementary\s+)?Table\s*S?\d"
    r"|^Suppl\.?\s+Fig",
    re.IGNORECASE,
)
# A block that opens with a label is body text, not a caption, when the label
# runs on as a sentence does: "Table 1 summarizes ...", "Figure 2, Table S3)."
# A caption puts a separator, a capital, a digit or a bracket after the label.
# Only an all-lowercase word counts, so that "Fig. 2 mRNA levels" stays a
# caption.  On the 204 PMC validation papers this dropped 123 blocks, every
# one of them a sentence.
_LABEL_RE = re.compile(
    r"(Supplementary\s+|Suppl\.?\s+)?(Fig(?:ure|\.)?|Table)\s*S?\d+[A-Za-z]?\s*",
    re.IGNORECASE,
)
_SENTENCE_TAIL_RE = re.compile(
    r"[,;)]"
    r"|\.\s*$"  # the label and a full stop, nothing after: the end of a sentence
    r"|(?!(?:continued|contd?)\b|in\s+(?:vitro|vivo|situ|silico)\b)[a-z]{2,}(?![\w-])"
)
MIN_IMG_DIM = 50
H_MARGIN = 20
V_MARGIN = 6
# A block this long, made up mostly of letters, is a paragraph of body text;
# the text inside a figure or a table is short labels and rows of numbers.
PROSE_MIN_CHARS = 200
PROSE_MIN_LETTERS = 0.75


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
        r"(Supplementary\s+|Suppl\.?\s+)?(Fig(?:ure|\.)?|Table)\s*(S?)(\d+)",
        text.strip(), re.IGNORECASE,
    )
    if not m:
        return None
    prefix = "Suppl_" if m.group(1) else ""
    kind = "Fig" if "fig" in m.group(2).lower() else "Table"
    return f"{prefix}{kind}_{m.group(3).upper()}{m.group(4)}"


def reads_as_running_text(text: str) -> bool:
    """Whether a block that starts with a figure or table label is a sentence."""
    text = text.strip()
    label = _LABEL_RE.match(text)
    return bool(label and _SENTENCE_TAIL_RE.match(text, label.end()))


def _is_prose(text: str) -> bool:
    """Whether a block reads as body text rather than as part of a figure."""
    if len(text) < PROSE_MIN_CHARS:
        return False
    solid = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in solid) >= PROSE_MIN_LETTERS * len(solid)


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


@dataclass
class _Caption:
    label: str
    kind: str  # "figure" or "table"
    text: str
    bbox: Rect
    column: str = "both"  # "left", "right" or "both"
    bounds: tuple[float, float] = (0.0, 0.0)  # the column, as an x range
    previous: int | None = None  # the caption before it in its column
    prev_y: float = 0.0  # ... and where that one ends
    next_y: float = 0.0  # where the caption after it begins


def _captions(text_blocks: list[dict], pw: float, ph: float) -> list[_Caption]:
    """The blocks that are captions, top to bottom, each with its neighbours."""
    captions: list[_Caption] = []
    for tb in text_blocks:
        if not CAPTION_RE.match(tb["text"]) or reads_as_running_text(tb["text"]):
            continue
        label = label_from_caption(tb["text"])
        if not label:
            continue
        kind = "table" if "table" in label.lower() else "figure"
        captions.append(_Caption(label, kind, tb["text"][:80], tb["bbox"]))
    captions.sort(key=lambda c: c.bbox.y0)

    two_col = _is_two_column(text_blocks, pw)
    for cap in captions:
        box = cap.bbox
        cap.bounds = _column_bounds(box.x0, box.x1, pw, two_col)
        # A caption that straddles the centre is in both columns.  That is
        # judged by its edges, not its midpoint: a short caption at the left
        # edge of the right column has its midpoint near the page centre.
        if not two_col or (box.x0 < pw * 0.45 and box.x1 > pw * 0.55):
            cap.column = "both"
        else:
            cap.column = "left" if (box.x0 + box.x1) / 2 < pw * 0.5 else "right"

    def shared(a: _Caption, b: _Caption) -> bool:
        return a.column == b.column or "both" in (a.column, b.column)

    for i, cap in enumerate(captions):
        before = [j for j in range(i) if shared(cap, captions[j])]
        after = [j for j in range(i + 1, len(captions)) if shared(cap, captions[j])]
        cap.previous = before[-1] if before else None
        cap.prev_y = captions[before[-1]].bbox.y1 if before else 0.0
        cap.next_y = captions[after[0]].bbox.y0 if after else ph
    return captions


def _table_crop(cap: _Caption, text_blocks: list[dict], pw: float, ph: float) -> Rect:
    """A table runs from its caption down through the text blocks under it."""
    box = cap.bbox
    mid_x = (box.x0 + box.x1) / 2
    half_w = (box.x1 - box.x0) / 2 + 15

    def scan(candidates: list[dict]) -> tuple[float, float, float]:
        bottom = box.y1
        lx0, lx1 = box.x0, box.x1
        max_gap = None
        for tb in sorted(candidates, key=lambda tb: tb["bbox"].y0):
            gap = max(0, tb["bbox"].y0 - bottom)
            if max_gap is not None and gap > max_gap + 2:
                break
            max_gap = gap if max_gap is None else max(max_gap, gap)
            bottom = max(bottom, tb["bbox"].y1)
            lx0 = min(lx0, tb["bbox"].x0)
            lx1 = max(lx1, tb["bbox"].x1)
        return bottom, lx0, lx1

    below = [
        tb for tb in text_blocks if box.y1 - 2 <= tb["bbox"].y0 < cap.next_y
    ]
    bottom, tx0, tx1 = scan([
        tb for tb in below
        if abs((tb["bbox"].x0 + tb["bbox"].x1) / 2 - mid_x) < half_w
    ])
    if bottom - box.y1 < 20:
        bottom, tx0, tx1 = scan(below)

    crop = _padded_rect(tx0, box.y0, tx1, bottom, pw, ph)
    if (tx1 - tx0) < pw * 0.55:
        crop.x0 = max(crop.x0, cap.bounds[0])
        crop.x1 = min(crop.x1, cap.bounds[1])
    return crop


def _assign_images(
    images: list[Rect], captions: list[_Caption], uppers: dict[int, float],
) -> tuple[dict[int, list[Rect]], list[Rect]]:
    """Give each image to the figure caption it belongs to.

    Returns the images of each figure caption, and those that belong to none.

    The images between two captions go together to one of them.  Which one
    follows from the page where it shows whether captions stand below their
    figures, as is usual, or above them, as in some journals; otherwise the
    images go to the caption below unless the one above is clearly nearer.
    An image is never shared, so the crop of one figure cannot take in the
    image of the next.
    """
    groups: dict[tuple[int | None, int | None], list[Rect]] = {}
    loose: list[Rect] = []
    for image in images:
        cx, cy = (image.x0 + image.x1) / 2, (image.y0 + image.y1) / 2
        # The captions it could belong to: those whose column it is centred
        # in, and those that fit under (or over) it, as a short caption does
        # under the left edge of a figure wider than the column.
        reach = [
            i for i, cap in enumerate(captions)
            if cap.bounds[0] <= cx <= cap.bounds[1]
            or min(image.x1, cap.bbox.x1) - max(image.x0, cap.bbox.x0)
            >= 0.8 * (cap.bbox.x1 - cap.bbox.x0)
        ]
        above = [i for i in reach if captions[i].bbox.y1 <= cy]
        below = [i for i in reach if captions[i].bbox.y0 >= cy]
        key = (above[-1] if above else None, below[0] if below else None)
        # Body text or a table between the image and the caption below it
        # means the image is not that caption's.
        if key[1] is not None and cy <= uppers.get(key[1], 0.0):
            key = (key[0], None)
        groups.setdefault(key, []).append(image)

    def gaps(a: int | None, b: int | None, group: list[Rect]) -> tuple[float, float]:
        top, bottom = min(r.y0 for r in group), max(r.y1 for r in group)
        return (
            math.inf if a is None else max(top - captions[a].bbox.y1, 0.0),
            math.inf if b is None else max(captions[b].bbox.y0 - bottom, 0.0),
        )

    # What the page as a whole says: an image above the first figure caption
    # means captions stand below their figures; failing that, an image right
    # under the last one means they stand above.
    figures = [i for i, cap in enumerate(captions) if cap.kind == "figure"]
    caption_first = None
    if figures and any(b == figures[0] for _, b in groups):
        caption_first = False
    elif figures and any(
        a == figures[-1] and (b is None or captions[b].kind == "table")
        and gaps(a, b, group)[0] <= 50
        for (a, b), group in groups.items()
    ):
        caption_first = True

    owned: dict[int, list[Rect]] = {}
    for (a, b), group in sorted(
        groups.items(), key=lambda item: min(r.y0 for r in item[1]),
    ):
        gap_above, gap_below = gaps(a, b, group)
        owner: int | None
        if caption_first is not None:
            owner = a if caption_first else b
        elif b is not None and (a in owned or gap_below <= 1.5 * gap_above + 5):
            owner = b
        elif a is not None and a not in owned and (b is not None or gap_above <= 50):
            owner = a
        else:
            owner = None
        if owner is not None and captions[owner].kind == "figure":
            owned.setdefault(owner, []).extend(group)
        else:
            loose.extend(group)
    return owned, loose


def _is_page_furniture(d: Rect, pw: float, ph: float) -> bool:
    """Whether a drawing is the rule or band of a running head or a footer."""
    return d.x1 - d.x0 >= 0.5 * pw and (d.y1 < 0.08 * ph or d.y0 > 0.92 * ph)


def _is_strip(r: Rect) -> bool:
    """Whether an image is a strip: too flat to count as a figure by the
    MIN_IMG_DIM rule, too large to be an icon.  A flow diagram can be 400 pt
    wide and 40 pt high."""
    width, height = r.x1 - r.x0, r.y1 - r.y0
    return (
        min(width, height) <= MIN_IMG_DIM < max(width, height)
        and min(width, height) >= 20 and width * height >= 5000
    )


def _figure_crop(
    cap: _Caption, images: list[Rect], loose: list[Rect], strips: list[Rect],
    drawings: list[Rect], upper: float, pw: float, ph: float,
) -> Rect:
    """A figure is its graphics and its caption."""
    box = cap.bbox
    # Drawings count when they lie above the caption and within its column,
    # or a little beyond a caption that is wider than its column.
    left = min(box.x0 - 60, cap.bounds[0])
    right = max(box.x1 + 60, cap.bounds[1])
    vector = [] if images else [
        d for d in drawings
        if d.y0 >= upper - 10 and (d.y0 + d.y1) / 2 <= box.y1
        and d.y1 <= box.y1 + 10 and left <= d.x0 and d.x1 <= right
        and not _is_page_furniture(d, pw, ph)
    ]
    # A figure that is one flat image, such as a flow diagram set as a strip.
    strips = [] if images or vector else [
        r for r in strips if upper < (r.y0 + r.y1) / 2 < box.y0
        and cap.bounds[0] <= (r.x0 + r.x1) / 2 <= cap.bounds[1]
    ]
    # Some journals set a wide figure with its caption in the column beside it.
    beside = [] if images or vector or strips else [
        r for r in loose
        if (r.x1 <= box.x0 or r.x0 >= box.x1)
        and min(r.y1, box.y1) - max(r.y0, box.y0)
        >= 0.5 * min(r.y1 - r.y0, box.y1 - box.y0)
    ]
    parts = images or vector or strips or beside
    if parts:
        x0 = min(r.x0 for r in parts)
        y0 = min(r.y0 for r in parts)
        x1 = max(r.x1 for r in parts)
        y1 = max(r.y1 for r in parts)
    else:
        x0, y0 = box.x0, upper
        x1, y1 = box.x1, box.y0
    if not beside:
        y0 = max(y0, upper)

    x0, y0 = min(x0, box.x0), min(y0, box.y0)
    x1, y1 = max(x1, box.x1), max(y1, box.y1)

    # The margin stays inside the caption's column; the figure itself is
    # never cut, also when it is wider than that column.
    crop = _padded_rect(x0, y0, x1, y1, pw, ph)
    crop.x0 = min(x0, max(crop.x0, cap.bounds[0]))
    crop.x1 = max(x1, min(crop.x1, cap.bounds[1], x1 + 4))
    # ... and below whatever the figure starts under.
    if not beside:
        crop.y0 = max(crop.y0, min(upper, y0))
    return crop


def find_figures(page: PdfPage) -> list[dict[str, Any]]:
    """Detect figures/tables on a PDF page via caption text.

    Returns list of dicts with keys: label, caption, crop_rect.
    """
    pw, ph = page.width, page.height
    text_blocks = page.text_blocks()
    captions = _captions(text_blocks, pw, ph)
    if not captions:
        return []

    # Tables first: a figure below a table starts where that table ends.
    crops: dict[int, Rect] = {
        i: _table_crop(cap, text_blocks, pw, ph)
        for i, cap in enumerate(captions) if cap.kind == "table"
    }

    prose = [
        tb["bbox"] for tb in text_blocks
        if _is_prose(tb["text"]) and not CAPTION_RE.match(tb["text"])
    ]
    # A figure lies between its caption and whatever ends above it: the
    # caption before, the table under that caption, or the last paragraph of
    # body text.
    uppers: dict[int, float] = {}
    for i, cap in enumerate(captions):
        if cap.kind != "figure":
            continue
        upper = cap.prev_y
        if cap.previous in crops:
            table_bottom = crops[cap.previous].y1 - V_MARGIN
            if table_bottom < cap.bbox.y0 - 20:
                upper = max(upper, table_bottom)
        uppers[i] = max([upper] + [
            b.y1 for b in prose
            if b.y1 <= cap.bbox.y0 + 2
            and b.x0 < cap.bounds[1] and b.x1 > cap.bounds[0]
        ])

    images = [
        r for r in page.image_rects()
        if r.x1 - r.x0 > MIN_IMG_DIM and r.y1 - r.y0 > MIN_IMG_DIM
    ]
    strips = [
        r for r in page.image_rects()
        if _is_strip(r) and not _is_page_furniture(r, pw, ph)
    ]
    owned, loose = _assign_images(images, captions, uppers)
    drawings = page.drawing_rects()
    for i, upper in uppers.items():
        crops[i] = _figure_crop(
            captions[i], owned.get(i, []), loose, strips, drawings, upper, pw, ph,
        )

    return [
        {"label": cap.label, "caption": cap.text, "crop_rect": crops[i]}
        for i, cap in enumerate(captions)
    ]
