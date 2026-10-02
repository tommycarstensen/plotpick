"""pypdfium2 backend for pdf_figures: page facts and rendering.

PDFium has no notion of a text block, so this module builds the blocks from
single characters, following the rules of MuPDF's text device (which the
detection heuristics in pdf_figures were tuned against):

- a character whose baseline is more than PARAGRAPH_DIST font sizes away from
  the previous character's starts a new block;
- so does the first line of a new text object that is indented against the
  line before it;
- text hidden by a clip path (the cropped-away part of an embedded figure)
  is dropped.

All boxes are returned in display coordinates: origin top-left, page rotation
and CropBox offset applied.

PDFium is not thread-safe, and Streamlit runs every session in its own
thread, so each call into the library is made under one module-wide lock and
every PDFium object is closed explicitly (a garbage-collector finaliser would
otherwise close it from whatever thread happens to run the collector).
"""

import math
import threading
from collections.abc import Iterator
from ctypes import c_double, c_float, c_int, c_void_p, cast
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from pdf_figures import Rect

# MuPDF's text-device thresholds, in font sizes.
PARAGRAPH_DIST = 1.5   # baseline jump that starts a new block
BASE_MAX_DIST = 0.8    # baseline jump that still counts as the same line
SPACE_MAX_DIST = 0.8   # gap along the line that starts a new line segment
# ... and in points: how far a new line must be indented to start a block.
INDENT_DIST = 0.5

MAX_FORM_DEPTH = 15

_LOCK = threading.RLock()

Box = tuple[float, float, float, float]
Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _address(pointer: Any) -> int | None:
    return cast(pointer, c_void_p).value


def _contains(box: Box, point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _intersect(a: Box, b: Box) -> Box:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def _compose(outer: Matrix, inner: Matrix) -> Matrix:
    """The matrix that applies inner first, then outer."""
    a, b, c, d, e, f = inner
    oa, ob, oc, od, oe, of = outer
    return (
        oa * a + oc * b, ob * a + od * b,
        oa * c + oc * d, ob * c + od * d,
        oa * e + oc * f + oe, ob * e + od * f + of,
    )


def _transform(matrix: Matrix, box: Box) -> Box:
    """Bounding box of a box after a matrix is applied to its corners."""
    if matrix is IDENTITY:
        return box
    a, b, c, d, e, f = matrix
    xs = [a * x + c * y + e for x in (box[0], box[2]) for y in (box[1], box[3])]
    ys = [b * x + d * y + f for x in (box[0], box[2]) for y in (box[1], box[3])]
    return min(xs), min(ys), max(xs), max(ys)


def _clip_box(obj: Any) -> Box | None:
    """Bounding box of an object's own clip path, in its container's space."""
    clip = pdfium_c.FPDFPageObj_GetClipPath(obj)
    if not clip:
        return None
    box: Box | None = None
    x, y = c_float(), c_float()
    for i in range(max(pdfium_c.FPDFClipPath_CountPaths(clip), 0)):
        xs: list[float] = []
        ys: list[float] = []
        for j in range(max(pdfium_c.FPDFClipPath_CountPathSegments(clip, i), 0)):
            segment = pdfium_c.FPDFClipPath_GetPathSegment(clip, i, j)
            if pdfium_c.FPDFPathSegment_GetPoint(segment, x, y):
                xs.append(x.value)
                ys.append(y.value)
        if xs:
            path = (min(xs), min(ys), max(xs), max(ys))
            box = path if box is None else _intersect(box, path)
    return box


class Page:
    def __init__(self, page: pdfium.PdfPage):
        self._page = page
        self._left, self._bottom, self._right, self._top = page.get_bbox()
        self._rotation = page.get_rotation()
        self.width, self.height = page.get_size()
        self._images: list[Rect] | None = None
        self._drawings: list[Rect] = []
        self._text_clips: dict[int | None, Box] = {}

    # -- coordinates -------------------------------------------------------

    def _point(self, x: float, y: float) -> tuple[float, float]:
        """A point in PDF page space, as it lies on the displayed page."""
        if self._rotation == 90:
            return y - self._bottom, x - self._left
        if self._rotation == 180:
            return self._right - x, y - self._bottom
        if self._rotation == 270:
            return self._top - y, self._right - x
        return x - self._left, self._top - y

    def _rect(self, box: Box) -> Rect:
        xa, ya = self._point(box[0], box[1])
        xb, yb = self._point(box[2], box[3])
        return Rect(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))

    # -- page objects ------------------------------------------------------

    def _scan(self) -> None:
        """Walk the page objects once: image boxes, drawing boxes, text clips."""
        if self._images is not None:
            return
        self._images = []
        self._walk(self._page, False, IDENTITY, None, 0)

    def _walk(
        self, parent: Any, in_form: bool, matrix: Matrix, clip: Box | None,
        depth: int,
    ) -> None:
        """Collect from one object list; matrix and clip map it to page space.

        Bounds and clip paths of an object inside a Form XObject are in that
        form's coordinates, hence the matrix.  A clip is reduced to the
        bounding box of its paths, intersected with the enclosing clips.
        """
        assert self._images is not None
        if in_form:
            count = pdfium_c.FPDFFormObj_CountObjects(parent)
            get = pdfium_c.FPDFFormObj_GetObject
        else:
            count = pdfium_c.FPDFPage_CountObjects(parent)
            get = pdfium_c.FPDFPage_GetObject
        left, bottom, right, top = c_float(), c_float(), c_float(), c_float()
        fill, stroke = c_int(), c_int()
        for i in range(max(count, 0)):
            obj = get(parent, i)
            if not obj:
                continue
            kind = pdfium_c.FPDFPageObj_GetType(obj)
            if kind in (pdfium_c.FPDF_PAGEOBJ_TEXT, pdfium_c.FPDF_PAGEOBJ_FORM):
                own = _clip_box(obj)
                inner_clip = clip
                if own is not None:
                    own = _transform(matrix, own)
                    inner_clip = own if clip is None else _intersect(clip, own)
                if kind == pdfium_c.FPDF_PAGEOBJ_TEXT:
                    if inner_clip is not None:
                        self._text_clips[_address(obj)] = inner_clip
                elif depth < MAX_FORM_DEPTH:
                    form = pdfium_c.FS_MATRIX()
                    inner = matrix
                    if pdfium_c.FPDFPageObj_GetMatrix(obj, form):
                        inner = _compose(matrix, (
                            form.a, form.b, form.c, form.d, form.e, form.f,
                        ))
                    self._walk(obj, True, inner, inner_clip, depth + 1)
                continue
            if kind == pdfium_c.FPDF_PAGEOBJ_PATH:
                # A path that is neither filled nor stroked draws nothing.
                pdfium_c.FPDFPath_GetDrawMode(obj, fill, stroke)
                if not fill.value and not stroke.value:
                    continue
                target = self._drawings
            elif kind == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                target = self._images
            else:
                continue
            if pdfium_c.FPDFPageObj_GetBounds(obj, left, bottom, right, top):
                box = (left.value, bottom.value, right.value, top.value)
                target.append(self._rect(_transform(matrix, box)))

    def image_rects(self) -> list[Rect]:
        with _LOCK:
            self._scan()
            assert self._images is not None
            return self._images

    def drawing_rects(self) -> list[Rect]:
        with _LOCK:
            self._scan()
            return self._drawings

    # -- text --------------------------------------------------------------

    def text_blocks(self) -> list[dict[str, Any]]:
        with _LOCK:
            self._scan()
            textpage = self._page.get_textpage()
            try:
                return self._blocks(textpage)
            finally:
                textpage.close()

    def _blocks(self, textpage: pdfium.PdfTextPage) -> list[dict[str, Any]]:
        """Group the page's characters, in content-stream order, into blocks."""
        rect = pdfium_c.FS_RECTF()
        matrix = pdfium_c.FS_MATRIX()
        ox, oy = c_double(), c_double()
        page_box = (self._left, self._bottom, self._right, self._top)
        blocks: list[tuple[list[str], list[float]]] = []
        chars: list[str] = []
        bbox: list[float] = []
        prev_origin = (0.0, 0.0)
        prev_dir = (1.0, 0.0)
        prev_advance = line_start = 0.0
        prev_owner: int | None = None
        styles: dict[int | None, tuple[float, float, float]] = {}

        for i in range(textpage.count_chars()):
            code = pdfium_c.FPDFText_GetUnicode(textpage, i)
            if code in (0, 0xFFFE, 0xFFFF):
                continue
            # PDFium reports a hyphen it removed at a line break as U+0002.
            char = "-" if code == 2 else chr(code)
            if char.isspace():
                if chars and chars[-1] != " ":
                    chars.append(" ")
                continue
            if not pdfium_c.FPDFText_GetLooseCharBox(textpage, i, rect):
                continue
            raw = (rect.left, rect.bottom, rect.right, rect.top)
            centre = ((raw[0] + raw[2]) / 2, (raw[1] + raw[3]) / 2)
            if not _contains(page_box, centre):
                continue
            owner = _address(pdfium_c.FPDFText_GetTextObject(textpage, i))
            clip = self._text_clips.get(owner)
            if clip is not None and not _contains(clip, centre):
                continue
            pdfium_c.FPDFText_GetCharOrigin(textpage, i, ox, oy)
            origin = (ox.value, oy.value)

            if owner not in styles:
                # The character matrix gives both the writing direction and
                # the scale that turns the nominal font size into points.
                size = pdfium_c.FPDFText_GetFontSize(textpage, i)
                cos, sin = 1.0, 0.0
                if pdfium_c.FPDFText_GetMatrix(textpage, i, matrix):
                    run = math.hypot(matrix.a, matrix.b)
                    if run > 0:
                        cos, sin = matrix.a / run, matrix.b / run
                    size *= math.hypot(matrix.c, matrix.d)
                styles[owner] = (size or max(raw[3] - raw[1], 1.0), cos, sin)
            size, cos, sin = styles[owner]

            # Offsets from the previous character, across and along the
            # writing direction, in font sizes (text may run sideways).
            dx, dy = origin[0] - prev_origin[0], origin[1] - prev_origin[1]
            across = abs(cos * dy - sin * dx) / size
            along = (cos * dx + sin * dy - prev_advance) / size
            turned = cos * prev_dir[0] + sin * prev_dir[1] < 0.999
            position = cos * origin[0] + sin * origin[1]
            if not chars or turned or across > PARAGRAPH_DIST:
                new_block = True
            elif across >= BASE_MAX_DIST:
                new_block = (
                    owner != prev_owner and position - line_start > INDENT_DIST
                )
                line_start = position
            else:
                new_block = False
                if abs(along) >= SPACE_MAX_DIST:
                    line_start = position

            box = self._rect(raw)
            if new_block:
                chars = [char]
                bbox = [box.x0, box.y0, box.x1, box.y1]
                blocks.append((chars, bbox))
                line_start = position
            else:
                chars.append(char)
                bbox[0] = min(bbox[0], box.x0)
                bbox[1] = min(bbox[1], box.y0)
                bbox[2] = max(bbox[2], box.x1)
                bbox[3] = max(bbox[3], box.y1)
            prev_origin, prev_dir, prev_owner = origin, (cos, sin), owner
            prev_advance = (
                abs(cos) * (raw[2] - raw[0]) + abs(sin) * (raw[3] - raw[1])
            )

        return [
            {"text": "".join(chars).strip(), "bbox": Rect(*bbox)}
            for chars, bbox in blocks
        ]

    # -- rendering ---------------------------------------------------------

    def render(self, clip: Rect | None, dpi: int) -> Image.Image:
        crop = (0.0, 0.0, 0.0, 0.0)
        if clip is not None:
            x0, y0 = max(clip.x0, 0.0), max(clip.y0, 0.0)
            x1, y1 = min(clip.x1, self.width), min(clip.y1, self.height)
            if x1 - x0 < 72 / dpi or y1 - y0 < 72 / dpi:
                raise ValueError(f"Empty clip {clip} on a {self.width} x "
                                 f"{self.height} pt page")
            crop = (x0, self.height - y1, self.width - x1, y0)
        scale = dpi / 72
        with _LOCK:
            # pypdfium2 is untyped, so pyright infers int from the default of 1.
            bitmap = self._page.render(
                scale=scale,  # pyright: ignore[reportArgumentType] -- float is valid
                crop=crop,
            )
            try:
                return bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()

    def close(self) -> None:
        with _LOCK:
            self._page.close()


class Document:
    def __init__(self, source: bytes | str | Path):
        with _LOCK:
            self._doc = pdfium.PdfDocument(source)

    def __len__(self) -> int:
        with _LOCK:
            return len(self._doc)

    def __iter__(self) -> Iterator[Page]:
        for index in range(len(self)):
            with _LOCK:
                page = Page(self._doc[index])
            try:
                yield page
            finally:
                page.close()

    def __enter__(self) -> "Document":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with _LOCK:
            self._doc.close()
