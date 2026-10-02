"""PyMuPDF backend for pdf_figures: page facts and rendering.

TEMPORARY: the reference implementation while the pypdfium2 backend is being
evaluated.  It reproduces what pdf_figures did before the backends were split
out, call for call, so that the two can be compared on the same PDFs.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from pdf_figures import Rect


class Page:
    def __init__(self, page: Any):
        self._page = page
        self.width: float = page.rect.width
        self.height: float = page.rect.height

    def text_blocks(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block in self._page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            full = ""
            for line in block["lines"]:
                full += "".join(span["text"] for span in line["spans"])
            blocks.append({"text": full.strip(), "bbox": Rect(*block["bbox"])})
        return blocks

    def image_rects(self) -> list[Rect]:
        return [
            Rect(*info["bbox"]) for info in self._page.get_image_info(xrefs=True)
        ]

    def drawing_rects(self) -> list[Rect]:
        return [Rect(*d["rect"]) for d in self._page.get_drawings()]

    def render(self, clip: Rect | None, dpi: int) -> Image.Image:
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = self._page.get_pixmap(
            matrix=matrix, clip=None if clip is None else pymupdf.Rect(*clip),
        )
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


class Document:
    def __init__(self, source: bytes | str | Path):
        if isinstance(source, bytes):
            self._doc = pymupdf.open(stream=source, filetype="pdf")
        else:
            self._doc = pymupdf.open(str(source))

    def __len__(self) -> int:
        return len(self._doc)

    def __iter__(self) -> Iterator[Page]:
        for index in range(len(self._doc)):
            yield Page(self._doc[index])

    def __enter__(self) -> "Document":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._doc.close()
