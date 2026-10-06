"""Turn uploaded files and PDFs into the figure images the app holds in memory.

Standalone module with no Streamlit dependency -- usable from tests and from
batch scripts.

Memory is the constraint: Streamlit Community Cloud restricts an app that
uses too much of it, and Streamlit reruns the whole script on every click.
Two rules keep a rerun from allocating anything large.

An upload is read once.  The uploader hands over the same files again on
every rerun, so sync_uploads() keeps what it made, keyed by the file's id.

The page is shown a preview made once.  A figure is held as the full render
and as a PNG no wider than MAX_DISPLAY_WIDTH; when the render already fits,
the two are the same object.

The full render is kept, and cut down to MAX_API_WIDTH only when it is sent,
because it is the smaller thing to hold: a page rendered at 300 DPI
compresses better as PNG than the same page scaled down (5.5 MB against
9.6 MB for the nine images of one eight-page paper).

A file that cannot be read never raises here.  Whatever could be read is
returned, and each file, archive entry, page or crop that could not is
described in a line added to the caller's `problems` list.  An exception
would end the Streamlit script, and the uploader would hand the same file
over again on the next rerun.
"""

import base64
import hashlib
import io
import weakref
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import Protocol

from PIL import Image

from pdf_figures import PdfError, find_figures, open_pdf
from process_memory import log_memory, return_freed_memory

MAX_API_WIDTH = 2000  # Max width for API images (balance quality vs tokens)

# st.image() decodes, shrinks and re-encodes any image wider than this
# (Streamlit's MAXIMUM_CONTENT_WIDTH), and does so again on every rerun.  An
# image that already fits is passed through untouched.
MAX_DISPLAY_WIDTH = 1460

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
)


@dataclass(frozen=True, eq=False)
class Figure:
    """One figure, at the two sizes the app needs."""

    label: str
    png: bytes  # the full render; image_to_base64() cuts it down for the model
    preview: bytes  # what the page shows
    # Names the image itself, which a label does not: two uploads can share a
    # file name, and a label can be given to another figure later.
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "digest", hashlib.sha256(self.png).hexdigest()[:16],
        )
        _LIVE.add(self)


# Every figure still held by any session, for held_summary().
_LIVE: "weakref.WeakSet[Figure]" = weakref.WeakSet()


def held_summary() -> str:
    """How many figures all sessions hold between them, and their size.

    A renamed copy from with_distinct_labels() shares its images with the
    original, so images are counted once each.
    """
    figures = list(_LIVE)
    images = {id(data): len(data) for f in figures for data in (f.png, f.preview)}
    count = len({id(f.png) for f in figures})
    return f"all sessions hold {count} figure(s), {sum(images.values()) / 1e6:.0f} MB"


def with_distinct_labels(figures: Sequence["Figure"]) -> list["Figure"]:
    """The figures, with a number added to every label already used.

    The app looks up selections, results and source images by label, so two
    figures sharing one (two uploads named fig1.png, or two "Figure 1"
    captions on a page) were extracted twice and kept one result.  The second
    becomes "fig1.png (2)"; the images are shared, not copied.
    """
    taken = {figure.label for figure in figures}
    seen: set[str] = set()
    distinct: list[Figure] = []
    for figure in figures:
        if figure.label in seen:
            number = 2
            while f"{figure.label} ({number})" in taken:
                number += 1
            figure = replace(figure, label=f"{figure.label} ({number})")
            taken.add(figure.label)
        seen.add(figure.label)
        distinct.append(figure)
    return distinct


class Upload(Protocol):
    """The part of Streamlit's UploadedFile that sync_uploads() relies on."""

    file_id: str
    name: str

    def read(self) -> bytes: ...


def _fit_width(img: Image.Image, max_width: int) -> Image.Image:
    """Scale image down to max_width, preserving aspect ratio."""
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_figure(label: str, img: Image.Image) -> Figure:
    """Hold an RGB image as PNG, with a preview if it is too wide to show."""
    png = _png_bytes(img)
    if img.width <= MAX_DISPLAY_WIDTH:
        return Figure(label, png, png)
    return Figure(label, png, _png_bytes(_fit_width(img, MAX_DISPLAY_WIDTH)))


def image_to_base64(png_bytes: bytes) -> str:
    """Encode PNG bytes as a base64 string for the API, resizing if needed."""
    img = Image.open(io.BytesIO(png_bytes))
    if img.width > MAX_API_WIDTH:
        png_bytes = _png_bytes(_fit_width(img, MAX_API_WIDTH))
    return base64.b64encode(png_bytes).decode("ascii")


def pdf_to_figures(
    data: bytes, name: str, problems: list[str] | None = None,
) -> list[Figure]:
    """Extract individual figures from a PDF.

    Uses caption detection to crop each figure/table separately.
    Falls back to full-page rendering when no captions are found.
    """
    if problems is None:
        problems = []
    results: list[Figure] = []
    dpi = 300  # for better readability
    pages_read = 0

    try:
        with open_pdf(data) as doc:
            for page_idx, page in enumerate(doc):
                where = f"{name} p.{page_idx + 1}"
                pages_read = page_idx + 1
                try:
                    elements = find_figures(page)
                except PdfError as exc:
                    problems.append(f"{where}: skipped ({exc})")
                    continue
                # No figures detected -- render full page as fallback
                regions = [
                    (f"{where} {elem['label']}", elem["crop_rect"])
                    for elem in elements
                ] or [(where, None)]
                for label, clip in regions:
                    try:
                        img = page.render(clip, dpi)
                    except PdfError as exc:
                        problems.append(f"{label}: skipped ({exc})")
                        continue
                    results.append(make_figure(label, img))
    except PdfError as exc:
        # The file did not open, or the next page did not load: keep what the
        # pages before it gave.
        if pages_read:
            problems.append(
                f"{name}: nothing after p.{pages_read} could be read ({exc})"
            )
        else:
            problems.append(f"{name}: could not be read as a PDF ({exc})")

    return_freed_memory()
    return results


def image_to_figure(raw: bytes, label: str) -> Figure:
    """Read an image in any supported format."""
    return make_figure(label, Image.open(io.BytesIO(raw)).convert("RGB"))


# What Pillow raises for a file that is not an image it can decode: OSError
# covers unknown formats and truncated files, the others broken headers and
# chunks, and images whose stated size is implausibly large.
_IMAGE_ERRORS = (OSError, ValueError, SyntaxError, Image.DecompressionBombError)


def _image_to_figures(raw: bytes, label: str, problems: list[str]) -> list[Figure]:
    try:
        return [image_to_figure(raw, label)]
    except _IMAGE_ERRORS as exc:
        problems.append(f"{label}: could not be read as an image ({exc})")
        return []


def zip_to_figures(
    data: bytes, zip_name: str, problems: list[str] | None = None,
) -> list[Figure]:
    """Extract the images and PDFs in a ZIP archive."""
    if problems is None:
        problems = []
    results: list[Figure] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        problems.append(f"{zip_name}: could not be read as a ZIP archive ({exc})")
        return results
    with archive as zf:
        for info in sorted(zf.infolist(), key=lambda info: info.filename):
            entry = info.filename
            suffix = PurePosixPath(entry).suffix.lower()
            label = f"{zip_name}/{entry}"
            if suffix != ".pdf" and suffix not in IMAGE_EXTENSIONS:
                continue
            if info.flag_bits & 0x1:
                problems.append(f"{label}: skipped (password-protected)")
                continue
            try:
                raw = zf.read(info)
            except (zipfile.BadZipFile, NotImplementedError) as exc:
                problems.append(f"{label}: could not be unpacked ({exc})")
                continue
            if suffix == ".pdf":
                results.extend(pdf_to_figures(raw, label, problems))
            else:
                results.extend(_image_to_figures(raw, label, problems))
    return results


def file_to_figures(
    name: str, data: bytes, problems: list[str] | None = None,
) -> list[Figure]:
    """Route a single file to the correct processor."""
    if problems is None:
        problems = []
    suffix = PurePosixPath(name).suffix.lower()

    if suffix == ".zip":
        return zip_to_figures(data, name, problems)
    if suffix == ".pdf":
        return pdf_to_figures(data, name, problems)
    if suffix in IMAGE_EXTENSIONS:
        return _image_to_figures(data, name, problems)
    return []


def sync_uploads(
    done: dict[str, list[Figure]],
    uploads: Sequence[Upload],
    problems: dict[str, list[str]] | None = None,
) -> list[Figure]:
    """Return the figures of the current uploads, reading only the new files.

    `done` maps a file id to that file's figures and is updated in place: a
    file not seen before is read, a file no longer uploaded is forgotten.
    `problems`, if given, is kept the same way: file id to what could not be
    read in that file, for as long as the file stays uploaded.
    """
    if problems is None:
        problems = {}
    current = {upload.file_id for upload in uploads}
    for file_id in list(done):
        if file_id not in current:
            del done[file_id]
            problems.pop(file_id, None)
    new = [upload for upload in uploads if upload.file_id not in done]
    for upload in new:
        notes: list[str] = []
        done[upload.file_id] = file_to_figures(upload.name, upload.read(), notes)
        if notes:
            problems[upload.file_id] = notes
    if new:
        log_memory(f"reading {len(new)} upload(s)", held_summary())
    return [figure for upload in uploads for figure in done[upload.file_id]]
