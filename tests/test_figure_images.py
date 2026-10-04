"""Tests for figure_images.py: image sizes, and reading each upload once."""

import base64
import gc
import io
import zipfile

import pytest
from PIL import Image

from figure_images import (
    MAX_API_WIDTH,
    MAX_DISPLAY_WIDTH,
    file_to_figures,
    held_summary,
    image_to_base64,
    make_figure,
    pdf_to_figures,
    sync_uploads,
)
from tests.pdf_builder import PageSpec, box, build_pdf, text


def size(png: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(png)).size


def png_file(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def pdf_file(*, caption: str | None = None) -> bytes:
    """A one-page US Letter PDF: body text, and optionally a captioned box."""
    content = text(72, 92, ["Body text without any caption."])
    if caption:
        content += f" {box(100, 542, 100, 150)} {text(100, 522, [caption])}"
    return build_pdf([PageSpec(content)])


class FakeUpload:
    """Stands in for Streamlit's UploadedFile and counts how often it is read."""

    def __init__(self, file_id: str, name: str, data: bytes):
        self.file_id = file_id
        self.name = name
        self.data = data
        self.reads = 0

    def read(self) -> bytes:
        self.reads += 1
        return self.data


class TestFigureSizes:
    def test_wide_image_is_kept_whole_with_a_preview_that_fits_the_page(self):
        figure = make_figure("wide", Image.new("RGB", (3000, 1500), "white"))
        assert size(figure.png) == (3000, 1500)
        assert size(figure.preview) == (MAX_DISPLAY_WIDTH, 730)

    def test_small_image_is_held_once_not_twice(self):
        figure = make_figure("small", Image.new("RGB", (800, 600), "white"))
        assert size(figure.png) == (800, 600)
        assert figure.preview is figure.png

    def test_model_is_sent_at_most_the_api_width(self):
        figure = make_figure("wide", Image.new("RGB", (3000, 1500), "white"))
        sent = base64.b64decode(image_to_base64(figure.png))
        assert size(sent) == (MAX_API_WIDTH, 1000)

    def test_model_is_sent_a_small_image_untouched(self):
        png = png_file(800, 600)
        assert base64.b64decode(image_to_base64(png)) == png

    def test_streamlit_shows_the_preview_without_re_encoding_it(self):
        """st.image() re-encodes anything wider than its own limit, every rerun."""
        image_utils = pytest.importorskip("streamlit.elements.lib.image_utils")
        limit = getattr(image_utils, "MAXIMUM_CONTENT_WIDTH", None)
        if limit is None:
            pytest.skip("this Streamlit no longer exposes MAXIMUM_CONTENT_WIDTH")
        assert limit >= MAX_DISPLAY_WIDTH


class TestHeldSummary:
    def test_counts_the_figures_that_are_still_held_and_their_size(self):
        gc.collect()
        before = held_summary()
        wide = make_figure("wide", Image.new("RGB", (3000, 1500), "white"))
        small = make_figure("small", Image.new("RGB", (800, 600), "white"))
        held = len(wide.png) + len(wide.preview) + len(small.png)

        def parse(summary: str) -> tuple[int, float]:
            words = summary.split()
            return int(words[3]), float(words[5])

        assert parse(held_summary())[0] == parse(before)[0] + 2
        assert parse(held_summary())[1] == pytest.approx(
            parse(before)[1] + held / 1e6, abs=1,
        )
        del wide, small
        gc.collect()
        assert held_summary() == before


class TestFilesToFigures:
    def test_page_without_a_caption_is_rendered_whole(self):
        (figure,) = pdf_to_figures(pdf_file(), "paper.pdf")
        assert figure.label == "paper.pdf p.1"
        assert size(figure.png)[0] == 2550  # 8.5 in at 300 DPI
        assert size(figure.preview)[0] == MAX_DISPLAY_WIDTH

    def test_captioned_figure_is_cropped_and_labelled(self):
        (figure,) = pdf_to_figures(pdf_file(caption="Figure 1. A box."), "paper.pdf")
        assert figure.label == "paper.pdf p.1 Fig_1"
        assert size(figure.png)[0] < 2550

    def test_zip_holds_images_and_pdfs(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("b.pdf", pdf_file())
            zf.writestr("a.png", png_file(40, 30))
            zf.writestr("notes.txt", "not a figure")
        figures = file_to_figures("batch.zip", buf.getvalue())
        assert [f.label for f in figures] == ["batch.zip/a.png", "batch.zip/b.pdf p.1"]

    def test_image_upload_is_converted_to_rgb_png(self):
        buf = io.BytesIO()
        Image.new("L", (40, 30), 128).save(buf, format="JPEG")
        (figure,) = file_to_figures("scan.JPG", buf.getvalue())
        image = Image.open(io.BytesIO(figure.png))
        assert (image.format, image.mode, image.size) == ("PNG", "RGB", (40, 30))

    def test_other_file_types_yield_nothing(self):
        assert file_to_figures("notes.txt", b"not a figure") == []


class TestSyncUploads:
    def test_an_upload_is_read_once_however_often_the_script_reruns(self):
        upload = FakeUpload("id-1", "a.png", png_file(40, 30))
        done: dict = {}
        first = sync_uploads(done, [upload])
        reruns = [sync_uploads(done, [upload]) for _ in range(5)]
        assert upload.reads == 1
        assert all(figures[0] is first[0] for figures in reruns)

    def test_adding_a_file_reads_only_the_new_one(self):
        a = FakeUpload("id-1", "a.png", png_file(40, 30))
        b = FakeUpload("id-2", "b.png", png_file(40, 30))
        done: dict = {}
        sync_uploads(done, [a])
        figures = sync_uploads(done, [a, b])
        assert (a.reads, b.reads) == (1, 1)
        assert [f.label for f in figures] == ["a.png", "b.png"]

    def test_figures_follow_the_order_of_the_uploader(self):
        a = FakeUpload("id-1", "a.png", png_file(40, 30))
        b = FakeUpload("id-2", "b.png", png_file(40, 30))
        done: dict = {}
        sync_uploads(done, [a, b])
        assert [f.label for f in sync_uploads(done, [b, a])] == ["b.png", "a.png"]

    def test_a_removed_file_is_forgotten(self):
        a = FakeUpload("id-1", "a.png", png_file(40, 30))
        b = FakeUpload("id-2", "b.png", png_file(40, 30))
        done: dict = {}
        sync_uploads(done, [a, b])
        assert [f.label for f in sync_uploads(done, [b])] == ["b.png"]
        assert list(done) == ["id-2"]
        assert sync_uploads(done, []) == []
        assert done == {}
