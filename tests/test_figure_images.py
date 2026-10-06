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
    Figure,
    file_to_figures,
    held_summary,
    image_to_base64,
    make_figure,
    pdf_to_figures,
    sync_uploads,
    with_distinct_labels,
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

    def test_a_renamed_copy_is_not_counted_twice(self):
        gc.collect()
        before = held_summary()
        plot = make_figure("plot.png", Image.new("RGB", (40, 30), "white"))
        after = held_summary()
        copies = with_distinct_labels([plot, plot])
        assert len(copies) == 2
        assert held_summary() == after != before


class TestDistinctLabels:
    @staticmethod
    def figures(*labels: str) -> list[Figure]:
        image = Image.new("RGB", (4, 3), "white")
        return [make_figure(label, image) for label in labels]

    def test_repeated_labels_get_a_number(self):
        figures = self.figures("fig1.png", "fig1.png", "b.pdf p.1 Fig_1", "fig1.png")
        distinct = with_distinct_labels(figures)
        assert [f.label for f in distinct] == [
            "fig1.png", "fig1.png (2)", "b.pdf p.1 Fig_1", "fig1.png (3)",
        ]
        assert distinct[1].png is figures[1].png

    def test_a_number_already_in_use_is_skipped(self):
        figures = self.figures("fig1.png", "fig1.png", "fig1.png (2)")
        labels = [f.label for f in with_distinct_labels(figures)]
        assert labels == ["fig1.png", "fig1.png (3)", "fig1.png (2)"]

    def test_distinct_labels_are_left_alone(self):
        figures = self.figures("a.png", "b.png")
        assert with_distinct_labels(figures) == figures


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


def zip_file(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestUnreadableFiles:
    """A bad file must not raise: that would end the script on every rerun."""

    @pytest.mark.parametrize("data", [b"", b"hello world", pdf_file()[:200]],
                             ids=["empty", "not-a-pdf", "truncated"])
    def test_a_file_that_is_not_a_pdf_is_reported(self, data):
        problems: list[str] = []
        assert file_to_figures("paper.pdf", data, problems) == []
        (problem,) = problems
        assert problem.startswith("paper.pdf: could not be read as a PDF")

    def test_a_file_that_is_not_an_image_is_reported(self):
        problems: list[str] = []
        assert file_to_figures("scan.png", b"not an image", problems) == []
        (problem,) = problems
        assert problem.startswith("scan.png: could not be read as an image")

    def test_a_truncated_image_is_reported(self):
        buf = io.BytesIO()
        Image.new("RGB", (400, 300), "red").save(buf, format="PNG")
        problems: list[str] = []
        assert file_to_figures("scan.png", buf.getvalue()[:150], problems) == []
        assert len(problems) == 1

    def test_a_file_that_is_not_a_zip_is_reported(self):
        problems: list[str] = []
        assert file_to_figures("batch.zip", b"not an archive", problems) == []
        (problem,) = problems
        assert problem.startswith("batch.zip: could not be read as a ZIP archive")

    def test_one_bad_entry_does_not_cost_the_rest_of_the_archive(self):
        archive = zip_file({
            "a.png": png_file(40, 30), "b_bad.pdf": b"hello world",
            "c_bad.png": b"not an image", "d.pdf": pdf_file(),
        })
        problems: list[str] = []
        figures = file_to_figures("batch.zip", archive, problems)
        assert [f.label for f in figures] == ["batch.zip/a.png", "batch.zip/d.pdf p.1"]
        assert [problem.split(":")[0] for problem in problems] == [
            "batch.zip/b_bad.pdf", "batch.zip/c_bad.png",
        ]

    def test_a_page_with_nothing_to_render_does_not_cost_the_other_pages(self):
        """A CropBox outside the MediaBox leaves a page of 0 x 0 points."""
        body = text(72, 92, ["Body text without any caption."])
        data = build_pdf([
            PageSpec(body),
            PageSpec(body, width=100, height=100, cropbox=(500, 500, 600, 600)),
            PageSpec(body),
        ])
        problems: list[str] = []
        figures = pdf_to_figures(data, "paper.pdf", problems)
        assert [f.label for f in figures] == ["paper.pdf p.1", "paper.pdf p.3"]
        (problem,) = problems
        assert problem.startswith("paper.pdf p.2: skipped")

    def test_problems_need_not_be_collected(self):
        assert file_to_figures("paper.pdf", b"hello world") == []


class TestSyncUploads:
    def test_a_file_that_cannot_be_read_is_not_read_again(self):
        bad = FakeUpload("id-1", "paper.pdf", b"hello world")
        good = FakeUpload("id-2", "a.png", png_file(40, 30))
        done: dict = {}
        problems: dict = {}
        sync_uploads(done, [bad, good], problems)
        sync_uploads(done, [bad, good], problems)
        figures = sync_uploads(done, [bad, good], problems)
        assert (bad.reads, good.reads) == (1, 1)
        assert [f.label for f in figures] == ["a.png"]
        assert list(problems) == ["id-1"]
        assert problems["id-1"][0].startswith("paper.pdf: could not be read")

    def test_the_problems_of_a_removed_file_are_forgotten(self):
        bad = FakeUpload("id-1", "paper.pdf", b"hello world")
        done: dict = {}
        problems: dict = {}
        sync_uploads(done, [bad], problems)
        sync_uploads(done, [], problems)
        assert done == {} and problems == {}

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
