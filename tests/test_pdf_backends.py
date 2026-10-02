"""Tests for figure detection and rendering, per PDF backend.

The fixtures are PDFs written by hand in pdf_builder.py.  Every test that is
parametrised over `backend` states behaviour both backends must share; the
PyMuPDF half disappears with the library once the port is done.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import pytest
from PIL import Image

from pdf_figures import BACKEND_ENV, PdfPage, Rect, find_figures, open_pdf
from tests.pdf_builder import (
    PAGE_H,
    PAGE_W,
    FormSpec,
    PageSpec,
    box,
    build_pdf,
    image,
    text,
)

BODY = text(72, 720, [
    "Body text introducing the result, long enough to be a paragraph.",
    "Figure 2 shows nothing here, this is a sentence inside the paragraph.",
    "A third line closes the paragraph.",
])
FIGURE = (
    image(100, 500, 200, 150)
    + " " + text(100, 480, ["Figure 1. A raster figure with a caption."])
)
TABLE = (
    text(72, 400, ["Table 1. Outcomes by arm."])
    + " " + text(72, 380, ["Arm      N     Mean", "Placebo  40    1.2",
                           "Active   41    2.3"])
)
UPRIGHT = PageSpec(f"{BODY} {FIGURE} {TABLE}")

HIDDEN = (
    text(10, 150, ["Table 9. Hidden caption outside the visible area."])
    + " " + text(10, 50, ["Visible text inside the form."])
)


@pytest.fixture(params=["pymupdf", "pdfium"])
def backend(request):
    pytest.importorskip("pymupdf" if request.param == "pymupdf" else "pypdfium2")
    return request.param


@contextmanager
def only_page(spec: PageSpec, backend: str) -> Iterator[PdfPage]:
    """The single page of a one-page fixture, open for the with-block."""
    with open_pdf(build_pdf([spec]), backend) as doc:
        pages = iter(doc)
        yield next(pages)


def detect(spec: PageSpec, backend: str) -> dict[str, Rect]:
    with only_page(spec, backend) as page:
        return {e["label"]: e["crop_rect"] for e in find_figures(page)}


def blocks(spec: PageSpec, backend: str) -> list[str]:
    with only_page(spec, backend) as page:
        return [b["text"] for b in page.text_blocks()]


def close_to(rect: Rect, expected: Iterable[float], tolerance: float = 2.5) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(rect, expected, strict=True))


def rgb(img: Image.Image, point: tuple[int, int]) -> tuple[int, ...]:
    pixel = img.getpixel(point)
    assert isinstance(pixel, tuple)
    return pixel


class TestDetection:
    def test_raster_figure_is_cropped_with_its_caption(self, backend):
        found = detect(UPRIGHT, backend)
        # Image spans x 100-300 and, from the top, y 142-292; the caption
        # baseline is 312 from the top.  20 pt margin left, 6 pt above.
        assert close_to(found["Fig_1"], (80, 136, 304, 320.5))

    def test_table_is_cropped_from_caption_to_last_row(self, backend):
        found = detect(UPRIGHT, backend)
        assert close_to(found["Table_1"], (52, 376, 212.6, 444.5))

    def test_a_caption_word_inside_a_paragraph_is_not_a_caption(self, backend):
        """ "Figure 2 shows ..." starts a line, but not a block."""
        assert set(detect(UPRIGHT, backend)) == {"Fig_1", "Table_1"}

    def test_vector_figure_is_found_from_its_drawings(self, backend):
        spec = PageSpec(
            box(320, 200, 100, 120) + " " + box(340, 220, 40, 60)
            + " " + text(320, 180, ["Figure 3. A vector figure."])
        )
        # Boxes span, from the top, y 472-592; caption baseline at 612.
        assert close_to(detect(spec, backend)["Fig_3"], (300, 465.5, 433.5, 620.5))

    def test_page_without_captions_gives_nothing(self, backend):
        assert detect(PageSpec(BODY), backend) == {}


class TestTextBlocks:
    def test_lines_of_one_paragraph_form_one_block(self, backend):
        (block,) = blocks(PageSpec(BODY), backend)
        assert block.startswith("Body text introducing")
        assert block.endswith("closes the paragraph.")

    def test_a_gap_of_two_lines_separates_blocks(self, backend):
        spec = PageSpec(text(72, 720, ["First paragraph."])
                        + " " + text(72, 700, ["Second paragraph."]))
        assert blocks(spec, backend) == ["First paragraph.", "Second paragraph."]

    def test_an_indented_first_line_separates_blocks(self, backend):
        spec = PageSpec(text(72, 720, ["End of one paragraph."])
                        + " " + text(84, 708, ["Indented start of the next."]))
        assert len(blocks(spec, backend)) == 2

    @pytest.mark.parametrize("spec", [
        # The form's own BBox cuts the caption off ...
        PageSpec("q 1 0 0 1 100 300 cm /Fm1 Do Q",
                 forms={"Fm1": FormSpec(HIDDEN, (0, 0, 300, 100))}),
        # ... or a clip set before the form is drawn (how pdfTeX crops) ...
        PageSpec("q 1 0 0 1 100 300 cm 0 0 300 100 re W n /Fm1 Do Q",
                 forms={"Fm1": FormSpec(HIDDEN, (0, 0, 400, 400))}),
        # ... or the BBox of a form around the form.
        PageSpec("q 1 0 0 1 100 300 cm /Fm2 Do Q",
                 forms={"Fm1": FormSpec(HIDDEN, (0, 0, 400, 400)),
                        "Fm2": FormSpec("/Fm1 Do", (0, 0, 300, 100))}),
    ], ids=["form-bbox", "clip-path", "outer-form-bbox"])
    def test_text_clipped_out_of_view_is_dropped(self, backend, spec):
        """A cropped embedded figure must not leak its original caption."""
        assert blocks(spec, backend) == ["Visible text inside the form."]
        assert detect(spec, backend) == {}


class TestRendering:
    def test_clip_is_rendered_at_the_requested_resolution(self, backend):
        with only_page(UPRIGHT, backend) as page:
            img = page.render(Rect(100, 142, 300, 292), 144)
        assert img.mode == "RGB"
        assert abs(img.width - 400) <= 2 and abs(img.height - 300) <= 2
        # The clip is exactly the 2 x 2 test image: red top-left, no white.
        red, green, blue = rgb(img, (50, 50))
        assert red > 150 and green < 80 and blue < 80
        assert rgb(img, (350, 250)) != (255, 255, 255)

    def test_whole_page_is_rendered_without_a_clip(self, backend):
        with only_page(UPRIGHT, backend) as page:
            img = page.render(None, 72)
        assert abs(img.width - PAGE_W) <= 1 and abs(img.height - PAGE_H) <= 1
        assert rgb(img, (5, 5)) == (255, 255, 255)

    def test_every_page_is_visited_in_order(self, backend):
        data = build_pdf([PageSpec(BODY), UPRIGHT, PageSpec(BODY)])
        with open_pdf(data, backend) as doc:
            assert len(doc) == 3
            counts = [len(find_figures(page)) for page in doc]
        assert counts == [0, 2, 0]

    def test_a_path_opens_like_bytes(self, backend, tmp_path):
        path = tmp_path / "sample.pdf"
        path.write_bytes(build_pdf([UPRIGHT]))
        with open_pdf(path, backend) as doc:
            assert len(doc) == 1


class TestPdfiumDisplayCoordinates:
    """Boxes follow the page as displayed.  PyMuPDF's path did not manage
    this on rotated pages, so these hold for the pdfium backend only."""

    @pytest.fixture(autouse=True)
    def _needs_pdfium(self):
        pytest.importorskip("pypdfium2")

    def test_rotated_page_matches_the_upright_page(self):
        # A landscape page shown upright by /Rotate 90: its content is the
        # upright page turned a quarter, (x, y) -> (792 - y, x).
        rotated = PageSpec(
            f"0 1 -1 0 {PAGE_H} 0 cm {UPRIGHT.content}",
            width=PAGE_H, height=PAGE_W, rotate=90,
        )
        upright = detect(UPRIGHT, "pdfium")
        turned = detect(rotated, "pdfium")
        assert set(turned) == {"Fig_1", "Table_1"}
        for label, rect in upright.items():
            assert close_to(turned[label], rect, tolerance=0.5)

    def test_rotated_page_renders_the_same_region(self):
        rotated = PageSpec(
            f"0 1 -1 0 {PAGE_H} 0 cm {UPRIGHT.content}",
            width=PAGE_H, height=PAGE_W, rotate=90,
        )
        clip = Rect(100, 142, 300, 292)
        images = []
        for spec in (UPRIGHT, rotated):
            with only_page(spec, "pdfium") as page:
                assert (round(page.width), round(page.height)) == (PAGE_W, PAGE_H)
                images.append(page.render(clip, 72))
        assert images[0].size == images[1].size
        for point in ((30, 30), (170, 30), (30, 120), (170, 120)):
            a, b = rgb(images[0], point), rgb(images[1], point)
            assert all(abs(x - y) <= 40 for x, y in zip(a, b, strict=True)), point

    def test_cropbox_origin_is_subtracted(self):
        cropped = PageSpec(UPRIGHT.content, cropbox=(50, 60, 562, 742))
        upright = detect(UPRIGHT, "pdfium")
        shifted = detect(cropped, "pdfium")
        # 50 pt cut off the left, 792 - 742 = 50 pt off the top.
        for label, rect in upright.items():
            expected = (rect.x0 - 50, rect.y0 - 50, rect.x1 - 50, rect.y1 - 50)
            assert close_to(shifted[label], expected, tolerance=0.5)

    def test_an_empty_clip_is_refused(self):
        with (
            only_page(UPRIGHT, "pdfium") as page,
            pytest.raises(ValueError, match="Empty clip"),
        ):
            page.render(Rect(300, 100, 200, 400), 300)

    def test_a_page_kept_past_its_turn_fails_loudly(self):
        with open_pdf(build_pdf([UPRIGHT, UPRIGHT]), "pdfium") as doc:
            pages = list(doc)
            with pytest.raises(RuntimeError, match="used after it was closed"):
                pages[0].text_blocks()


class TestBackendSelection:
    def test_environment_variable_picks_the_backend(self, monkeypatch):
        pytest.importorskip("pypdfium2")
        monkeypatch.setenv(BACKEND_ENV, "pdfium")
        with open_pdf(build_pdf([UPRIGHT])) as doc:
            assert type(doc).__module__ == "pdf_backend_pdfium"

    def test_unknown_backend_is_an_error(self):
        with pytest.raises(ValueError, match="Unknown PDF backend"):
            open_pdf(build_pdf([UPRIGHT]), "ghostscript")


class TestLegacyEntryPoint:
    """validation/pipeline/match_pairs.py opens PDFs with PyMuPDF itself."""

    def test_crop_rect_is_a_rectangle_pymupdf_honours(self):
        pymupdf = pytest.importorskip("pymupdf")
        from pdf_figures import find_figures_on_page

        page = pymupdf.open(stream=build_pdf([UPRIGHT]), filetype="pdf")[0]
        (figure,) = [e for e in find_figures_on_page(page) if e["label"] == "Fig_1"]
        assert isinstance(figure["crop_rect"], pymupdf.Rect)
        pix = page.get_pixmap(clip=figure["crop_rect"])
        # Anything PyMuPDF does not recognise as a rectangle renders the
        # whole 612 x 792 page instead of the crop.
        assert (pix.width, pix.height) == (224, 185)
