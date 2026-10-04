"""Tests for figure detection and rendering on PDFs read with pypdfium2.

The fixtures are PDFs written by hand in pdf_builder.py.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import pytest
from PIL import Image

import pdf_backend_pdfium
from pdf_figures import PdfError, PdfPage, Rect, find_figures, open_pdf
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


@contextmanager
def only_page(spec: PageSpec) -> Iterator[PdfPage]:
    """The single page of a one-page fixture, open for the with-block."""
    with open_pdf(build_pdf([spec])) as doc:
        pages = iter(doc)
        yield next(pages)


def detect(spec: PageSpec) -> dict[str, Rect]:
    with only_page(spec) as page:
        return {e["label"]: e["crop_rect"] for e in find_figures(page)}


def blocks(spec: PageSpec) -> list[str]:
    with only_page(spec) as page:
        return [b["text"] for b in page.text_blocks()]


def close_to(rect: Rect, expected: Iterable[float], tolerance: float = 2.5) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(rect, expected, strict=True))


def rgb(img: Image.Image, point: tuple[int, int]) -> tuple[int, ...]:
    pixel = img.getpixel(point)
    assert isinstance(pixel, tuple)
    return pixel


class TestDetection:
    def test_raster_figure_is_cropped_with_its_caption(self):
        found = detect(UPRIGHT)
        # Image spans x 100-300 and, from the top, y 142-292; the caption
        # baseline is 312 from the top.  20 pt margin left, 6 pt above.
        assert close_to(found["Fig_1"], (80, 136, 304, 320.5))

    def test_table_is_cropped_from_caption_to_last_row(self):
        found = detect(UPRIGHT)
        assert close_to(found["Table_1"], (52, 376, 212.6, 444.5))

    def test_a_caption_word_inside_a_paragraph_is_not_a_caption(self):
        """ "Figure 2 shows ..." starts a line, but not a block."""
        assert set(detect(UPRIGHT)) == {"Fig_1", "Table_1"}

    def test_a_paragraph_that_opens_with_a_label_is_not_a_caption(self):
        """An indented first line starts a block of its own, so the block
        starts with the label, as a caption does."""
        spec = PageSpec(
            f"{UPRIGHT.content} "
            + text(72, 300, ["The paragraph before ends here, on a full line."])
            + " " + text(84, 288, ["Table 1 summarizes the outcomes by arm, and",
                                   "the paragraph carries on below."])
        )
        with only_page(spec) as page:
            found = find_figures(page)
            starts = [b["text"][:18] for b in page.text_blocks()]
        assert "Table 1 summarizes" in starts
        assert [e["label"] for e in found] == ["Fig_1", "Table_1"]
        assert found[1]["caption"].startswith("Table 1. Outcomes")

    def test_vector_figure_is_found_from_its_drawings(self):
        spec = PageSpec(
            box(320, 200, 100, 120) + " " + box(340, 220, 40, 60)
            + " " + text(320, 180, ["Figure 3. A vector figure."])
        )
        # Boxes span, from the top, y 472-592; caption baseline at 612.
        assert close_to(detect(spec)["Fig_3"], (300, 465.5, 433.5, 620.5))

    def test_page_without_captions_gives_nothing(self):
        assert detect(PageSpec(BODY)) == {}


class TestTextBlocks:
    def test_lines_of_one_paragraph_form_one_block(self):
        (block,) = blocks(PageSpec(BODY))
        assert block.startswith("Body text introducing")
        assert block.endswith("closes the paragraph.")

    def test_a_gap_of_two_lines_separates_blocks(self):
        spec = PageSpec(text(72, 720, ["First paragraph."])
                        + " " + text(72, 700, ["Second paragraph."]))
        assert blocks(spec) == ["First paragraph.", "Second paragraph."]

    def test_an_indented_first_line_separates_blocks(self):
        spec = PageSpec(text(72, 720, ["End of one paragraph."])
                        + " " + text(84, 708, ["Indented start of the next."]))
        assert len(blocks(spec)) == 2

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
    def test_text_clipped_out_of_view_is_dropped(self, spec):
        """A cropped embedded figure must not leak its original caption."""
        assert blocks(spec) == ["Visible text inside the form."]
        assert detect(spec) == {}


class TestTextOrder:
    def test_sideways_caption_between_body_lines_is_read_whole(self):
        """A landscape table on a portrait page: its caption runs bottom-up
        in two pieces, and the first starts level with a body line drawn just
        before it.  PDFium's own reading order sorts that piece into the body
        line ("TABLE", the body line, then "2 | Sleep ...")."""
        spec = PageSpec(
            text(300, 100, ["low symptom severity, although differences"])
            + " " + text(60, 100, ["TABLE"], size=8, sideways=True)
            + " " + text(60, 131, ["2 | Sleep onset and wake-up time"], size=8,
                         sideways=True)
        )
        assert blocks(spec) == [
            "low symptom severity, although differences",
            "TABLE 2 | Sleep onset and wake-up time",
        ]
        assert "Table_2" in detect(spec)

    def test_characters_beyond_the_basic_plane_arrive_whole(self):
        """PDFium counts in UTF-16 units and hands such a character out as
        two surrogates, which cannot be encoded."""
        (block,) = blocks(PageSpec(text(72, 720, ["A = 1"], font="F2")))
        assert block.replace(" ", "") == "\U0001d746=1"
        block.encode("utf-8")


class TestWordSpacing:
    def test_letter_spaced_caption_is_still_a_caption(self):
        """Journals set FIGURE in spaced capitals; a fixed gap threshold
        reads that as "F I G U R E 1"."""
        spec = PageSpec(
            image(100, 500, 200, 150)
            + " " + text(100, 480, ["FIGURE 1"], spacing=2)
            + " " + text(190, 480, ["Distribution of cases by age."])
        )
        assert "Fig_1" in detect(spec)


class TestDrawings:
    def test_spiky_stroke_does_not_reach_beyond_its_points(self):
        """PDFium's own bounds add the mitre of every corner: here some
        250 pt past the tip of the spike at x = 300."""
        spec = PageSpec("5 w 100 M 100 300 m 300 301 l 100 302 l S")
        with only_page(spec) as page:
            (rect,) = page.drawing_rects()
        assert rect.x1 == pytest.approx(300, abs=1)
        assert rect.x0 == pytest.approx(100, abs=1)


class TestRendering:
    def test_clip_is_rendered_at_the_requested_resolution(self):
        with only_page(UPRIGHT) as page:
            img = page.render(Rect(100, 142, 300, 292), 144)
        assert img.mode == "RGB"
        assert abs(img.width - 400) <= 2 and abs(img.height - 300) <= 2
        # The clip is exactly the 2 x 2 test image: red top-left, no white.
        red, green, blue = rgb(img, (50, 50))
        assert red > 150 and green < 80 and blue < 80
        assert rgb(img, (350, 250)) != (255, 255, 255)

    def test_whole_page_is_rendered_without_a_clip(self):
        with only_page(UPRIGHT) as page:
            img = page.render(None, 72)
        assert abs(img.width - PAGE_W) <= 1 and abs(img.height - PAGE_H) <= 1
        assert rgb(img, (5, 5)) == (255, 255, 255)

    def test_every_page_is_visited_in_order(self):
        data = build_pdf([PageSpec(BODY), UPRIGHT, PageSpec(BODY)])
        with open_pdf(data) as doc:
            assert len(doc) == 3
            counts = [len(find_figures(page)) for page in doc]
        assert counts == [0, 2, 0]

    def test_a_path_opens_like_bytes(self, tmp_path):
        path = tmp_path / "sample.pdf"
        path.write_bytes(build_pdf([UPRIGHT]))
        with open_pdf(path) as doc:
            assert len(doc) == 1


class TestDisplayCoordinates:
    """Boxes follow the page as displayed: /Rotate and the CropBox applied."""

    def test_rotated_page_matches_the_upright_page(self):
        # A landscape page shown upright by /Rotate 90: its content is the
        # upright page turned a quarter, (x, y) -> (792 - y, x).
        rotated = PageSpec(
            f"0 1 -1 0 {PAGE_H} 0 cm {UPRIGHT.content}",
            width=PAGE_H, height=PAGE_W, rotate=90,
        )
        upright = detect(UPRIGHT)
        turned = detect(rotated)
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
            with only_page(spec) as page:
                assert (round(page.width), round(page.height)) == (PAGE_W, PAGE_H)
                images.append(page.render(clip, 72))
        assert images[0].size == images[1].size
        for point in ((30, 30), (170, 30), (30, 120), (170, 120)):
            a, b = rgb(images[0], point), rgb(images[1], point)
            assert all(abs(x - y) <= 40 for x, y in zip(a, b, strict=True)), point

    def test_cropbox_origin_is_subtracted(self):
        cropped = PageSpec(UPRIGHT.content, cropbox=(50, 60, 562, 742))
        upright = detect(UPRIGHT)
        shifted = detect(cropped)
        # 50 pt cut off the left, 792 - 742 = 50 pt off the top.
        for label, rect in upright.items():
            expected = (rect.x0 - 50, rect.y0 - 50, rect.x1 - 50, rect.y1 - 50)
            assert close_to(shifted[label], expected, tolerance=0.5)


class TestUnreadable:
    """What PDFium cannot read arrives as PdfError, whatever PDFium raised."""

    @pytest.mark.parametrize("data", [
        b"", b"hello, this is not a PDF", build_pdf([UPRIGHT])[:200],
    ], ids=["empty", "not-a-pdf", "truncated"])
    def test_a_file_that_is_not_a_readable_pdf(self, data):
        with pytest.raises(PdfError, match="Failed to load document"):
            open_pdf(data)

    @pytest.mark.parametrize("clip", [
        Rect(300, 100, 200, 400),     # inside out
        Rect(100, 100, 100.25, 150),  # a quarter point wide: under one pixel
        Rect(700, 900, 800, 1000),    # off the page
    ], ids=["inside-out", "thin", "off-page"])
    def test_a_clip_that_covers_no_pixel(self, clip):
        with only_page(UPRIGHT) as page, pytest.raises(PdfError, match="no pixel"):
            page.render(clip, 300)

    def test_a_thin_clip_that_still_covers_a_pixel_is_rendered(self):
        with only_page(UPRIGHT) as page:
            assert page.render(Rect(100, 100, 100.5, 150), 300).width >= 1

    def test_a_page_with_no_visible_area(self):
        """A CropBox outside the MediaBox leaves a page of 0 x 0 points."""
        spec = PageSpec(BODY, width=100, height=100, cropbox=(500, 500, 600, 600))
        with only_page(spec) as page:
            assert find_figures(page) == []
            with pytest.raises(PdfError, match="no pixel"):
                page.render(None, 300)


class TestRenderSize:
    def test_a_region_too_large_for_the_dpi_is_rendered_coarser(self, monkeypatch):
        """An A0 poster at 300 DPI would need 139 million pixels."""
        monkeypatch.setattr(pdf_backend_pdfium, "MAX_RENDER_PIXELS", 1_000_000)
        poster = PageSpec(box(100, 100, 2000, 3000), width=2384, height=3370)
        with only_page(poster) as page:
            img = page.render(None, 300)
            crop = page.render(Rect(0, 0, 2384, 1685), 300)
        assert 900_000 < img.width * img.height <= 1_010_000
        assert img.width / img.height == pytest.approx(2384 / 3370, rel=0.01)
        assert 900_000 < crop.width * crop.height <= 1_010_000

    def test_an_ordinary_page_is_rendered_at_the_dpi_asked_for(self):
        with only_page(UPRIGHT) as page:
            width, height = page.render(None, 300).size
        # 8.5 x 11 in; the height may round up by a pixel.
        assert width == 2550 and height in (3300, 3301)


class TestMisuse:
    def test_a_page_kept_past_its_turn_fails_loudly(self):
        with open_pdf(build_pdf([UPRIGHT, UPRIGHT])) as doc:
            pages = list(doc)
            with pytest.raises(RuntimeError, match="used after it was closed"):
                pages[0].text_blocks()

    def test_a_page_kept_past_its_document_fails_loudly(self):
        doc = open_pdf(build_pdf([UPRIGHT]))
        pages = iter(doc)
        page = next(pages)
        doc.close()
        for use in (page.text_blocks, page.image_rects, lambda: page.render(None, 72)):
            with pytest.raises(RuntimeError, match="used after it was closed"):
                use()
