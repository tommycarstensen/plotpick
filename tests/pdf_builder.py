"""Write small PDFs by hand, so PDF tests need no PDF library to make fixtures.

Coordinates here are raw PDF ones: points, origin at the bottom-left of the
MediaBox.  Text uses the built-in Helvetica, so nothing has to be embedded.
"""

from dataclasses import dataclass, field

PAGE_W = 612
PAGE_H = 792


@dataclass
class FormSpec:
    """A Form XObject: its content stream and its BBox."""

    content: str
    bbox: tuple[float, float, float, float]


@dataclass
class PageSpec:
    content: str
    width: float = PAGE_W
    height: float = PAGE_H
    rotate: int = 0
    cropbox: tuple[float, float, float, float] | None = None
    forms: dict[str, FormSpec] = field(default_factory=dict)


def text(x: float, y: float, lines: list[str], size: float = 10,
         leading: float = 12, *, sideways: bool = False, spacing: float = 0,
         font: str = "F1") -> str:
    """One text object; y is the baseline of the first line.

    sideways turns the text a quarter, to read bottom-up; spacing is extra
    space after every character, in points; font F2 is a copy of the
    Helvetica F1 whose letter A reads as U+1D746, outside the basic plane.
    """
    turn = "0 1 -1 0" if sideways else "1 0 0 1"
    ops = [f"BT /{font} {size} Tf {leading} TL {spacing} Tc {turn} {x} {y} Tm"]
    for i, line in enumerate(lines):
        if i:
            ops.append("T*")
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({escaped}) Tj")
    ops.append("ET")
    return " ".join(ops)


def image(x: float, y: float, width: float, height: float) -> str:
    """The shared 2 x 2 pixel image, drawn width x height with corner (x, y)."""
    return f"q {width} 0 0 {height} {x} {y} cm /Im1 Do Q"


def box(x: float, y: float, width: float, height: float) -> str:
    """A stroked rectangle with bottom-left corner (x, y)."""
    return f"{x} {y} {width} {height} re S"


def _stream(entries: str, data: bytes) -> bytes:
    head = f"<< {entries} /Length {len(data)} >>\nstream\n".encode()
    return head + data + b"\nendstream"


def build_pdf(pages: list[PageSpec]) -> bytes:
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")
    tree = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    mapping = add(_stream("", (
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
        b"/CMapName /Test def /CMapType 2 def "
        b"1 begincodespacerange <00> <FF> endcodespacerange "
        b"1 beginbfchar <41> <D835DF46> endbfchar "
        b"endcmap CMapName currentdict /CMap defineresource pop end end"
    )))
    mapped = add(
        f"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        f"/ToUnicode {mapping} 0 R >>".encode()
    )
    pixels = bytes([200, 30, 30, 30, 200, 30, 30, 30, 200, 220, 220, 40])
    picture = add(_stream(
        "/Type /XObject /Subtype /Image /Width 2 /Height 2 "
        "/ColorSpace /DeviceRGB /BitsPerComponent 8", pixels,
    ))
    resources = f"/Font << /F1 {font} 0 R /F2 {mapped} 0 R >>"

    kids: list[int] = []
    for spec in pages:
        xobjects = f"/Im1 {picture} 0 R"
        for name, form in spec.forms.items():
            x0, y0, x1, y1 = form.bbox
            number = add(_stream(
                f"/Type /XObject /Subtype /Form /BBox [{x0} {y0} {x1} {y1}] "
                f"/Resources << {resources} /XObject << {xobjects} >> >>",
                form.content.encode(),
            ))
            xobjects += f" /{name} {number} 0 R"
        content = add(_stream("", spec.content.encode()))
        extra = f" /Rotate {spec.rotate}" if spec.rotate else ""
        if spec.cropbox:
            extra += " /CropBox [{} {} {} {}]".format(*spec.cropbox)
        kids.append(add((
            f"<< /Type /Page /Parent {tree} 0 R "
            f"/MediaBox [0 0 {spec.width} {spec.height}]{extra} "
            f"/Contents {content} 0 R "
            f"/Resources << {resources} /XObject << {xobjects} >> >> >>"
        ).encode()))

    objects[catalog - 1] = f"<< /Type /Catalog /Pages {tree} 0 R >>".encode()
    refs = " ".join(f"{k} 0 R" for k in kids)
    objects[tree - 1] = (
        f"<< /Type /Pages /Kids [{refs}] /Count {len(kids)} >>".encode()
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)
