"""Detect and extract figures/tables from PDF pages using caption heuristics.

Standalone module with no Streamlit dependency -- usable from both the app
and from batch scripts.
"""

import re
from typing import Any

import pymupdf

CAPTION_RE = re.compile(
    r"^(Supplementary\s+)?Fig(ure|\.)\s*\d"
    r"|^(Supplementary\s+)?Table\s*\d"
    r"|^Suppl\.?\s+Fig",
    re.IGNORECASE,
)
MIN_IMG_DIM = 50
H_MARGIN = 20
V_MARGIN = 6


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
) -> pymupdf.Rect:
    return pymupdf.Rect(
        max(0, x0 - H_MARGIN), max(0, y0 - V_MARGIN),
        min(pw, x1 + H_MARGIN), min(ph, y1 + V_MARGIN),
    )


def find_figures_on_page(page: Any) -> list[dict[str, Any]]:
    """Detect figures/tables on a PDF page via caption text.

    Returns list of dicts with keys: label, caption, crop_rect.
    """
    pw, ph = page.rect.width, page.rect.height

    text_blocks: list[dict] = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        full = ""
        for line in block["lines"]:
            full += "".join(span["text"] for span in line["spans"])
        text_blocks.append({"text": full.strip(), "bbox": pymupdf.Rect(block["bbox"])})

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
        pymupdf.Rect(info["bbox"])
        for info in page.get_image_info(xrefs=True)
        if (info["bbox"][2] - info["bbox"][0] > MIN_IMG_DIM
            and info["bbox"][3] - info["bbox"][1] > MIN_IMG_DIM)
    ]
    drawings = page.get_drawings()
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
                    if (d["rect"].y0 >= prev_y - 10
                        and d["rect"].y1 <= cap_y1 + 10
                        and d["rect"].x0 >= cap_x0 - 60
                        and d["rect"].x1 <= cap_x1 + 60)
                ]
                if nearby:
                    x0 = min(d["rect"].x0 for d in nearby)
                    y0 = min(d["rect"].y0 for d in nearby)
                    x1 = max(d["rect"].x1 for d in nearby)
                    y1 = max(d["rect"].y1 for d in nearby)
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

            def _scan(candidates: list[dict]) -> tuple[float, float, float]:
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

        elements.append({"label": cap["label"], "caption": cap["text"][:80], "crop_rect": crop})
    return elements
