"""
AutoExtract -- AI-powered batch extraction of data from graph images.

Upload images, PDFs, or ZIP archives.  Each figure is sent to Claude's
vision API with a structured extraction prompt.  Results are displayed
as tables and can be exported in multiple formats.

Run with:  streamlit run streamlit_app.py

Requires an Anthropic API key in .streamlit/secrets.toml:
    ANTHROPIC_API_KEY = "sk-ant-..."
"""

import base64
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import anthropic
import pandas as pd
import pymupdf
import streamlit as st
import streamlit.components.v1
from PIL import Image

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# ---------------------------------------------------------------------------
# Region Hovedstaden palette
# ---------------------------------------------------------------------------
NAVY = "#002555"
BLUE = "#007dbb"
LIGHT_BLUE = "#ccd3dd"
TEXT_LIGHT = "#e5e9ee"

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
)
ACCEPTED_TYPES: list[str] = [
    "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "pdf", "zip",
]

# ---------------------------------------------------------------------------
# Extraction prompt (derived from figure_extraction_instructions.md)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert at reading scientific figures.  You extract numerical
data from boxplots, bar charts, and line plots with high accuracy.

Follow these rules strictly:

1. INVENTORY first: identify figure type, axis labels, scale (linear or
   log), legend, number of groups/subplots, and any significance markers.
   For multi-panel figures, list each panel/subplot separately.
2. Define the output columns BEFORE extracting values:
   - Boxplots: biomarker, group, timepoint, median, q1, q3
   - Bar charts: biomarker, group, timepoint, n, mean, error, error_type
   - Line plots: biomarker, group, timepoint, mean_or_median, error, error_type
   Include ALL columns even if some are null.  Use "timepoint" for any
   time-based grouping (Baseline, Week 6, etc.).
3. Extract EVERY box / bar / point in the figure.  Go subplot by subplot,
   left-to-right within each subplot, reading all groups and timepoints.
   Interpolate numeric values from the nearest axis ticks.
4. The "data" array MUST contain one object per box / bar / data point.
   NEVER return an empty "data" array -- if you can see boxes or bars,
   you MUST extract values even if approximate.  Use null for any single
   value you truly cannot read, but still include the row.
5. Every numeric field must be a JSON number (not a string).
6. Include a top-level "confidence" field (0-100) reflecting how
   precisely you could read the values.  Use the FULL range:
   - 95-100: crisp axes, clear ticks, easy to read exactly
   - 80-94: good axes but some interpolation needed
   - 60-79: small figure, overlapping elements, or missing ticks
   - below 60: largely guessing
   Report the ACTUAL precision, not a round number.
7. Include a "notes" string listing any specific values that were hard
   to read and explaining why.

Example -- multi-panel boxplot with timepoints:
{
  "figure_type": "boxplot",
  "y_axis": "various (see per-row biomarker)",
  "scale": "linear",
  "confidence": 76,
  "notes": "IL-6 Week 12 boxes overlap; Q1/Q3 approximate",
  "data": [
    {"biomarker": "IL-6", "group": "Responders", "timepoint": "Baseline", "median": 9.5, "q1": 7.0, "q3": 12.0},
    {"biomarker": "IL-6", "group": "Responders", "timepoint": "Week 6", "median": 8.0, "q1": 6.5, "q3": 10.5},
    {"biomarker": "IL-6", "group": "Non-Responders", "timepoint": "Baseline", "median": 10.0, "q1": 8.0, "q3": 13.0},
    {"biomarker": "CRP", "group": "Responders", "timepoint": "Baseline", "median": 3.2, "q1": 2.5, "q3": 4.0}
  ]
}

IMPORTANT: Return ONLY valid JSON. No text before or after the JSON object.
The "data" array must NEVER be empty if the figure contains any visual elements.
"""

USER_PROMPT = """\
Extract ALL numerical data from this figure.  There should be one row
per box/bar/point.  Return structured JSON only.
"""


# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoExtract",
    page_icon="\U0001f916",
    layout="wide",
)

_CSS_PATH = Path(__file__).parent / "style.css"
st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers -- file processing
# ---------------------------------------------------------------------------
MAX_API_WIDTH = 2000  # Max width for API images (balance quality vs tokens)


def _resize_for_api(img: Image.Image) -> Image.Image:
    """Scale image down to MAX_API_WIDTH for the API, preserving aspect ratio."""
    if img.width > MAX_API_WIDTH:
        ratio = MAX_API_WIDTH / img.width
        new_size = (MAX_API_WIDTH, int(img.height * ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img


_CAPTION_RE = re.compile(
    r"^(Supplementary\s+)?Fig(ure|\.)\s*\d"
    r"|^(Supplementary\s+)?Table\s*\d"
    r"|^Suppl\.?\s+Fig",
    re.IGNORECASE,
)
_MIN_IMG_DIM = 50
_H_MARGIN = 20   # horizontal padding
_V_MARGIN = 6    # vertical padding (tight to avoid headers/body text)


def _label_from_caption(text: str) -> str | None:
    """Turn caption text into a short label, e.g. 'Fig_2', 'Table_1'."""
    m = re.match(
        r"(Supplementary\s+|Suppl\.?\s+)?(Fig(?:ure|\.)?|Table)\s*(\d+)",
        text.strip(), re.IGNORECASE,
    )
    if not m:
        return None
    prefix = "Suppl_" if m.group(1) else ""
    kind = "Fig" if "fig" in m.group(2).lower() else "Table"
    return f"{prefix}{kind}_{m.group(3)}"


def _is_two_column(text_blocks: list[dict[str, Any]], pw: float) -> bool:
    """Detect whether a page uses a two-column layout."""
    left = right = 0
    for tb in text_blocks:
        w = tb["bbox"].x1 - tb["bbox"].x0
        if w > pw * 0.55 or w < 30:
            continue
        if len(tb["text"]) < 10:
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
    """Return (col_left, col_right) limits for the caption's column."""
    if not two_col:
        return 0, pw
    mid = (cap_x0 + cap_x1) / 2
    center = pw * 0.5
    if mid < pw * 0.4:
        return 0, center + 4
    elif mid > pw * 0.6:
        return center - 4, pw
    return 0, pw


def _padded_rect(
    x0: float, y0: float, x1: float, y1: float, pw: float, ph: float,
) -> pymupdf.Rect:
    """Create a Rect with padding, clamped to page bounds."""
    return pymupdf.Rect(
        max(0, x0 - _H_MARGIN),
        max(0, y0 - _V_MARGIN),
        min(pw, x1 + _H_MARGIN),
        min(ph, y1 + _V_MARGIN),
    )


def _find_figures_on_page(
    page: Any,
) -> list[dict[str, Any]]:
    """Detect figures/tables on a PDF page via caption text.

    Returns list of dicts with keys: label, caption, crop_rect.
    """
    pw, ph = page.rect.width, page.rect.height

    # Collect text blocks
    text_blocks: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        full = ""
        for line in block["lines"]:
            full += "".join(span["text"] for span in line["spans"])
        text_blocks.append({
            "text": full.strip(),
            "bbox": pymupdf.Rect(block["bbox"]),
        })

    # Find captions
    captions: list[dict[str, Any]] = []
    for tb in text_blocks:
        if not _CAPTION_RE.match(tb["text"]):
            continue
        label = _label_from_caption(tb["text"])
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

    # Embedded images
    img_rects = [
        pymupdf.Rect(info["bbox"])
        for info in page.get_image_info(xrefs=True)
        if (info["bbox"][2] - info["bbox"][0] > _MIN_IMG_DIM
            and info["bbox"][3] - info["bbox"][1] > _MIN_IMG_DIM)
    ]
    drawings = page.get_drawings()
    two_col = _is_two_column(text_blocks, pw)

    def _same_column(a: dict, b: dict) -> bool:
        if not two_col:
            return True
        a_mid = (a["bbox"].x0 + a["bbox"].x1) / 2
        b_mid = (b["bbox"].x0 + b["bbox"].x1) / 2
        return (a_mid < pw * 0.5) == (b_mid < pw * 0.5)

    elements: list[dict[str, Any]] = []
    for i, cap in enumerate(captions):
        # Vertical neighbours in the same column
        prev_y = 0
        for j in range(i - 1, -1, -1):
            if _same_column(cap, captions[j]):
                prev_y = captions[j]["bbox"].y1
                break
        next_y = ph
        for j in range(i + 1, len(captions)):
            if _same_column(cap, captions[j]):
                next_y = captions[j]["bbox"].y0
                break

        cap_x0, cap_x1 = cap["bbox"].x0, cap["bbox"].x1
        cap_y0, cap_y1 = cap["bbox"].y0, cap["bbox"].y1
        col_left, col_right = _column_bounds(cap_x0, cap_x1, pw, two_col)

        if cap["type"] == "figure":
            # Raster images between same-column neighbours
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
                # Vector drawings in the caption's column region
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

            # Include the caption bbox
            x0 = min(x0, cap_x0)
            y0 = min(y0, cap_y0)
            x1 = max(x1, cap_x1)
            y1 = max(y1, cap_y1)

            # Clamp vertically to same-column neighbours
            y0 = max(y0, prev_y)

            crop = _padded_rect(x0, y0, x1, y1, pw, ph)
            crop.x0 = max(crop.x0, col_left)
            crop.x1 = min(crop.x1, col_right, x1 + 4)
        else:
            # Table: find bottom by detecting gap after table rows.
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
                    if max_gap is None:
                        max_gap = gap
                    else:
                        max_gap = max(max_gap, gap)
                    bottom = max(bottom, tb["bbox"].y1)
                    lx0 = min(lx0, tb["bbox"].x0)
                    lx1 = max(lx1, tb["bbox"].x1)
                return bottom, lx0, lx1

            below = sorted(
                [tb for tb in text_blocks
                 if tb["bbox"].y0 >= cap_y1 - 2
                 and tb["bbox"].y0 < next_y
                 and abs((tb["bbox"].x0 + tb["bbox"].x1) / 2 - cap_mid_x)
                 < cap_half_w],
                key=lambda tb: tb["bbox"].y0,
            )
            table_bottom, tx0, tx1 = _scan(below)
            if table_bottom - cap_y1 < 20:
                below_all = sorted(
                    [tb for tb in text_blocks
                     if tb["bbox"].y0 >= cap_y1 - 2
                     and tb["bbox"].y0 < next_y],
                    key=lambda tb: tb["bbox"].y0,
                )
                table_bottom, tx0, tx1 = _scan(below_all)

            crop = _padded_rect(tx0, cap_y0, tx1, table_bottom, pw, ph)
            # Column clamp -- skip if table is full-width
            if (tx1 - tx0) < pw * 0.55:
                crop.x0 = max(crop.x0, col_left)
                crop.x1 = min(crop.x1, col_right)

        elements.append({
            "label": cap["label"],
            "caption": cap["text"][:80],
            "crop_rect": crop,
        })
    return elements


def _pdf_to_images(data: bytes, name: str) -> list[tuple[str, Image.Image]]:
    """Extract individual figures from a PDF.

    Uses caption detection to crop each figure/table separately.
    Falls back to full-page rendering when no captions are found.
    """
    results: list[tuple[str, Image.Image]] = []
    doc = pymupdf.open(stream=data, filetype="pdf")
    mat = pymupdf.Matrix(300 / 72, 300 / 72)  # 300 DPI for better readability

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        elements = _find_figures_on_page(page)

        if elements:
            for elem in elements:
                pix = page.get_pixmap(matrix=mat, clip=elem["crop_rect"])
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                label = f"{name} p.{page_idx + 1} {elem['label']}"
                results.append((label, img))
        else:
            # No figures detected -- render full page as fallback
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            results.append((f"{name} p.{page_idx + 1}", img))

    doc.close()
    return results


def _process_zip(
    data: bytes,
    zip_name: str,
) -> list[tuple[str, Image.Image]]:
    """Extract images and PDFs from a ZIP archive."""
    results: list[tuple[str, Image.Image]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for entry in sorted(zf.namelist()):
            suffix = PurePosixPath(entry).suffix.lower()
            label = f"{zip_name}/{entry}"
            if suffix == ".pdf":
                results.extend(_pdf_to_images(zf.read(entry), label))
            elif suffix in IMAGE_EXTENSIONS:
                img = Image.open(io.BytesIO(zf.read(entry))).convert("RGB")
                results.append((label, img))
    return results


def _process_upload(
    uploaded: "UploadedFile",
) -> list[tuple[str, Image.Image]]:
    """Route a single uploaded file to the correct processor."""
    name: str = uploaded.name
    data: bytes = uploaded.read()
    suffix = PurePosixPath(name).suffix.lower()

    if suffix == ".zip":
        return _process_zip(data, name)
    if suffix == ".pdf":
        return _pdf_to_images(data, name)
    if suffix in IMAGE_EXTENSIONS:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return [(name, img)]
    return []


def _image_to_base64(img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string for the API."""
    img = _resize_for_api(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Helpers -- Claude API
# ---------------------------------------------------------------------------
def _extract_from_image(
    client: anthropic.Anthropic,
    img: Image.Image,
    model: str,
) -> dict[str, Any]:
    """Send a single image to Claude and parse the JSON response."""
    b64 = _image_to_base64(img)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT,
                    },
                ],
            },
        ],
    )

    raw_text = response.content[0].text  # type: ignore[union-attr]

    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        # Attach a snippet of the raw response for debugging
        snippet = raw_text[:300] + ("..." if len(raw_text) > 300 else "")
        raise json.JSONDecodeError(
            f"Claude returned invalid JSON. Raw response (first 300 chars):\n"
            f"{snippet}\n\nOriginal error: {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc


def _dataframe_to_r(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to an R data.frame() assignment."""
    lines = [
        "# AutoExtract output -- source directly in R",
        f"# Generated {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "dat <- data.frame(",
    ]
    for col_idx, col in enumerate(df.columns):
        vals = df[col].tolist()
        if df[col].dtype == object:
            escaped = [
                "NA" if pd.isna(v) else f'"{str(v)}"' for v in vals
            ]
            vec = f"  {col} = c({', '.join(escaped)})"
        else:
            formatted = [
                "NA" if pd.isna(v) else str(v) for v in vals
            ]
            vec = f"  {col} = c({', '.join(formatted)})"
        vec += "," if col_idx < len(df.columns) - 1 else ""
        lines.append(vec)
    lines.append("  stringsAsFactors = FALSE")
    lines.append(")")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_SESSION_DEFAULTS: dict[str, object] = {
    "all_images": [],
    "results": {},
}
for _key, _default in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://avatars.githubusercontent.com/u/243538387?s=200&v=4",
        width=80,
    )
    st.markdown("### \U0001f916 AutoExtract")
    st.caption("Copenhagen Biological & Precision Psychiatry")

    # API key: user can type their own, otherwise fall back to secrets.toml
    default_key: str = ""
    try:
        default_key = st.secrets["ANTHROPIC_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass

    user_key: str = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="Leave blank to use default key" if default_key else "sk-ant-...",
        help="Paste your own sk-ant-... key, or leave blank to use the shared default.",
    )
    api_key: str = user_key or default_key

    _MODEL_OPTIONS: dict[str, str] = {
        "Sonnet 4.6 \U0001f4b2\U0001f4b2\U0001f4b2": "claude-sonnet-4-6",
        "Haiku 4.5 \U0001f4b2": "claude-haiku-4-5-20251001",
        "Opus 4.6 \U0001f4b2\U0001f4b2\U0001f4b2\U0001f4b2\U0001f4b2": "claude-opus-4-6",
    }
    model_label: str = st.selectbox(
        "Model",
        list(_MODEL_OPTIONS),
        index=0,
        help="Sonnet 4.6 is a good balance of speed and accuracy.",
    ) or list(_MODEL_OPTIONS)[0]
    model: str = _MODEL_OPTIONS[model_label]

    st.divider()

    uploaded_files: list["UploadedFile"] = st.file_uploader(  # type: ignore[assignment]
        "Upload files",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        help="Images, PDFs, or ZIP archives (may contain PDFs and images).",
    )

    if uploaded_files:
        all_imgs: list[tuple[str, Image.Image]] = []
        for uf in uploaded_files:
            all_imgs.extend(_process_upload(uf))
        if all_imgs:
            st.session_state.all_images = all_imgs

    # Read selection state from checkbox widget keys (updated by Streamlit
    # before the script reruns, so this is always current).
    n_loaded = len(st.session_state.all_images)
    selected_labels: set[str] = {
        lbl
        for i, (lbl, _) in enumerate(st.session_state.all_images)
        if st.session_state.get(f"sel_{i}", False)
    }
    n_selected = len(selected_labels)

    if n_loaded:
        st.caption(f"{n_loaded} image(s) loaded, {n_selected} selected")

    st.divider()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_all = st.button(
            "\U0001f680 Extract all",
            disabled=not api_key or n_loaded == 0,
            use_container_width=True,
        )
    with btn_col2:
        run_selected = st.button(
            "\U0001f3af Extract selected",
            disabled=not api_key or n_selected == 0,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    f'<h2 style="color:{TEXT_LIGHT}; margin-bottom:0;">AutoExtract</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:{LIGHT_BLUE}; font-size:0.95rem;">'
    "Upload figures, then click <b>Extract all</b>.  Each image is sent to "
    "Claude for automatic data extraction.</p>",
    unsafe_allow_html=True,
)

if not st.session_state.all_images:
    st.info("Upload files in the sidebar to get started.")
    st.stop()

# -- Determine which images to extract --------------------------------------
images_to_extract: list[tuple[str, Image.Image]] = []
if run_all:
    images_to_extract = list(st.session_state.all_images)
elif run_selected:
    images_to_extract = [
        (lbl, img) for lbl, img in st.session_state.all_images
        if lbl in selected_labels
    ]

if images_to_extract:
    client = anthropic.Anthropic(api_key=api_key)
    total = len(images_to_extract)
    results: dict[str, dict[str, Any]] = dict(st.session_state.results)

    with st.status(
        f"Extracting {total} figure(s) with {model_label.split(chr(0x1f4b2))[0].strip()}...",
        expanded=True,
    ) as status:
        progress = st.progress(0)
        for i, (label, img) in enumerate(images_to_extract):
            st.write(f"\U0001f50d  **[{i + 1}/{total}]** {label}")
            progress.progress((i + 1) / total)
            try:
                result = _extract_from_image(client, img, model)
                results[label] = result
                n_rows = len(result.get("data", []))
                st.write(f"\u2705  {n_rows} row(s) extracted")
            except (json.JSONDecodeError, anthropic.APIError) as exc:
                results[label] = {"error": str(exc), "data": []}
                st.write(f"\u274c  Failed: {exc}")

        n_ok = sum(1 for r in results.values() if "error" not in r)
        n_err = len(results) - n_ok
        msg = f"Done -- {n_ok} figure(s) extracted successfully."
        if n_err:
            msg += f"  {n_err} failed."
        status.update(label=msg, state="complete", expanded=False)

    st.session_state.results = results
    # Auto-switch to the Results tab via JS (Streamlit has no Python API
    # for programmatic tab selection).
    streamlit.components.v1.html(
        """<script>
        const tabs = window.parent.document.querySelectorAll(
            'button[data-baseweb="tab"]'
        );
        if (tabs.length > 1) tabs[1].click();
        </script>""",
        height=0,
    )

# -- Display results --------------------------------------------------------
tab_gallery, tab_results, tab_export = st.tabs(
    ["\U0001f5c2  Images", "\U0001f4cb  Results", "\U0001f4e5  Export"]
)

with tab_gallery:
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        if st.button("Select all"):
            for j in range(len(st.session_state.all_images)):
                st.session_state[f"sel_{j}"] = True
    with sel_col2:
        if st.button("Deselect all"):
            for j in range(len(st.session_state.all_images)):
                st.session_state[f"sel_{j}"] = False

    gallery_cols = st.columns(min(len(st.session_state.all_images), 3))
    for i, (label, img) in enumerate(st.session_state.all_images):
        with gallery_cols[i % 3]:
            st.checkbox(label, key=f"sel_{i}")
            st.image(img, caption=label, use_container_width=True)

with tab_results:
    if not st.session_state.results:
        st.info("Click 'Extract all' in the sidebar to run the AI extraction.")
    else:
        with st.expander("How is the confidence score calculated?"):
            st.markdown(
                "The confidence score (0--100) is the AI model's self-assessed "
                "estimate of how precisely it could read numeric values from the "
                "figure. It is **not** a validated accuracy metric.\n\n"
                "| Range | Meaning |\n"
                "|-------|---------|\n"
                "| 95--100 | Crisp axes, clear ticks, values easy to read exactly |\n"
                "| 80--94 | Good axes but some interpolation between ticks needed |\n"
                "| 60--79 | Small figure, overlapping elements, or missing ticks |\n"
                "| < 60 | Largely guessing -- consider manual verification |\n\n"
                "**Tip:** Always cross-check extracted values against the original "
                "figure, especially when confidence is below 80."
            )
        for label, result in st.session_state.results.items():
            st.markdown(
                f'<h4 style="color:{LIGHT_BLUE};">{label}</h4>',
                unsafe_allow_html=True,
            )

            if "error" in result:
                st.error(f"Extraction failed: {result['error']}")
                continue

            # Metadata as a compact inline summary
            fig_type = result.get("figure_type", "?").capitalize()
            y_ax = result.get("y_axis", "?")
            scale = result.get("scale", "?").capitalize()
            conf = result.get("confidence", "?")
            notes = result.get("notes", "")
            st.markdown(
                f'<div style="background:{NAVY};border-left:4px solid {BLUE};'
                f'padding:0.6rem 1rem;border-radius:4px;margin-bottom:0.5rem;'
                f'font-size:0.9rem;color:{TEXT_LIGHT};'
                f'display:flex;flex-wrap:wrap;gap:0.2rem 1rem;">'
                f'<span><b>Type:</b> {fig_type}</span>'
                f'<span><b>Y-axis:</b> {y_ax}</span>'
                f'<span><b>Scale:</b> {scale}</span>'
                f'<span><b>Confidence:</b> {conf}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if notes:
                st.caption(f"Notes: {notes}")

            # Data table
            data_rows = result.get("data", [])
            if data_rows:
                df = pd.DataFrame(data_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No data rows extracted.")

            st.divider()

with tab_export:
    if not st.session_state.results:
        st.info("No results to export yet.")
    else:
        # Combine all data rows into one dataframe
        all_rows: list[dict[str, Any]] = []
        for label, result in st.session_state.results.items():
            for row in result.get("data", []):
                all_rows.append({"source": label, **row})

        if not all_rows:
            st.warning("No data rows found across all extractions.")
        else:
            combined = pd.DataFrame(all_rows)
            st.dataframe(combined, use_container_width=True, hide_index=True)

            # Format picker (2 rows of 3 -- stacks on mobile via CSS)
            fmt_row1 = st.columns(3)
            with fmt_row1[0]:
                want_md = st.checkbox("Markdown", value=True)
            with fmt_row1[1]:
                want_xlsx = st.checkbox("Excel", value=True)
            with fmt_row1[2]:
                want_csv = st.checkbox("CSV", value=False)
            fmt_row2 = st.columns(3)
            with fmt_row2[0]:
                want_latex = st.checkbox("LaTeX", value=False)
            with fmt_row2[1]:
                want_json = st.checkbox("JSON", value=False)
            with fmt_row2[2]:
                want_r = st.checkbox("R script", value=False)

            timestamp = f"{datetime.now():%Y%m%d_%H%M}"

            if want_md:
                md_text = combined.to_markdown(index=False)
                st.code(md_text, language="markdown")
                st.download_button(
                    "\U0001f4e5 Download Markdown",
                    data=md_text.encode("utf-8"),
                    file_name=f"autoextract_{timestamp}.md",
                    mime="text/markdown",
                )

            if want_xlsx:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    combined.to_excel(
                        writer, index=False, sheet_name="All"
                    )
                    # One sheet per source
                    for label in combined["source"].unique():
                        sheet = label[:31]  # Excel sheet name limit
                        subset = combined[combined["source"] == label]
                        subset.to_excel(writer, index=False, sheet_name=sheet)
                st.download_button(
                    "\U0001f4e5 Download Excel",
                    data=buf.getvalue(),
                    file_name=f"autoextract_{timestamp}.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-"
                        "officedocument.spreadsheetml.sheet"
                    ),
                )

            if want_csv:
                csv_bytes = combined.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "\U0001f4e5 Download CSV",
                    data=csv_bytes,
                    file_name=f"autoextract_{timestamp}.csv",
                    mime="text/csv",
                )

            if want_latex:
                latex_text = combined.to_latex(index=False)
                st.code(latex_text, language="latex")
                st.download_button(
                    "\U0001f4e5 Download LaTeX",
                    data=latex_text.encode("utf-8"),
                    file_name=f"autoextract_{timestamp}.tex",
                    mime="text/plain",
                )

            if want_json:
                # Include full results with metadata
                full_json = json.dumps(
                    st.session_state.results, indent=2, ensure_ascii=False,
                )
                st.code(full_json, language="json")
                st.download_button(
                    "\U0001f4e5 Download JSON",
                    data=full_json.encode("utf-8"),
                    file_name=f"autoextract_{timestamp}.json",
                    mime="application/json",
                )

            if want_r:
                r_script = _dataframe_to_r(combined)
                st.code(r_script, language="r")
                st.download_button(
                    "\U0001f4e5 Download R script",
                    data=r_script.encode("utf-8"),
                    file_name=f"autoextract_{timestamp}.R",
                    mime="text/plain",
                )
