"""
PlotPick -- AI-powered batch extraction of data from graph images.

Upload images, PDFs, ZIP archives, or enter PubMed IDs to fetch figures
from PMC Open Access.  Each figure is sent to Claude's vision API with
a structured extraction prompt.  Results are displayed as tables and can
be exported in multiple formats.

Run with:  streamlit run streamlit_app.py

Requires an Anthropic API key, supplied either as the ANTHROPIC_API_KEY
environment variable or in .streamlit/secrets.toml:
    ANTHROPIC_API_KEY = "sk-ant-..."
"""

import io
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic
import pandas as pd
import requests
import streamlit as st

from figure_images import Figure, image_to_base64, pdf_to_figures, sync_uploads
from models import (
    DEFAULT_MODEL,
    MODEL_LABELS,
    MODELS,
    MODELS_BY_LABEL,
    ExtractionError,
    extract_blocked_reason,
    figure_summary,
    missing_key_message,
    reply_text,
    resolve_api_key,
    shared_key_from_environment,
)
from pmc import download_pmc_pdf

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# ---------------------------------------------------------------------------
# Region Hovedstaden palette
# ---------------------------------------------------------------------------
NAVY = "#002555"
BLUE = "#007dbb"
LIGHT_BLUE = "#ccd3dd"
TEXT_LIGHT = "#e5e9ee"

ACCEPTED_TYPES: list[str] = [
    "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "pdf", "zip",
]

TAB_IMAGES = "\U0001f5c2  Images"
TAB_RESULTS = "\U0001f4cb  Results"
TAB_EXPORT = "\U0001f4e5  Export"

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
8. For EACH data row, include an "uncertain" field: a list of column
   names whose values you are unsure about (e.g. overlapping boxes,
   blurry region, ambiguous tick alignment, very small figure).
   Use an empty list [] when all values in that row are confident.

Example -- multi-panel boxplot with timepoints:
{
  "figure_type": "boxplot",
  "y_axis": "various (see per-row biomarker)",
  "scale": "linear",
  "confidence": 76,
  "notes": "IL-6 Week 12 boxes overlap; Q1/Q3 approximate",
  "data": [
    {"biomarker": "IL-6", "group": "Responders", "timepoint": "Baseline", "median": 9.5, "q1": 7.0, "q3": 12.0, "uncertain": []},
    {"biomarker": "IL-6", "group": "Responders", "timepoint": "Week 6", "median": 8.0, "q1": 6.5, "q3": 10.5, "uncertain": ["q1", "q3"]},
    {"biomarker": "IL-6", "group": "Non-Responders", "timepoint": "Baseline", "median": 10.0, "q1": 8.0, "q3": 13.0, "uncertain": []},
    {"biomarker": "CRP", "group": "Responders", "timepoint": "Baseline", "median": 3.2, "q1": 2.5, "q3": 4.0, "uncertain": ["median"]}
  ]
}

IMPORTANT: Return ONLY valid JSON. No text before or after the JSON object.
The "data" array must NEVER be empty if the figure contains any visual elements.
"""  # noqa: E501 -- example JSON rows; wrapping them would corrupt the example.

USER_PROMPT = """\
Extract ALL numerical data from this figure.  There should be one row
per box/bar/point.  Return structured JSON only.
"""


# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PlotPick",
    page_icon="\U0001f916",
    layout="wide",
)

_CSS_PATH = Path(__file__).parent / "style.css"
st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers -- PubMed / PMC
# ---------------------------------------------------------------------------
_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _parse_pubmed_ids(text: str) -> list[str]:
    """Extract PubMed IDs or PMCIDs from free-text input.

    Accepts: PMID (numeric), PMC + digits, or full PubMed/PMC URLs.
    Returns normalised IDs like '12345678' or 'PMC1234567'.
    """
    ids: list[str] = []
    for token in re.split(r"[,;\s]+", text.strip()):
        if not token:
            continue
        # Full URL: https://pubmed.ncbi.nlm.nih.gov/12345678/
        m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", token)
        if m:
            ids.append(m.group(1))
            continue
        # Full URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/
        m = re.search(r"pmc/articles/(PMC\d+)", token, re.IGNORECASE)
        if m:
            ids.append(m.group(1).upper())
            continue
        # Bare PMCID
        m = re.fullmatch(r"(PMC\d+)", token, re.IGNORECASE)
        if m:
            ids.append(m.group(1).upper())
            continue
        # Bare PMID (numeric)
        if re.fullmatch(r"\d{5,12}", token):
            ids.append(token)
            continue
    return ids


def _pmids_to_pmcids(pmids: list[str]) -> dict[str, str | None]:
    """Convert PMIDs to PMCIDs via NCBI ID Converter API.

    Returns {pmid: pmcid_or_None}.  PMCIDs are passed through unchanged.
    """
    result: dict[str, str | None] = {}
    to_convert: list[str] = []
    for pid in pmids:
        if pid.upper().startswith("PMC"):
            result[pid] = pid.upper()
        else:
            to_convert.append(pid)

    if not to_convert:
        return result

    try:
        r = requests.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={"ids": ",".join(to_convert), "format": "json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        for rec in data.get("records", []):
            pmid = rec.get("pmid", "")
            pmcid = rec.get("pmcid")
            if pmid in to_convert:
                result[pmid] = pmcid  # None if no PMC record
    except (requests.RequestException, ValueError) as exc:
        # Only network and JSON-decode failures are expected here.  Anything
        # else is a bug and must propagate rather than be reported to the user
        # as "no PMC record found", which is what a blanket handler did.
        st.warning(f"NCBI ID lookup failed ({exc}). Could not resolve: "
                   f"{', '.join(to_convert)}")
        for pid in to_convert:
            result.setdefault(pid, None)
    return result


def _download_pmc_pdf(pmcid: str) -> bytes | None:
    """Download a PDF from PMC Open Access. Returns bytes or None."""
    try:
        return download_pmc_pdf(pmcid)
    except (requests.RequestException, ET.ParseError) as exc:
        # As above: narrow to the failures this function can actually cause.
        # A blanket handler turned every bug into "PDF not available", which
        # is also how the retired oa.fcgi endpoint went unnoticed.
        st.warning(f"Download failed for {pmcid} ({exc}).")
        return None


# ---------------------------------------------------------------------------
# Helpers -- Claude API
# ---------------------------------------------------------------------------
def _extract_from_image(
    client: anthropic.Anthropic,
    png_bytes: bytes,
    model: str,
) -> dict[str, Any]:
    """Send a single image to Claude and parse the JSON response."""
    b64 = image_to_base64(png_bytes)

    response = client.messages.create(
        model=model,
        max_tokens=16384,
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

    raw_text = reply_text(response)

    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    text = text.removesuffix("```")
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
        "# PlotPick output -- source directly in R",
        f"# Generated {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "dat <- data.frame(",
    ]
    for col_idx, col in enumerate(df.columns):
        vals = df[col].tolist()
        if df[col].dtype == object:
            escaped = [
                "NA" if pd.isna(v) else f'"{v!s}"' for v in vals
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


def _one_type_per_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return the table as it is shown: a column mixing text and numbers as text.

    The model sometimes answers "Baseline" in one row and 6 in the next.
    st.dataframe() sends the table as Arrow, which takes one type per column;
    given such a column, Streamlit logs a traceback and then makes this same
    conversion itself.  Exports keep the values as the model gave them.
    """
    shown = df.copy()
    for name, column in df.items():
        cells = list(zip(column.tolist(), column.notna().tolist(), strict=True))
        if {isinstance(value, str) for value, present in cells if present} == {
            True, False,
        }:
            shown[name] = [str(value) if present else value for value, present in cells]
    return shown


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_SESSION_DEFAULTS: dict[str, object] = {
    "upload_figures": {},  # file id -> figures, see figure_images.sync_uploads
    "pubmed_figures": {},  # PMCID -> figures
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
    st.markdown("### \U0001f916 PlotPick")
    st.caption("Copenhagen Biological & Precision Psychiatry")

    # API key: the app owner's, from secrets.toml or else the
    # ANTHROPIC_API_KEY environment variable, pays for Sonnet and Haiku.
    # Opus runs only on a key the visitor pastes in (see models.py).
    #
    # load_if_toml_exists() reports a missing secrets file as False rather
    # than raising, and still raises on a *malformed* one -- so this reads
    # the optional file without a try/except that would also swallow a real
    # parse error.
    shared_key: str = ""
    if st.secrets.load_if_toml_exists() and "ANTHROPIC_API_KEY" in st.secrets:
        shared_key = str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    if not shared_key:
        shared_key = shared_key_from_environment()

    model_label: str = st.selectbox(
        "Model",
        MODEL_LABELS,
        index=0,
        help="\n\n".join(f"**{m.short_name}** -- {m.blurb}" for m in MODELS),
    ) or DEFAULT_MODEL.label
    selected_model = MODELS_BY_LABEL[model_label]
    model: str = selected_model.model_id
    st.caption(selected_model.blurb)

    # The key box exists only for bring-your-own-key models, so visitors
    # using Sonnet or Haiku are never asked for a key.
    user_key: str = ""
    if selected_model.needs_own_key:
        user_key = st.text_input(
            "Your Anthropic API key",
            type="password",
            placeholder="sk-ant-...",
            help=f"{selected_model.short_name} is billed to your own key.",
        )

    api_key = resolve_api_key(selected_model, user_key, shared_key)
    if not api_key:
        # Never leave the Extract buttons disabled without saying why.
        st.warning(missing_key_message(selected_model))

    st.divider()

    uploaded_files: list["UploadedFile"] = st.file_uploader(  # type: ignore[assignment]
        "Upload files",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        help="Images, PDFs, or ZIP archives (may contain PDFs and images).",
    )

    # The script reruns on every click, and the uploader hands over the same
    # files each time.  An upload is rendered once and kept until it is removed
    # from the uploader; rendering it again on every rerun costs tens of MB per
    # PDF page, which Community Cloud's memory limit does not allow for.
    upload_figures = sync_uploads(
        st.session_state.upload_figures, uploaded_files or [],
    )

    st.markdown("**-- or --**")

    pubmed_input: str = st.text_area(
        "PubMed IDs",
        placeholder="e.g. 38275943, PMC10817106\nor paste PubMed URLs",
        height=80,
        help="Enter PMIDs, PMCIDs, or PubMed URLs (one per line or comma-separated). "
        "PDFs will be fetched from PMC Open Access where available.",
    )
    fetch_pubmed = st.button(
        "\U0001f4e1 Fetch from PubMed",
        disabled=not pubmed_input.strip(),
    )

    if fetch_pubmed and pubmed_input.strip():
        raw_ids = _parse_pubmed_ids(pubmed_input)
        if not raw_ids:
            st.error("No valid PubMed IDs found in input.")
        else:
            with st.status(f"Fetching {len(raw_ids)} ID(s)...", expanded=True):
                mapping = _pmids_to_pmcids(raw_ids)
                for orig_id, pmcid in mapping.items():
                    if pmcid is None:
                        st.write(f"\u26a0\ufe0f  {orig_id} -- no PMC record found")
                        continue
                    st.write(f"\U0001f50d  {orig_id} -> {pmcid} -- downloading PDF...")
                    pdf_bytes = _download_pmc_pdf(pmcid)
                    if pdf_bytes is None:
                        st.write(
                            f"\u274c  {pmcid} -- PDF not available (not open access?)"
                        )
                        continue
                    figures = pdf_to_figures(pdf_bytes, pmcid)
                    if figures:
                        st.session_state.pubmed_figures[pmcid] = figures
                        st.write(
                            f"\u2705  {pmcid} -- {len(figures)} figure(s) extracted"
                        )
                    else:
                        st.write(
                            f"\u26a0\ufe0f  {pmcid} -- PDF downloaded but no figures "
                            "detected"
                        )

    st.session_state.all_images = upload_figures + [
        figure
        for figures in st.session_state.pubmed_figures.values()
        for figure in figures
    ]

    # Read selection state from checkbox widget keys (updated by Streamlit
    # before the script reruns, so this is always current).
    n_loaded = len(st.session_state.all_images)
    selected_labels: set[str] = {
        figure.label
        for i, figure in enumerate(st.session_state.all_images)
        if st.session_state.get(f"sel_{i}", False)
    }
    n_selected = len(selected_labels)

    if n_loaded:
        st.caption(f"{n_loaded} image(s) loaded, {n_selected} selected")

    st.divider()

    # A disabled button must always explain itself -- see issue #2, where the
    # only symptom was a not-allowed cursor and no message anywhere.
    _all_blocked = extract_blocked_reason(
        api_key, n_loaded, n_selected, selected_only=False
    )
    _sel_blocked = extract_blocked_reason(
        api_key, n_loaded, n_selected, selected_only=True
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_all = st.button(
            "\U0001f680 Extract all",
            disabled=_all_blocked is not None,
            help=_all_blocked or f"Extract all {n_loaded} figure(s).",
            width="stretch",
        )
    with btn_col2:
        run_selected = st.button(
            "\U0001f3af Extract selected",
            disabled=_sel_blocked is not None,
            help=_sel_blocked or f"Extract the {n_selected} selected figure(s).",
            width="stretch",
        )

    if _all_blocked and n_loaded:
        st.caption(f"\u26a0\ufe0f  {_all_blocked}")

    st.divider()

    with st.expander("Benchmarks"):
        st.image(
            str(Path(__file__).parent / "assets" / "chartx_by_type.png"),
            caption=(
                "Chart-to-table extraction accuracy by plot type (ChartX benchmark)"
            ),
        )
        st.image(
            str(Path(__file__).parent / "assets" / "heatmap_model_type.png"),
            caption="Recall (%) by model and chart type",
        )
        st.caption(
            "Recall scores on life-science figure types from ChartX. "
            "All eight VLMs from four providers outperform DePlot, "
            "a dedicated chart-to-table model. "
            "Labeled figures (with text annotations) are consistently "
            "easier for all models."
        )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    f'<h2 style="color:{TEXT_LIGHT}; margin-bottom:0;">PlotPick</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:{LIGHT_BLUE}; font-size:0.95rem;">'
    "Upload figures or enter PubMed IDs, then click <b>Extract all</b>.  "
    "Each image is sent to Claude for automatic data extraction.</p>",
    unsafe_allow_html=True,
)

if not st.session_state.all_images:
    st.info("Upload files or enter PubMed IDs in the sidebar to get started.")
    st.stop()

# -- Determine which images to extract --------------------------------------
images_to_extract: list[Figure] = []
if run_all:
    images_to_extract = list(st.session_state.all_images)
elif run_selected:
    images_to_extract = [
        figure for figure in st.session_state.all_images
        if figure.label in selected_labels
    ]


@st.cache_resource(max_entries=8)
def _get_client(key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=key)


if images_to_extract:
    client = _get_client(api_key)
    total = len(images_to_extract)
    results: dict[str, dict[str, Any]] = dict(st.session_state.results)

    with st.status(
        f"Extracting {total} figure(s) with {selected_model.short_name}...",
        expanded=True,
    ) as status:
        progress = st.progress(0)
        for i, figure in enumerate(images_to_extract):
            label = figure.label
            st.write(f"\U0001f50d  **[{i + 1}/{total}]** {label}")
            progress.progress((i + 1) / total)
            try:
                result = _extract_from_image(client, figure.png, model)
                results[label] = result
                n_rows = len(result.get("data", []))
                st.write(f"\u2705  {n_rows} row(s) extracted")
            except (json.JSONDecodeError, ExtractionError, anthropic.APIError) as exc:
                results[label] = {"error": str(exc), "data": []}
                st.write(f"\u274c  Failed: {exc}")

        n_ok = sum(1 for r in results.values() if "error" not in r)
        n_err = len(results) - n_ok
        msg = f"Done -- {n_ok} figure(s) extracted successfully."
        if n_err:
            msg += f"  {n_err} failed."
        status.update(label=msg, state="complete", expanded=False)

    st.session_state.results = results
    # Show the Results tab: the tabs below read their selection from this key.
    st.session_state.active_tab = TAB_RESULTS

# -- Display results --------------------------------------------------------
# on_change="rerun" is what makes the tabs keep their selection in Session
# State, so that an extraction can select Results.  It replaces a script that
# clicked the tab through st.components.v1.html, which Streamlit is removing.
tab_gallery, tab_results, tab_export = st.tabs(
    [TAB_IMAGES, TAB_RESULTS, TAB_EXPORT], key="active_tab", on_change="rerun",
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
    for i, figure in enumerate(st.session_state.all_images):
        with gallery_cols[i % 3]:
            st.checkbox(figure.label, key=f"sel_{i}")
            st.image(figure.preview, caption=figure.label, width="stretch")

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
                "figure, especially when confidence is below 80.\n\n"
                "Cells highlighted in amber are individual values the model "
                "flagged as uncertain (e.g. overlapping boxes, blurry regions)."
            )
        # Build a lookup from label -> PNG bytes for showing source figures
        _img_lookup: dict[str, bytes] = {
            figure.label: figure.preview for figure in st.session_state.all_images
        }

        for label, result in st.session_state.results.items():
            st.markdown(
                f'<h4 style="color:{LIGHT_BLUE};">{label}</h4>',
                unsafe_allow_html=True,
            )

            if "error" in result:
                st.error(f"Extraction failed: {result['error']}")
                continue

            # Two-column layout: source image | metadata + table
            # Columns stack vertically on mobile automatically
            col_img, col_data = st.columns([2, 3])

            with col_img:
                src_img = _img_lookup.get(label)
                if src_img is not None:
                    st.image(src_img, caption=label, width="stretch")

            with col_data:
                # Metadata as a compact inline summary
                summary = figure_summary(result)
                fig_type = summary["figure_type"]
                y_ax = summary["y_axis"]
                scale = summary["scale"]
                conf = summary["confidence"]
                notes = summary["notes"]
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

                # Data table with uncertain-value highlighting
                data_rows = result.get("data", [])
                if data_rows:
                    # Extract per-row uncertain lists before building the DF
                    uncertain_lists: list[list[str]] = [
                        row.pop("uncertain", []) or [] for row in data_rows
                    ]
                    df = pd.DataFrame(data_rows)

                    # Build a mask of cells that were flagged as uncertain
                    uncertain_mask = pd.DataFrame(
                        False, index=df.index, columns=df.columns,
                    )
                    for i, cols in enumerate(uncertain_lists):
                        for col in cols:
                            if col in uncertain_mask.columns:
                                uncertain_mask.at[i, col] = True

                    df = _one_type_per_column(df)
                    if uncertain_mask.any().any():
                        styled = df.style.apply(
                            lambda col, mask=uncertain_mask: [
                                "background-color: rgba(255, 170, 0, 0.25)"
                                if mask.at[idx, col.name] else ""
                                for idx in col.index
                            ],
                            axis=0,
                        )
                        st.dataframe(
                            styled, width="stretch", hide_index=True,
                        )
                    else:
                        st.dataframe(
                            df, width="stretch", hide_index=True,
                        )
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
            st.dataframe(
                _one_type_per_column(combined), width="stretch", hide_index=True,
            )

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
                    file_name=f"plotpick_{timestamp}.md",
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
                    file_name=f"plotpick_{timestamp}.xlsx",
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
                    file_name=f"plotpick_{timestamp}.csv",
                    mime="text/csv",
                )

            if want_latex:
                latex_text = combined.to_latex(index=False)
                st.code(latex_text, language="latex")
                st.download_button(
                    "\U0001f4e5 Download LaTeX",
                    data=latex_text.encode("utf-8"),
                    file_name=f"plotpick_{timestamp}.tex",
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
                    file_name=f"plotpick_{timestamp}.json",
                    mime="application/json",
                )

            if want_r:
                r_script = _dataframe_to_r(combined)
                st.code(r_script, language="r")
                st.download_button(
                    "\U0001f4e5 Download R script",
                    data=r_script.encode("utf-8"),
                    file_name=f"plotpick_{timestamp}.R",
                    mime="text/plain",
                )
