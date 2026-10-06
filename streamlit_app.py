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

import html
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anthropic
import pandas as pd
import requests
import streamlit as st

from exports import dataframe_to_excel, dataframe_to_latex, dataframe_to_r
from extraction import extract_from_image
from figure_images import (
    Figure,
    held_summary,
    pdf_to_figures,
    sync_uploads,
    with_distinct_labels,
)
from models import (
    DEFAULT_MODEL,
    MODEL_LABELS,
    MODELS,
    MODELS_BY_LABEL,
    ExtractionError,
    extract_blocked_reason,
    figure_summary,
    missing_key_message,
    resolve_api_key,
    shared_key_from_environment,
)
from pmc import download_pmc_pdf, parse_pubmed_ids, pmids_to_pmcids
from process_memory import log_memory

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
def _pmids_to_pmcids(pmids: list[str]) -> dict[str, str | None]:
    """Each ID's PMCID, or None.  PMCIDs are passed through unchanged."""
    result: dict[str, str | None] = {
        pid: pid.upper() for pid in pmids if pid.upper().startswith("PMC")
    }
    to_convert = [pid for pid in pmids if pid not in result]
    if not to_convert:
        return result
    try:
        result.update(pmids_to_pmcids(to_convert))
    except (requests.RequestException, ValueError) as exc:
        # Only network and JSON-decode failures are expected here.  Anything
        # else is a bug and must propagate rather than be reported to the user
        # as "no PMC record found", which is what a blanket handler did.
        st.warning(f"NCBI ID lookup failed ({exc}). Could not resolve: "
                   f"{', '.join(to_convert)}")
        for pid in to_convert:
            result.setdefault(pid, None)
    return {pid: result[pid] for pid in pmids}  # in the order they were entered


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


def _values_only(row: dict[str, Any]) -> dict[str, Any]:
    """An extracted row without the model's list of uncertain columns.

    The list is shown as highlighting and kept in the JSON export; in a table
    it would be a column of lists.
    """
    return {key: value for key, value in row.items() if key != "uncertain"}


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
    "upload_problems": {},  # file id -> what could not be read in that file
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
        st.session_state.upload_problems,
    )
    for uploaded in uploaded_files or []:
        for problem in st.session_state.upload_problems.get(uploaded.file_id, []):
            st.warning(problem)

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
        raw_ids, ignored = parse_pubmed_ids(pubmed_input)
        if ignored:
            st.warning(
                f"Not a PubMed ID or PMCID, so ignored: {', '.join(ignored)}"
            )
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
                    problems: list[str] = []
                    figures = pdf_to_figures(pdf_bytes, pmcid, problems)
                    for problem in problems:
                        st.write(f"\u26a0\ufe0f  {problem}")
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
            log_memory(f"fetching {len(raw_ids)} PubMed ID(s)", held_summary())

    st.session_state.all_images = with_distinct_labels(upload_figures + [
        figure
        for figures in st.session_state.pubmed_figures.values()
        for figure in figures
    ])

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
                "Numeric F1 by chart type, ChartX validation split "
                "(synthetic charts, 50 per type; one fewer for Haiku 4.5 on "
                "plain bar charts), with 95% intervals"
            ),
        )
        st.image(
            str(Path(__file__).parent / "assets" / "chartx_heatmap.png"),
            caption=(
                "Numeric F1 (%) by model and chart type. Orange outlines mark "
                "the six cells where a model scores below DePlot"
            ),
        )
        st.caption(
            "Of the nine models in these charts the app offers only Haiku "
            "4.5; its default, Sonnet 5.5, is not shown. "
            "Numeric F1 is the F1 between the numbers a model extracts and "
            "the numbers in the ground truth, matched within 5%; it does not "
            "check that a value sits in the right group. Across six ChartX "
            "chart types all nine vision-language models tested score above "
            "DePlot, a dedicated chart-to-table model, mostly because of "
            "box plots; the two weakest trail it on some chart types. On a "
            "second benchmark, a subset of PlotQA that is mostly horizontal "
            "bar charts, scored leniently on the best-matching series of "
            "each reply, DePlot scores 87.0%: two of six models are level "
            "with it or slightly above it and four fall below it, Haiku 4.5 "
            "at 70.2%. "
            "These are synthetic charts read with a two-sentence prompt: "
            "the benchmark did not test this app's own prompt, did not "
            "include Sonnet 5.5 or Opus 5.5, and says nothing yet about "
            "real biomedical figures. Check every extracted value against "
            "its figure."
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
    # Held in Session State from the start, so that a figure that raises
    # does not take the batch's earlier results, already paid for, with it.
    st.session_state.results = results

    with st.status(
        f"Extracting {total} figure(s) with {selected_model.short_name}...",
        expanded=True,
    ) as status:
        progress = st.progress(0)
        for i, figure in enumerate(images_to_extract):
            label = figure.label
            st.write(f"\U0001f50d  **[{i + 1}/{total}]** {label}")
            progress.progress((i + 1) / total)
            # Which model read the figure, and when (UTC), travel with the
            # result into the JSON export and every tabular export.
            provenance = {
                "model": model,
                "extracted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            try:
                result = extract_from_image(client, figure.png, model)
                results[label] = {**result, **provenance}
                n_rows = len(result.get("data", []))
                st.write(f"\u2705  {n_rows} row(s) extracted")
            except (json.JSONDecodeError, ExtractionError, anthropic.APIError) as exc:
                results[label] = {"error": str(exc), "data": [], **provenance}
                st.write(f"\u274c  Failed: {exc}")

        n_ok = sum(1 for r in results.values() if "error" not in r)
        n_err = len(results) - n_ok
        msg = f"Done -- {n_ok} figure(s) extracted successfully."
        if n_err:
            msg += f"  {n_err} failed."
        status.update(label=msg, state="complete", expanded=False)

    log_memory(f"extracting {total} figure(s)", held_summary())
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
        st.caption(
            "Every value below is a model's reading of the figure. Check each "
            "one against its figure before you use it."
        )
        with st.expander("What is the confidence score?"):
            st.markdown(
                "The confidence score (0-100) is the number the AI model "
                "itself reports for how precisely it thinks it read the "
                "figure. It is **not** a probability and has not been checked "
                "against true values, so it is not known whether it is higher "
                "for figures that were read correctly.\n\n"
                "The model was told to choose its number like this:\n\n"
                "| Range | What the model was told each range means |\n"
                "|-------|---------|\n"
                "| 95-100 | Crisp axes, clear ticks, values easy to read exactly |\n"
                "| 80-94 | Good axes but some interpolation between ticks needed |\n"
                "| 60-79 | Small figure, overlapping elements, or missing ticks |\n"
                "| < 60 | Largely guessing |\n\n"
                "Check every extracted value against the original figure, "
                "whatever the score, and do not skip a value because it is "
                "not highlighted. Cells highlighted in amber are values the "
                "model itself flagged as uncertain (e.g. overlapping boxes, "
                "blurry regions); the other cells are unverified too."
            )
        # Build a lookup from label -> PNG bytes for showing source figures
        _img_lookup: dict[str, bytes] = {
            figure.label: figure.preview for figure in st.session_state.all_images
        }

        for label, result in st.session_state.results.items():
            # The label is a file name and the summary is the model's reading
            # of a figure: both are escaped before they go into HTML.
            st.markdown(
                f'<h4 style="color:{LIGHT_BLUE};">{html.escape(label)}</h4>',
                unsafe_allow_html=True,
            )

            if "error" in result:
                # The error can quote the model's reply: shown as code, so
                # that none of it renders as Markdown.
                st.error("Extraction failed:")
                st.code(result["error"], language=None)
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
                fig_type = html.escape(summary["figure_type"])
                y_ax = html.escape(summary["y_axis"])
                scale = html.escape(summary["scale"])
                conf = html.escape(summary["confidence"])
                notes = summary["notes"]
                st.markdown(
                    f'<div style="background:{NAVY};border-left:4px solid {BLUE};'
                    f'padding:0.6rem 1rem;border-radius:4px;margin-bottom:0.5rem;'
                    f'font-size:0.9rem;color:{TEXT_LIGHT};'
                    f'display:flex;flex-wrap:wrap;gap:0.2rem 1rem;">'
                    f'<span><b>Type:</b> {fig_type}</span>'
                    f'<span><b>Y-axis:</b> {y_ax}</span>'
                    f'<span><b>Scale:</b> {scale}</span>'
                    f'<span><b>Confidence (model\'s own):</b> {conf}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if notes:
                    # The model's own words: plain text, never Markdown.
                    st.text(f"Notes: {notes}")

                # Data table with uncertain-value highlighting
                data_rows = result.get("data", [])
                if data_rows:
                    # Read the flags without removing them: this script runs
                    # again on every interaction, and rows stripped here would
                    # lose their highlighting on the next one.
                    uncertain_lists: list[list[str]] = [
                        row.get("uncertain") or [] for row in data_rows
                    ]
                    df = pd.DataFrame([_values_only(row) for row in data_rows])

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
                all_rows.append({
                    "source": label,
                    "model": result.get("model"),
                    "extracted_at": result.get("extracted_at"),
                    **_values_only(row),
                })

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
                st.download_button(
                    "\U0001f4e5 Download Excel",
                    data=dataframe_to_excel(combined),
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
                latex_text = dataframe_to_latex(combined)
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
                r_script = dataframe_to_r(combined)
                st.code(r_script, language="r")
                st.download_button(
                    "\U0001f4e5 Download R script",
                    data=r_script.encode("utf-8"),
                    file_name=f"plotpick_{timestamp}.R",
                    mime="text/plain",
                )
