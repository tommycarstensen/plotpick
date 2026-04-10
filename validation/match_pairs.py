"""Match table-figure pairs and extract figure images from PDFs.

Parses cross-ref snippets from candidates.csv to identify which Table N and
Figure M are paired, extracts the figure as PNG, and writes pairs.json.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdf_figures import find_figures_on_page

HERE = Path(__file__).resolve().parent
PDF_DIR = Path.home() / "camilla2026_local" / "pdfs"
TABLE_DIR = HERE / "tables"
FIG_DIR = Path.home() / "camilla2026_local" / "figures"
PAIRS_PATH = HERE / "pairs.json"
SECRETS_PATH = HERE.parent / ".streamlit" / "secrets.toml"

SCREEN_SYSTEM = """\
You are a scientific literature expert. Judge whether a figure and a table in \
the same paper present the same numerical data. Answer with JSON only:
{"match": true or false, "reason": "one sentence"}
"match" is true if both show the same measured values. False if they clearly \
present different quantities (e.g. figure shows survival rates but table shows \
a risk formula, or figure is a schematic). When in doubt, answer true.\
"""

CLASSIFY_SYSTEM = """\
You classify scientific figures. Given a figure image, respond with JSON only:
{"plot_type": "<type>", "is_life_science": true/false, "exclude": true/false, "reason": "one sentence"}

Assign one plot_type from: bar_chart, line_chart, scatter_plot, box_plot, \
violin_plot, forest_plot, kaplan_meier, roc_curve, heatmap, histogram, \
funnel_plot, bland_altman, path_diagram, confusion_matrix, other_2d, \
3d_plot, flowchart, schematic, photograph, table_as_figure, text, other.

Set "is_life_science" to true if the content is clearly biomedical, clinical, \
or biological (e.g. patient data, biomarkers, survival, genomics, epidemiology). \
False for engineering, physics, materials science, computer science benchmarks, etc.

Set "exclude" to true if any of these apply:
- 3D plots (surface, contour, scatter)
- Photographs or microscopy images without extractable numeric data
- Schematics or diagrams without numeric data
- Flowcharts (CONSORT, cohort selection, etc.)
- Text passages that are not actual figures
- Not life-science content
Otherwise false.\
"""

# Plot types that are always excluded
EXCLUDED_PLOT_TYPES = {
    "3d_plot", "flowchart", "schematic", "photograph",
    "table_as_figure", "text", "other",
}

# Target quotas per plot type — 100 per type for statistical power.
# Set lower for rare types that are hard to find automatically.
PLOT_TYPE_QUOTAS = {
    "forest_plot": 100,
    "bar_chart": 100,
    "line_chart": 100,
    "kaplan_meier": 100,
    "box_plot": 100,
    "scatter_plot": 100,
    "violin_plot": 100,
    "histogram": 100,
    "funnel_plot": 100,
}
TOTAL_TARGET = 900


def _load_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if SECRETS_PATH.exists():
        for line in SECRETS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def classify_figure(client: anthropic.Anthropic, img: Image.Image) -> dict:
    """Classify a figure image and decide whether to include it."""
    import base64
    import io
    buf = io.BytesIO()
    img_resized = img.copy()
    img_resized.thumbnail((512, 512))
    img_resized.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png", "data": b64}},
                {"type": "text", "text": "Classify this figure."},
            ]}],
        )
        text = resp.content[0].text.strip().strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        return json.loads(text)
    except Exception:
        return {"plot_type": "unknown", "is_life_science": True, "exclude": False}


def screen_pair(client: anthropic.Anthropic, cross_ref: str, table: dict,
                table_num: int, figure_num: int) -> bool:
    prompt = (
        f"Cross-reference: \"{cross_ref}\"\n"
        f"Table label: {table.get('label', '')}\n"
        f"Table caption: {table.get('caption', '')[:200]}\n"
        f"Table headers: {table.get('headers', [])}\n"
        f"Figure {figure_num} vs Table {table_num}: same numerical data?"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=SCREEN_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip().strip("```").strip()
        return bool(json.loads(text).get("match", True))
    except Exception:
        return True  # keep on error

PAIR_RE = re.compile(
    r"Table\s+(\d+).*?Figure\s+(\d+)|Figure\s+(\d+).*?Table\s+(\d+)", re.I,
)


def parse_pair(snippet: str) -> tuple[int, int] | None:
    """Return (table_num, figure_num) or None."""
    m = PAIR_RE.search(snippet)
    if not m:
        return None
    if m.group(1):
        return int(m.group(1)), int(m.group(2))
    return int(m.group(4)), int(m.group(3))


def extract_figure(pdf_path: Path, figure_num: int) -> Image.Image | None:
    """Crop a numbered figure from a PDF using caption-based detection."""
    doc = pymupdf.open(str(pdf_path))
    mat = pymupdf.Matrix(300 / 72, 300 / 72)
    target = f"Fig_{figure_num}"
    for page in doc:
        for elem in find_figures_on_page(page):
            if elem["label"] == target:
                pix = page.get_pixmap(matrix=mat, clip=elem["crop_rect"])
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                doc.close()
                return img
    doc.close()
    return None


def fetch_figure_from_pmc(pmcid: str, figure_num: int) -> Image.Image | None:
    """Fetch a figure image from PMC by scraping the full article page.

    Finds all <figure> blocks, matches by caption text (e.g. "Figure 3",
    "Fig. 3"), and downloads the CDN image.  Falls back to the per-figure
    URL pattern if the article page has no <figure> tags.
    """
    import io
    import requests

    _HDR = {"User-Agent": "Mozilla/5.0"}
    _CDN_RE = re.compile(
        r'src="(https://cdn\.ncbi\.nlm\.nih\.gov/[^"]*\.(?:jpg|png|gif))"'
    )
    _FIG_LABEL = re.compile(
        r"(?:Figure|Fig\.?)\s*" + str(figure_num) + r"\b", re.I
    )

    try:
        # --- Strategy 1: scrape full article HTML for <figure> blocks ---
        article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
        r = requests.get(article_url, timeout=30, headers=_HDR)
        if r.ok:
            figures = re.findall(
                r"<figure[^>]*>(.*?)</figure>", r.text, re.DOTALL
            )
            for fig_html in figures:
                if _FIG_LABEL.search(fig_html):
                    m = _CDN_RE.search(fig_html)
                    if m:
                        img_r = requests.get(m.group(1), timeout=30)
                        if img_r.ok:
                            return Image.open(
                                io.BytesIO(img_r.content)
                            ).convert("RGB")

        # --- Strategy 2: direct figure page URLs (publisher-dependent) ---
        for slug in [f"fig{figure_num}", f"fig-{figure_num}",
                     f"F{figure_num}", f"figure-{figure_num}",
                     f"fig0{figure_num}"]:
            url2 = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/figure/{slug}/"
            r2 = requests.get(url2, timeout=10, headers=_HDR)
            if r2.ok:
                m = _CDN_RE.search(r2.text)
                if m:
                    img_r = requests.get(m.group(1), timeout=30)
                    if img_r.ok:
                        return Image.open(
                            io.BytesIO(img_r.content)
                        ).convert("RGB")
    except Exception:
        pass
    return None


def find_table(tables: list[dict], num: int) -> dict | None:
    target = f"table {num}"
    for t in tables:
        label = t.get("label", "").strip().lower()
        if label == target:
            return t
        m = re.search(r"(\d+)", label)
        if m and int(m.group(1)) == num:
            return t
    return None


def has_numeric_data(table: dict, threshold: int = 3) -> bool:
    count = sum(
        1
        for row in table.get("rows", [])
        for cell in row
        if re.search(r"\d+\.?\d*", cell)
    )
    return count >= threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Process first N only")
    parser.add_argument("--no-screen", action="store_true",
                        help="Skip LLM pre-screen for figure-table correspondence")
    args = parser.parse_args()

    csv_path = HERE / "candidates.csv"
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found.")

    FIG_DIR.mkdir(exist_ok=True)

    api_key = _load_api_key()
    client = anthropic.Anthropic(api_key=api_key) if (api_key and not args.no_screen) else None

    with open(csv_path, encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    if args.limit:
        candidates = candidates[: args.limit]

    pairs: list[dict] = []
    if PAIRS_PATH.exists():
        pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    seen = {p["pmcid"] for p in pairs}

    # Count existing pairs by plot type for quota tracking
    type_counts: dict[str, int] = {}
    for p in pairs:
        pt = p.get("plot_type", "")
        if pt:
            type_counts[pt] = type_counts.get(pt, 0) + 1

    def quota_full(plot_type: str) -> bool:
        quota = PLOT_TYPE_QUOTAS.get(plot_type, 0)
        if quota == 0:
            # Types not in the quota table: accept up to 2 overflow to keep
            # diversity, but only if total target not reached
            return type_counts.get(plot_type, 0) >= 2
        return type_counts.get(plot_type, 0) >= quota

    def all_quotas_met() -> bool:
        total = sum(type_counts.values())
        if total >= TOTAL_TARGET:
            return True
        return all(
            type_counts.get(pt, 0) >= q
            for pt, q in PLOT_TYPE_QUOTAS.items() if q > 0
        )

    if all_quotas_met():
        total = sum(type_counts.values())
        print(f"All quotas met ({total} pairs). Nothing to do.")
        print("Current distribution:")
        for pt, n in sorted(type_counts.items(), key=lambda x: -x[1]):
            quota = PLOT_TYPE_QUOTAS.get(pt, 0)
            print(f"  {pt}: {n}/{quota}")
        return

    stats = {"matched": 0, "no_pair": 0, "no_pdf": 0, "no_fig": 0,
             "no_table": 0, "no_numeric": 0, "screened_out": 0,
             "excluded_plot_type": 0, "not_life_science": 0,
             "quota_full": 0}

    for row in candidates:
        pmcid = row["pmcid"]
        if pmcid in seen:
            stats["matched"] += 1
            continue

        pair = parse_pair(row.get("example_ref", ""))
        if not pair:
            stats["no_pair"] += 1
            continue
        table_num, figure_num = pair

        pdf_path = PDF_DIR / f"{pmcid}.pdf"
        table_path = TABLE_DIR / f"{pmcid}.json"
        has_pdf = pdf_path.exists()
        if not table_path.exists():
            stats["no_table"] += 1
            continue

        tables = json.loads(table_path.read_text(encoding="utf-8"))
        table = find_table(tables, table_num)
        if not table:
            stats["no_table"] += 1
            continue
        if not has_numeric_data(table):
            stats["no_numeric"] += 1
            continue

        fig_path = FIG_DIR / f"{pmcid}_Fig_{figure_num}.png"
        img = None
        if fig_path.exists():
            img = Image.open(fig_path)
        elif has_pdf:
            try:
                img = extract_figure(pdf_path, figure_num)
            except Exception as e:
                print(f"  figure error {pmcid}: {e}")
        # Fallback: fetch from PMC CDN if no PDF or PDF extraction failed
        if img is None:
            img = fetch_figure_from_pmc(pmcid, figure_num)
        if img is None:
            stats["no_fig"] += 1
            continue

        classification = {}
        if client is not None:
            classification = classify_figure(client, img)
            plot_type = classification.get("plot_type", "unknown")
            if plot_type in EXCLUDED_PLOT_TYPES or classification.get("exclude"):
                reason = classification.get("reason", plot_type)
                print(f"  [excluded] {pmcid}: {reason}")
                stats["excluded_plot_type"] += 1
                time.sleep(0.3)
                continue
            if not classification.get("is_life_science", True):
                print(f"  [not life-sci] {pmcid}: {classification.get('reason', '')}")
                stats["not_life_science"] += 1
                time.sleep(0.3)
                continue
            if quota_full(plot_type):
                print(f"  [quota full] {pmcid}: {plot_type} ({type_counts.get(plot_type, 0)} already)")
                stats["quota_full"] += 1
                time.sleep(0.3)
                continue

            cross_ref = row.get("example_ref", "")[:120]
            if not screen_pair(client, cross_ref, table, table_num, figure_num):
                print(f"  [screened out] {pmcid}: figure/table present different data")
                stats["screened_out"] += 1
                time.sleep(0.3)
                continue

        plot_type = classification.get("plot_type", "")
        img.save(str(fig_path))
        pairs.append({
            "pmcid": pmcid,
            "doi": row.get("doi", ""),
            "title": row.get("title", "")[:200],
            "table_num": table_num,
            "figure_num": figure_num,
            "figure_path": f"figures/{pmcid}_Fig_{figure_num}.png",
            "table_label": table.get("label", ""),
            "table_caption": table.get("caption", "")[:200],
            "table_headers": table.get("headers", []),
            "table_n_rows": len(table.get("rows", [])),
            "cross_ref": row.get("example_ref", "")[:120],
            "plot_type": plot_type,
        })
        type_counts[plot_type] = type_counts.get(plot_type, 0) + 1
        stats["matched"] += 1

        title = row.get("title", "")[:60].encode("ascii", "replace").decode()
        filled = sum(type_counts.values())
        print(f"  [{filled:3d}/{TOTAL_TARGET}] {pmcid} Table {table_num} <-> "
              f"Fig {figure_num} [{plot_type}]: {title}")

        if stats["matched"] % 20 == 0:
            PAIRS_PATH.write_text(
                json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8",
            )

        if all_quotas_met():
            print(f"\n*** All quotas met! ({filled} pairs) ***")
            break

    PAIRS_PATH.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    total = sum(type_counts.values())
    print(f"\nResults: {total} pairs -> {PAIRS_PATH}")
    print(f"  (new this run: {stats['matched']})")
    for key in ("no_pair", "no_pdf", "no_fig", "no_table", "no_numeric",
                "excluded_plot_type", "not_life_science", "quota_full",
                "screened_out"):
        if stats.get(key, 0):
            print(f"  {stats[key]} {key.replace('_', ' ')}")

    print("\nPlot type distribution:")
    for pt, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        quota = PLOT_TYPE_QUOTAS.get(pt, 0)
        status = "FULL" if n >= quota and quota > 0 else ""
        print(f"  {pt:20s}: {n:3d}/{quota:2d}  {status}")

    missing = {pt: q - type_counts.get(pt, 0)
               for pt, q in PLOT_TYPE_QUOTAS.items()
               if q > 0 and type_counts.get(pt, 0) < q}
    if missing:
        print(f"\nStill needed ({sum(missing.values())} pairs):")
        for pt, n in sorted(missing.items(), key=lambda x: -x[1]):
            print(f"  {pt}: {n} more")


if __name__ == "__main__":
    main()
