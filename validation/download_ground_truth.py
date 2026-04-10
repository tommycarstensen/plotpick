"""Download PDFs and extract structured table data from PMC XML.

Reads candidates.csv; writes pdfs/<PMCID>.pdf and tables/<PMCID>.json.
"""

import argparse
import csv
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"
PDF_LISTS = ["oa_comm_use_file_list.csv", "oa_non_comm_use_pdf.csv"]
HERE = Path(__file__).resolve().parent
PDF_DIR = Path.home() / "camilla2026_local" / "pdfs"
TABLE_DIR = HERE / "tables"
PDF_INDEX_CACHE = HERE / ".pdf_index.json"
DELAY = 0.35

_REPLACE = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2212": "-", "\u00a0": " ",
})


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_REPLACE)).strip()


def cell_text(el: ET.Element) -> str:
    return normalise(ET.tostring(el, encoding="unicode", method="text"))


def parse_table(tw: ET.Element) -> dict:
    label_el = tw.find("label")
    label = cell_text(label_el) if label_el is not None else ""

    caption_el = tw.find("caption")
    caption = cell_text(caption_el) if caption_el is not None else ""

    table_el = tw.find(".//table")
    if table_el is None:
        return {"label": label, "caption": caption, "headers": [], "rows": []}

    headers: list[str] = []
    rows: list[list[str]] = []

    thead = table_el.find("thead")
    if thead is not None:
        for tr in thead.findall("tr"):
            headers = [cell_text(c) for c in (tr.findall("th") or tr.findall("td"))]

    for tbody in table_el.findall("tbody") or [table_el]:
        for tr in tbody.findall("tr"):
            cells = tr.findall("td") or tr.findall("th")
            row = [cell_text(c) for c in cells]
            if row != headers:
                rows.append(row)

    if not headers and rows:
        headers = rows.pop(0)

    return {"label": label, "caption": caption[:300], "headers": headers, "rows": rows}


def _build_pdf_index(pmcids: set[str]) -> dict[str, str]:
    """Build a PMCID -> FTP path lookup from PMC OA file lists.

    Streams the CSV files and only keeps entries matching our PMCIDs.
    """
    if PDF_INDEX_CACHE.exists():
        return json.loads(PDF_INDEX_CACHE.read_text(encoding="utf-8"))

    print("  Building PDF index from PMC FTP file lists (one-time)...")
    index: dict[str, str] = {}
    for csv_name in PDF_LISTS:
        url = FTP_BASE + csv_name
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or line.startswith("File"):
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                accession = parts[2].strip()
                if accession in pmcids:
                    ftp_path = parts[0].strip()
                    index[accession] = FTP_BASE + ftp_path
            r.close()
        except Exception as e:
            print(f"  warning: could not fetch {csv_name}: {e}")

    # Also check the main oa_file_list for tar.gz packages
    try:
        remaining = pmcids - set(index.keys())
        if remaining:
            r = requests.get(FTP_BASE + "oa_file_list.csv", timeout=120, stream=True)
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or line.startswith("File"):
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                accession = parts[2].strip()
                if accession in remaining and accession not in index:
                    ftp_path = parts[0].strip()
                    index[accession] = FTP_BASE + ftp_path
            r.close()
    except Exception as e:
        print(f"  warning: could not fetch oa_file_list.csv: {e}")

    PDF_INDEX_CACHE.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"  Index: {len(index)} PMCIDs mapped to FTP paths")
    return index


def _download_from_url(url: str, out: Path) -> bool:
    """Download a PDF from a direct URL or tar.gz package."""
    try:
        r = requests.get(url, timeout=120, allow_redirects=True)
        if not r.ok:
            return False
        if url.endswith(".tar.gz"):
            import io
            import tarfile
            with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".pdf"):
                        f = tar.extractfile(member)
                        if f:
                            out.write_bytes(f.read())
                            return True
        else:
            out.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def _download_via_oa_api(pmcid: str, out: Path) -> bool:
    """Download PDF via the PMC OA web service (no FTP index needed)."""
    oa_url = (
        f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        f"?id={pmcid}&format=pdf"
    )
    try:
        r = requests.get(oa_url, timeout=30)
        if not r.ok:
            return False
        import xml.etree.ElementTree as _ET
        root = _ET.fromstring(r.content)
        link = root.find(".//link[@format='pdf']")
        if link is None:
            # Try tgz fallback
            link = root.find(".//link[@format='tgz']")
        if link is None:
            return False
        href = link.get("href", "")
        if not href:
            return False
        # OA API returns FTP URLs; convert to HTTPS
        href = href.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
        return _download_from_url(href, out)
    except Exception:
        return False


def download_pdf(pmcid: str, out: Path, pdf_index: dict[str, str]) -> bool:
    if out.exists() and out.stat().st_size > 5000:
        return True
    # Try FTP index first
    url = pdf_index.get(pmcid)
    if url:
        if _download_from_url(url, out):
            return True
    # Fall back to OA web service API
    return _download_via_oa_api(pmcid, out)


def extract_tables(pmcid: str) -> list[dict]:
    numeric_id = pmcid.removeprefix("PMC")
    r = requests.get(
        f"{NCBI}/efetch.fcgi",
        params={"db": "pmc", "id": numeric_id, "retmode": "xml"},
        timeout=60,
    )
    r.raise_for_status()
    root = ET.fromstring(r.content)
    return [parse_table(tw) for tw in root.findall(".//table-wrap")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Process first N only")
    args = parser.parse_args()

    csv_path = HERE / "candidates.csv"
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found. Run find_validation_papers.py first.")

    PDF_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)

    with open(csv_path, encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    if args.limit:
        candidates = candidates[: args.limit]

    all_pmcids = {row["pmcid"] for row in candidates}
    pdf_index = _build_pdf_index(all_pmcids)

    n_pdf = n_tables = 0

    for i, row in enumerate(candidates):
        pmcid = row["pmcid"]
        pdf_path = PDF_DIR / f"{pmcid}.pdf"
        table_path = TABLE_DIR / f"{pmcid}.json"

        pdf_ok = download_pdf(pmcid, pdf_path, pdf_index)
        n_pdf += pdf_ok

        if not table_path.exists():
            try:
                tables = extract_tables(pmcid)
                table_path.write_text(
                    json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8",
                )
                n_tables += 1
            except Exception as e:
                print(f"  table error {pmcid}: {e}")
        else:
            n_tables += 1

        tag = "PDF+tables" if pdf_ok else "tables only"
        title = row.get("title", "")[:70].encode("ascii", "replace").decode()
        print(f"  [{i + 1:3d}/{len(candidates)}] {pmcid} ({tag}): {title}")
        time.sleep(DELAY)

    print(f"\nDone: {n_pdf} PDFs, {n_tables} table JSONs")


if __name__ == "__main__":
    main()
