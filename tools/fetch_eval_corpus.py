"""Download the PDFs of the PMC validation papers, as a PDF-backend test corpus.

TEMPORARY: part of the PyMuPDF -> pypdfium2 evaluation on the pdfium-eval
branch; delete together with tools/compare_pdf_backends.py after the port.

    python tools/fetch_eval_corpus.py [--pairs PAIRS.json] [--out DIR]
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pmc import download_pmc_pdf  # noqa: E402

DATA = Path(os.environ.get("PLOTPICK_DATA", Path.home() / "plotpick_data"))
DEFAULT_OUT = DATA / "pdf_backend_eval" / "pmc"
DEFAULT_PAIRS = REPO.parent / "validation" / "pairs.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the PMC validation papers as a PDF test corpus."
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pause", type=float, default=0.3)
    args = parser.parse_args()

    pmcids = sorted({p["pmcid"] for p in json.loads(args.pairs.read_text())})
    args.out.mkdir(parents=True, exist_ok=True)
    got = skipped = missing = failed = 0
    for i, pmcid in enumerate(pmcids, 1):
        target = args.out / f"{pmcid}.pdf"
        if target.exists():
            skipped += 1
            continue
        try:
            data = download_pmc_pdf(pmcid)
        except (requests.RequestException, ET.ParseError) as exc:
            failed += 1
            print(f"[{i}/{len(pmcids)}] {pmcid} FAILED {exc}", flush=True)
            continue
        if data is None:
            missing += 1
            print(f"[{i}/{len(pmcids)}] {pmcid} no PDF in Open Access", flush=True)
            continue
        target.write_bytes(data)
        got += 1
        print(f"[{i}/{len(pmcids)}] {pmcid} {len(data) / 1e6:.1f} MB", flush=True)
        time.sleep(args.pause)
    print(f"downloaded={got} already_there={skipped} no_pdf={missing} "
          f"failed={failed} -> {args.out}")


if __name__ == "__main__":
    main()
