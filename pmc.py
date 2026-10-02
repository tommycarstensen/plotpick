"""Fetch article PDFs from the PMC Open Access subset.

Standalone module with no Streamlit dependency -- usable from both the app
and from batch scripts.
"""

import re
import xml.etree.ElementTree as ET

import requests

# NCBI retired the OA web service (oa.fcgi) and the FTP article packages in
# August 2026.  The Open Access subset is now served only from the PMC Cloud
# Service, one folder per article version: PMC123.1/PMC123.1.pdf.
PMC_CLOUD_URL = "https://pmc-oa-opendata.s3.amazonaws.com/"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _versions(pmcid: str) -> list[int]:
    """Version numbers the Open Access subset holds for an article, newest first."""
    r = requests.get(
        PMC_CLOUD_URL,
        params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"},
        timeout=15,
    )
    r.raise_for_status()
    versions: list[int] = []
    root = ET.fromstring(r.content)
    for node in root.iterfind("s3:CommonPrefixes/s3:Prefix", _S3_NS):
        m = re.fullmatch(rf"{re.escape(pmcid)}\.(\d+)/", node.text or "")
        if m:
            versions.append(int(m.group(1)))
    return sorted(versions, reverse=True)


def download_pmc_pdf(pmcid: str) -> bytes | None:
    """Download the newest PDF of an article from PMC Open Access.

    Returns None when the article is not in the Open Access subset, or is
    there without a PDF.  Network and parse failures are raised
    (requests.RequestException, ET.ParseError) so that the caller can tell
    "not available" from "could not ask".
    """
    for version in _versions(pmcid):
        stem = f"{pmcid}.{version}"
        r = requests.get(f"{PMC_CLOUD_URL}{stem}/{stem}.pdf", timeout=120)
        if r.status_code in (403, 404):
            continue
        r.raise_for_status()
        return r.content
    return None
