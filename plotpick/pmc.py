"""Read PubMed identifiers and fetch article PDFs from the PMC Open Access subset.

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

# NCBI's ID Converter moved here from www.ncbi.nlm.nih.gov/pmc/utils/idconv/
# in 2026; the old address redirects.  It takes at most 200 IDs a request.
IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
_IDCONV_BATCH = 200


# Article URLs: PMC (current and old address) and Europe PMC name the PMCID;
# PubMed (current and old address) and Europe PMC name the PMID.
_URL_PMCID = re.compile(r"(PMC\d+)", re.IGNORECASE)
_URL_PMID = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/"
    r"|europepmc\.org/(?:article|abstract)/MED/)(\d+)",
    re.IGNORECASE,
)


def _read_token(token: str) -> str | None:
    """One token as a PMID or PMCID, or None."""
    if "/" in token:
        m = _URL_PMCID.search(token) or _URL_PMID.search(token)
        return m.group(1).upper() if m else None
    if re.fullmatch(r"PMC\d+", token, re.IGNORECASE):
        return token.upper()
    if re.fullmatch(r"\d{5,12}", token):
        return token
    return None


def parse_pubmed_ids(text: str) -> tuple[list[str], list[str]]:
    """The PubMed IDs and PMCIDs in free text, and the tokens that were neither.

    Accepts PMIDs ("31452104", "PMID: 31452104"), PMCIDs ("PMC6711232",
    "PMC 6711232") and PubMed, PMC and Europe PMC article URLs, separated by
    commas, semicolons or white space, in brackets or followed by a full stop.
    Returns IDs like '31452104' or 'PMC6711232' in order and without repeats,
    and the tokens it could not read, so that the caller can say what it
    ignored.
    """
    # Join what white space would split.  "PMC 6711232" was read as the
    # PMID 6711232, a different article.
    text = re.sub(r"\b(PMC)\s+(\d)", r"\1\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPMID\s*:?\s*(\d)", r"\1", text, flags=re.IGNORECASE)
    ids: list[str] = []
    ignored: list[str] = []
    for raw in re.split(r"[,;\s]+", text.strip()):
        token = raw.strip("[](){}<>.:'\"")
        if not token:
            continue
        found = _read_token(token)
        if found is None:
            ignored.append(raw)
        elif found not in ids:
            ids.append(found)
    return ids, ignored


def pmids_to_pmcids(pmids: list[str]) -> dict[str, str | None]:
    """Each PubMed ID's PMCID, or None when PubMed Central has no record of it.

    Network and JSON-decode failures are raised (requests.RequestException,
    ValueError) so that the caller can tell "no record" from "could not ask".
    """
    found: dict[str, str | None] = {}
    for start in range(0, len(pmids), _IDCONV_BATCH):
        r = requests.get(
            IDCONV_URL,
            params={
                "ids": ",".join(pmids[start:start + _IDCONV_BATCH]),
                "format": "json",
                "tool": "plotpick",
            },
            timeout=15,
        )
        r.raise_for_status()
        for record in r.json().get("records", []):
            # "pmid" comes back as a number; "requested-id" echoes the ID as
            # it was sent.  Matching the number against the strings sent
            # found nothing, so every PMID was silently dropped.
            asked = str(record.get("requested-id") or record.get("pmid") or "")
            found[asked] = record.get("pmcid")
    return {pmid: found.get(pmid) for pmid in pmids}


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
