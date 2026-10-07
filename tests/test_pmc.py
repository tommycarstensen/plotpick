"""Tests for pmc.py: reading PubMed identifiers, and locating an article's PDF
in the PMC Cloud Service."""

import pytest
import requests

from plotpick import pmc


def ids(text: str) -> list[str]:
    return pmc.parse_pubmed_ids(text)[0]


def test_parses_bare_pmids_and_pmcids_with_any_separator():
    text = " 12345678, pmc7654321;23456789\nPMC111 "
    assert ids(text) == ["12345678", "PMC7654321", "23456789", "PMC111"]


def test_parses_pubmed_pmc_and_europe_pmc_urls():
    text = (
        "https://pubmed.ncbi.nlm.nih.gov/31452104/ "
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6711232/ "
        "https://www.ncbi.nlm.nih.gov/pmc/articles/pmc6711233/ "
        "https://www.ncbi.nlm.nih.gov/pubmed/31452105 "
        "https://europepmc.org/article/MED/31452106 "
        "https://europepmc.org/article/PMC/PMC6711234"
    )
    assert ids(text) == [
        "31452104", "PMC6711232", "PMC6711233", "31452105", "31452106",
        "PMC6711234",
    ]


def test_a_spaced_pmcid_is_not_read_as_a_pmid():
    """"PMC 6711232" was read as the PMID 6711232, a different article."""
    assert ids("PMC 6711232") == ["PMC6711232"]


def test_labels_brackets_and_punctuation_are_allowed():
    text = "PMID:31452104 PMID: 31452105 31452106. (31452107) [PMC6711232],"
    assert ids(text) == [
        "31452104", "31452105", "31452106", "31452107", "PMC6711232",
    ]


def test_repeats_are_dropped_and_the_rest_reported():
    """A DOI, a word or a four-digit year is not a PubMed identifier."""
    assert pmc.parse_pubmed_ids("31452104 2019 trial 31452104 10.1000/xyz") == (
        ["31452104"], ["2019", "trial", "10.1000/xyz"],
    )
    assert pmc.parse_pubmed_ids("   ") == ([], [])


class FakeIdconv:
    """The ID Converter's reply, as the service gave it on 6 October 2026."""

    def __init__(self, records: list[dict], status_code: int = 200):
        self.records = records
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self) -> dict:
        return {"status": "ok", "records": self.records}


def test_pmids_are_matched_although_the_service_answers_numbers(monkeypatch):
    """It sends "pmid" as a number; matching that against the strings sent
    found nothing, so every PMID was dropped without a word."""
    sent = []

    def fake_get(url, params=None, timeout=None):
        del timeout
        sent.append((url, params))
        return FakeIdconv([
            {"doi": "10.3390/v16010008", "pmcid": "PMC10821221",
             "pmid": 38275943, "requested-id": "38275943"},
            {"pmid": 1, "requested-id": "1", "status": "error",
             "errmsg": "Identifier not found in PMC"},
        ])

    monkeypatch.setattr(pmc.requests, "get", fake_get)
    assert pmc.pmids_to_pmcids(["38275943", "1", "99999999"]) == {
        "38275943": "PMC10821221", "1": None, "99999999": None,
    }
    ((url, params),) = sent
    assert url == pmc.IDCONV_URL
    assert params["ids"] == "38275943,1,99999999"
    assert params["tool"] == "plotpick"


def test_pmid_lookup_is_sent_in_batches_of_200(monkeypatch):
    batches = []

    def fake_get(url, params: dict[str, str], timeout=None):
        del url, timeout
        ids = params["ids"].split(",")
        batches.append(len(ids))
        return FakeIdconv([{"pmcid": f"PMC{i}", "requested-id": i} for i in ids])

    monkeypatch.setattr(pmc.requests, "get", fake_get)
    pmids = [str(10000 + i) for i in range(450)]
    assert pmc.pmids_to_pmcids(pmids)["10449"] == "PMC10449"
    assert batches == [200, 200, 50]


def test_a_failed_pmid_lookup_is_raised_not_reported_as_no_record(monkeypatch):
    monkeypatch.setattr(
        pmc.requests, "get", lambda *args, **kwargs: FakeIdconv([], 503),
    )
    with pytest.raises(requests.HTTPError):
        pmc.pmids_to_pmcids(["38275943"])


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def listing(*prefixes: str) -> bytes:
    """An S3 ListObjectsV2 reply holding the given version folders."""
    items = "".join(
        f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<Name>pmc-oa-opendata</Name>{items}</ListBucketResult>"
    ).encode()


@pytest.fixture
def cloud(monkeypatch):
    """Serve a fake bucket: a listing plus whichever PDF keys are given."""
    requested: list[str] = []

    def install(listing_reply: FakeResponse, pdfs: dict[str, bytes]):
        def fake_get(url, params=None, timeout=None):
            del timeout
            if params is not None:
                requested.append(f"list:{params['prefix']}")
                return listing_reply
            key = url.removeprefix(pmc.PMC_CLOUD_URL)
            requested.append(key)
            if key in pdfs:
                return FakeResponse(200, pdfs[key])
            return FakeResponse(404)

        monkeypatch.setattr(pmc.requests, "get", fake_get)
        return requested

    return install


def test_downloads_the_newest_version(cloud):
    requested = cloud(
        FakeResponse(200, listing("PMC123.1/", "PMC123.2/")),
        {"PMC123.1/PMC123.1.pdf": b"old", "PMC123.2/PMC123.2.pdf": b"new"},
    )
    assert pmc.download_pmc_pdf("PMC123") == b"new"
    assert requested == ["list:PMC123.", "PMC123.2/PMC123.2.pdf"]


def test_falls_back_when_the_newest_version_has_no_pdf(cloud):
    cloud(
        FakeResponse(200, listing("PMC123.1/", "PMC123.2/")),
        {"PMC123.1/PMC123.1.pdf": b"old"},
    )
    assert pmc.download_pmc_pdf("PMC123") == b"old"


def test_versions_sort_numerically(cloud):
    requested = cloud(
        FakeResponse(200, listing("PMC123.10/", "PMC123.9/")),
        {"PMC123.10/PMC123.10.pdf": b"ten"},
    )
    assert pmc.download_pmc_pdf("PMC123") == b"ten"
    assert requested[1] == "PMC123.10/PMC123.10.pdf"


def test_article_outside_open_access_gives_none(cloud):
    cloud(FakeResponse(200, listing()), {})
    assert pmc.download_pmc_pdf("PMC123") is None


def test_other_articles_sharing_the_prefix_are_ignored(cloud):
    """The listing prefix "PMC123." must not match "PMC1234.1/"."""
    cloud(
        FakeResponse(200, listing("PMC1234.1/")),
        {"PMC1234.1/PMC1234.1.pdf": b"other"},
    )
    assert pmc.download_pmc_pdf("PMC123") is None


def test_a_failed_listing_is_raised_not_reported_as_unavailable(cloud):
    """The retired endpoint answered 404 and that was shown as "not open access"."""
    cloud(FakeResponse(500), {})
    with pytest.raises(requests.HTTPError):
        pmc.download_pmc_pdf("PMC123")
