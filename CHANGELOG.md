# Changelog

Notable changes to PlotPick. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- PDFs are read with pypdfium2 (BSD-3-Clause or Apache-2.0); PyMuPDF (AGPL) is no longer a dependency. A figure's region is bounded anew, and a sentence that opens with a label ("Table 1 summarizes ...") is no longer taken for a caption.
- Captions in the style "Fig 1" and supplementary "Figure S1" or "Table S2" are found.
- Claude Sonnet 5.5 is the default model. Sonnet and Claude Haiku 4.5 run on the app's key; Claude Opus 5.5 runs only on a key the user pastes in.
- Open-access PDFs are fetched from the PMC Cloud Service, since NCBI retired the OA web service. PubMed identifier parsing moved to `pmc.py` and accepts the current PMC article URLs.
- The benchmark statements in the app and the README follow version 2 of the companion preprint (arXiv:2605.06021).
- Each upload is read once instead of on every rerun.

### Added

- Every disabled Extract button states its reason (issue #2); an upload that cannot be read is explained instead of crashing the app.
- Every result and every exported row records the model and the time of extraction (UTC).
- The model's uncertainty flags are kept in the stored results and the JSON export.
- ruff, pycodestyle and pyright in CI; smoke tests that run the Streamlit script with a fake model; tests of the PDF backend, the exports and PubMed identifier parsing.
- The process's memory is logged after heavy work.

### Fixed

- A PubMed ID (PMID) was dropped without a message: NCBI's ID Converter moved and now answers the PMID as a number, which never matched the ID sent. The lookup moved to `pmc.py`, uses the new address, sends batches of 200 and is tested.
- The R script export did not run, the Excel export failed for ZIP uploads, and the LaTeX export did not compile.
- Two figures with the same label (two uploads named alike, or two "Figure 1" captions on a page) kept one result between them.
- File names and the model's text are escaped before they go into HTML.
- A crash on null metadata fields in a model reply, and a column mixing text and numbers that could not be shown.

## [0.1.0] - 2026-04-23

First tagged release: the Streamlit application with PyMuPDF figure detection, PubMed identifier input, six export formats, tests and CI, and CONTRIBUTING.md.
