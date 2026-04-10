---
title: 'PlotPick: AI-powered batch extraction of numerical data from scientific figures'
tags:
  - Python
  - systematic review
  - meta-analysis
  - data extraction
  - large language models
authors:
  - name: Camilla ???
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Tommy Carstensen
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Region Hovedstaden, Copenhagen, Denmark
    index: 1
date: 2026-03-19
bibliography: paper.bib
---

# Summary

Systematic reviews and meta-analyses frequently require numerical data
that authors report only as figures. Manual digitisation using tools such
as WebPlotDigitizer is slow, error-prone, and does not scale to large
corpora. **PlotPick** is an open-source Streamlit application that uses
Claude's vision API [@anthropic2024] to batch-extract structured data from
boxplots, bar charts, and line plots embedded in images, PDFs, or ZIP
archives. Results are displayed as interactive tables and exported as
Excel, CSV, LaTeX, or JSON.

# Statement of Need

<!-- TODO: quantify the burden -- how many systematic reviews rely on
figure-only data? cite a survey if available -->

Data locked in figures is a well-known obstacle in evidence synthesis
[@glasziou2001]. Existing digitisation tools require manual point-clicking
per data series and do not produce structured, labelled output suitable
for downstream meta-analysis. PlotPick addresses this by combining
automatic figure detection from PDFs with a structured vision-language
prompt that names biomarkers, groups, and timepoints before reading values.

# Implementation

PlotPick is a single-file Streamlit application. Uploaded PDFs are parsed
with PyMuPDF [@pymupdf]; individual figures are detected by locating
figure captions via regular expression and cropping the surrounding region.
Images are encoded as base64 and sent to the Claude API with a two-stage
prompt: an inventory stage that identifies axis labels, scale (linear or
log), and legend entries, followed by an extraction stage that reads values
for each group. A confidence score accompanies each row of output.

<!-- TODO: insert architecture figure -->
<!-- ![PlotPick architecture.\label{fig:arch}](figures/architecture.png) -->

# Validation

<!-- TODO: fill in after find_candidates.py run -->

We validated PlotPick against table-reported ground truth from $N = \ldots$
open-access PMC articles in which the same data appeared in both a figure
and a table [@candidates2026]. Candidate articles were identified
programmatically by searching for sentence-level cross-references between
figures and tables. Median absolute error was $\ldots$ and $\ldots$% of
extracted values agreed with the table value within 10%.

# Acknowledgements

<!-- TODO -->

# References
