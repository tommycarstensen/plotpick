"""Tests for pdf_figures.py caption detection and figure extraction."""

import pytest

from pdf_figures import CAPTION_RE, reads_as_running_text


class TestCaptionRegex:
    """Test that the caption regex matches expected patterns."""

    @pytest.mark.parametrize("text", [
        "Figure 1",
        "Figure 1.",
        "Figure 12 —",
        "Fig. 1",
        "Fig. 1:",
        "Fig. 2A",
        "Figure 3 shows the results",
        "Table 1",
        "Table 2.",
        "Table 10: Patient demographics",
        "Supplementary Figure 1",
        "Supplementary Fig. 3",
        "Supplementary Table 2",
        "Suppl. Fig 1",
    ])
    def test_caption_matches(self, text):
        assert CAPTION_RE.match(text), f"Should match: {text!r}"

    @pytest.mark.parametrize("text", [
        "The figure shows",
        "table of contents",
        "See Figure 1",
        "in Figure 2",
        "Results",
        "Methods",
        "",
        "1. Introduction",
    ])
    def test_caption_rejects(self, text):
        assert not CAPTION_RE.match(text), f"Should not match: {text!r}"


class TestRunningText:
    """A paragraph may open with a label; that does not make it a caption."""

    @pytest.mark.parametrize("text", [
        "Table 1 summarizes the baseline characteristics of participants",
        "Figure 3 shows the results",
        "Figure 6 displays the local SHAP value distributions",
        "TABLE 2 summarizes the outcomes",
        "Fig 1a shows the flow of participants",
        "Figure 3 is formulated as:",
        "Table 4 and Figure 2 show the same pattern",
        "Table 1 for demographic and clinical characteristics",
        "Figure 2, Table S12). Pedogenic oxide contents exhibited",
        "Figure 4H). The specific Rmax and EC50 values",
        "Figure 2C,D show the temporal changes",
        "Supplementary Figure 10, the therapeutic impact",
        "Table 2). These results highlight",
        "Figure 4.",
    ])
    def test_sentences(self, text):
        assert reads_as_running_text(text), f"Should be a sentence: {text!r}"

    @pytest.mark.parametrize("text", [
        "Table 1",
        "Figure 12",
        "Table 2. Cont.",
        "Table 1. Demographic and clinical characteristics",
        "Table 3: results",
        "FIGURE 1 | Flowchart of data collection.",
        "Fig. 1 - Descriptives of male and female participants.",
        "Figure 1 The modified Delphi process",
        "TABLE 2 Summary of findings",
        "FIGURE 3 SHAP values",
        "TABLE 1 (Continued) Baseline characteristics",
        "Table 2 continued",
        "Fig. 2 a Kaplan-Meier curves",
        "Fig. 2 mRNA expression of IL-6",
        "Fig. 4 miR-21 expression",
        "Figure 2 p53 levels",
        "Fig. 5 pH dependence",
        "Fig. 3 in vitro release profile",
        "Fig. 2 non-linear dose response",
        "Figure 1 [68Ga]PentixaFor PET/CT",
    ])
    def test_captions(self, text):
        assert CAPTION_RE.match(text)
        assert not reads_as_running_text(text), f"Should be a caption: {text!r}"
