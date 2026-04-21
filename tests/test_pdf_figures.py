"""Tests for pdf_figures.py caption detection and figure extraction."""

import re
import sys
from pathlib import Path

import pytest

# Add parent dir so we can import pdf_figures
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdf_figures import CAPTION_RE, find_figures_on_page


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


class TestExtractNumbers:
    """Test number extraction from extracted text."""

    def test_integers(self):
        from tests.test_helpers import extract_numbers
        assert extract_numbers("100\t200\t300") == {100.0, 200.0, 300.0}

    def test_floats(self):
        from tests.test_helpers import extract_numbers
        assert extract_numbers("3.14\t2.718") == {3.14, 2.718}

    def test_scientific_notation(self):
        from tests.test_helpers import extract_numbers
        assert extract_numbers("2.2e+10\t1.5E-3") == {2.2e10, 1.5e-3}

    def test_comma_separated(self):
        from tests.test_helpers import extract_numbers
        assert extract_numbers("500,000\t1,200,000") == {500000.0, 1200000.0}

    def test_mixed(self):
        from tests.test_helpers import extract_numbers
        nums = extract_numbers("Year\tValue\n2020\t1,500\n2021\t2.5e3")
        assert 2020.0 in nums
        assert 1500.0 in nums
        assert 2500.0 in nums


class TestComputeRecall:
    """Test recall computation with tolerance."""

    def test_perfect_match(self):
        from tests.test_helpers import compute_recall
        assert compute_recall({100, 200, 300}, {100, 200, 300}) == 1.0

    def test_within_tolerance(self):
        from tests.test_helpers import compute_recall
        # 105 is within 5% of 100
        assert compute_recall({105, 200, 300}, {100, 200, 300}) == 1.0

    def test_outside_tolerance(self):
        from tests.test_helpers import compute_recall
        # 120 is NOT within 5% of 100
        assert compute_recall({120, 200, 300}, {100, 200, 300}) < 1.0

    def test_missing_values(self):
        from tests.test_helpers import compute_recall
        assert compute_recall({100}, {100, 200, 300}) == pytest.approx(1/3)

    def test_empty(self):
        from tests.test_helpers import compute_recall
        assert compute_recall(set(), {100, 200}) == 0.0
