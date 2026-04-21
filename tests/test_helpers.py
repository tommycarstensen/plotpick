"""Helper functions for tests -- mirrors metrics from validation/benchmarks/shared.py."""

import re

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")


def extract_numbers(text):
    """Extract all numeric values from a string."""
    return set(
        float(m.group().replace(",", ""))
        for m in NUM_RE.finditer(text)
    )


def compute_recall(extracted, ground_truth, tolerance=0.05):
    """Fraction of ground-truth values found in extraction (within tolerance)."""
    if not ground_truth:
        return 0.0
    matched = 0
    for gt in ground_truth:
        for ex in extracted:
            if gt == 0:
                if abs(ex) < 0.01:
                    matched += 1
                    break
            elif abs(ex - gt) / abs(gt) <= tolerance:
                matched += 1
                break
    return matched / len(ground_truth)
