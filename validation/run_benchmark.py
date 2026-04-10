"""Benchmark PlotPick figure extraction against ground-truth tables.

Sends each figure image through Claude's vision API with the table structure
as context, extracts values cell-by-cell, and compares them 1:1 against the
corresponding PMC XML table (positional matching, no bag-of-numbers search).

Usage:
    python run_benchmark.py                     # default: haiku, all pairs
    python run_benchmark.py --model sonnet      # use Sonnet
    python run_benchmark.py --limit 20          # first 20 pairs only
    python run_benchmark.py --report            # skip extraction, just report
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
from PIL import Image

HERE = Path(__file__).resolve().parent
SECRETS_PATH = HERE.parent / ".streamlit" / "secrets.toml"
TABLE_DIR = HERE / "tables"
FIG_DIR = Path.home() / "camilla2026_local" / "figures"
PAIRS_PATH = HERE / "pairs.json"
RESULTS_DIR = HERE / "results"  # subdir per model: results/haiku/, results/sonnet/

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-5-20251101",
}

SYSTEM_PROMPT = """\
You are an expert at reading scientific figures. A figure from a research paper \
is shown alongside the structure of its corresponding data table.

Your task: fill in the table by transcribing numbers that are explicitly visible \
as text labels or annotations in the figure. This is a pure reading task — \
no arithmetic, no estimation from axis scales, no unit conversion.

Return a JSON object:
{
  "figure_type": "string describing the chart type",
  "rows": [
    ["row_label_or_value", value_col2, value_col3, ...],
    ...
  ],
  "confidence": integer 0-100,
  "notes": "any difficulties"
}

Rules:
- Return exactly one row per table row, in the same order as the template
- Return exactly one value per column, in the same column order
- Copy the row label (first column) from the template unchanged
- For each data cell: transcribe the number exactly as it appears as a text \
  label or annotation in the figure (e.g. a bar label, a data point annotation, \
  a cell value in an embedded table). Report it as a string preserving the \
  original notation, e.g. "0.63 (0.33-1.18)", "94.7", "p=0.03"
- Use null if no explicit text label for that cell is visible in the figure
- Do NOT read values off axes by visual estimation
- Do NOT calculate, multiply, divide, or transform any numbers
- Return ONLY valid JSON
"""

MAX_WIDTH = 2000
TOLERANCE = 0.05  # 5% relative error
DELAY = 0.5


def _load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if SECRETS_PATH.exists():
        for line in SECRETS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"')
    sys.exit("No API key found. Set ANTHROPIC_API_KEY or add to .streamlit/secrets.toml")


def load_pairs(limit: int = 0) -> list[dict]:
    pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    return pairs[:limit] if limit else pairs


def load_table(pmcid: str, table_num: int) -> dict | None:
    path = TABLE_DIR / f"{pmcid}.json"
    if not path.exists():
        return None
    tables = json.loads(path.read_text(encoding="utf-8"))
    target = f"table {table_num}"
    for t in tables:
        label = t.get("label", "").strip().lower()
        if label == target:
            return t
        m = re.search(r"(\d+)", label)
        if m and int(m.group(1)) == table_num:
            return t
    return None


def image_to_base64(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        img = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_template(table: dict) -> list[list]:
    """Build a row template: first column has the row label, rest are '?'."""
    n_cols = len(table.get("headers", []))
    rows = table.get("rows", [])
    template = []
    for row in rows:
        if not row:
            template.append(["?"] * max(n_cols, 1))
        else:
            label = row[0] if row else "?"
            template.append([label] + ["?"] * max(n_cols - 1, 0))
    return template


def extract_from_figure(client: anthropic.Anthropic, fig_path: Path,
                        table: dict, model: str) -> dict:
    b64 = image_to_base64(fig_path)
    template = make_template(table)
    user_prompt = (
        f"Table headers: {json.dumps(table.get('headers', []))}\n"
        f"Table template (fill in '?' cells with values from the figure):\n"
        f"{json.dumps(template)}\n\n"
        "Return the filled table as JSON."
    )
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": user_prompt},
            ],
        }],
    )
    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    # Try direct parse first, then extract outermost { ... }
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"error": "invalid_json", "raw": text[:500]}


def parse_numbers(cell) -> list[float]:
    """Extract numbers from a cell value in order of appearance."""
    if cell is None:
        return []
    nums = []
    for m in re.finditer(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", str(cell)):
        try:
            nums.append(float(m.group()))
        except ValueError:
            pass
    return nums


def is_match(truth: float, extracted: float) -> bool:
    """5% relative error only. No absolute fallback."""
    if truth == 0:
        return extracted == 0
    return abs(extracted - truth) / abs(truth) <= TOLERANCE


def compare(table: dict, extraction: dict) -> dict:
    """1:1 positional comparison: truth[row][col][n] vs extracted[row][col][n]."""
    truth_rows = table.get("rows", [])
    ext_rows = extraction.get("rows", [])

    n_correct = 0
    n_total = 0
    cell_details = []

    for row_idx, truth_row in enumerate(truth_rows):
        ext_row = ext_rows[row_idx] if row_idx < len(ext_rows) else []
        for col_idx, truth_cell in enumerate(truth_row):
            truth_nums = parse_numbers(truth_cell)
            if not truth_nums:
                continue  # skip non-numeric cells (labels, ND, etc.)

            ext_cell = ext_row[col_idx] if col_idx < len(ext_row) else None
            ext_nums = parse_numbers(ext_cell)

            for num_idx, truth_val in enumerate(truth_nums):
                n_total += 1
                ext_val = ext_nums[num_idx] if num_idx < len(ext_nums) else None
                matched = ext_val is not None and is_match(truth_val, ext_val)
                if matched:
                    n_correct += 1
                cell_details.append({
                    "row": row_idx, "col": col_idx, "pos": num_idx,
                    "truth": truth_val, "extracted": ext_val, "match": matched,
                })

    recall = n_correct / n_total if n_total else None
    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "recall": round(recall, 3) if recall is not None else None,
        "cells": cell_details,
    }


def run_extractions(pairs: list[dict], model_key: str) -> None:
    model = MODELS[model_key]
    results_dir = RESULTS_DIR / model_key
    results_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=_load_api_key())

    done = 0
    for i, pair in enumerate(pairs):
        pmcid = pair["pmcid"]
        out_path = results_dir / f"{pmcid}_T{pair['table_num']}_F{pair['figure_num']}.json"
        if out_path.exists():
            done += 1
            continue

        fig_path = FIG_DIR / Path(pair["figure_path"]).name
        if not fig_path.exists():
            sys.stdout.buffer.write(
                f"  [{i+1:3d}/{len(pairs)}] {pmcid}: figure missing, skipping\n"
                .encode("utf-8"))
            continue

        table = load_table(pmcid, pair["table_num"])
        if not table:
            sys.stdout.buffer.write(
                f"  [{i+1:3d}/{len(pairs)}] {pmcid}: table missing, skipping\n"
                .encode("utf-8"))
            continue

        extraction = None
        for attempt in range(4):
            try:
                extraction = extract_from_figure(client, fig_path, table, model)
                break
            except Exception as e:
                if attempt < 3 and ("529" in str(e) or "overloaded" in str(e).lower()
                                    or "500" in str(e)):
                    wait = 2 ** attempt * 5  # 5, 10, 20 seconds
                    sys.stdout.buffer.write(
                        f"  [{i+1:3d}/{len(pairs)}] {pmcid}: retry {attempt+1}/3 "
                        f"in {wait}s ({e.__class__.__name__})\n".encode("utf-8"))
                    time.sleep(wait)
                else:
                    sys.stdout.buffer.write(
                        f"  [{i+1:3d}/{len(pairs)}] {pmcid}: API error: {e}\n"
                        .encode("utf-8"))
                    break
        if extraction is None:
            continue

        metrics = compare(table, extraction)
        result = {
            "pmcid": pmcid,
            "table_num": pair["table_num"],
            "figure_num": pair["figure_num"],
            "model": model_key,
            "confidence": extraction.get("confidence"),
            "figure_type": extraction.get("figure_type"),
            "metrics": metrics,
            "extraction": extraction,
        }
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        done += 1

        recall = metrics["recall"]
        tag = f"recall={recall:.0%}" if recall is not None else "no numeric truth"
        title = pair.get("title", "")[:50].encode("ascii", "replace").decode()
        sys.stdout.buffer.write(
            f"  [{done:3d}/{len(pairs)}] {pmcid} ({tag}): {title}\n"
            .encode("utf-8"))
        time.sleep(DELAY)

    sys.stdout.buffer.write(
        f"\nExtraction complete: {done}/{len(pairs)} results in {results_dir}\n"
        .encode("utf-8"))


def generate_report(model_key: str) -> None:
    results_dir = RESULTS_DIR / model_key
    if not results_dir.exists():
        sys.exit(f"No results directory for model '{model_key}'. Run extractions first.")

    results = []
    for path in sorted(results_dir.glob("*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))

    if not results:
        sys.exit("No result files found.")

    recalls = [r["metrics"]["recall"] for r in results
               if r["metrics"]["recall"] is not None]
    confidences = [r["confidence"] for r in results if r.get("confidence")]

    n = len(results)
    n_with_recall = len(recalls)

    lines = [
        "# PlotPick Validation Benchmark",
        "",
        f"**Model:** {model_key}",
        f"**Pairs evaluated:** {n}",
        f"**Pairs with numeric ground truth:** {n_with_recall}",
        f"**Tolerance:** {TOLERANCE:.0%} relative error",
        "",
        "## Aggregate Metrics",
        "",
    ]

    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        med_recall = sorted(recalls)[len(recalls) // 2]
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Mean recall | {avg_recall:.1%} |")
        lines.append(f"| Median recall | {med_recall:.1%} |")
        if confidences:
            lines.append(f"| Mean confidence | {sum(confidences)/len(confidences):.0f} |")
        lines.append("")

    if recalls:
        bins = {">=90%": 0, "70-89%": 0, "50-69%": 0, "<50%": 0}
        for r in recalls:
            if r >= 0.9:
                bins[">=90%"] += 1
            elif r >= 0.7:
                bins["70-89%"] += 1
            elif r >= 0.5:
                bins["50-69%"] += 1
            else:
                bins["<50%"] += 1

        lines.append("## Recall Distribution")
        lines.append("")
        lines.append("| Bin | Count | Pct |")
        lines.append("|-----|-------|-----|")
        for label, count in bins.items():
            pct = count / n_with_recall
            lines.append(f"| {label} | {count} | {pct:.0%} |")
        lines.append("")

    type_recalls: dict[str, list[float]] = {}
    for r in results:
        ft = r.get("figure_type", "unknown") or "unknown"
        rec = r["metrics"]["recall"]
        if rec is not None:
            type_recalls.setdefault(ft, []).append(rec)

    if type_recalls:
        lines.append("## By Figure Type")
        lines.append("")
        lines.append("| Type | N | Mean Recall |")
        lines.append("|------|---|-------------|")
        for ft in sorted(type_recalls, key=lambda k: -len(type_recalls[k])):
            vals = type_recalls[ft]
            avg = sum(vals) / len(vals)
            lines.append(f"| {ft} | {len(vals)} | {avg:.1%} |")
        lines.append("")

    ranked = sorted(
        [r for r in results if r["metrics"]["recall"] is not None],
        key=lambda r: r["metrics"]["recall"],
    )
    if ranked:
        lines.append("## Lowest Recall (bottom 10)")
        lines.append("")
        lines.append("| PMCID | Table | Fig | Recall | N values | Type |")
        lines.append("|-------|-------|-----|--------|----------|------|")
        for r in ranked[:10]:
            m = r["metrics"]
            lines.append(
                f"| {r['pmcid']} | {r['table_num']} | {r['figure_num']} "
                f"| {m['recall']:.0%} | {m['n_total']} "
                f"| {r.get('figure_type', '')} |"
            )
        lines.append("")

    report_path = HERE / f"benchmark_report_{model_key}.md"
    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    sys.stdout.buffer.write((report + "\n").encode("utf-8"))
    sys.stdout.buffer.write(f"\nReport saved to {report_path}\n".encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), default="haiku",
                        help="Claude model to use (default: haiku)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process first N pairs only")
    parser.add_argument("--report", action="store_true",
                        help="Generate report from existing results (no API calls)")
    args = parser.parse_args()

    if args.report:
        generate_report(args.model)
        return

    pairs = load_pairs(args.limit)
    sys.stdout.buffer.write(
        f"Benchmark: {len(pairs)} pairs, model={args.model}\n".encode("utf-8"))
    run_extractions(pairs, args.model)
    generate_report(args.model)


if __name__ == "__main__":
    main()
