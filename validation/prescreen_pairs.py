"""Pre-screen figure-table pairs for numerical correspondence.

For each pair, sends the table metadata and cross-reference text to Haiku
(text only, no image) and asks whether the figure and table likely present
the same numerical data. Pairs judged 'no' are written to excluded_pairs.json
and removed from pairs.json.

Usage:
    python prescreen_pairs.py              # screen all pairs
    python prescreen_pairs.py --dry-run   # print verdicts without modifying files
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
PAIRS_PATH = HERE / "pairs.json"
TABLE_DIR = HERE / "tables"
EXCLUDED_PATH = HERE / "excluded_pairs.json"
SECRETS_PATH = HERE.parent / ".streamlit" / "secrets.toml"
DELAY = 0.3

SYSTEM_PROMPT = """\
You are a scientific literature expert. Your task is to judge whether a figure \
and a table in the same paper present the same numerical data.

You will be given:
- A cross-reference sentence from the paper body that mentions both
- The table label, caption, and column headers

Answer with a JSON object:
{
  "match": true or false,
  "reason": "one sentence explanation"
}

"match" is true if the figure and table are likely showing the same measured \
values (e.g. both show patient outcomes, both show measurement results). \
"match" is false if they clearly present different quantities (e.g. figure shows \
survival rates but table shows a risk scoring formula, or figure is a schematic \
with no data). When in doubt, answer true.
"""


def _load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if SECRETS_PATH.exists():
        for line in SECRETS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"')
    sys.exit("No API key found.")


def load_table_meta(pmcid: str, table_num: int) -> dict:
    path = TABLE_DIR / f"{pmcid}.json"
    if not path.exists():
        return {}
    tables = json.loads(path.read_text(encoding="utf-8"))
    import re
    target = f"table {table_num}"
    for t in tables:
        label = t.get("label", "").strip().lower()
        if label == target:
            return t
        m = re.search(r"(\d+)", label)
        if m and int(m.group(1)) == table_num:
            return t
    return {}


def build_prompt(pair: dict, table: dict) -> str:
    cross_ref = pair.get("cross_ref", "").strip()
    label = table.get("label", "")
    caption = table.get("caption", "")[:300]
    headers = table.get("headers", [])
    return (
        f"Cross-reference sentence: \"{cross_ref}\"\n\n"
        f"Table label: {label}\n"
        f"Table caption: {caption}\n"
        f"Table column headers: {headers}\n\n"
        f"Figure {pair['figure_num']} vs Table {pair['table_num']}: "
        f"do they present the same numerical data?"
    )


def screen_pair(client: anthropic.Anthropic, pair: dict) -> tuple[bool, str]:
    table = load_table_meta(pair["pmcid"], pair["table_num"])
    if not table:
        return True, "no table metadata — keeping by default"
    prompt = build_prompt(pair, table)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        obj = json.loads(text.strip())
        return bool(obj.get("match", True)), obj.get("reason", "")
    except json.JSONDecodeError:
        # If parsing fails, keep the pair
        return True, f"parse error: {text[:100]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print verdicts without modifying files")
    args = parser.parse_args()

    pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    sys.stdout.buffer.write(f"Screening {len(pairs)} pairs...\n\n".encode("utf-8"))

    client = anthropic.Anthropic(api_key=_load_api_key())

    kept, excluded = [], []
    for i, pair in enumerate(pairs):
        match, reason = screen_pair(client, pair)
        tag = "KEEP" if match else "EXCLUDE"
        title = pair.get("title", "")[:60].encode("ascii", "replace").decode()
        line = f"  [{i+1:3d}/{len(pairs)}] {tag} {pair['pmcid']}: {reason[:80]}"
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
        if match:
            kept.append(pair)
        else:
            excluded.append({**pair, "exclusion_reason": reason})
        time.sleep(DELAY)

    sys.stdout.buffer.write(f"\nKept: {len(kept)}, Excluded: {len(excluded)}\n".encode("utf-8"))

    if not args.dry_run:
        PAIRS_PATH.write_text(json.dumps(kept, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        EXCLUDED_PATH.write_text(json.dumps(excluded, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        sys.stdout.buffer.write(f"pairs.json updated ({len(kept)} pairs)\n".encode("utf-8"))
        sys.stdout.buffer.write(f"excluded_pairs.json written ({len(excluded)} pairs)\n".encode("utf-8"))


if __name__ == "__main__":
    main()
