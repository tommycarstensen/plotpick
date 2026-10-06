"""The extraction step: one figure image to Claude, one table back.

No Streamlit dependency.  The prompt asks for JSON with the fields a
meta-analyst records; parse_reply() turns the model's text into that JSON and
refuses anything that is not an object with a list of rows, so that one odd
reply fails one figure instead of the batch.
"""

import json
from typing import Any

import anthropic

from figure_images import image_to_base64
from models import ExtractionError, reply_text

MAX_TOKENS = 16384

# Derived from figure_extraction_instructions.md.
SYSTEM_PROMPT = """\
You are an expert at reading scientific figures.  You extract numerical
data from boxplots, bar charts, and line plots with high accuracy.

Follow these rules strictly:

1. INVENTORY first: identify figure type, axis labels, scale (linear or
   log), legend, number of groups/subplots, and any significance markers.
   For multi-panel figures, list each panel/subplot separately.
2. Define the output columns BEFORE extracting values:
   - Boxplots: biomarker, group, timepoint, median, q1, q3
   - Bar charts: biomarker, group, timepoint, n, mean, error, error_type
   - Line plots: biomarker, group, timepoint, mean_or_median, error, error_type
   Include ALL columns even if some are null.  Use "timepoint" for any
   time-based grouping (Baseline, Week 6, etc.).
3. Extract EVERY box / bar / point in the figure.  Go subplot by subplot,
   left-to-right within each subplot, reading all groups and timepoints.
   Interpolate numeric values from the nearest axis ticks.
4. The "data" array MUST contain one object per box / bar / data point.
   NEVER return an empty "data" array -- if you can see boxes or bars,
   you MUST extract values even if approximate.  Use null for any single
   value you truly cannot read, but still include the row.
5. Every numeric field must be a JSON number (not a string).
6. Include a top-level "confidence" field (0-100) reflecting how
   precisely you could read the values.  Use the FULL range:
   - 95-100: crisp axes, clear ticks, easy to read exactly
   - 80-94: good axes but some interpolation needed
   - 60-79: small figure, overlapping elements, or missing ticks
   - below 60: largely guessing
   Report the ACTUAL precision, not a round number.
7. Include a "notes" string listing any specific values that were hard
   to read and explaining why.
8. For EACH data row, include an "uncertain" field: a list of column
   names whose values you are unsure about (e.g. overlapping boxes,
   blurry region, ambiguous tick alignment, very small figure).
   Use an empty list [] when all values in that row are confident.

Example -- multi-panel boxplot with timepoints:
{
  "figure_type": "boxplot",
  "y_axis": "various (see per-row biomarker)",
  "scale": "linear",
  "confidence": 76,
  "notes": "IL-6 Week 12 boxes overlap; Q1/Q3 approximate",
  "data": [
    {"biomarker": "IL-6", "group": "Responders", "timepoint": "Baseline", "median": 9.5, "q1": 7.0, "q3": 12.0, "uncertain": []},
    {"biomarker": "IL-6", "group": "Responders", "timepoint": "Week 6", "median": 8.0, "q1": 6.5, "q3": 10.5, "uncertain": ["q1", "q3"]},
    {"biomarker": "IL-6", "group": "Non-Responders", "timepoint": "Baseline", "median": 10.0, "q1": 8.0, "q3": 13.0, "uncertain": []},
    {"biomarker": "CRP", "group": "Responders", "timepoint": "Baseline", "median": 3.2, "q1": 2.5, "q3": 4.0, "uncertain": ["median"]}
  ]
}

IMPORTANT: Return ONLY valid JSON. No text before or after the JSON object.
The "data" array must NEVER be empty if the figure contains any visual elements.
"""  # noqa: E501 -- example JSON rows; wrapping them would corrupt the example.

USER_PROMPT = """\
Extract ALL numerical data from this figure.  There should be one row
per box/bar/point.  Return structured JSON only.
"""


def parse_reply(raw_text: str) -> dict[str, Any]:
    """The model's reply as a result: a JSON object whose "data" is a list.

    Raises json.JSONDecodeError for text that is not JSON, and
    ExtractionError for JSON of another shape.
    """
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    text = text.removesuffix("```")
    text = text.strip()

    snippet = raw_text[:300] + ("..." if len(raw_text) > 300 else "")
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        # Attach a snippet of the raw response for debugging
        raise json.JSONDecodeError(
            f"Claude returned invalid JSON. Raw response (first 300 chars):\n"
            f"{snippet}\n\nOriginal error: {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc
    if not isinstance(result, dict):
        raise ExtractionError(
            f"Claude's reply was not a JSON object. It began: {snippet}"
        )
    rows = result.setdefault("data", [])
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ExtractionError(
            f'Claude\'s "data" was not a list of rows. The reply began: {snippet}'
        )
    return result


def extract_from_image(
    client: anthropic.Anthropic,
    png_bytes: bytes,
    model: str,
) -> dict[str, Any]:
    """Send a single image to Claude and parse the JSON response."""
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_to_base64(png_bytes),
                        },
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT,
                    },
                ],
            },
        ],
    )
    return parse_reply(reply_text(response))
