"""Write extracted tables as R, Excel, LaTeX, Markdown or JSON.

No Streamlit dependency, so that every export can be tested on its own.  The
app builds one table of all extracted rows, with a "source" column naming the
figure each row came from, and hands it to these functions.

Each function also takes `about`, lines that say which PlotPick version and
model made the table, when, and how to cite PlotPick (plotpick.citation
writes them).  They go where the format has room outside the data: comments
in R and LaTeX, a footer in Markdown, a last sheet in Excel and a
"provenance" object in JSON.  CSV has no such place and gets none, so that
the data reads into any program unchanged.
"""

import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

# Excel refuses these characters in a sheet name (an apostrophe only at
# either end), names longer than 31 characters and the name "History".
_SHEET_FORBIDDEN = re.compile(r"[\[\]:*?/\\']")
SHEET_MAX = 31


def _missing(value: object) -> bool:
    """True for None and for pandas' and numpy's missing-value markers."""
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, float) and math.isnan(value))
    )


def _r_value(value: object) -> str:
    """One cell as an R literal: a number, TRUE/FALSE, NA or a quoted string."""
    if _missing(value):
        return "NA"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return repr(value)
    # json.dumps escapes quotes, backslashes and control characters the way
    # an R string literal expects them.
    return json.dumps(str(value), ensure_ascii=False)


def dataframe_to_r(df: pd.DataFrame, about: Sequence[str] = ()) -> str:
    """The table as an R script that recreates it as a data.frame."""
    columns = []
    for name, column in df.items():
        values = column.tolist()
        if not pd.api.types.is_numeric_dtype(column):
            # A text column, or one mixing text and numbers: all text in R.
            values = [None if _missing(v) else str(v) for v in values]
        quoted = str(name).replace("\\", "\\\\").replace("`", "\\`")
        cells = ", ".join(_r_value(v) for v in values)
        columns.append(f"  `{quoted}` = c({cells}),")
    return "\n".join([
        "# PlotPick output -- source directly in R",
        f"# Generated {datetime.now():%Y-%m-%d %H:%M}",
        *(f"# {line}" for line in about),
        "",
        "dat <- data.frame(",
        *columns,
        "  check.names = FALSE,",
        "  stringsAsFactors = FALSE",
        ")",
        "",
    ])


def sheet_names(labels: list[str]) -> dict[str, str]:
    """A valid, unique Excel sheet name for each source label.

    Forbidden characters become "_".  A long label keeps its end, which names
    the page and the figure; the start is often the same file name.  A name
    already taken, ignoring case, gets a number.
    """
    taken = {"all", "history", "about"}
    names: dict[str, str] = {}
    for label in labels:
        base = _SHEET_FORBIDDEN.sub("_", label) or "Sheet"
        name = base[-SHEET_MAX:]
        number = 2
        while name.lower() in taken:
            suffix = f" ({number})"
            name = base[-(SHEET_MAX - len(suffix)):] + suffix
            number += 1
        taken.add(name.lower())
        names[label] = name
    return names


def dataframe_to_excel(df: pd.DataFrame, about: Sequence[str] = ()) -> bytes:
    """A workbook with all rows on one sheet and one sheet per source.

    The `about` lines go on a last sheet, "About", so that the data sheets
    hold data only.

    Text is written as text: openpyxl stores a string that starts with "=" as
    a formula, so a group label such as "=A1+1" from the model or a file
    name would be evaluated when the workbook is opened.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="All")
        sources = df["source"].astype(str)
        for label, sheet in sheet_names(list(sources.unique())).items():
            df[sources == label].to_excel(writer, index=False, sheet_name=sheet)
        if about:
            about_sheet = writer.book.create_sheet("About")
            for line in about:
                about_sheet.append([line])
            about_sheet.column_dimensions["A"].width = 100
        for worksheet in writer.book.worksheets:
            for row in worksheet.iter_rows():
                for cell in row:
                    if cell.data_type == "f":
                        cell.data_type = "s"
    return buf.getvalue()


LATEX_NOTE = (
    "% PlotPick output. Needs \\usepackage{booktabs}. Characters such as\n"
    "% Greek letters or >= signs need xelatex or lualatex, or pdflatex with\n"
    "% \\usepackage{newunicodechar} and a definition for each.\n"
)


def dataframe_to_latex(df: pd.DataFrame, about: Sequence[str] = ()) -> str:
    """The table as a LaTeX tabular, with special characters escaped.

    Numbers keep up to ten significant digits: pandas' default printed 1e-7
    as 0.000000, and a group size in a column with a gap as 12.000000.
    """
    comments = "".join(f"% {line}\n" for line in about)
    return LATEX_NOTE + comments + df.to_latex(
        index=False, escape=True, na_rep="",
        float_format=lambda value: format(value, ".10g"),
    )


def dataframe_to_markdown(df: pd.DataFrame, about: Sequence[str] = ()) -> str:
    """The table as a Markdown pipe table, the `about` lines below it."""
    footer = "".join(f"\n*{line}*\n" for line in about)
    return df.to_markdown(index=False) + "\n" + footer


def results_to_json(
        results: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    """Every figure's full result by source label, after its provenance."""
    return json.dumps(
        {"provenance": provenance, "results": results},
        indent=2, ensure_ascii=False,
    )
