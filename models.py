"""Model catalogue and API-key resolution for the PlotPick app.

Deliberately free of Streamlit imports so the selection rules can be unit
tested without spinning up a Streamlit script run.

Which models run on the app owner's key is a policy decision, not a
technical one: the owner's key (secrets.toml or ANTHROPIC_API_KEY) pays for
Sonnet and Haiku, and visitors never see a key box for those.  Opus is
bring-your-own-key -- the owner does not pay for it -- so the key box
appears only when Opus is selected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

ENV_VAR = "ANTHROPIC_API_KEY"
SECRETS_PATH = ".streamlit/secrets.toml"


@dataclass(frozen=True)
class Model:
    """One selectable Claude model."""

    short_name: str
    model_id: str
    needs_own_key: bool
    blurb: str

    @property
    def label(self) -> str:
        """Text shown in the model dropdown."""
        if self.needs_own_key:
            return f"{self.short_name} (bring your own key)"
        return self.short_name


# Ordered -- the first entry is the default selection.
#
# Sonnet leads.  The ChartX validation split (299 paired figures) was run on
# the previous Sonnet, 4.6, which reached 92.2% mean recall against Haiku
# 4.5's 88.5% -- a +3.7 point gap, 95% bootstrap CI [+2.7, +4.8], holding
# across all six chart types (validation/results/final_val/).  Sonnet 5.5
# replaces it at a lower per-token price and has not been benchmarked yet.
MODELS: tuple[Model, ...] = (
    Model(
        short_name="Sonnet 5.5",
        model_id="claude-sonnet-5-5",
        needs_own_key=False,
        blurb="Latest Sonnet. Its predecessor reached 92.2% mean recall on ChartX.",
    ),
    Model(
        short_name="Haiku 4.5",
        model_id="claude-haiku-4-5-20251001",
        needs_own_key=False,
        blurb="Faster and cheaper -- 88.5% mean recall on ChartX.",
    ),
    Model(
        short_name="Opus 5.5",
        model_id="claude-opus-5-5",
        needs_own_key=True,
        blurb="Anthropic's most capable Opus. Runs on your own API key.",
    ),
)

MODELS_BY_LABEL: dict[str, Model] = {m.label: m for m in MODELS}
MODEL_LABELS: list[str] = [m.label for m in MODELS]
DEFAULT_MODEL: Model = MODELS[0]


def shared_key_from_environment() -> str:
    """Read the app's API key from the environment.

    Streamlit secrets are handled by the caller (importing ``st`` here would
    defeat the point of this module); this covers the equally common case of
    an exported ``ANTHROPIC_API_KEY``.
    """
    return os.environ.get(ENV_VAR, "").strip()


def resolve_api_key(model: Model, user_key: str, shared_key: str) -> str:
    """Return the key to call the API with, or "" when none is available.

    The owner's key pays for every model except those that need the
    visitor's own key; those get only the key the visitor typed, never the
    owner's.
    """
    if model.needs_own_key:
        return (user_key or "").strip()
    return (shared_key or "").strip()


def missing_key_message(model: Model) -> str:
    """Explain what to do when `resolve_api_key` came back empty.

    The app used to disable the Extract buttons with no explanation at all,
    which reads as a broken UI rather than as missing configuration
    (github.com/tommycarstensen/plotpick/issues/2).
    """
    if model.needs_own_key:
        alternatives = " or ".join(
            m.short_name for m in MODELS if not m.needs_own_key
        )
        return (
            f"{model.short_name} runs on your own Anthropic API key. Paste one "
            f"above, or switch to {alternatives}."
        )
    # Visitors cannot supply a key for these models, so this is addressed to
    # whoever deploys the app.
    return (
        "No Anthropic API key is configured, so extraction is disabled. Set "
        f"{ENV_VAR} in the environment, in {SECRETS_PATH}, or in the app's "
        "Secrets settings on Streamlit Community Cloud."
    )


class ExtractionError(Exception):
    """The API call succeeded but returned no usable answer."""


def reply_text(response: Any) -> str:
    """Return the text of a Messages API response.

    Sonnet 5.5 and Opus 5.5 think by default, so the first content block can
    be a ``thinking`` block with empty text -- reading ``content[0]`` by
    position then hands an empty string to the JSON parser.  Blocks are read
    by type instead, and the stop reasons that leave no usable text are
    reported as such rather than as a JSON syntax error.
    """
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise ExtractionError(
            "Claude declined to process this figure"
            + (f" (category: {category})." if category else ".")
        )
    text = "".join(b.text for b in response.content if b.type == "text")
    if response.stop_reason == "max_tokens":
        raise ExtractionError(
            "Claude's answer hit the output-token limit before it finished, "
            "so the table would be incomplete."
        )
    if not text.strip():
        raise ExtractionError("Claude returned no text for this figure.")
    return text


def extract_blocked_reason(
    api_key: str,
    n_loaded: int,
    n_selected: int,
    *,
    selected_only: bool,
) -> str | None:
    """Why an Extract button is disabled, or None when it is clickable.

    Returned verbatim as the button's tooltip so a disabled button always
    says what is missing.
    """
    if not api_key:
        return "No Anthropic API key available -- see the sidebar."
    if n_loaded == 0:
        return "Upload a figure or enter a PubMed ID first."
    if selected_only and n_selected == 0:
        return "Tick at least one figure in the gallery first."
    return None


def figure_summary(result: dict) -> dict[str, str]:
    """Display strings for the metadata banner above an extracted table.

    Every field can legitimately come back as JSON ``null``: the extraction
    prompt tells the model to use null for anything it cannot read.  A
    ``dict.get`` default only fires for a *missing* key, so a present-but-null
    value used to reach ``.capitalize()`` and raise AttributeError, crashing
    the Results tab after a successful extraction.
    """
    confidence = result.get("confidence")
    return {
        "figure_type": str(result.get("figure_type") or "?").capitalize(),
        "y_axis": str(result.get("y_axis") or "?"),
        "scale": str(result.get("scale") or "?").capitalize(),
        "confidence": "?" if confidence is None else str(confidence),
        "notes": str(result.get("notes") or ""),
    }
