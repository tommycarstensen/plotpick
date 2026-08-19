"""Model catalogue and API-key resolution for the PlotPick app.

Deliberately free of Streamlit imports so the selection rules can be unit
tested without spinning up a Streamlit script run.

Which models need the user's own key is a policy decision, not a technical
one: the shared key that ships with the hosted demo covers the two cheaper
models, and Opus is left as bring-your-own-key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
# Sonnet 4.6 leads because it is measurably the most accurate of the two
# shared-key models: on the ChartX validation split (299 paired figures) it
# reaches 92.2% mean recall against Haiku's 88.5%, a +3.7 point gap whose
# 95% bootstrap CI is [+2.7, +4.8] and which holds across all six chart
# types.  See validation/results/final_val/.
MODELS: tuple[Model, ...] = (
    Model(
        short_name="Sonnet 4.6",
        model_id="claude-sonnet-4-6",
        needs_own_key=False,
        blurb="Most accurate shared-key model -- 92.2% mean recall on ChartX.",
    ),
    Model(
        short_name="Haiku 4.5",
        model_id="claude-haiku-4-5-20251001",
        needs_own_key=False,
        blurb="Faster and cheaper -- 88.5% mean recall on ChartX.",
    ),
    Model(
        short_name="Opus 5",
        model_id="claude-opus-5",
        needs_own_key=True,
        blurb="Anthropic's most capable model. Requires your own API key.",
    ),
)

MODELS_BY_LABEL: dict[str, Model] = {m.label: m for m in MODELS}
MODEL_LABELS: list[str] = [m.label for m in MODELS]
DEFAULT_MODEL: Model = MODELS[0]


def shared_key_from_environment() -> str:
    """Read the fallback API key from the environment.

    Streamlit secrets are handled by the caller (importing ``st`` here would
    defeat the point of this module); this covers the equally common case of
    an exported ``ANTHROPIC_API_KEY``.
    """
    return os.environ.get(ENV_VAR, "").strip()


def resolve_api_key(model: Model, user_key: str, shared_key: str) -> str:
    """Return the key to call the API with, or "" when none is available.

    A key typed into the sidebar always wins.  The shared key is only offered
    to models that do not require the user to bring their own.
    """
    user_key = (user_key or "").strip()
    if model.needs_own_key:
        return user_key
    return user_key or (shared_key or "").strip()


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
            f"{model.short_name} requires your own Anthropic API key. "
            f"Paste one above, or switch to {alternatives}."
        )
    return (
        "No Anthropic API key found, so extraction is disabled. Paste a key "
        f"above, or set {ENV_VAR} in your environment, or add it to "
        f"{SECRETS_PATH}."
    )


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
        return "Add an Anthropic API key in the sidebar first."
    if n_loaded == 0:
        return "Upload a figure or enter a PubMed ID first."
    if selected_only and n_selected == 0:
        return "Tick at least one figure in the gallery first."
    return None
