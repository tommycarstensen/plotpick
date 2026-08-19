"""Tests for models.py -- model catalogue and API-key resolution.

Regression cover for issue #2, where Extract stayed disabled with no
explanation because no API key had been configured.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (  # noqa: E402
    DEFAULT_MODEL,
    ENV_VAR,
    MODEL_LABELS,
    MODELS,
    MODELS_BY_LABEL,
    extract_blocked_reason,
    missing_key_message,
    resolve_api_key,
    shared_key_from_environment,
)

SHARED = "sk-ant-shared"
OWN = "sk-ant-own"

BYOK = next(m for m in MODELS if m.needs_own_key)
FREE = next(m for m in MODELS if not m.needs_own_key)


# -- catalogue --------------------------------------------------------------

def test_default_is_sonnet_not_haiku():
    """Sonnet beats Haiku on ChartX (92.2% vs 88.5%), so it is the default."""
    assert DEFAULT_MODEL.short_name == "Sonnet 4.6"
    assert DEFAULT_MODEL.model_id == "claude-sonnet-4-6"
    assert not DEFAULT_MODEL.needs_own_key


def test_only_opus_requires_own_key():
    byok = {m.short_name for m in MODELS if m.needs_own_key}
    assert byok == {"Opus 5"}


def test_haiku_available_without_own_key():
    haiku = MODELS_BY_LABEL["Haiku 4.5"]
    assert not haiku.needs_own_key
    assert resolve_api_key(haiku, "", SHARED) == SHARED


def test_labels_are_unique_and_flag_byok():
    assert len(MODEL_LABELS) == len(set(MODEL_LABELS)) == len(MODELS)
    assert MODELS_BY_LABEL["Opus 5 (bring your own key)"].needs_own_key
    assert "bring your own key" not in MODELS_BY_LABEL["Haiku 4.5"].label


# -- key resolution ---------------------------------------------------------

def test_shared_key_used_for_non_byok_models():
    assert resolve_api_key(FREE, "", SHARED) == SHARED


def test_shared_key_never_used_for_byok_model():
    assert resolve_api_key(BYOK, "", SHARED) == ""


def test_own_key_wins_over_shared():
    assert resolve_api_key(FREE, OWN, SHARED) == OWN
    assert resolve_api_key(BYOK, OWN, SHARED) == OWN


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_keys_are_normalised(blank):
    assert resolve_api_key(FREE, blank, blank) == ""


def test_whitespace_is_stripped():
    assert resolve_api_key(FREE, f"  {OWN}  ", "") == OWN


def test_env_var_is_read(monkeypatch):
    monkeypatch.setenv(ENV_VAR, f"  {SHARED}  ")
    assert shared_key_from_environment() == SHARED
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert shared_key_from_environment() == ""


# -- disabled-button messaging (issue #2) -----------------------------------

def test_missing_key_message_names_the_env_var():
    msg = missing_key_message(FREE)
    assert ENV_VAR in msg and "secrets.toml" in msg


def test_missing_key_message_for_byok_suggests_alternatives():
    msg = missing_key_message(BYOK)
    assert "Opus 5" in msg
    assert "Sonnet 4.6" in msg and "Haiku 4.5" in msg


def test_no_key_blocks_both_buttons_with_a_reason():
    for selected_only in (False, True):
        reason = extract_blocked_reason(
            "", n_loaded=5, n_selected=2, selected_only=selected_only
        )
        assert reason and "API key" in reason


def test_missing_images_reported_before_selection():
    reason = extract_blocked_reason(
        SHARED, n_loaded=0, n_selected=0, selected_only=False
    )
    assert reason and "Upload" in reason


def test_selected_only_requires_a_selection():
    assert extract_blocked_reason(
        SHARED, n_loaded=3, n_selected=0, selected_only=True
    ) is not None
    assert extract_blocked_reason(
        SHARED, n_loaded=3, n_selected=0, selected_only=False
    ) is None


def test_button_enabled_when_key_and_images_present():
    assert extract_blocked_reason(
        SHARED, n_loaded=3, n_selected=1, selected_only=True
    ) is None
    assert extract_blocked_reason(
        SHARED, n_loaded=3, n_selected=1, selected_only=False
    ) is None
