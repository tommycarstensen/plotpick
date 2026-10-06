"""Tests for models.py -- model catalogue and API-key resolution.

Regression cover for issue #2, where Extract stayed disabled with no
explanation because no API key had been configured.
"""

from types import SimpleNamespace

import pytest

from models import (
    DEFAULT_MODEL,
    ENV_VAR,
    MODEL_LABELS,
    MODELS,
    MODELS_BY_LABEL,
    extract_blocked_reason,
    figure_summary,
    ExtractionError,
    missing_key_message,
    reply_text,
    resolve_api_key,
    shared_key_from_environment,
)

SHARED = "sk-ant-shared"
OWN = "sk-ant-own"

BYOK = next(m for m in MODELS if m.needs_own_key)
FREE = next(m for m in MODELS if not m.needs_own_key)


# -- catalogue --------------------------------------------------------------

def test_default_is_latest_sonnet():
    assert DEFAULT_MODEL.short_name == "Sonnet 5.5"
    assert DEFAULT_MODEL.model_id == "claude-sonnet-5-5"
    assert not DEFAULT_MODEL.needs_own_key


def test_model_ids_are_current():
    assert [m.model_id for m in MODELS] == [
        "claude-sonnet-5-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-5-5",
    ]


def test_only_opus_requires_own_key():
    """The owner pays for Sonnet and Haiku only."""
    byok = {m.short_name for m in MODELS if m.needs_own_key}
    assert byok == {"Opus 5.5"}


def test_labels_are_unique_and_flag_byok():
    assert len(MODEL_LABELS) == len(set(MODEL_LABELS)) == len(MODELS)
    assert MODEL_LABELS == [
        "Sonnet 5.5",
        "Haiku 4.5",
        "Opus 5.5 (bring your own key)",
    ]
    assert set(MODELS_BY_LABEL) == set(MODEL_LABELS)


# -- key resolution ---------------------------------------------------------

@pytest.mark.parametrize("model", [m for m in MODELS if not m.needs_own_key])
def test_shared_key_pays_for_sonnet_and_haiku(model):
    assert resolve_api_key(model, "", SHARED) == SHARED


def test_shared_key_never_used_for_byok_model():
    assert resolve_api_key(BYOK, "", SHARED) == ""


def test_byok_model_uses_the_visitors_key():
    assert resolve_api_key(BYOK, OWN, SHARED) == OWN


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_keys_are_normalised(blank):
    assert resolve_api_key(FREE, "", blank) == ""
    assert resolve_api_key(BYOK, blank, SHARED) == ""


def test_whitespace_is_stripped():
    assert resolve_api_key(FREE, "", f"  {SHARED}  ") == SHARED
    assert resolve_api_key(BYOK, f"  {OWN}  ", "") == OWN


def test_env_var_is_read(monkeypatch):
    monkeypatch.setenv(ENV_VAR, f"  {SHARED}  ")
    assert shared_key_from_environment() == SHARED
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert shared_key_from_environment() == ""


# -- disabled-button messaging (issue #2) -----------------------------------

def test_missing_shared_key_message_names_the_config_locations():
    msg = missing_key_message(FREE)
    assert ENV_VAR in msg and "secrets.toml" in msg
    assert "paste" not in msg.lower()


def test_missing_key_message_for_byok_suggests_alternatives():
    msg = missing_key_message(BYOK)
    assert "Opus 5.5" in msg
    assert "Sonnet 5.5" in msg and "Haiku 4.5" in msg


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


# -- reply_text: Sonnet 5.5 / Opus 5.5 lead with a thinking block -----------

def _block(kind, text=""):
    return SimpleNamespace(type=kind, text=text, thinking="")


def _response(*blocks, stop_reason="end_turn", category=None):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
    )


def test_reply_text_skips_a_leading_thinking_block():
    """content[0] is an empty thinking block under adaptive thinking."""
    response = _response(_block("thinking"), _block("text", '{"data": []}'))
    assert reply_text(response) == '{"data": []}'


def test_reply_text_plain_text_response():
    assert reply_text(_response(_block("text", "{}"))) == "{}"


def test_refusal_is_reported_with_its_category():
    with pytest.raises(ExtractionError, match=r"declined.*bio"):
        reply_text(_response(stop_reason="refusal", category="bio"))


def test_truncated_answer_is_reported():
    response = _response(_block("text", '{"data": [1, 2'), stop_reason="max_tokens")
    with pytest.raises(ExtractionError, match="output-token limit"):
        reply_text(response)


def test_empty_answer_is_reported():
    with pytest.raises(ExtractionError, match="no text"):
        reply_text(_response(_block("thinking")))


# -- figure_summary: JSON null crashed the Results tab ----------------------

def test_null_fields_do_not_crash():
    """`"scale": null` reached .capitalize() and raised AttributeError."""
    summary = figure_summary(
        {"figure_type": None, "y_axis": None, "scale": None,
         "confidence": None, "notes": None}
    )
    assert summary == {
        "figure_type": "?", "y_axis": "?", "scale": "?",
        "confidence": "not given", "notes": "",
    }


def test_missing_fields_fall_back():
    assert figure_summary({})["scale"] == "?"


def test_populated_fields_are_capitalised():
    summary = figure_summary(
        {"figure_type": "boxplot", "y_axis": "IL-6", "scale": "linear",
         "confidence": 76, "notes": "overlapping"}
    )
    assert summary["figure_type"] == "Boxplot"
    assert summary["scale"] == "Linear"
    assert summary["y_axis"] == "IL-6"
    assert summary["confidence"] == "76/100"
    assert summary["notes"] == "overlapping"


def test_white_space_is_collapsed():
    """A blank line ended the HTML banner, and what followed rendered as
    Markdown, an image link included."""
    summary = figure_summary(
        {"y_axis": "mg/L\n\n![](https://example.invalid/p.png)",
         "confidence": "7\n\n0", "notes": "a\n\nb"}
    )
    assert summary["y_axis"] == "mg/L ![](https://example.invalid/p.png)"
    assert summary["confidence"] == "7 0/100"
    assert summary["notes"] == "a b"


def test_zero_confidence_is_not_reported_as_unknown():
    """0 is falsy but a real score -- it must not become "not given"."""
    assert figure_summary({"confidence": 0})["confidence"] == "0/100"
