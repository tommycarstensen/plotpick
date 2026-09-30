"""End-to-end smoke tests that actually execute the Streamlit script.

These exist because two bugs shipped that no unit test could have caught:
issue #2 (Extract buttons disabled with no explanation) and an
AttributeError that crashed the Results tab whenever the model returned
`"scale": null`.  Both were only visible by running the app.
"""

import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"
sys.path.insert(0, str(APP.parent))

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="needs streamlit"
).AppTest


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Run the app from a scratch cwd so no local secrets.toml leaks in."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    return lambda: AppTest.from_file(str(APP), default_timeout=90)


def test_app_starts_without_a_key(app):
    """A missing key must not raise -- it used to just silently disable."""
    at = app().run()
    assert not at.exception


def test_missing_key_is_explained_not_silent(app):
    """Issue #2: the only symptom was a not-allowed cursor."""
    at = app().run()
    warnings = [w.value for w in at.warning]
    assert any("ANTHROPIC_API_KEY" in w for w in warnings), warnings


def test_disabled_extract_buttons_state_a_reason(app):
    at = app().run()
    extract = [b for b in at.button if "Extract" in b.label]
    assert len(extract) == 2
    for button in extract:
        assert button.disabled
        assert button.help, f"{button.label} is disabled with no explanation"
        assert "API key" in button.help


def test_env_var_key_is_picked_up(app, monkeypatch):
    """ANTHROPIC_API_KEY used to be ignored -- secrets.toml was the only source."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    at = app().run()
    assert not at.exception
    assert not [w for w in at.warning if "API key" in w.value]
    # Now only the absence of images blocks extraction.
    for button in (b for b in at.button if "Extract" in b.label):
        assert "Upload" in button.help


OPUS = "Opus 5.5 (bring your own key)"


def test_model_choices_and_default(app):
    """Visitors land on Sonnet."""
    at = app().run()
    options = at.selectbox[0].options
    assert options == ["Sonnet 5.5", "Haiku 4.5", OPUS]
    assert at.selectbox[0].value == "Sonnet 5.5"


@pytest.mark.parametrize("model", ["Sonnet 5.5", "Haiku 4.5"])
def test_sonnet_and_haiku_run_on_the_app_key(app, monkeypatch, model):
    """No key box, no warning: the owner's key pays."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    at = app().run()
    at.selectbox[0].set_value(model).run()
    assert not at.exception
    assert not [t for t in at.text_input if "API key" in t.label]
    assert not [w for w in at.warning if "API key" in w.value]
    for button in (b for b in at.button if "Extract" in b.label):
        assert "Upload" in button.help


def test_opus_refuses_the_app_key(app, monkeypatch):
    """The owner does not pay for Opus: it needs the visitor's own key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    at = app().run()
    at.selectbox[0].set_value(OPUS).run()
    assert not at.exception
    assert [t for t in at.text_input if "API key" in t.label]
    assert any("your own" in w.value for w in at.warning)
    for button in (b for b in at.button if "Extract" in b.label):
        assert "API key" in button.help


def test_opus_runs_on_a_pasted_key(app, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    at = app().run()
    at.selectbox[0].set_value(OPUS).run()
    key_box = next(t for t in at.text_input if "API key" in t.label)
    key_box.input("sk-ant-visitor").run()
    assert not at.exception
    assert not [w for w in at.warning if "API key" in w.value]
    for button in (b for b in at.button if "Extract" in b.label):
        assert "Upload" in button.help
