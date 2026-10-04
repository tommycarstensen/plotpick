"""End-to-end smoke tests that actually execute the Streamlit script.

These exist because two bugs shipped that no unit test could have caught:
issue #2 (Extract buttons disabled with no explanation) and an
AttributeError that crashed the Results tab whenever the model returned
`"scale": null`.  Both were only visible by running the app.
"""

import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest
from PIL import Image

import figure_images

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


def png_file(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def upload(at, name: str, content: bytes):
    """Put a file in the uploader and rerun, as a visitor's upload does."""
    if not hasattr(at, "file_uploader"):
        pytest.skip("this Streamlit's AppTest cannot simulate an upload")
    return at.file_uploader[0].upload(name, content).run()


def test_upload_is_rendered_once_not_on_every_rerun(app, monkeypatch):
    """Rendering every upload again on each click took the app over its memory limit."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    rendered: list[str] = []
    real = figure_images.file_to_figures

    def counting(name: str, data: bytes):
        rendered.append(name)
        return real(name, data)

    monkeypatch.setattr(figure_images, "file_to_figures", counting)
    at = upload(app().run(), "plot.png", png_file(3000, 1500))
    assert not at.exception
    assert any("1 image(s) loaded" in c.value for c in at.caption)

    next(b for b in at.button if b.label == "Select all").click().run()
    next(c for c in at.checkbox if c.label == "plot.png").uncheck().run()
    at.selectbox[0].set_value("Haiku 4.5").run()
    assert not at.exception
    assert rendered == ["plot.png"]


def test_removing_an_upload_removes_its_figures(app, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    at = upload(app().run(), "plot.png", png_file(40, 30))
    assert any("1 image(s) loaded" in c.value for c in at.caption)
    at.file_uploader[0].set_value(None).run()
    assert not at.exception
    assert not any("image(s) loaded" in c.value for c in at.caption)
    assert any("Upload files" in i.value for i in at.info)


def test_extraction_sends_the_upload_at_the_api_width(app, monkeypatch):
    """End to end: an upload wider than the API limit is cut down when sent."""
    sent: list[dict] = []
    answer = {
        "figure_type": "bar chart", "y_axis": "mg/L", "scale": "linear",
        "confidence": 90, "notes": "",
        "data": [{"group": "A", "mean": 1.5, "uncertain": []}],
    }

    class FakeAnthropic:
        def __init__(self, api_key: str):
            del api_key
            self.messages = self

        def create(self, **request):
            sent.append(request)
            text = SimpleNamespace(type="text", text=json.dumps(answer))
            return SimpleNamespace(stop_reason="end_turn", content=[text])

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-extraction")
    at = upload(app().run(), "plot.png", png_file(3000, 1500))
    next(b for b in at.button if "Extract all" in b.label).click().run()
    assert not at.exception

    (request,) = sent
    source = request["messages"][0]["content"][0]["source"]
    image = Image.open(io.BytesIO(base64.b64decode(source["data"])))
    assert image.size == (figure_images.MAX_API_WIDTH, 1000)
    assert at.dataframe
