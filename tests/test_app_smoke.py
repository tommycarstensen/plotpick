"""End-to-end smoke tests that actually execute the Streamlit script.

These exist because two bugs shipped that no unit test could have caught:
issue #2 (Extract buttons disabled with no explanation) and an
AttributeError that crashed the Results tab whenever the model returned
`"scale": null`.  Both were only visible by running the app.
"""

import base64
import io
import json
import logging
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest
from PIL import Image

import figure_images
import process_memory

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

    def counting(name: str, data: bytes, problems: list[str]):
        rendered.append(name)
        return real(name, data, problems)

    monkeypatch.setattr(figure_images, "file_to_figures", counting)
    at = upload(app().run(), "plot.png", png_file(3000, 1500))
    assert not at.exception
    assert any("1 image(s) loaded" in c.value for c in at.caption)

    next(b for b in at.button if b.label == "Select all").click().run()
    next(c for c in at.checkbox if c.label == "plot.png").uncheck().run()
    at.selectbox[0].set_value("Haiku 4.5").run()
    assert not at.exception
    assert rendered == ["plot.png"]


def test_an_unreadable_upload_is_explained_not_a_crash(app, monkeypatch):
    """A file PDFium cannot open used to end the script with a traceback, and
    again on every rerun, until the visitor took the file out of the uploader."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    at = app().run()
    if not hasattr(at, "file_uploader"):
        pytest.skip("this Streamlit's AppTest cannot simulate an upload")
    at = at.file_uploader[0].set_value([
        ("paper.pdf", b"hello, this is not a PDF", "application/pdf"),
        ("plot.png", png_file(40, 30), "image/png"),
    ]).run()
    # The explanation stays while the file does, and the good file still loads.
    for run in (at, at.run()):
        assert not run.exception
        warnings = [w.value for w in run.warning]
        assert any("paper.pdf: could not be read as a PDF" in w for w in warnings)
        assert any("1 image(s) loaded" in c.value for c in run.caption)


def test_removing_an_upload_removes_its_figures(app, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    at = upload(app().run(), "plot.png", png_file(40, 30))
    assert any("1 image(s) loaded" in c.value for c in at.caption)
    at.file_uploader[0].set_value(None).run()
    assert not at.exception
    assert not any("image(s) loaded" in c.value for c in at.caption)
    assert any("Upload files" in i.value for i in at.info)


TABS = ["\U0001f5c2  Images", "\U0001f4cb  Results", "\U0001f4e5  Export"]


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Stand in for the API: record each request, reply with `answer` as JSON."""
    fake = SimpleNamespace(
        sent=[],
        answer={
            "figure_type": "bar chart", "y_axis": "mg/L", "scale": "linear",
            "confidence": 90, "notes": "",
            "data": [{"group": "A", "mean": 1.5, "uncertain": []}],
        },
    )

    class Client:
        def __init__(self, api_key: str):
            del api_key
            self.messages = self

        def create(self, **request):
            fake.sent.append(request)
            text = SimpleNamespace(type="text", text=json.dumps(fake.answer))
            return SimpleNamespace(stop_reason="end_turn", content=[text])

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    # The app caches one client per key, so each test needs a key of its own.
    monkeypatch.setenv("ANTHROPIC_API_KEY", f"sk-ant-fake-{uuid.uuid4().hex}")
    return fake


def extract_all(at):
    return next(b for b in at.button if "Extract all" in b.label).click().run()


def test_extraction_sends_the_upload_at_the_api_width(app, fake_anthropic):
    """End to end: an upload wider than the API limit is cut down when sent."""
    at = extract_all(upload(app().run(), "plot.png", png_file(3000, 1500)))
    assert not at.exception

    (request,) = fake_anthropic.sent
    source = request["messages"][0]["content"][0]["source"]
    image = Image.open(io.BytesIO(base64.b64decode(source["data"])))
    assert image.size == (figure_images.MAX_API_WIDTH, 1000)
    assert at.dataframe


def test_two_uploads_with_one_name_keep_two_results(app, fake_anthropic):
    """Both were sent to the model and the second result replaced the first."""
    at = app().run()
    if not hasattr(at, "file_uploader"):
        pytest.skip("this Streamlit's AppTest cannot simulate an upload")
    at = extract_all(at.file_uploader[0].set_value([
        ("fig1.png", png_file(60, 40), "image/png"),
        ("fig1.png", png_file(80, 50), "image/png"),
    ]).run())
    assert not at.exception
    assert len(fake_anthropic.sent) == 2
    assert sorted(at.session_state.results) == ["fig1.png", "fig1.png (2)"]


def test_extraction_shows_the_results_tab(app, fake_anthropic):
    """The tab is selected through Session State, not by a script in an iframe."""
    del fake_anthropic
    at = upload(app().run(), "plot.png", png_file(40, 30))
    assert [tab.label for tab in at.tabs] == TABS
    assert at.session_state.active_tab == TABS[0]

    at = extract_all(at)
    assert not at.exception
    assert at.session_state.active_tab == TABS[1]


@pytest.fixture
def arrow_complaints():
    """What Streamlit logs when it cannot send a table to the browser as it is."""
    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    logger = logging.getLogger("streamlit.dataframe_util")
    handler = Collect()
    logger.addHandler(handler)
    yield messages
    logger.removeHandler(handler)


def test_column_mixing_text_and_numbers_is_shown_as_text(
    app, fake_anthropic, arrow_complaints,
):
    """Cloud log, 2 Oct 2026: "Conversion failed for column timepoint"."""
    fake_anthropic.answer["data"] = [
        {"group": "A", "timepoint": "Baseline", "mean": 1.5, "uncertain": []},
        {"group": "A", "timepoint": 6, "mean": 2.5, "uncertain": []},
        {"group": "A", "timepoint": 70.0, "mean": None, "uncertain": ["mean"]},
    ]
    at = extract_all(upload(app().run(), "plot.png", png_file(40, 30)))
    assert not at.exception
    assert arrow_complaints == []
    shown = [frame.value["timepoint"].tolist() for frame in at.dataframe]
    assert shown == [["Baseline", "6", "70.0"]] * 2  # Results tab, Export tab


def test_uncertain_flags_outlive_the_first_display(app, fake_anthropic):
    """The Results tab popped them from the stored rows, so a rerun lost them."""
    fake_anthropic.answer["data"] = [
        {"group": "A", "mean": 1.5, "uncertain": ["mean"]},
    ]
    at = extract_all(upload(app().run(), "plot.png", png_file(40, 30)))
    at = at.run()  # any interaction runs the script again
    assert not at.exception
    (result,) = at.session_state.results.values()
    assert result["data"] == fake_anthropic.answer["data"]
    shown = [list(frame.value.columns) for frame in at.dataframe]
    assert shown == [["group", "mean"], ["source", "group", "mean"]]


def test_heavy_work_leaves_a_memory_line_in_the_log(app, fake_anthropic, monkeypatch):
    """The Cloud log of the outage had no memory figure in it at all."""
    del fake_anthropic
    lines: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    handler = Collect()
    monkeypatch.setattr(process_memory, "resident_mb", lambda: 500.0)
    process_memory.LOG.addHandler(handler)
    try:
        at = upload(app().run(), "plot.png", png_file(40, 30))
        at.selectbox[0].set_value("Haiku 4.5").run()  # a rerun: nothing heavy
        at = extract_all(at)
    finally:
        process_memory.LOG.removeHandler(handler)
    assert not at.exception
    assert [line.split(":")[0] for line in lines] == [
        "memory after reading 1 upload(s)",
        "memory after extracting 1 figure(s)",
    ]
    assert all("all sessions hold" in line for line in lines)
