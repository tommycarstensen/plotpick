# Contributing to PlotPick

Thanks for your interest in contributing to PlotPick! It has one maintainer, Tommy Carstensen, and is maintained on a best-effort basis.

## Getting help

Ask questions by opening an issue at https://github.com/tommycarstensen/plotpick/issues. Say what you did, what you expected and what happened.

## Reporting bugs

Open an issue at https://github.com/tommycarstensen/plotpick/issues with:

- Steps to reproduce
- Expected vs actual behaviour
- Whether you used the hosted app or a local copy, and the model you selected
- The chart image or PDF, if you can share it

Never paste an API key into an issue.

## Suggesting features

Feature requests are welcome as issues. Please describe the use case and why existing functionality does not cover it.

## Development setup

```bash
git clone https://github.com/tommycarstensen/plotpick.git && cd plotpick && pip install -r requirements.txt pytest ruff pycodestyle pyright
```

Run the app locally:

```bash
streamlit run streamlit_app.py
```

Run what continuous integration runs (the tests need no API key):

```bash
pytest tests/ -v && ruff check . && pycodestyle . && pyright models.py streamlit_app.py pdf_figures.py figure_images.py process_memory.py pdf_backend_pdfium.py pmc.py exports.py tests
```

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Add tests for any new functionality.
3. Make sure pytest, ruff, pycodestyle and pyright pass; CI runs the tests on Python 3.12 and 3.13 and the linters on 3.12.
4. Keep pull requests focused -- one feature or fix per PR.

## API keys

PlotPick requires an Anthropic API key; it has no other backend. If your contribution adds another provider, add its API caller to `streamlit_app.py` and document the required key in the README.

## Code style

- Python 3.12+
- ruff (`ruff.toml`) and pycodestyle (`setup.cfg`), 88 columns; pyright with no errors. A new module goes on the pyright line in `.github/workflows/test.yml`.
- No non-ASCII characters in plain text.
