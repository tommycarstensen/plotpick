# Contributing to PlotPick

Thanks for your interest in contributing to PlotPick!

## Reporting bugs

Open an issue at https://github.com/tommycarstensen/plotpick/issues with:

- Steps to reproduce
- Expected vs actual behaviour
- The chart image (if possible) and the model/provider used

## Suggesting features

Feature requests are welcome as issues. Please describe the use case and
why existing functionality does not cover it.

## Development setup

```bash
git clone https://github.com/tommycarstensen/plotpick.git
cd plotpick
pip install -r requirements.txt
pip install pytest
```

Run the app locally:

```bash
streamlit run streamlit_app.py
```

Run tests:

```bash
pytest tests/ -v
```

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Add tests for any new functionality.
3. Make sure all tests pass (`pytest tests/ -v`).
4. Keep pull requests focused -- one feature or fix per PR.

## API keys

PlotPick requires a VLM API key (Anthropic, OpenAI, or Google). If your
contribution involves a new provider, add the API caller to
`streamlit_app.py` and document the required key in the README.

## Code style

- Python 3.12+
- No additional linting configuration -- keep it readable.
