"""Tests for extraction.py: reading the model's reply as a result."""

import json

import pytest

from extraction import SYSTEM_PROMPT, USER_PROMPT, parse_reply
from models import ExtractionError

REPLY = {"figure_type": "bar chart", "data": [{"group": "A", "mean": 1.5}]}


def test_plain_json_is_read():
    assert parse_reply(json.dumps(REPLY)) == REPLY


def test_code_fences_are_stripped():
    assert parse_reply(f"```json\n{json.dumps(REPLY)}\n```") == REPLY


def test_a_reply_without_rows_gets_an_empty_list():
    assert parse_reply('{"figure_type": "flow chart"}')["data"] == []


def test_text_that_is_not_json_is_reported_with_its_start():
    with pytest.raises(json.JSONDecodeError, match=r"(?s)Raw response.*I cannot read"):
        parse_reply("I cannot read this figure.")


@pytest.mark.parametrize("reply", [
    "[1, 2, 3]",
    '"just a string"',
    '{"data": null}',
    '{"data": {"group": "A"}}',
    '{"data": [1.5, 2.5]}',
])
def test_json_of_another_shape_fails_that_figure(reply):
    """It used to raise a TypeError outside the handled errors and lose the
    batch's earlier results."""
    with pytest.raises(ExtractionError, match="Claude"):
        parse_reply(reply)


def test_the_prompt_describes_the_three_chart_types():
    assert "boxplots, bar charts, and line plots" in SYSTEM_PROMPT
    assert "JSON" in USER_PROMPT
