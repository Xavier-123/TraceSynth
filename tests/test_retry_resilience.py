from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import MagicMock, patch

from openai import APIError, RateLimitError

from tracesynth.configuration import ModelConfiguration
from tracesynth.functions.call_llms import ParseError, call_and_parse, is_retryable_api_error
from tracesynth.functions.fuzzy_task import _parse_fuzzy_task_response
from tracesynth.functions.mock_tools import _parse_mock_tool_response
from tracesynth.functions.mock_user import _parse_mock_user_response
from tracesynth.functions.tool_check import _parse_checked_tools
from tracesynth.graph.graph_virtual_tools import get_tool_call_max_retries, should_call_tool, validate_tool_call


def test_is_retryable_api_error():
    assert is_retryable_api_error(
        RateLimitError("rate", response=MagicMock(status_code=429), body=None)
    )
    err500 = APIError("server", request=MagicMock(), body=None)
    err500.status_code = 500
    assert is_retryable_api_error(err500)
    err400 = APIError("bad", request=MagicMock(), body=None)
    err400.status_code = 400
    assert not is_retryable_api_error(err400)


def test_parse_checked_tools():
    tools = _parse_checked_tools(
        '<tools>[{"name":"t","parameters":{"type":"object","properties":{}}}]</tools>'
    )
    assert len(tools) == 1
    try:
        _parse_checked_tools("no tags")
        raise AssertionError("expected ParseError")
    except ParseError:
        pass


def test_parse_fuzzy_task():
    task, bg = _parse_fuzzy_task_response("<task>hello</task><background>world</background>")
    assert task == "hello"
    assert bg == "world"


def test_call_and_parse_resampling():
    class Cfg:
        api_base = "http://test"
        api_key = "k"
        model_name = "m"
        max_tokens = 100
        temperature = 0.1
        use_thinking = False
        api_max_retries = 1
        api_retry_base = 0.01
        parse_max_retries = 2

    responses = ["bad", "bad", "<reply>ok</reply>"]
    call_count = {"n": 0}

    def fake_call_llm_messages(**kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        msgs = list(kwargs["messages"])
        msgs.append({"role": "assistant", "content": responses[idx]})
        return msgs

    with patch("tracesynth.functions.call_llms.call_llm_messages", side_effect=fake_call_llm_messages):
        result, _ = call_and_parse(
            Cfg(),
            [{"role": "user", "content": "hi"}],
            _parse_mock_user_response,
            step_name="test",
        )
        assert result == "ok"
        assert call_count["n"] == 3


def test_call_and_parse_feedback_on_retry():
    class Cfg:
        api_base = "http://test"
        api_key = "k"
        model_name = "m"
        max_tokens = 100
        temperature = 0.1
        use_thinking = False
        api_max_retries = 1
        api_retry_base = 0.01
        parse_max_retries = 2

    responses = ["bad", "bad", "<reply>ok</reply>"]
    call_count = {"n": 0}
    captured_messages = []

    def fake_call_llm_messages(**kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        captured_messages.append(list(kwargs["messages"]))
        msgs = list(kwargs["messages"])
        msgs.append({"role": "assistant", "content": responses[idx]})
        return msgs

    with patch("tracesynth.functions.call_llms.call_llm_messages", side_effect=fake_call_llm_messages):
        result, _ = call_and_parse(
            Cfg(),
            [{"role": "user", "content": "hi"}],
            _parse_mock_user_response,
            step_name="test",
        )
        assert result == "ok"
        assert call_count["n"] == 3
        assert len(captured_messages[1]) == 3
        assert captured_messages[1][1]["role"] == "assistant"
        assert captured_messages[1][1]["content"] == "bad"
        assert "上一次输出无法解析" in captured_messages[1][2]["content"]
        assert len(captured_messages[2]) == 5
        assert "上一次输出无法解析" in captured_messages[2][4]["content"]

def test_call_and_parse_reraises_api_errors():
    class Cfg:
        api_base = "http://test"
        api_key = "k"
        model_name = "m"
        max_tokens = 100
        temperature = 0.1
        use_thinking = False
        api_max_retries = 1
        api_retry_base = 0.01
        parse_max_retries = 2

    api_error = RuntimeError("upstream unavailable")

    with patch("tracesynth.functions.call_llms.call_llm_messages", side_effect=api_error):
        try:
            call_and_parse(
                Cfg(),
                [{"role": "user", "content": "hi"}],
                _parse_mock_user_response,
                step_name="test",
            )
            raise AssertionError("expected API error to be re-raised")
        except RuntimeError as exc:
            assert exc is api_error


def test_parse_mock_tool_response_lenient():
    response, new_bg = _parse_mock_tool_response(
        "<TOOL_RESPONSE>hello</TOOL_RESPONSE><new_bg_introduced>NO</new_bg_introduced>"
    )
    assert response == "hello"
    assert new_bg is False

    response, new_bg = _parse_mock_tool_response("<tool_response attr='x'>open only")
    assert response == "open only"
    assert new_bg is True

    response, new_bg = _parse_mock_tool_response("plain tool output without tags")
    assert response == "plain tool output without tags"
    assert new_bg is True

    try:
        _parse_mock_tool_response("<tools>not a tool response</tools>")
        raise AssertionError("expected ParseError")
    except ParseError:
        pass


def test_model_configuration_defaults():
    cfg = ModelConfiguration(model_name="m")
    assert cfg.api_max_retries == 3
    assert cfg.parse_max_retries == 2
    assert cfg.tool_call_max_retries == 3


def test_graph_helpers():
    assert get_tool_call_max_retries({"configurable": {"retry": {"tool_call_max_retries": 5}}}) == 5
    assert should_call_tool({"breaked": False, "task_finished": "Retry solve"}) == "retry_solve"
    valid, _ = validate_tool_call('{"name":"t","arguments":{}}', [{"name": "t", "parameters": {}}])
    assert valid


if __name__ == "__main__":
    test_is_retryable_api_error()
    test_parse_checked_tools()
    test_parse_fuzzy_task()
    test_call_and_parse_resampling()
    test_call_and_parse_feedback_on_retry()
    test_call_and_parse_reraises_api_errors()
    test_parse_mock_tool_response_lenient()
    test_model_configuration_defaults()
    test_graph_helpers()
    print("All verification tests passed")
