from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import MagicMock, patch

from openai import APIError, RateLimitError

from tracesynth.configuration import ModelConfiguration
from tracesynth.functions.call_llms import (
    ParseError,
    call_and_parse,
    create_chat_completion_with_retry,
    is_retryable_api_error,
    messages_for_chat_completion,
)
from tracesynth.functions.fuzzy_task import _parse_fuzzy_task_response
from tracesynth.functions.mock_tools import _parse_mock_tool_response, mock_tool_response
from tracesynth.functions.mock_user import _parse_mock_user_response
from tracesynth.functions.solve_task import _parse_solver_response
from tracesynth.functions.tool_check import _parse_checked_tools
from tracesynth.graph.graph_virtual_tools import (
    execute_plan_node,
    should_continue_execution,
    should_execute_or_replan,
    validate_tool_call,
)
from tracesynth.graph.node_utils import get_tool_call_max_retries


class RetryCfg:
    api_base = "http://test"
    api_key = "k"
    model_name = "m"
    max_tokens = 100
    temperature = 0.1
    use_thinking = False
    api_max_retries = 1
    api_retry_base = 0.01
    parse_max_retries = 2


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
        '{"tools":[{"name":"t","parameters":{"type":"object","properties":{}}}]}'
    )
    assert len(tools) == 1
    try:
        _parse_checked_tools("no json")
        raise AssertionError("expected ParseError")
    except ParseError:
        pass


def test_parse_fuzzy_task():
    task, bg = _parse_fuzzy_task_response('{"task":"hello","background":"world"}')
    assert task == "hello"
    assert bg == "world"


def test_call_and_parse_resampling():
    responses = ["bad", "bad", '{"reply":"ok"}']
    call_count = {"n": 0}

    def fake_call_llm_messages(**kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        msgs = list(kwargs["messages"])
        msgs.append({"role": "assistant", "content": responses[idx]})
        return msgs

    with patch("tracesynth.functions.call_llms.call_llm_messages", side_effect=fake_call_llm_messages):
        result, _ = call_and_parse(
            RetryCfg(),
            [{"role": "user", "content": "hi"}],
            _parse_mock_user_response,
            step_name="test",
        )
        assert result == "ok"
        assert call_count["n"] == 3


def test_call_and_parse_feedback_on_retry():
    responses = ["bad", "bad", '{"reply":"ok"}']
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
            RetryCfg(),
            [{"role": "user", "content": "hi"}],
            _parse_mock_user_response,
            step_name="test",
        )
        assert result == "ok"
        assert call_count["n"] == 3
        assert len(captured_messages[1]) == 3
        assert captured_messages[1][1]["role"] == "assistant"
        assert captured_messages[1][1]["content"] == "bad"
        assert "Previous output could not be parsed" in captured_messages[1][2]["content"]
        assert len(captured_messages[2]) == 5
        assert "Previous output could not be parsed" in captured_messages[2][4]["content"]


def test_call_and_parse_returns_last_messages_after_parse_exhaustion():
    class Cfg(RetryCfg):
        parse_max_retries = 1

    responses = ["bad-one", "bad-two"]
    call_count = {"n": 0}

    def fake_call_llm_messages(**kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        msgs = list(kwargs["messages"])
        msgs.append({"role": "assistant", "content": responses[idx]})
        return msgs

    with patch("tracesynth.functions.call_llms.call_llm_messages", side_effect=fake_call_llm_messages):
        result, messages = call_and_parse(
            Cfg(),
            [{"role": "user", "content": "hi"}],
            _parse_mock_user_response,
            step_name="test",
        )
        assert result is None
        assert messages[-1] == {"role": "assistant", "content": "bad-two"}


def test_call_and_parse_json_mode_sets_response_format():
    captured = []

    def fake_call_llm_messages(**kwargs):
        captured.append(kwargs)
        msgs = list(kwargs["messages"])
        msgs.append({"role": "assistant", "content": '{"reply":"ok"}'})
        return msgs

    with patch("tracesynth.functions.call_llms.call_llm_messages", side_effect=fake_call_llm_messages):
        result, _ = call_and_parse(
            RetryCfg(),
            [{"role": "user", "content": "hi"}],
            _parse_mock_user_response,
            step_name="test",
            json_mode=True,
        )
    assert result == "ok"
    assert captured[0]["llm_params"]["response_format"] == {"type": "json_object"}


def test_call_and_parse_json_mode_keeps_configured_response_format():
    class Cfg(RetryCfg):
        llm_params = {"response_format": {"type": "text"}}

    captured = []

    def fake_call_llm_messages(**kwargs):
        captured.append(kwargs)
        msgs = list(kwargs["messages"])
        msgs.append({"role": "assistant", "content": '{"reply":"ok"}'})
        return msgs

    with patch("tracesynth.functions.call_llms.call_llm_messages", side_effect=fake_call_llm_messages):
        result, _ = call_and_parse(
            Cfg(),
            [{"role": "user", "content": "hi"}],
            _parse_mock_user_response,
            step_name="test",
            json_mode=True,
        )
    assert result == "ok"
    assert captured[0]["llm_params"]["response_format"] == {"type": "text"}


def test_messages_for_chat_completion_converts_pseudo_tool_role():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "tool", "content": '{"tool_response":"answer"}'},
    ]

    converted = messages_for_chat_completion(messages)

    assert messages[1]["role"] == "tool"
    assert converted[0] == messages[0]
    assert converted[1] == {
        "role": "user",
        "content": '{"tool_response":"answer"}',
    }


def test_parse_solver_response_json_actions():
    content, tool_call = _parse_solver_response(
        '{"action":"tool_call","reasoning":"need search","tool_call":{"name":"Search","arguments":{"query":"q"}}}'
    )
    assert content.startswith('{"action"')
    assert tool_call == '{"name": "Search", "arguments": {"query": "q"}}'

    content, tool_call = _parse_solver_response(
        '{"action":"ask_user","reasoning":"missing time","message":"Which year?"}'
    )
    assert tool_call is None

    content, tool_call = _parse_solver_response(
        '{"action":"final_answer","answer":"done"}'
    )
    assert tool_call is None

    for bad_response in (
        '{"action":"tool_call","tool_call":{"name":"Search","arguments":[]}}',
        '{"action":"ask_user","message":""}',
        '{"action":"final_answer","answer":""}',
    ):
        try:
            _parse_solver_response(bad_response)
            raise AssertionError("expected ParseError")
        except ParseError:
            pass


def test_call_and_parse_reraises_api_errors():
    api_error = RuntimeError("upstream unavailable")

    with patch("tracesynth.functions.call_llms.call_llm_messages", side_effect=api_error):
        try:
            call_and_parse(
                RetryCfg(),
                [{"role": "user", "content": "hi"}],
                _parse_mock_user_response,
                step_name="test",
            )
            raise AssertionError("expected API error to be re-raised")
        except RuntimeError as exc:
            assert exc is api_error


def test_parse_mock_tool_response_json_contract():
    response, new_bg = _parse_mock_tool_response(
        '{"tool_response":"hello","new_bg_introduced":"NO"}'
    )
    assert response == "hello"
    assert new_bg is False

    response, new_bg = _parse_mock_tool_response(
        '```json\n{"tool_response":"open only","new_bg_introduced":"YES"}\n```'
    )
    assert response == "open only"
    assert new_bg is True

    for bad_response in ("plain tool output without json", '{"new_bg_introduced":"NO"}', '["bad"]'):
        try:
            _parse_mock_tool_response(bad_response)
            raise AssertionError("expected ParseError")
        except ParseError:
            pass


def test_mock_tool_response_resamples_after_invalid_json():
    responses = [
        "not json",
        '{"new_bg_introduced":"NO"}',
        '{"tool_response":"ok","new_bg_introduced":"YES"}',
    ]
    call_count = {"n": 0}

    def fake_call_llm_messages(**kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        msgs = list(kwargs["messages"])
        msgs.append({"role": "assistant", "content": responses[idx]})
        return msgs

    with patch("tracesynth.functions.call_llms.call_llm_messages", side_effect=fake_call_llm_messages):
        result = mock_tool_response(RetryCfg(), "{}", [], [])
        assert result == ("ok", True)
        assert call_count["n"] == 3


def test_openai_client_internal_retries_are_disabled():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]

    with patch("tracesynth.functions.call_llms.OpenAI") as openai_cls:
        openai_cls.return_value.chat.completions.create.return_value = fake_response
        result = create_chat_completion_with_retry(
            api_base="http://test",
            api_key="k",
            model_name="m",
            messages=[],
            max_tokens=10,
            temperature=0.1,
            api_max_retries=1,
        )
        assert result == "ok"
        openai_cls.assert_called_once_with(api_key="k", base_url="http://test", max_retries=0)


def test_solver_turn_limit_returns_failure_without_extra_model_call():
    state = {
        "breaked": False,
        "seed_info": {"label": ""},
        "checked_tools": [{"name": "known", "parameters": {}}],
        "fuzzy_task": "task",
        "restrict": "",
        "solve_history": [{"role": "assistant", "content": "still searching"}],
        "tool_call_history": ["previous call"],
        "tool_call_retry_count": 0,
        "solver_turn_count": 1,
        "plan": [{"tool_name": "known", "arguments": {}}],
        "current_plan_step": 0,
    }
    config = {
        "configurable": {
            "processing": {"max_solver_turns": 1},
            "step_models": {
                "SolveAgent": {
                    "name": "m",
                    "api_base": "http://test",
                    "api_key_env": "EMPTY",
                }
            },
        }
    }

    with patch("tracesynth.graph.graph_virtual_tools._generate_final_answer_from_plan") as mock_solve:
        update = execute_plan_node(state, config)

    mock_solve.assert_not_called()
    assert update["breaked"] is True
    assert update["task_finished"] == "Terminated"
    assert "max_solver_turns=1" in update["failure_reason"]
    assert update["solve_history"] == state["solve_history"]
    assert update["tool_call_history"] == state["tool_call_history"]


def test_model_configuration_defaults():
    cfg = ModelConfiguration(model_name="m")
    assert cfg.api_max_retries == 3
    assert cfg.parse_max_retries == 2
    assert cfg.tool_call_max_retries == 3


def test_graph_helpers():
    assert get_tool_call_max_retries({"configurable": {"retry": {"tool_call_max_retries": 5}}}) == 5
    assert should_continue_execution({"breaked": False, "task_finished": "Need replan"}) == "replan"
    assert should_execute_or_replan({"breaked": True, "plan_is_valid": False, "plan_revision_count": 0}) == "end"
    valid, _ = validate_tool_call('{"name":"t","arguments":{}}', [{"name": "t", "parameters": {}}])
    assert valid


if __name__ == "__main__":
    test_is_retryable_api_error()
    test_parse_checked_tools()
    test_parse_fuzzy_task()
    test_call_and_parse_resampling()
    test_call_and_parse_feedback_on_retry()
    test_call_and_parse_returns_last_messages_after_parse_exhaustion()
    test_call_and_parse_json_mode_sets_response_format()
    test_call_and_parse_json_mode_keeps_configured_response_format()
    test_messages_for_chat_completion_converts_pseudo_tool_role()
    test_parse_solver_response_json_actions()
    test_call_and_parse_reraises_api_errors()
    test_parse_mock_tool_response_json_contract()
    test_mock_tool_response_resamples_after_invalid_json()
    test_openai_client_internal_retries_are_disabled()
    test_solver_turn_limit_returns_failure_without_extra_model_call()
    test_model_configuration_defaults()
    test_graph_helpers()
    print("All verification tests passed")
