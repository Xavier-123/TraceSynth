import json

from .call_llms import ParseError, call_and_parse, parse_json_object
from .prompt import mock_user_prompt


def _parse_mock_user_response(content: str) -> str:
    payload = parse_json_object(content)
    reply = payload.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ParseError("missing required JSON field: reply")
    return reply.strip()


def mock_user_response(cfg, task, background, restrict, interaction):
    prompt = mock_user_prompt.format(
        task=task,
        background=background,
        restrict=restrict,
        interaction=json.dumps(interaction),
    )
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    parsed, _ = call_and_parse(
        cfg,
        messages,
        _parse_mock_user_response,
        step_name="MockUserAgent",
        json_mode=True,
    )
    return parsed
