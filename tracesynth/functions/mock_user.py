import json
import re
from .call_llms import ParseError, call_and_parse
from .prompt import mock_user_prompt


def _parse_mock_user_response(content: str) -> str:
    user_response_matches = re.findall(r"<reply>(.+?)</reply>", content, re.DOTALL)
    if not user_response_matches:
        raise ParseError("missing <reply> tag")
    return user_response_matches[-1].strip()


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
    )
    return parsed
