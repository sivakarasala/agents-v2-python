from evals.utils import build_messages, build_mocked_tools
from evals.mocks.tools import (
    create_mock_read_file,
    create_mock_write_file,
    create_mock_list_files,
    create_mock_delete_file,
    create_mock_shell,
)


def test_build_messages_uses_system_prompt():
    msgs = build_messages({"prompt": "hi"})
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_build_messages_custom_system_prompt():
    msgs = build_messages({"prompt": "hi", "system_prompt": "you are X"})
    assert msgs[0]["content"] == "you are X"


def test_build_mocked_tools():
    defs, execs = build_mocked_tools({
        "lookup": {
            "description": "look stuff up",
            "parameters": ["query"],
            "mock_return": "the result",
        }
    })
    # Responses API flat format: name + parameters at top level, no nested "function"
    assert defs[0]["type"] == "function"
    assert defs[0]["name"] == "lookup"
    assert defs[0]["parameters"]["properties"]["query"]["type"] == "string"
    assert execs["lookup"]({"query": "anything"}) == "the result"


def test_mock_executors_return_mocked_values():
    assert create_mock_read_file("hello world")({"path": "x"}) == "hello world"
    assert "Successfully wrote" in create_mock_write_file()({"path": "x", "content": "yo"})
    assert create_mock_write_file("custom")({"path": "x", "content": ""}) == "custom"
    assert create_mock_list_files(["a", "b"])({"directory": "."}) == "a\nb"
    assert "Successfully deleted" in create_mock_delete_file()({"path": "f"})
    assert create_mock_shell("hello\n")({"command": "echo hello"}) == "hello\n"
