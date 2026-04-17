import json
from typing import Any
from src.agent.system.prompt import SYSTEM_PROMPT


def build_messages(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """Build message array from eval data.

    Returns a Responses API input list. The system prompt is also returned in
    the array (as a system message) so existing tests that index msgs[0] /
    msgs[1] keep working — single_turn_executor pulls it out and passes it via
    `instructions` instead.
    """
    system_prompt = data.get("system_prompt") or SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["prompt"]},
    ]


def build_mocked_tools(
    mock_tools: dict[str, dict[str, Any]],
) -> tuple[list[dict], dict[str, callable]]:
    """Build Responses API tool definitions and executors from mock config.

    Returns:
        (tool_definitions, executor_map)
    """
    tool_definitions = []
    executor_map = {}

    for name, config in mock_tools.items():
        properties = {}
        for param_name in config["parameters"]:
            properties[param_name] = {"type": "string"}

        tool_def = {
            "type": "function",
            "name": name,
            "description": config["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }
        tool_definitions.append(tool_def)

        # Create executor that returns the mock value
        mock_return = config["mock_return"]
        executor_map[name] = lambda args, ret=mock_return: ret

    return tool_definitions, executor_map
