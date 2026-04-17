import json
from typing import Any
from openai import OpenAI
from src.agent.system.prompt import SYSTEM_PROMPT
from src.agent.tools import ALL_TOOLS, TOOL_EXECUTORS
from evals.types import EvalData, SingleTurnResult
from evals.utils import build_messages

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def single_turn_executor(
    data: dict[str, Any],
    available_tools: list[dict],
) -> SingleTurnResult:
    """Run a single-turn evaluation. Gets tool selection without executing.

    Uses the Responses API. `available_tools` is a list of flat-format tool
    definitions ({"type": "function", "name": ..., ...}).
    """
    msgs = build_messages(data)
    # build_messages returns [system, user]; pull system out into `instructions`
    system_prompt = msgs[0]["content"]
    input_items = msgs[1:]

    # Filter to only the tools the eval wants to expose
    tool_names_wanted = set(data["tools"])
    tools = [t for t in available_tools if t.get("name") in tool_names_wanted]

    model = "gpt-5-mini"
    if data.get("config") and data["config"].get("model"):
        model = data["config"]["model"]

    response = _get_client().responses.create(
        model=model,
        instructions=system_prompt,
        input=input_items,
        tools=tools if tools else None,
    )

    tool_calls = []
    tool_names = []
    for item in response.output:
        item_dict = item.model_dump(exclude_none=True)
        if item_dict.get("type") == "function_call":
            try:
                args = json.loads(item_dict.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"tool_name": item_dict["name"], "args": args})
            tool_names.append(item_dict["name"])

    return SingleTurnResult(
        tool_calls=tool_calls,
        tool_names=tool_names,
        selected_any=len(tool_names) > 0,
    )
