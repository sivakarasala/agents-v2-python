import json
from typing import Any
from openai import OpenAI
from src.agent.system.prompt import SYSTEM_PROMPT
from src.agent.tools import ALL_TOOLS, TOOL_EXECUTORS
from evals.types import EvalData, SingleTurnResult, MultiTurnEvalData, MultiTurnResult
from evals.utils import build_messages, build_mocked_tools

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


def multi_turn_with_mocks(data: dict[str, Any]) -> MultiTurnResult:
    """Run a multi-turn evaluation with mocked tools, using the Responses API."""
    tool_definitions, executor_map = build_mocked_tools(data["mock_tools"])

    # Build initial input items. If the test provides explicit messages, use
    # them as-is (assumed to already be in Responses-API shape).
    if "messages" in data and data["messages"]:
        input_items = list(data["messages"])
    else:
        input_items = [{"role": "user", "content": data["prompt"]}]

    model = "gpt-5-mini"
    max_steps = 20
    if data.get("config"):
        model = data["config"].get("model", model)
        max_steps = data["config"].get("max_steps", max_steps)

    all_tool_calls: list[str] = []
    steps: list[dict[str, Any]] = []
    final_text = ""

    for _step in range(max_steps):
        response = _get_client().responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=input_items,
            tools=tool_definitions if tool_definitions else None,
        )

        step_data: dict[str, Any] = {}
        step_tool_calls = []
        step_tool_results = []
        had_function_call = False

        for item in response.output:
            item_dict = item.model_dump(exclude_none=True)
            input_items.append(item_dict)

            if item_dict.get("type") == "function_call":
                had_function_call = True
                tool_name = item_dict["name"]
                try:
                    args = json.loads(item_dict.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                all_tool_calls.append(tool_name)
                step_tool_calls.append({"tool_name": tool_name, "args": args})

                executor = executor_map.get(tool_name)
                result = executor(args) if executor else f"Unknown tool: {tool_name}"
                step_tool_results.append({"tool_name": tool_name, "result": result})

                input_items.append({
                    "type": "function_call_output",
                    "call_id": item_dict["call_id"],
                    "output": result,
                })

        # Capture final assistant text from this step
        text = getattr(response, "output_text", "") or ""
        if text:
            step_data["text"] = text
            final_text = text

        if step_tool_calls:
            step_data["tool_calls"] = step_tool_calls
            step_data["tool_results"] = step_tool_results

        steps.append(step_data)

        if not had_function_call:
            break

    return MultiTurnResult(
        text=final_text,
        steps=steps,
        tools_used=list(set(all_tool_calls)),
        tool_call_order=all_tool_calls,
    )
