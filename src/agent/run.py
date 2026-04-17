import json
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import ALL_TOOLS
from src.agent.execute_tool import execute_tool
from src.agent.system.prompt import SYSTEM_PROMPT
from src.agent.system.filter_messages import filter_compatible_messages
from src.types import AgentCallbacks, ToolCallInfo

load_dotenv()

_client: OpenAI | None = None
MODEL_NAME = "gpt-5-mini"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def run_agent(
    user_message: str,
    conversation_history: list[dict[str, Any]],
    callbacks: AgentCallbacks,
) -> list[dict[str, Any]]:
    working_history = filter_compatible_messages(conversation_history)

    input_items: list[dict[str, Any]] = [
        *working_history,
        {"role": "user", "content": user_message},
    ]

    response = _get_client().responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_PROMPT,
        input=input_items,
        tools=ALL_TOOLS if ALL_TOOLS else None,
    )

    full_response = response.output_text or ""
    callbacks.on_token(full_response)

    for item in response.output:
        item_dict = item.model_dump(exclude_none=True)
        input_items.append(item_dict)

        if item_dict.get("type") == "function_call":
            args = json.loads(item_dict.get("arguments") or "{}")
            callbacks.on_tool_call_start(item_dict["name"], args)
            result = execute_tool(item_dict["name"], args)
            callbacks.on_tool_call_end(item_dict["name"], result)
            input_items.append({
                "type": "function_call_output",
                "call_id": item_dict["call_id"],
                "output": result,
            })

    callbacks.on_complete(full_response)
    return input_items
