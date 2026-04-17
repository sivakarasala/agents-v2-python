import asyncio
import inspect
import json
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import ALL_TOOLS
from src.agent.execute_tool import execute_tool
from src.agent.system.prompt import SYSTEM_PROMPT
from src.agent.context import (
    estimate_messages_tokens,
    get_model_limits,
    is_over_threshold,
    calculate_usage_percentage,
    compact_conversation,
    DEFAULT_THRESHOLD,
)
from src.agent.system.filter_messages import filter_compatible_messages
from src.types import AgentCallbacks, ToolCallInfo, TokenUsageInfo

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
    """Run the agent loop using the OpenAI Responses API.

    Conversation history is a list of Responses API "input items":
      - {"role": "user"|"assistant", "content": "..."}
      - {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}
      - {"type": "function_call_output", "call_id": "...", "output": "..."}
      - provider-managed items like {"type": "web_search_call", ...} kept verbatim

    The system prompt is sent via the `instructions` parameter, not as a message.
    """
    model_limits = get_model_limits(MODEL_NAME)

    # Compact if we're over the context budget
    working_history = filter_compatible_messages(conversation_history)
    pre_check_tokens = estimate_messages_tokens([
        {"role": "user", "content": SYSTEM_PROMPT},
        *working_history,
        {"role": "user", "content": user_message},
    ])
    if is_over_threshold(pre_check_tokens.total, model_limits.context_window):
        working_history = compact_conversation(working_history, MODEL_NAME)

    input_items: list[dict[str, Any]] = [
        *working_history,
        {"role": "user", "content": user_message},
    ]

    def report_token_usage():
        if callbacks.on_token_usage:
            usage = estimate_messages_tokens(
                [{"role": "user", "content": SYSTEM_PROMPT}, *input_items]
            )
            callbacks.on_token_usage(TokenUsageInfo(
                input_tokens=usage.input,
                output_tokens=usage.output,
                total_tokens=usage.total,
                context_window=model_limits.context_window,
                threshold=DEFAULT_THRESHOLD,
                percentage=calculate_usage_percentage(
                    usage.total, model_limits.context_window
                ),
            ))

    report_token_usage()

    full_response = ""

    while True:
        stream = _get_client().responses.create(
            model=MODEL_NAME,
            instructions=SYSTEM_PROMPT,
            input=input_items,
            tools=ALL_TOOLS if ALL_TOOLS else None,
            stream=True,
        )

        # Stream text deltas to the UI; capture the final response object on
        # `response.completed` so we can read its full output items.
        final_response = None
        current_text = ""

        for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    current_text += delta
                    callbacks.on_token(delta)

            elif event_type == "response.completed":
                final_response = getattr(event, "response", None)

        full_response += current_text

        if final_response is None:
            # Stream ended without a completed event — nothing more to do
            break

        # Walk the output items: append everything (assistant text, reasoning,
        # web_search_call, function_call) to history so the next turn has full
        # context, and collect any function_call items we need to execute.
        function_calls: list[ToolCallInfo] = []

        for item in final_response.output:
            item_dict = item.model_dump(exclude_none=True)
            input_items.append(item_dict)

            if item_dict.get("type") == "function_call":
                try:
                    args = json.loads(item_dict.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                function_calls.append(ToolCallInfo(
                    tool_call_id=item_dict["call_id"],
                    tool_name=item_dict["name"],
                    args=args,
                ))

        # No function calls → the model gave a final answer; we're done
        if not function_calls:
            break

        for tc in function_calls:
            callbacks.on_tool_call_start(tc.tool_name, tc.args)

        # Execute each function call (with optional approval) and append the
        # corresponding function_call_output item back into the input.
        rejected = False
        for tc in function_calls:
            approval = callbacks.on_tool_approval(tc.tool_name, tc.args)
            if inspect.isawaitable(approval):
                approved = asyncio.run(approval)
            else:
                approved = approval

            if not approved:
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tc.tool_call_id,
                    "output": "User rejected this tool call.",
                })
                rejected = True
                break

            result = execute_tool(tc.tool_name, tc.args)
            callbacks.on_tool_call_end(tc.tool_name, result)

            input_items.append({
                "type": "function_call_output",
                "call_id": tc.tool_call_id,
                "output": result,
            })

            report_token_usage()

        if rejected:
            break

    callbacks.on_complete(full_response)
    return input_items
