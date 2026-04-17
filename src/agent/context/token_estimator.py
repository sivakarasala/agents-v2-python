import json
from typing import Any
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Estimate token count using character division.
    Uses 3.75 as the divisor (midpoint of 3.5-4 range).
    """
    return max(1, len(text) // 4 + 1)


def extract_message_text(message: dict[str, Any]) -> str:
    """Extract text content from a Responses API input item.

    Handles:
      - role-based messages: {"role": ..., "content": str | list}
      - typed items: function_call, function_call_output, web_search_call, …
      - legacy Chat Completions shapes: assistant with tool_calls, role="tool"
    """
    item_type = message.get("type")

    # Responses API typed items
    if item_type == "function_call":
        return f"{message.get('name', '')}({message.get('arguments', '')})"
    if item_type == "function_call_output":
        return str(message.get("output", ""))
    if item_type and "content" not in message:
        # other typed items (web_search_call, reasoning, etc.) — fall back to dump
        return json.dumps(message)

    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part["text"]))
                elif "value" in part:
                    parts.append(str(part["value"]))
                else:
                    parts.append(json.dumps(part))
        return " ".join(parts)

    if content is None:
        # Legacy Chat Completions tool_calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            return json.dumps(tool_calls)
        return ""

    return json.dumps(content)


@dataclass
class TokenUsage:
    input: int
    output: int
    total: int


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> TokenUsage:
    """Estimate token counts for a Responses API input item array.
    Separates input (user/system/function results) from output (assistant text,
    function calls, model-generated typed items).
    """
    input_tokens = 0
    output_tokens = 0

    for message in messages:
        text = extract_message_text(message)
        tokens = estimate_tokens(text)

        item_type = message.get("type")
        role = message.get("role")

        is_output = (
            role == "assistant"
            or item_type == "function_call"
            or item_type == "reasoning"
            or item_type == "web_search_call"
        )

        if is_output:
            output_tokens += tokens
        else:
            input_tokens += tokens

    return TokenUsage(
        input=input_tokens,
        output=output_tokens,
        total=input_tokens + output_tokens,
    )
