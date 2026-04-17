---
layout: default
title: "Human-in-the-Loop (HITL)"
---

# Human-in-the-Loop (HITL)

## Overview

Human-in-the-Loop (HITL) is a design pattern where human judgment is integrated into automated systems at critical decision points. This lesson focuses on **runtime approval flows** for AI agents.

## Why Runtime Approvals Matter

- **Trust**: Users won't adopt agents that act without oversight
- **Accountability**: Clear audit trail of who approved what
- **Reversibility**: Safety net for irreversible operations (deleting files, shell commands)
- **Learning**: Approval patterns teach you about the agent's behavior

## Approval Flow: Synchronous

Our implementation: the agent loop pauses while waiting for human input. Simple, agent retains full context, no persistence layer needed. Best for CLI tools and local development.

## Approval Granularity

- **Per-Tool**: Approve every call to `run_command`, auto-approve `read_file`
- **Input-Based**: Auto-approve `run_command` if it's `ls`, require approval for `rm`
- **Session-Based**: "Trust this tool for the rest of this session"

## Code

### src/agent/run.py

Add approval logic before executing each tool call. The key change: process tool calls **sequentially** so we can stop if any approval is rejected.

Replace the tool execution section of the loop with:

```python
import asyncio
import inspect
```

Add these imports, then replace the tool execution block:

```python
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
```

Key implementation details:
- `callbacks.on_tool_approval(...)` returns a value (or awaitable) that resolves when the user decides
- `inspect.isawaitable` handles both sync and async approval callbacks
- If rejected, we still append a `function_call_output` (with rejection message) so the API gets a response for each `function_call`
- `rejected = True` + `break` exits both the inner loop (tool calls) and the outer `while True` loop

### The Full Updated run.py

For reference, the complete `run_agent` function with HITL:

```python
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
    """Run the agent loop using the OpenAI Responses API."""
    model_limits = get_model_limits(MODEL_NAME)

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
            break

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

        if not function_calls:
            break

        for tc in function_calls:
            callbacks.on_tool_call_start(tc.tool_name, tc.args)

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
```

## The Approval Pattern

The `on_tool_approval` callback is the key abstraction. The UI implements it:

1. Agent wants to call `run_command` with args `{"command": "rm -rf temp/"}`
2. Agent loop calls `callbacks.on_tool_approval("run_command", {"command": "rm -rf temp/"})`
3. UI shows a prompt: "Allow run_command with rm -rf temp/? [y/n]"
4. User types `y` or `n`
5. Callback returns `True` or `False`
6. Agent either executes or stops

The `inspect.isawaitable` check handles both sync (testing) and async (UI) callbacks.
