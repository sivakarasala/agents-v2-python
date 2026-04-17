# The Agent Loop

## Overview

The agent loop is what separates an agent from a simple LLM call. It's the mechanism that allows an AI to take action, observe results, and decide what to do next - repeatedly - until a task is complete.

## LLM vs Workflow vs Agent

### LLM (Single Call)
One input, one output. No tools, no iteration.

### Workflow (Orchestrated Pipeline)
A predefined sequence of steps. You decide the order. Predictable but inflexible.

### Agent (Autonomous Loop)
The LLM decides what to do. It can call tools, observe results, and choose the next action. The loop continues until the agent decides it's done.

## What Is the Loop?

```
while not done:
  1. Send messages to LLM
  2. LLM responds (text and/or tool calls)
  3. If tool calls: execute them, add results to messages
  4. If no tool calls: we're done
```

## Streaming in the Loop

We use `stream=True` instead of a regular call for real-time output:

```python
stream = client.responses.create(
    model=MODEL_NAME,
    instructions=SYSTEM_PROMPT,
    input=input_items,
    tools=ALL_TOOLS,
    stream=True,
)

for event in stream:
    event_type = getattr(event, "type", None)
    if event_type == "response.output_text.delta":
        delta = getattr(event, "delta", "")
        callbacks.on_token(delta)
    elif event_type == "response.completed":
        final_response = getattr(event, "response", None)
```

Users see text as it's generated, not after. This makes the agent feel responsive even when it's thinking.

## Code

### src/agent/run.py

The complete agent loop with streaming. Replace the non-streaming single-turn version:

```python
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
            result = execute_tool(tc.tool_name, tc.args)
            callbacks.on_tool_call_end(tc.tool_name, result)

            input_items.append({
                "type": "function_call_output",
                "call_id": tc.tool_call_id,
                "output": result,
            })

    callbacks.on_complete(full_response)
    return input_items
```

## Breaking Down the Loop

### 1. Setup Input Items
```python
input_items = [*working_history, {"role": "user", "content": user_message}]
```
Start with conversation history, add the new user message. The system prompt goes via `instructions=`, not as a message.

### 2. Stream the Response
```python
stream = client.responses.create(..., stream=True)
```
Call the model with streaming. We iterate events to get text deltas and the final response.

### 3. Process Events
- `response.output_text.delta` - Stream text to UI immediately
- `response.completed` - Capture the final response object with all output items

### 4. Check If Done
```python
if not function_calls:
    break
```
If the model didn't request any function calls, we're done.

### 5. Execute Tools
```python
for tc in function_calls:
    result = execute_tool(tc.tool_name, tc.args)
    input_items.append({
        "type": "function_call_output",
        "call_id": tc.tool_call_id,
        "output": result,
    })
```
Execute each tool, append `function_call_output` items back to `input_items`. The model will see these results on the next iteration.

### 6. Loop Again
Back to step 2. The model now has tool results and can decide what to do next.

## The Responses API Message Format

Tool results use `function_call_output`:
```python
{
    "type": "function_call_output",
    "call_id": "call_abc123",  # Links to the original function_call
    "output": "file contents here...",
}
```

The `call_id` links the result back to the specific function call.

## Common Pitfalls

- **Infinite Loops** - Always have a max iteration limit in production
- **Lost Tool Results** - Forgetting to append `function_call_output` items
- **Not Handling Errors** - Tool execution can fail; return error strings so the model can adapt
