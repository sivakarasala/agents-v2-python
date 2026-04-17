# Tool Calling

## Overview

This lesson covers tool calling - the mechanism that allows LLMs to interact with the outside world. We'll create our first tool and wire it up to our agent.

## What is Tool Calling?

Tool calling (also called function calling) lets LLMs request to execute functions. Instead of just generating text, the model can:

1. Recognize when a task requires external capabilities
2. Select the appropriate tool(s) to use
3. Generate the correct arguments for that tool
4. Receive the result and incorporate it into its response

The model NEVER executes tools itself. It only generates a structured request. You execute the code.

## Anatomy of a Tool (OpenAI Responses API)

In the Responses API, tools use a flat format (no nested `"function"` key):

```python
{
    "type": "function",
    "name": "get_date_time",
    "description": "Get the current date and time",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}
```

- **name** - The function name the model will call
- **description** - Critical for the model to understand when to use this tool
- **parameters** - JSON Schema defining the parameters

## Code

### src/agent/tools/date_time.py

Create your first tool - getting the current date and time:

```python
from datetime import datetime
from typing import Any


def get_date_time_execute(args: dict[str, Any]) -> str:
    return datetime.now().isoformat()


GET_DATE_TIME_TOOL = {
    "type": "function",
    "name": "get_date_time",
    "description": "Get the current date and time",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}
```

### src/agent/tools/__init__.py

Register the tool in the tools index:

```python
from src.agent.tools.date_time import get_date_time_execute, GET_DATE_TIME_TOOL

TOOL_EXECUTORS: dict[str, callable] = {
    "get_date_time": get_date_time_execute,
}

ALL_TOOLS = [GET_DATE_TIME_TOOL]
FILE_TOOLS: list = []
FILE_TOOL_EXECUTORS: dict = {}
SHELL_TOOLS: list = []
SHELL_TOOL_EXECUTORS: dict = {}
```

### src/agent/execute_tool.py

Create a helper to execute tools by name:

```python
from typing import Any
from src.agent.tools import TOOL_EXECUTORS


def execute_tool(name: str, args: dict[str, Any]) -> str:
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return f"Unknown tool: {name}"
    try:
        result = executor(args)
        return str(result)
    except Exception as e:
        return f"Error executing {name}: {e}"
```

### src/agent/run.py

Add tools to the agent and handle tool calls. Still a standalone script:

```python
import json
from openai import OpenAI
from dotenv import load_dotenv
from src.agent.system.prompt import SYSTEM_PROMPT
from src.agent.tools import ALL_TOOLS
from src.agent.execute_tool import execute_tool

load_dotenv()

MODEL_NAME = "gpt-5-mini"


def run_agent(user_message: str):
    client = OpenAI()
    response = client.responses.create(
        model=MODEL_NAME,
        input=user_message,
        instructions=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
    )

    print(response.output_text)

    for item in response.output:
        item_dict = item.model_dump(exclude_none=True)
        if item_dict.get("type") == "function_call":
            args = json.loads(item_dict.get("arguments") or "{}")
            result = execute_tool(item_dict["name"], args)
            print(f"Tool: {item_dict['name']}, Result: {result}")


run_agent("What is the current date and time?")
```

### tests/test_execute_tool.py

```python
from src.agent.execute_tool import execute_tool


def test_unknown_tool():
    result = execute_tool("nonexistent", {})
    assert "Unknown tool" in result


def test_execute_date_time():
    result = execute_tool("get_date_time", {})
    assert len(result) > 0
```

## Key Points

1. **Tools are declarative** - You describe what they do, the model decides when to use them
2. **Good descriptions matter** - The model relies on your description to select the right tool
3. **Flat format** - The Responses API uses `{"type": "function", "name": ...}` not nested `{"type": "function", "function": {...}}`
4. **You control execution** - The model requests, you execute. This is a security boundary.
