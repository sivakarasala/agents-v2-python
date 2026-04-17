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
