# Single Turn Evals

## Overview

This lesson covers evaluating AI agents with a focus on single-turn tool selection evaluations. We'll also transition `run.py` from a standalone script to a proper function that integrates with the UI.

## Why We Need Evals

1. **Agents are non-deterministic** - The same input can produce different outputs
2. **Quality regression** - Model updates, prompt changes, or tool modifications can silently degrade performance
3. **Confidence in deployment** - You need quantifiable metrics before shipping changes
4. **Debugging** - When something goes wrong, you need to understand *why*

## Eval Categories

We use three categories:

1. **Golden** - Must select exactly the expected tools. No ambiguity.
2. **Secondary** - Likely selects certain tools, but there's flexibility. Scored on precision/recall.
3. **Negative** - Must NOT select forbidden tools. Tests that the agent doesn't over-reach.

## Code

### src/agent/run.py

Transition from standalone script to a proper function. No more direct `run_agent()` call at the bottom - the UI will call this. Non-streaming, single turn with callbacks:

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
```

Key changes from lesson 02:
- No standalone `run_agent("...")` call at the bottom
- Accepts `conversation_history` and `callbacks` parameters
- Uses `filter_compatible_messages` for history cleanup
- Returns `input_items` (Responses API format) for conversation continuity
- Singleton client pattern with `_get_client()`

### evals/types.py

Define the data structures for evaluations:

```python
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvalData:
    """Input data for single-turn tool selection evaluations."""
    prompt: str
    tools: list[str]
    system_prompt: Optional[str] = None
    config: Optional[dict[str, Any]] = None


@dataclass
class EvalTarget:
    """Target expectations for single-turn evaluations."""
    category: str  # "golden", "secondary", or "negative"
    expected_tools: Optional[list[str]] = None
    forbidden_tools: Optional[list[str]] = None


@dataclass
class SingleTurnResult:
    """Result from single-turn executor."""
    tool_calls: list[dict[str, Any]]
    tool_names: list[str]
    selected_any: bool


@dataclass
class MockToolConfig:
    """Mock tool configuration for multi-turn evaluations."""
    description: str
    parameters: dict[str, str]
    mock_return: str


@dataclass
class MultiTurnEvalData:
    """Input data for multi-turn agent evaluations."""
    mock_tools: dict[str, MockToolConfig]
    prompt: Optional[str] = None
    messages: Optional[list[dict[str, Any]]] = None
    config: Optional[dict[str, Any]] = None


@dataclass
class MultiTurnTarget:
    """Target expectations for multi-turn evaluations."""
    original_task: str
    mock_tool_results: dict[str, str]
    category: str  # "task-completion", "conversation-continuation", "negative"
    expected_tool_order: Optional[list[str]] = None
    forbidden_tools: Optional[list[str]] = None


@dataclass
class MultiTurnResult:
    """Result from multi-turn executor."""
    text: str
    steps: list[dict[str, Any]]
    tools_used: list[str]
    tool_call_order: list[str]
```

### evals/evaluators.py

Scorer functions for evaluating tool selection:

```python
from typing import Union

from evals.types import (
    EvalTarget,
    SingleTurnResult,
    MultiTurnTarget,
    MultiTurnResult,
)


def tools_selected(
    output: Union[SingleTurnResult, MultiTurnResult],
    target: Union[EvalTarget, MultiTurnTarget],
) -> float:
    """Check if all expected tools were selected. Returns 1 or 0."""
    expected = getattr(target, "expected_tools", None) or getattr(
        target, "expected_tool_order", None
    )
    if not expected:
        return 1.0

    selected = set(
        output.tool_names if hasattr(output, "tool_names") else output.tools_used
    )
    return 1.0 if all(t in selected for t in expected) else 0.0


def tools_avoided(
    output: Union[SingleTurnResult, MultiTurnResult],
    target: Union[EvalTarget, MultiTurnTarget],
) -> float:
    """Check if forbidden tools were avoided. Returns 1 or 0."""
    forbidden = target.forbidden_tools
    if not forbidden:
        return 1.0

    selected = set(
        output.tool_names if hasattr(output, "tool_names") else output.tools_used
    )
    return 0.0 if any(t in selected for t in forbidden) else 1.0


def tool_selection_score(
    output: SingleTurnResult,
    target: EvalTarget,
) -> float:
    """Precision/recall F1 score for tool selection. Returns 0 to 1."""
    if not target.expected_tools:
        return 0.5 if output.selected_any else 1.0

    expected = set(target.expected_tools)
    selected = set(output.tool_names)

    hits = len([t for t in output.tool_names if t in expected])
    precision = hits / len(selected) if selected else 0.0
    recall = hits / len(expected) if expected else 0.0

    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
```

### evals/executors.py

The single-turn executor using the Responses API:

```python
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
    """Run a single-turn evaluation. Gets tool selection without executing."""
    msgs = build_messages(data)
    system_prompt = msgs[0]["content"]
    input_items = msgs[1:]

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
```

### evals/utils.py

Helper functions for building messages and mocked tools:

```python
import json
from typing import Any
from src.agent.system.prompt import SYSTEM_PROMPT


def build_messages(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """Build message array from eval data."""
    system_prompt = data.get("system_prompt") or SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["prompt"]},
    ]


def build_mocked_tools(
    mock_tools: dict[str, dict[str, Any]],
) -> tuple[list[dict], dict[str, callable]]:
    """Build Responses API tool definitions and executors from mock config."""
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

        mock_return = config["mock_return"]
        executor_map[name] = lambda args, ret=mock_return: ret

    return tool_definitions, executor_map
```

### evals/mocks/tools.py

Mock tool executors for testing:

```python
from typing import Any


def create_mock_read_file(mock_content: str):
    """Create a mock read_file executor."""
    def execute(args: dict[str, Any]) -> str:
        return mock_content
    return execute


def create_mock_write_file(mock_response: str = None):
    """Create a mock write_file executor."""
    def execute(args: dict[str, Any]) -> str:
        if mock_response:
            return mock_response
        content = args.get("content", "")
        path = args.get("path", "unknown")
        return f"Successfully wrote {len(content)} characters to {path}"
    return execute


def create_mock_list_files(mock_files: list[str]):
    """Create a mock list_files executor."""
    def execute(args: dict[str, Any]) -> str:
        return "\n".join(mock_files)
    return execute


def create_mock_delete_file(mock_response: str = None):
    """Create a mock delete_file executor."""
    def execute(args: dict[str, Any]) -> str:
        if mock_response:
            return mock_response
        return f"Successfully deleted {args.get('path', 'unknown')}"
    return execute


def create_mock_shell(mock_output: str):
    """Create a mock shell command executor."""
    def execute(args: dict[str, Any]) -> str:
        return mock_output
    return execute
```

### evals/file_tools_eval.py

The file tools evaluation runner:

```python
import json
import os
from dotenv import load_dotenv

from src.agent.tools import FILE_TOOLS
from evals.executors import single_turn_executor
from evals.evaluators import tools_selected, tools_avoided, tool_selection_score
from evals.types import EvalTarget, SingleTurnResult

load_dotenv()


def load_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def run_eval():
    dataset = load_dataset("evals/data/file_tools.json")
    results = []

    for i, entry in enumerate(dataset):
        data = entry["data"]
        target_data = entry["target"]

        target = EvalTarget(
            category=target_data["category"],
            expected_tools=target_data.get("expected_tools"),
            forbidden_tools=target_data.get("forbidden_tools"),
        )

        output = single_turn_executor(data, FILE_TOOLS)

        scores = {}
        if target.category == "golden":
            scores["tools_selected"] = tools_selected(output, target)
        elif target.category == "negative":
            scores["tools_avoided"] = tools_avoided(output, target)
        elif target.category == "secondary":
            scores["selection_score"] = tool_selection_score(output, target)

        results.append({
            "prompt": data["prompt"],
            "category": target.category,
            "selected": output.tool_names,
            "scores": scores,
        })

        status = "✓" if all(v >= 1.0 for v in scores.values()) else "✗"
        print(f"  {status} [{target.category}] {data['prompt']}")
        print(f"    Selected: {output.tool_names}")
        print(f"    Scores: {scores}")
        print()

    all_scores = [s for r in results for s in r["scores"].values()]
    avg = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"Average score: {avg:.2f}")


if __name__ == "__main__":
    print("File Tools Evaluation")
    print("=" * 40)
    run_eval()
```

### tests/test_evals_utils.py

```python
from evals.utils import build_messages, build_mocked_tools
from evals.mocks.tools import (
    create_mock_read_file,
    create_mock_write_file,
    create_mock_list_files,
    create_mock_delete_file,
    create_mock_shell,
)


def test_build_messages_uses_system_prompt():
    msgs = build_messages({"prompt": "hi"})
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_build_messages_custom_system_prompt():
    msgs = build_messages({"prompt": "hi", "system_prompt": "you are X"})
    assert msgs[0]["content"] == "you are X"


def test_build_mocked_tools():
    defs, execs = build_mocked_tools({
        "lookup": {
            "description": "look stuff up",
            "parameters": ["query"],
            "mock_return": "the result",
        }
    })
    assert defs[0]["type"] == "function"
    assert defs[0]["name"] == "lookup"
    assert defs[0]["parameters"]["properties"]["query"]["type"] == "string"
    assert execs["lookup"]({"query": "anything"}) == "the result"


def test_mock_executors_return_mocked_values():
    assert create_mock_read_file("hello world")({"path": "x"}) == "hello world"
    assert "Successfully wrote" in create_mock_write_file()({"path": "x", "content": "yo"})
    assert create_mock_write_file("custom")({"path": "x", "content": ""}) == "custom"
    assert create_mock_list_files(["a", "b"])({"directory": "."}) == "a\nb"
    assert "Successfully deleted" in create_mock_delete_file()({"path": "f"})
    assert create_mock_shell("hello\n")({"command": "echo hello"}) == "hello\n"
```

## Running Evals

```bash
# Run file tools eval
python evals/file_tools_eval.py
```

Results will show pass/fail for each test case with scores per evaluator.
