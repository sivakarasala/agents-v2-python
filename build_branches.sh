#!/usr/bin/env bash
set -euo pipefail

REPO="/Users/sivakrishna/agentic/agents-v2-python"
SRC="/Users/sivakrishna/agentic/building-ai-agents-python"

cd "$REPO"

# Clean everything except .git
find . -maxdepth 1 -not -name '.git' -not -name '.' -not -name 'build_branches.sh' -exec rm -rf {} + 2>/dev/null || true

###############################################################################
# Helper: copy base files from source (everything except excluded dirs/files)
###############################################################################
copy_base() {
    rsync -a \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='.env' \
        --exclude='*.egg-info' \
        "$SRC/" "$REPO/"
}

###############################################################################
# Helper: create notes directory with placeholders
###############################################################################
create_notes() {
    mkdir -p notes
    echo "# Intro to Agents" > notes/01-Intro-to-Agents.md
    echo "# Tool Calling" > notes/02-Tool-Calling.md
    echo "# Single Turn Evals" > notes/03-Single-Turn-Evals.md
    echo "# The Agent Loop" > notes/04-The-Agent-Loop.md
    echo "# Multi-turn Evals" > notes/05-Multi-turn-Evals.md
    echo "# File System Tools" > notes/06-File-System-Tools.md
    echo "# Web Search and Context Management" > notes/07-Web-Search-Context-Management.md
    echo "# Shell Tool" > notes/08-Shell-Tool.md
    echo "# Human in the Loop" > notes/09-Human-in-the-Loop.md
    echo "# Course" > notes/course.md
}

###############################################################################
# LESSON 01: Standalone intro script
###############################################################################
echo "=== Building lesson-01 ==="
find . -maxdepth 1 -not -name '.git' -not -name '.' -not -name 'build_branches.sh' -exec rm -rf {} + 2>/dev/null || true
copy_base
create_notes

# Remove tool files (keep __init__.py)
rm -f src/agent/tools/file.py
rm -f src/agent/tools/shell.py
rm -f src/agent/tools/code_execution.py
rm -f src/agent/tools/web_search.py
rm -f src/agent/tools/date_time.py

# Remove execute_tool.py
rm -f src/agent/execute_tool.py

# Stub tools/__init__.py
cat > src/agent/tools/__init__.py << 'PYEOF'
TOOL_EXECUTORS: dict[str, callable] = {}
ALL_TOOLS: list = []
FILE_TOOLS: list = []
FILE_TOOL_EXECUTORS: dict = {}
SHELL_TOOLS: list = []
SHELL_TOOL_EXECUTORS: dict = {}
PYEOF

# Standalone run.py
cat > src/agent/run.py << 'PYEOF'
from openai import OpenAI
from dotenv import load_dotenv
from src.agent.system.prompt import SYSTEM_PROMPT

load_dotenv()

MODEL_NAME = "gpt-5-mini"


def run_agent(user_message: str):
    client = OpenAI()
    response = client.responses.create(
        model=MODEL_NAME,
        input=user_message,
        instructions=SYSTEM_PROMPT,
    )

    print(response.output_text)


run_agent("Who won the 2025 IPL final")
PYEOF

# Stub compaction.py
cat > src/agent/context/compaction.py << 'PYEOF'
from typing import Any
from src.agent.context.token_estimator import extract_message_text

SUMMARIZATION_PROMPT = ""


def messages_to_text(messages: list[dict[str, Any]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = extract_message_text(msg)
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def compact_conversation(
    messages: list[dict[str, Any]],
    model: str = "gpt-5-mini",
) -> list[dict[str, Any]]:
    return list(messages)
PYEOF

# Remove evals (keep empty __init__.py)
rm -f evals/evaluators.py evals/executors.py evals/types.py evals/utils.py
rm -f evals/file_tools_eval.py evals/shell_tools_eval.py evals/agent_multiturn_eval.py
rm -rf evals/data evals/mocks
cat > evals/__init__.py << 'PYEOF'
# Empty init file
PYEOF

# Remove test files except __init__.py and test_context.py
rm -f tests/test_execute_tool.py tests/test_file_tools.py tests/test_shell_and_code.py tests/test_evals_utils.py

git add -A
git commit -m "lesson-01: standalone intro script with basic LLM call"

git branch lesson-01

###############################################################################
# LESSON 02: Tool calling (still standalone)
###############################################################################
echo "=== Building lesson-02 ==="

# Add date_time tool
cat > src/agent/tools/date_time.py << 'PYEOF'
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
PYEOF

# Update tools/__init__.py
cat > src/agent/tools/__init__.py << 'PYEOF'
from src.agent.tools.date_time import get_date_time_execute, GET_DATE_TIME_TOOL

TOOL_EXECUTORS: dict[str, callable] = {
    "get_date_time": get_date_time_execute,
}

ALL_TOOLS = [GET_DATE_TIME_TOOL]
FILE_TOOLS: list = []
FILE_TOOL_EXECUTORS: dict = {}
SHELL_TOOLS: list = []
SHELL_TOOL_EXECUTORS: dict = {}
PYEOF

# Add execute_tool.py
cat > src/agent/execute_tool.py << 'PYEOF'
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
PYEOF

# Update run.py (still standalone)
cat > src/agent/run.py << 'PYEOF'
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
PYEOF

# Add test_execute_tool.py
cat > tests/test_execute_tool.py << 'PYEOF'
from src.agent.execute_tool import execute_tool


def test_unknown_tool():
    result = execute_tool("nonexistent", {})
    assert "Unknown tool" in result


def test_execute_date_time():
    result = execute_tool("get_date_time", {})
    assert len(result) > 0
PYEOF

git add -A
git commit -m "lesson-02: tool calling with date_time tool (standalone)"

git branch lesson-02

###############################################################################
# LESSON 03: Evals + UI transition
###############################################################################
echo "=== Building lesson-03 ==="

# Transition run.py to UI function
cat > src/agent/run.py << 'PYEOF'
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
PYEOF

# Add evals types.py
cat > evals/types.py << 'PYEOF'
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
PYEOF

# Add evals evaluators.py (only single-turn evaluators)
cat > evals/evaluators.py << 'PYEOF'
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
PYEOF

# Add evals utils.py
cat > evals/utils.py << 'PYEOF'
import json
from typing import Any
from src.agent.system.prompt import SYSTEM_PROMPT


def build_messages(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """Build message array from eval data.

    Returns a Responses API input list. The system prompt is also returned in
    the array (as a system message) so existing tests that index msgs[0] /
    msgs[1] keep working — single_turn_executor pulls it out and passes it via
    `instructions` instead.
    """
    system_prompt = data.get("system_prompt") or SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["prompt"]},
    ]


def build_mocked_tools(
    mock_tools: dict[str, dict[str, Any]],
) -> tuple[list[dict], dict[str, callable]]:
    """Build Responses API tool definitions and executors from mock config.

    Returns:
        (tool_definitions, executor_map)
    """
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

        # Create executor that returns the mock value
        mock_return = config["mock_return"]
        executor_map[name] = lambda args, ret=mock_return: ret

    return tool_definitions, executor_map
PYEOF

# Add evals executors.py (only single_turn_executor)
cat > evals/executors.py << 'PYEOF'
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
PYEOF

# Add evals/data and file_tools eval data
mkdir -p evals/data
cp "$SRC/evals/data/file_tools.json" evals/data/file_tools.json

# Add file_tools_eval.py
cp "$SRC/evals/file_tools_eval.py" evals/file_tools_eval.py

# Add evals/mocks
mkdir -p evals/mocks
cat > evals/mocks/__init__.py << 'PYEOF'
# Empty init file
PYEOF
cp "$SRC/evals/mocks/tools.py" evals/mocks/tools.py

# Add tests/test_evals_utils.py
cp "$SRC/tests/test_evals_utils.py" tests/test_evals_utils.py

# Update evals/__init__.py
cat > evals/__init__.py << 'PYEOF'
# Empty init file
PYEOF

git add -A
git commit -m "lesson-03: evals framework and UI transition (non-streaming)"

git branch lesson-03

###############################################################################
# LESSON 04: Agent loop with streaming
###############################################################################
echo "=== Building lesson-04 ==="

cat > src/agent/run.py << 'PYEOF'
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
PYEOF

git add -A
git commit -m "lesson-04: streaming agent loop with while True"

git branch lesson-04

###############################################################################
# LESSON 05: Multi-turn evals
###############################################################################
echo "=== Building lesson-05 ==="

# Update evaluators.py to add multi-turn evaluators
cat > evals/evaluators.py << 'PYEOF'
import json
from typing import Any, Union
from openai import OpenAI
from pydantic import BaseModel

from evals.types import (
    EvalTarget,
    SingleTurnResult,
    MultiTurnTarget,
    MultiTurnResult,
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


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


def tool_order_correct(
    output: MultiTurnResult,
    target: MultiTurnTarget,
) -> float:
    """Check if tools were called in the expected order.
    Returns the fraction of expected tools found in sequence.
    """
    if not target.expected_tool_order:
        return 1.0

    actual_order = output.tool_call_order
    expected_idx = 0

    for tool_name in actual_order:
        if tool_name == target.expected_tool_order[expected_idx]:
            expected_idx += 1
            if expected_idx == len(target.expected_tool_order):
                break

    return expected_idx / len(target.expected_tool_order)


def llm_judge(
    output: MultiTurnResult,
    target: MultiTurnTarget,
) -> float:
    """Use an LLM to judge output quality. Returns 0-1."""
    class JudgeResult(BaseModel):
        score: int  # 1-10
        reason: str

    response = _get_client().responses.parse(
        model="gpt-5.1",
        text_format=JudgeResult,
        instructions="""You are an evaluation judge. Score the agent's response on a scale of 1-10.

Scoring criteria:
- 10: Response fully addresses the task using tool results correctly
- 7-9: Response is mostly correct with minor issues
- 4-6: Response partially addresses the task
- 1-3: Response is mostly incorrect or irrelevant""",
        input=f"""Task: {target.original_task}

Tools called: {json.dumps(output.tool_call_order)}
Tool results provided: {json.dumps(target.mock_tool_results)}

Agent's final response:
{output.text}

Evaluate if this response correctly uses the tool results to answer the task.""",
    )

    return response.output_parsed.score / 10
PYEOF

# Update executors.py to add multi_turn_with_mocks
cat > evals/executors.py << 'PYEOF'
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
PYEOF

# Add agent_multiturn_eval.py
cp "$SRC/evals/agent_multiturn_eval.py" evals/agent_multiturn_eval.py

# Add agent_multiturn.json
cp "$SRC/evals/data/agent_multiturn.json" evals/data/agent_multiturn.json

git add -A
git commit -m "lesson-05: multi-turn evals with LLM judge and mock tools"

git branch lesson-05

###############################################################################
# LESSON 06: File system tools
###############################################################################
echo "=== Building lesson-06 ==="

# Add file.py
cp "$SRC/src/agent/tools/file.py" src/agent/tools/file.py

# Remove date_time.py
rm -f src/agent/tools/date_time.py

# Update tools/__init__.py with file tools
cat > src/agent/tools/__init__.py << 'PYEOF'
from src.agent.tools.file import (
    read_file_execute, write_file_execute,
    list_files_execute, delete_file_execute,
    READ_FILE_TOOL, WRITE_FILE_TOOL,
    LIST_FILES_TOOL, DELETE_FILE_TOOL,
)

TOOL_EXECUTORS: dict[str, callable] = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
}

ALL_TOOLS = [
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    LIST_FILES_TOOL,
    DELETE_FILE_TOOL,
]

FILE_TOOLS = [READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_FILES_TOOL, DELETE_FILE_TOOL]
FILE_TOOL_EXECUTORS = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
}

SHELL_TOOLS: list = []
SHELL_TOOL_EXECUTORS: dict = {}
PYEOF

# Update test_execute_tool.py for file tools
cat > tests/test_execute_tool.py << 'PYEOF'
from src.agent.execute_tool import execute_tool


def test_unknown_tool():
    result = execute_tool("nonexistent", {})
    assert "Unknown tool" in result


def test_execute_read_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    assert execute_tool("read_file", {"path": str(f)}) == "hello"
PYEOF

# Add test_file_tools.py
cp "$SRC/tests/test_file_tools.py" tests/test_file_tools.py

# Add shell_tools eval data and eval
cp "$SRC/evals/data/shell_tools.json" evals/data/shell_tools.json
cp "$SRC/evals/shell_tools_eval.py" evals/shell_tools_eval.py

git add -A
git commit -m "lesson-06: file system tools (read, write, list, delete)"

git branch lesson-06

###############################################################################
# LESSON 07: Web search + context management
###############################################################################
echo "=== Building lesson-07 ==="

# Add web_search.py
cp "$SRC/src/agent/tools/web_search.py" src/agent/tools/web_search.py

# Update tools/__init__.py to include web_search
cat > src/agent/tools/__init__.py << 'PYEOF'
from src.agent.tools.file import (
    read_file_execute, write_file_execute,
    list_files_execute, delete_file_execute,
    READ_FILE_TOOL, WRITE_FILE_TOOL,
    LIST_FILES_TOOL, DELETE_FILE_TOOL,
)
from src.agent.tools.web_search import WEB_SEARCH_TOOL, web_search_execute

TOOL_EXECUTORS: dict[str, callable] = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
    "web_search": web_search_execute,
}

ALL_TOOLS = [
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    LIST_FILES_TOOL,
    DELETE_FILE_TOOL,
    WEB_SEARCH_TOOL,
]

FILE_TOOLS = [READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_FILES_TOOL, DELETE_FILE_TOOL]
FILE_TOOL_EXECUTORS = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
}

SHELL_TOOLS: list = []
SHELL_TOOL_EXECUTORS: dict = {}
PYEOF

# Fill in real compaction.py
cp "$SRC/src/agent/context/compaction.py" src/agent/context/compaction.py

# Update run.py with context management (no tool approval yet)
cat > src/agent/run.py << 'PYEOF'
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

            report_token_usage()

    callbacks.on_complete(full_response)
    return input_items
PYEOF

git add -A
git commit -m "lesson-07: web search tool and context window management"

git branch lesson-07

###############################################################################
# LESSON 08: Shell tool + code execution
###############################################################################
echo "=== Building lesson-08 ==="

# Add shell.py and code_execution.py
cp "$SRC/src/agent/tools/shell.py" src/agent/tools/shell.py
cp "$SRC/src/agent/tools/code_execution.py" src/agent/tools/code_execution.py

# Update tools/__init__.py to include shell and code execution
cat > src/agent/tools/__init__.py << 'PYEOF'
from src.agent.tools.file import (
    read_file_execute, write_file_execute,
    list_files_execute, delete_file_execute,
    READ_FILE_TOOL, WRITE_FILE_TOOL,
    LIST_FILES_TOOL, DELETE_FILE_TOOL,
)
from src.agent.tools.shell import run_command_execute, RUN_COMMAND_TOOL
from src.agent.tools.code_execution import execute_code_execute, EXECUTE_CODE_TOOL
from src.agent.tools.web_search import WEB_SEARCH_TOOL, web_search_execute

TOOL_EXECUTORS: dict[str, callable] = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
    "run_command": run_command_execute,
    "execute_code": execute_code_execute,
    "web_search": web_search_execute,
}

ALL_TOOLS = [
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    LIST_FILES_TOOL,
    DELETE_FILE_TOOL,
    RUN_COMMAND_TOOL,
    EXECUTE_CODE_TOOL,
    WEB_SEARCH_TOOL,
]

FILE_TOOLS = [READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_FILES_TOOL, DELETE_FILE_TOOL]
FILE_TOOL_EXECUTORS = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
}

SHELL_TOOLS = [RUN_COMMAND_TOOL]
SHELL_TOOL_EXECUTORS = {
    "run_command": run_command_execute,
}
PYEOF

# Update test_execute_tool.py for full tools
cp "$SRC/tests/test_execute_tool.py" tests/test_execute_tool.py

# Add test_shell_and_code.py
cp "$SRC/tests/test_shell_and_code.py" tests/test_shell_and_code.py

git add -A
git commit -m "lesson-08: shell command and code execution tools"

git branch lesson-08

###############################################################################
# LESSON 09: HITL (Human in the loop)
###############################################################################
echo "=== Building lesson-09 ==="

# Update run.py with full HITL approval logic
cp "$SRC/src/agent/run.py" src/agent/run.py

git add -A
git commit -m "lesson-09: human-in-the-loop tool approval"

git branch lesson-09

###############################################################################
# DONE: Polish + CLAUDE.md
###############################################################################
echo "=== Building done ==="

cat > CLAUDE.md << 'PYEOF'
# Building AI Agents - Python

## Project Overview
This is a Python CLI AI agent built from scratch, serving as the companion code for the Building AI Agents course.

## Branch Strategy
- `done` - The complete, finished app
- Each lesson branch builds forward from the previous lesson
- `lesson-01` through `lesson-09` represent progressive feature additions

## Running
```bash
python -m src.main    # or: agi (after pip install -e .)
```

## Testing
```bash
pytest
```

## Evals
```bash
python -m evals.file_tools_eval
python -m evals.shell_tools_eval
python -m evals.agent_multiturn_eval
```
PYEOF

git add -A
git commit -m "done: complete app with CLAUDE.md"

# Clean up build script
rm -f build_branches.sh
git add -A
git commit -m "chore: remove build script"

echo ""
echo "=== All branches created ==="
git branch -a
echo ""
echo "=== Commit log ==="
git log --oneline --all --graph
