# Multi-turn Evals

## Overview

Single-turn evals test tool selection. Multi-turn evals test the full agent loop - did the agent accomplish the task across multiple steps? This is where we evaluate agent behavior end-to-end.

## Why Multi-Turn Evals Matter

Multi-turn evals catch failures that single-turn evals miss:
- Agent picks right first tool but wrong second tool
- Agent gets stuck in loops
- Agent misinterprets tool results
- Agent gives up too early

## LLM-as-Judge

The solution for evaluating non-deterministic output: use another LLM to evaluate.

Instead of checking `output == expected`, we ask a judge model:
- "Given this task and these tool results, is this response correct?"
- "Does this answer make sense?"

### Making LLM-as-Judge Reliable

- **Use structured output**: `responses.parse()` with a Pydantic model
- **Use a stronger model**: The judge should be at least as capable as the agent
- **Clear criteria**: Define exactly what 1-10 means
- **Provide full context**: task, tools called, tool results, agent response

## Code

### evals/evaluators.py

Add `tool_order_correct` and `llm_judge` evaluators:

```python
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
```

Key details:
- `responses.parse()` with `text_format=JudgeResult` gives structured output
- 1-10 scale converted to 0-1 for consistency
- `tool_order_correct` checks subsequence matching, not exact match

### evals/executors.py

Add the multi-turn executor with mocked tools:

```python
def multi_turn_with_mocks(data: dict[str, Any]) -> MultiTurnResult:
    """Run a multi-turn evaluation with mocked tools, using the Responses API."""
    tool_definitions, executor_map = build_mocked_tools(data["mock_tools"])

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
```

Key details:
- Uses `build_mocked_tools` to create tools with fixed return values
- Supports both fresh prompts and pre-filled message history
- `max_steps` prevents infinite loops
- Non-streaming (evals don't need streaming)

### evals/agent_multiturn_eval.py

The multi-turn evaluation runner:

```python
import json
from dotenv import load_dotenv

from evals.executors import multi_turn_with_mocks
from evals.evaluators import tool_order_correct, tools_avoided, llm_judge
from evals.types import MultiTurnTarget, MultiTurnResult

load_dotenv()


def load_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def run_eval():
    dataset = load_dataset("evals/data/agent_multiturn.json")

    for i, entry in enumerate(dataset):
        data = entry["data"]
        target_data = entry["target"]

        target = MultiTurnTarget(
            original_task=target_data["original_task"],
            mock_tool_results=target_data.get("mock_tool_results", {}),
            category=target_data["category"],
            expected_tool_order=target_data.get("expected_tool_order"),
            forbidden_tools=target_data.get("forbidden_tools"),
        )

        output = multi_turn_with_mocks(data)

        scores = {}
        if target.expected_tool_order:
            scores["tool_order"] = tool_order_correct(output, target)
        if target.forbidden_tools:
            scores["tools_avoided"] = tools_avoided(output, target)

        scores["output_quality"] = llm_judge(output, target)

        prompt = data.get("prompt", "(mid-conversation)")
        status = "✓" if all(v >= 0.7 for v in scores.values()) else "✗"
        print(f"  {status} [{target.category}] {prompt}")
        print(f"    Tools called: {output.tool_call_order}")
        print(f"    Scores: {scores}")
        print()


if __name__ == "__main__":
    print("Multi-Turn Agent Evaluation")
    print("=" * 40)
    run_eval()
```

## Running Evals

```bash
# Run multi-turn eval
python evals/agent_multiturn_eval.py
```

## Why Mock Tools in Evals?

- **Reproducibility**: Mocks return the same value every time
- **Speed**: No actual I/O or network calls
- **Safety**: Can't accidentally delete files during testing
- **Edge cases**: Easy to test "file not found" by setting mock results
- **Isolation**: Each test case is independent
