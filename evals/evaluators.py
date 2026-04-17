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
