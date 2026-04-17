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
