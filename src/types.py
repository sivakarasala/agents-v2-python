from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional


@dataclass
class ToolCallInfo:
    """Metadata about a tool the LLM wants to call."""
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass
class ModelLimits:
    """Token limits for a model."""
    input_limit: int
    output_limit: int
    context_window: int


@dataclass
class TokenUsageInfo:
    """Current token usage for display."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_window: int
    threshold: float
    percentage: float


@dataclass
class AgentCallbacks:
    """How the agent communicates back to the UI."""
    on_token: Callable[[str], None]
    on_tool_call_start: Callable[[str, Any], None]
    on_tool_call_end: Callable[[str, str], None]
    on_complete: Callable[[str], None]
    on_tool_approval: Callable[[str, Any], Awaitable[bool]]
    on_token_usage: Optional[Callable[[TokenUsageInfo], None]] = None


@dataclass
class ToolApprovalRequest:
    """A pending tool approval for the UI to display."""
    tool_name: str
    args: Any
    resolve: Callable[[bool], None]
