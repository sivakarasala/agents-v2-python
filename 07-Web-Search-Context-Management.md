---
layout: default
title: "Web Search + Context Management"
---

# Web Search + Context Management

## Overview

This lesson covers two related topics: giving your agent access to the web, and managing the inevitable context window bloat that comes from accumulating tool results.

## Part 1: Web Search for Agents

An LLM's knowledge is frozen at its training cutoff. Web search gives your agent access to current information.

### Provider-Managed Web Search

The Responses API has a built-in `web_search` tool type. It's provider-managed - OpenAI handles the search and inlines results into the model's reasoning. We never see a `function_call` for it.

```python
WEB_SEARCH_TOOL = {
    "type": "web_search",
}
```

One line. That's it.

## Part 2: The Context Window Problem

### What Is a Context Window?

The maximum number of tokens a model can process in a single request. Current limits (2025):
- GPT-5: 400K tokens
- GPT-5-mini: 400K tokens

It adds up fast with agents that loop multiple times.

### Our Strategy: Compaction

1. Estimate token usage before each turn
2. If over threshold (80% of context window), trigger compaction
3. Summarize conversation history into a condensed form
4. Replace history with summary
5. Continue conversation

## Code

### src/agent/tools/web_search.py

Provider-managed web search tool:

```python
from typing import Any

# Web search is a provider-managed tool on the Responses API — OpenAI runs it
# server-side and inlines the results into the model's reasoning. We just
# declare it in the tools list; we never see a function_call for it and never
# need to return a function_call_output.
WEB_SEARCH_TOOL = {
    "type": "web_search",
}


def web_search_execute(args: dict[str, Any]) -> str:
    """Provider tools are executed by OpenAI, not us. This stub exists only so
    the registry has something to look up if the model ever surfaces it."""
    return "Provider tool web_search - executed by model provider"
```

### src/agent/tools/__init__.py

Add web search to the tool registry:

```python
from src.agent.tools.web_search import WEB_SEARCH_TOOL, web_search_execute
```

Add to `TOOL_EXECUTORS`:
```python
"web_search": web_search_execute,
```

Add to `ALL_TOOLS`:
```python
WEB_SEARCH_TOOL,
```

### src/agent/context/compaction.py

The summarization prompt and compaction logic. Fill in the stub:

```python
from typing import Any
from openai import OpenAI
from src.agent.context.token_estimator import extract_message_text

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

SUMMARIZATION_PROMPT = """You are a conversation summarizer. Your task is to create a concise summary of the conversation so far that preserves:

1. Key decisions and conclusions reached
2. Important context and facts mentioned
3. Any pending tasks or questions
4. The overall goal of the conversation

Be concise but complete. The summary should allow the conversation to continue naturally.

Conversation to summarize:
"""


def messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Format messages as readable text for summarization."""
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
    """Compact a conversation by summarizing it with an LLM."""
    conversation_messages = [
        m for m in messages
        if m.get("role") not in ("system", "developer")
    ]

    if not conversation_messages:
        return []

    conversation_text = messages_to_text(conversation_messages)

    response = _get_client().responses.create(
        model=model,
        input=SUMMARIZATION_PROMPT + conversation_text,
    )

    summary = response.output_text

    return [
        {
            "role": "user",
            "content": (
                f"[CONVERSATION SUMMARY]\n"
                f"The following is a summary of our conversation so far:\n\n"
                f"{summary}\n\n"
                f"Please continue from where we left off."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "I understand. I've reviewed the summary of our conversation "
                "and I'm ready to continue. How can I help you next?"
            ),
        },
    ]
```

### src/agent/run.py

Add context management logic. Before the loop, check if we need to compact. Add token usage reporting throughout:

```python
from src.agent.context import (
    estimate_messages_tokens,
    get_model_limits,
    is_over_threshold,
    calculate_usage_percentage,
    compact_conversation,
    DEFAULT_THRESHOLD,
)
from src.types import AgentCallbacks, ToolCallInfo, TokenUsageInfo
```

Before the loop starts:
```python
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
```

Token usage reporting helper:
```python
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
```

Call `report_token_usage()` after each tool result is appended.

## The Compaction Strategy Explained

1. **Pre-check**: Before starting a turn, estimate total tokens
2. **Threshold**: If over 80% of context window, compact
3. **Summarize**: Use the LLM itself to summarize the conversation
4. **Replace**: Swap detailed history with compact summary
5. **Seed**: Start with a summary message + acknowledgment

### Why 80%?

We need headroom for the new user message, tool calls and results, the assistant's response, and safety margin for estimation errors.
