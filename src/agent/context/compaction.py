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
    """Compact a conversation by summarizing it with an LLM.

    Takes a Responses API input-item array and returns a fresh, much shorter
    one (summary as a user message + acknowledgment as an assistant message)
    that the next call can build on.
    """
    # Drop system/developer messages — system prompt is sent separately
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
