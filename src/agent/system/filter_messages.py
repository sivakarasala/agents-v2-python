from typing import Any


def filter_compatible_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter conversation history into a clean Responses API input list.

    The Responses API uses a list of "input items":
      - role-based messages: {"role": "user"|"assistant"|"system", "content": ...}
      - typed items: {"type": "function_call", ...}, {"type": "function_call_output", ...},
        {"type": "web_search_call", ...}, etc.

    We drop empty assistant messages (no useful content) but keep all typed
    items so function_call / function_call_output pairs stay intact for the
    next turn.
    """
    filtered: list[dict[str, Any]] = []

    for msg in messages:
        # Typed items (function_call, function_call_output, web_search_call, …)
        # are always kept verbatim.
        if "type" in msg and "role" not in msg:
            filtered.append(msg)
            continue

        role = msg.get("role")

        if role in ("user", "system", "developer"):
            filtered.append(msg)
            continue

        if role == "assistant":
            content = msg.get("content")
            has_text = False
            if isinstance(content, str) and content.strip():
                has_text = True
            elif isinstance(content, list) and content:
                has_text = True

            if has_text:
                filtered.append(msg)
                continue

        # Anything else (e.g. legacy "tool" role from old transcripts) — skip
        # silently rather than crashing the next request.

    return filtered
