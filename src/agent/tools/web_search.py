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
