import asyncio
from typing import Any
from rich.console import Console
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.patch_stdout import patch_stdout

from src.agent.run import run_agent
from src.types import AgentCallbacks, TokenUsageInfo
from src.ui.message_list import print_message
from src.ui.tool_call import print_tool_start, print_tool_end
from src.ui.tool_approval import request_approval
from src.ui.token_usage import print_token_usage
from src.ui.spinner import Spinner

console = Console()


def run_app():
    """Main application loop."""
    console.print("[bold magenta]🤖 AI Agent[/bold magenta] [dim](type 'exit' to quit)[/dim]")
    console.print()

    conversation_history: list[dict[str, Any]] = []
    token_usage_info: TokenUsageInfo | None = None

    while True:
        # Get user input
        try:
            user_input = pt_prompt("> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("Goodbye!")
            break

        print_message("user", user_input)

        # Track streaming state
        streaming_text = ""
        spinner = Spinner()
        spinner_active = False

        def on_token(token: str):
            nonlocal streaming_text, spinner_active
            if spinner_active:
                spinner.stop()
                spinner_active = False
                console.print("[bold green]› Assistant[/bold green]")
                console.print("  ", end="")
            streaming_text += token
            console.print(token, end="", highlight=False)

        def on_tool_call_start(name: str, args: Any):
            nonlocal spinner_active
            if spinner_active:
                spinner.stop()
                spinner_active = False
            print_tool_start(name, args if isinstance(args, dict) else {})

        def on_tool_call_end(name: str, result: str):
            print_tool_end(name, result)

        def on_complete(response: str):
            nonlocal spinner_active
            if spinner_active:
                spinner.stop()
                spinner_active = False
            if streaming_text:
                console.print()  # Newline after streamed text
            console.print()

        async def on_tool_approval(name: str, args: Any) -> bool:
            return request_approval(name, args if isinstance(args, dict) else {})

        def on_token_usage(usage: TokenUsageInfo):
            nonlocal token_usage_info
            token_usage_info = usage

        # Start spinner
        spinner.start()
        spinner_active = True

        try:
            new_history = run_agent(
                user_input,
                conversation_history,
                AgentCallbacks(
                    on_token=on_token,
                    on_tool_call_start=on_tool_call_start,
                    on_tool_call_end=on_tool_call_end,
                    on_complete=on_complete,
                    on_tool_approval=on_tool_approval,
                    on_token_usage=on_token_usage,
                ),
            )
            conversation_history = new_history
        except Exception as e:
            if spinner_active:
                spinner.stop()
            console.print(f"\n  [red]Error: {e}[/red]")
            console.print()

        # Show token usage
        if token_usage_info:
            print_token_usage(token_usage_info)

        streaming_text = ""
