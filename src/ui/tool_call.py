from rich.console import Console
from rich.text import Text

console = Console()


def print_tool_start(name: str, args: dict = None) -> None:
    """Show a tool call starting."""
    summary = ""
    if args:
        for key in ("path", "command", "query", "code", "content"):
            if key in args and isinstance(args[key], str):
                value = args[key]
                if len(value) > 50:
                    value = value[:50] + "..."
                summary = f"({value})"
                break

    console.print(f"  ⚡ [bold yellow]{name}[/bold yellow]{summary} ...", end="")


def print_tool_end(name: str, result: str) -> None:
    """Show a tool call completed."""
    console.print(" [green]✓[/green]")
    truncated = result[:100] + "..." if len(result) > 100 else result
    console.print(f"    [dim]→ {truncated}[/dim]")
