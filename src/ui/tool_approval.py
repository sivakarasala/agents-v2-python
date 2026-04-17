import json
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import prompt

console = Console()

MAX_PREVIEW_LINES = 5


def format_args_preview(args: dict) -> tuple[str, int]:
    """Format args as JSON preview with line limit."""
    formatted = json.dumps(args, indent=2)
    lines = formatted.split("\n")

    if len(lines) <= MAX_PREVIEW_LINES:
        return formatted, 0

    preview = "\n".join(lines[:MAX_PREVIEW_LINES])
    extra = len(lines) - MAX_PREVIEW_LINES
    return preview, extra


def get_args_summary(args) -> str:
    """Get a one-line summary of the most meaningful arg."""
    if not isinstance(args, dict):
        return str(args)

    for key in ("path", "filePath", "command", "query", "code", "content"):
        if key in args and isinstance(args[key], str):
            value = args[key]
            if len(value) > 50:
                return value[:50] + "..."
            return value

    keys = list(args.keys())
    if keys and isinstance(args[keys[0]], str):
        value = args[keys[0]]
        if len(value) > 50:
            return value[:50] + "..."
        return value

    return ""


def request_approval(tool_name: str, args: dict) -> bool:
    """Show tool approval prompt and return True if approved."""
    console.print()
    console.print("[bold yellow]Tool Approval Required[/bold yellow]")

    summary = get_args_summary(args)
    summary_text = f" [dim]({summary})[/dim]" if summary else ""
    console.print(f"  [bold cyan]{tool_name}[/bold cyan]{summary_text}")

    preview, extra = format_args_preview(args)
    console.print(f"    [dim]{preview}[/dim]")
    if extra > 0:
        console.print(f"    [dim]... +{extra} more lines[/dim]")

    console.print()

    while True:
        try:
            answer = prompt("  Approve? [Y/n] ").strip().lower()
            if answer in ("", "y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            console.print("  [dim]Please enter Y or N[/dim]")
        except (KeyboardInterrupt, EOFError):
            return False
