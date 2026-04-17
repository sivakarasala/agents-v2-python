from rich.console import Console
from rich.text import Text


console = Console()


def print_message(role: str, content: str) -> None:
    """Print a chat message with color coding."""
    if role == "user":
        label = Text("› You", style="bold blue")
    else:
        label = Text("› Assistant", style="bold green")

    console.print(label)
    console.print(f"  {content}")
    console.print()
