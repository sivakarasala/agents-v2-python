from rich.console import Console
from rich.panel import Panel
from src.types import TokenUsageInfo

console = Console()


def print_token_usage(usage: TokenUsageInfo) -> None:
    """Display token usage with color-coded percentage."""
    threshold_percent = round(usage.threshold * 100)
    usage_percent = f"{usage.percentage:.1f}"

    # Color based on usage
    if usage.percentage >= usage.threshold * 100:
        color = "red"
    elif usage.percentage >= usage.threshold * 100 * 0.75:
        color = "yellow"
    else:
        color = "green"

    text = f"Tokens: [{color} bold]{usage_percent}%[/{color} bold] [dim](threshold: {threshold_percent}%)[/dim]"
    console.print(Panel(text, border_style="dim"))
