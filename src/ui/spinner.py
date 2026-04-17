from rich.console import Console
from rich.spinner import Spinner as RichSpinner
from rich.live import Live


class Spinner:
    """A terminal spinner for showing loading state."""

    def __init__(self, label: str = "Thinking..."):
        self.console = Console()
        self.label = label
        self.live = None

    def start(self):
        self.live = Live(
            RichSpinner("dots", text=f" {self.label}"),
            console=self.console,
            refresh_per_second=10,
        )
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()
            self.live = None
