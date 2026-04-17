import subprocess
from typing import Any


def run_command_execute(args: dict[str, Any]) -> str:
    """Execute a shell command and return its output."""
    command = args["command"]
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr

        if result.returncode != 0:
            return f"Command failed (exit code {result.returncode}):\n{output}"

        return output or "Command completed successfully (no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {e}"


RUN_COMMAND_TOOL = {
    "type": "function",
    "name": "run_command",
    "description": "Execute a shell command and return its output. Use this for system operations, running scripts, or interacting with the operating system.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            }
        },
        "required": ["command"],
    },
}
