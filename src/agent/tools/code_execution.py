import os
import tempfile
import subprocess
from typing import Any


def execute_code_execute(args: dict[str, Any]) -> str:
    """Execute code by writing to a temp file and running it."""
    code = args["code"]
    language = args.get("language", "python")

    extensions = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
    }

    commands = {
        "python": lambda f: f"python3 {f}",
        "javascript": lambda f: f"node {f}",
        "typescript": lambda f: f"npx tsx {f}",
    }

    ext = extensions.get(language, ".py")
    get_command = commands.get(language)

    if not get_command:
        return f"Unsupported language: {language}"

    # Write code to temp file
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_file = f.name

        # Execute
        command = get_command(tmp_file)
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
            return f"Execution failed (exit code {result.returncode}):\n{output}"

        return output or "Code executed successfully (no output)"

    except subprocess.TimeoutExpired:
        return "Error: Execution timed out after 30 seconds"
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        # Clean up temp file
        if tmp_file:
            try:
                os.unlink(tmp_file)
            except OSError:
                pass


EXECUTE_CODE_TOOL = {
    "type": "function",
    "name": "execute_code",
    "description": "Execute code for anything you need compute for. Supports Python, JavaScript, and TypeScript. Returns the output of the execution.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The code to execute",
            },
            "language": {
                "type": "string",
                "enum": ["python", "javascript", "typescript"],
                "description": "The programming language of the code",
                "default": "python",
            },
        },
        "required": ["code"],
    },
}
