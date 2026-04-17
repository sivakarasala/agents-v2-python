# Shell Tool

## Overview

This lesson gives your agent a computer. When an agent can execute shell commands and run code, it transforms from a text generator into something that can actually *do things*.

## Two Approaches: Shell vs Code Execution

### Shell Tool (`run_command`)
Direct access to the terminal. The agent provides a command string, and we execute it via subprocess.

### Code Execution Tool (`execute_code`)
A higher-level abstraction. The agent provides code in a specific language, and we handle writing it to a temp file and executing with the appropriate runtime.

## Safety Considerations

The implementation runs commands **directly on your host machine** with your user's full permissions. This is NOT production-ready. For production, use sandboxed execution (Docker, gVisor, Firecracker).

## Code

### src/agent/tools/shell.py

The shell tool executes arbitrary commands using `subprocess`:

```python
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
```

### src/agent/tools/code_execution.py

The code execution tool - a composite tool that writes to a temp file and runs it:

```python
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

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_file = f.name

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
```

### src/agent/tools/__init__.py

Register both shell tools:

```python
from src.agent.tools.shell import run_command_execute, RUN_COMMAND_TOOL
from src.agent.tools.code_execution import execute_code_execute, EXECUTE_CODE_TOOL
```

Add to `TOOL_EXECUTORS`:
```python
"run_command": run_command_execute,
"execute_code": execute_code_execute,
```

Add to `ALL_TOOLS`:
```python
RUN_COMMAND_TOOL,
EXECUTE_CODE_TOOL,
```

Update shell tool exports:
```python
SHELL_TOOLS = [RUN_COMMAND_TOOL]
SHELL_TOOL_EXECUTORS = {
    "run_command": run_command_execute,
}
```

### tests/test_shell_and_code.py

```python
from src.agent.tools.shell import run_command_execute
from src.agent.tools.code_execution import execute_code_execute


def test_run_command_basic():
    out = run_command_execute({"command": "echo hello"})
    assert "hello" in out


def test_run_command_failure():
    out = run_command_execute({"command": "false"})
    assert "Command failed" in out


def test_execute_python_code():
    out = execute_code_execute({"code": "print(2 + 2)", "language": "python"})
    assert "4" in out


def test_execute_unsupported_language():
    out = execute_code_execute({"code": "x", "language": "ruby"})
    assert "Unsupported" in out


def test_execute_python_error():
    out = execute_code_execute({"code": "raise ValueError('boom')"})
    assert "Execution failed" in out
```

### tests/test_execute_tool.py

Updated to verify the full tool registry:

```python
from src.agent.execute_tool import execute_tool
from src.agent.tools import ALL_TOOLS, TOOL_EXECUTORS, FILE_TOOLS, SHELL_TOOLS


def test_execute_tool_unknown():
    out = execute_tool("nope", {})
    assert "Unknown tool" in out


def test_execute_tool_known(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    assert execute_tool("read_file", {"path": str(f)}) == "hello"


def test_execute_tool_handles_exception():
    out = execute_tool("read_file", {"path": "/definitely/not/a/real/path/xyz"})
    assert out.startswith("Error")


def test_registry_has_expected_tools():
    names = set(TOOL_EXECUTORS.keys())
    assert {"read_file", "write_file", "list_files", "delete_file",
            "run_command", "execute_code", "web_search"} <= names


def test_function_tool_definitions_well_formed():
    for tool in ALL_TOOLS:
        assert "type" in tool
        if tool["type"] == "function":
            assert "name" in tool
            assert "parameters" in tool


def test_file_and_shell_subsets():
    file_names = {t["name"] for t in FILE_TOOLS}
    assert file_names == {"read_file", "write_file", "list_files", "delete_file"}
    shell_names = {t["name"] for t in SHELL_TOOLS}
    assert shell_names == {"run_command"}
```

## Composite Tools vs Orchestrated Tools

The `execute_code` tool bundles write-file + execute + cleanup into one. Trade-offs:

**Composite (what we did):** Fewer round trips, cleaner mental model, agent doesn't need to know about temp files.

**Orchestrated (let agent coordinate):** Agent has full control over each step, can inspect intermediate results, but uses more tokens and agent might forget cleanup.
