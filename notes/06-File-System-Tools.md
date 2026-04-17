# File System Tools

## Overview

File system access is one of the most powerful capabilities you can give an agent. Files become the agent's memory, workspace, and interface to the broader system. This lesson replaces the `date_time` tool with real file tools.

## Why Files Matter for Agents

- **Reading source code** to understand a codebase
- **Writing code files** when implementing features
- **Files as memory** - persistent state across sessions
- **Files as scratch pad** - external working memory for large data
- **Files as audit trail** - logging agent decisions

## The Four Core Operations

1. **Read** - Essential for understanding anything
2. **Write** - Creates new files or overwrites existing ones
3. **List** - Navigate the file system, discover what's available
4. **Delete** - Clean up temporary files, remove outdated content

## Code

### src/agent/tools/file.py

The complete file tools implementation:

```python
import os
from typing import Any


def read_file_execute(args: dict[str, Any]) -> str:
    """Execute the read_file tool."""
    file_path = args["path"]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file_execute(args: dict[str, Any]) -> str:
    """Execute the write_file tool."""
    file_path = args["path"]
    content = args["content"]
    try:
        # Create parent directories if they don't exist
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_files_execute(args: dict[str, Any]) -> str:
    """Execute the list_files tool."""
    directory = args.get("directory", ".")
    try:
        entries = os.listdir(directory)
        items = []
        for entry in sorted(entries):
            full_path = os.path.join(directory, entry)
            entry_type = "[dir]" if os.path.isdir(full_path) else "[file]"
            items.append(f"{entry_type} {entry}")
        return "\n".join(items) if items else f"Directory {directory} is empty"
    except FileNotFoundError:
        return f"Error: Directory not found: {directory}"
    except Exception as e:
        return f"Error listing directory: {e}"


def delete_file_execute(args: dict[str, Any]) -> str:
    """Execute the delete_file tool."""
    file_path = args["path"]
    try:
        os.unlink(file_path)
        return f"Successfully deleted {file_path}"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error deleting file: {e}"


# Tool definitions in OpenAI Responses API format (flat — no nested "function" key)
READ_FILE_TOOL = {
    "type": "function",
    "name": "read_file",
    "description": "Read the contents of a file at the specified path. Use this to examine file contents.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to read",
            }
        },
        "required": ["path"],
    },
}

WRITE_FILE_TOOL = {
    "type": "function",
    "name": "write_file",
    "description": "Write content to a file at the specified path. Creates the file if it doesn't exist, overwrites if it does.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
}

LIST_FILES_TOOL = {
    "type": "function",
    "name": "list_files",
    "description": "List all files and directories in the specified directory path.",
    "parameters": {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "The directory path to list contents of",
                "default": ".",
            }
        },
    },
}

DELETE_FILE_TOOL = {
    "type": "function",
    "name": "delete_file",
    "description": "Delete a file at the specified path. Use with caution as this is irreversible.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to delete",
            }
        },
        "required": ["path"],
    },
}
```

### src/agent/tools/__init__.py

Replace date_time tool with file tools:

```python
from src.agent.tools.file import (
    read_file_execute, write_file_execute,
    list_files_execute, delete_file_execute,
    READ_FILE_TOOL, WRITE_FILE_TOOL,
    LIST_FILES_TOOL, DELETE_FILE_TOOL,
)

TOOL_EXECUTORS: dict[str, callable] = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
}

ALL_TOOLS = [
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    LIST_FILES_TOOL,
    DELETE_FILE_TOOL,
]

FILE_TOOLS = [READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_FILES_TOOL, DELETE_FILE_TOOL]
FILE_TOOL_EXECUTORS = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
}

SHELL_TOOLS: list = []
SHELL_TOOL_EXECUTORS: dict = {}
```

Note: `date_time.py` is removed at this step. File tools replace it entirely.

### evals/data/shell_tools.json

Shell tools eval dataset (used in lesson 06 for the shell eval runner, even though shell tools aren't implemented yet - the eval tests that the model selects `run_command` from available tools):

```json
[
  {
    "data": {
      "prompt": "Run npm install to install the dependencies",
      "tools": ["run_command"]
    },
    "target": {
      "expected_tools": ["run_command"],
      "category": "golden"
    }
  },
  {
    "data": {
      "prompt": "Check the git status of this repository",
      "tools": ["run_command"]
    },
    "target": {
      "expected_tools": ["run_command"],
      "category": "golden"
    }
  },
  {
    "data": {
      "prompt": "What is TypeScript used for?",
      "tools": ["run_command"]
    },
    "target": {
      "forbidden_tools": ["run_command"],
      "category": "negative"
    }
  }
]
```

### evals/shell_tools_eval.py

```python
import json
from dotenv import load_dotenv

from src.agent.tools import SHELL_TOOLS
from evals.executors import single_turn_executor
from evals.evaluators import tools_selected, tools_avoided, tool_selection_score
from evals.types import EvalTarget

load_dotenv()


def run_eval():
    with open("evals/data/shell_tools.json", "r") as f:
        dataset = json.load(f)

    for entry in dataset:
        data = entry["data"]
        target_data = entry["target"]

        target = EvalTarget(
            category=target_data["category"],
            expected_tools=target_data.get("expected_tools"),
            forbidden_tools=target_data.get("forbidden_tools"),
        )

        output = single_turn_executor(data, SHELL_TOOLS)

        scores = {}
        if target.category == "golden":
            scores["tools_selected"] = tools_selected(output, target)
        elif target.category == "negative":
            scores["tools_avoided"] = tools_avoided(output, target)

        status = "✓" if all(v >= 1.0 for v in scores.values()) else "✗"
        print(f"  {status} [{target.category}] {data['prompt']}")
        print(f"    Selected: {output.tool_names}  Scores: {scores}")
        print()


if __name__ == "__main__":
    print("Shell Tools Evaluation")
    print("=" * 40)
    run_eval()
```

### tests/test_file_tools.py

```python
import os
import pytest
from src.agent.tools.file import (
    read_file_execute,
    write_file_execute,
    list_files_execute,
    delete_file_execute,
)


def test_write_then_read(tmp_path):
    f = tmp_path / "hello.txt"
    result = write_file_execute({"path": str(f), "content": "hi there"})
    assert "Successfully wrote" in result
    assert read_file_execute({"path": str(f)}) == "hi there"


def test_write_creates_parent_dirs(tmp_path):
    f = tmp_path / "nested" / "deep" / "file.txt"
    write_file_execute({"path": str(f), "content": "x"})
    assert f.exists()


def test_read_missing_file_returns_error(tmp_path):
    out = read_file_execute({"path": str(tmp_path / "nope.txt")})
    assert out.startswith("Error: File not found")


def test_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "sub").mkdir()
    out = list_files_execute({"directory": str(tmp_path)})
    assert "[file] a.txt" in out
    assert "[file] b.txt" in out
    assert "[dir] sub" in out


def test_list_files_missing(tmp_path):
    out = list_files_execute({"directory": str(tmp_path / "nope")})
    assert out.startswith("Error: Directory not found")


def test_delete_file(tmp_path):
    f = tmp_path / "to-delete.txt"
    f.write_text("bye")
    result = delete_file_execute({"path": str(f)})
    assert "Successfully deleted" in result
    assert not f.exists()


def test_delete_missing_file(tmp_path):
    out = delete_file_execute({"path": str(tmp_path / "nope.txt")})
    assert out.startswith("Error: File not found")
```

### tests/test_execute_tool.py

Updated to test file tools instead of date_time:

```python
from src.agent.execute_tool import execute_tool


def test_unknown_tool():
    result = execute_tool("nonexistent", {})
    assert "Unknown tool" in result


def test_execute_read_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    assert execute_tool("read_file", {"path": str(f)}) == "hello"
```

## Design Decisions

### Why Separate Tools vs One "File" Tool?
Separate tools because:
- **Clearer intent**: Model knows exactly what each tool does
- **Simpler schemas**: Each tool has only relevant parameters
- **Easier permissions**: Can allow read but block write

### Why Return Errors as Strings Instead of Throwing?
The agent needs to handle errors gracefully. By returning error strings, the agent sees the error in tool output and can decide how to proceed.
