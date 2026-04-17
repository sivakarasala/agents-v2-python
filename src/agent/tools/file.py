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
