from typing import Any


def create_mock_read_file(mock_content: str):
    """Create a mock read_file executor."""
    def execute(args: dict[str, Any]) -> str:
        return mock_content
    return execute


def create_mock_write_file(mock_response: str = None):
    """Create a mock write_file executor."""
    def execute(args: dict[str, Any]) -> str:
        if mock_response:
            return mock_response
        content = args.get("content", "")
        path = args.get("path", "unknown")
        return f"Successfully wrote {len(content)} characters to {path}"
    return execute


def create_mock_list_files(mock_files: list[str]):
    """Create a mock list_files executor."""
    def execute(args: dict[str, Any]) -> str:
        return "\n".join(mock_files)
    return execute


def create_mock_delete_file(mock_response: str = None):
    """Create a mock delete_file executor."""
    def execute(args: dict[str, Any]) -> str:
        if mock_response:
            return mock_response
        return f"Successfully deleted {args.get('path', 'unknown')}"
    return execute


def create_mock_shell(mock_output: str):
    """Create a mock shell command executor."""
    def execute(args: dict[str, Any]) -> str:
        return mock_output
    return execute
