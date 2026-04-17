from src.agent.tools.file import (
    read_file_execute, write_file_execute,
    list_files_execute, delete_file_execute,
    READ_FILE_TOOL, WRITE_FILE_TOOL,
    LIST_FILES_TOOL, DELETE_FILE_TOOL,
)
from src.agent.tools.web_search import WEB_SEARCH_TOOL, web_search_execute

TOOL_EXECUTORS: dict[str, callable] = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
    "web_search": web_search_execute,
}

ALL_TOOLS = [
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    LIST_FILES_TOOL,
    DELETE_FILE_TOOL,
    WEB_SEARCH_TOOL,
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
