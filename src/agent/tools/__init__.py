from src.agent.tools.date_time import get_date_time_execute, GET_DATE_TIME_TOOL

TOOL_EXECUTORS: dict[str, callable] = {
    "get_date_time": get_date_time_execute,
}

ALL_TOOLS = [GET_DATE_TIME_TOOL]
FILE_TOOLS: list = []
FILE_TOOL_EXECUTORS: dict = {}
SHELL_TOOLS: list = []
SHELL_TOOL_EXECUTORS: dict = {}
