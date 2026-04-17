from datetime import datetime
from typing import Any


def get_date_time_execute(args: dict[str, Any]) -> str:
    return datetime.now().isoformat()


GET_DATE_TIME_TOOL = {
    "type": "function",
    "name": "get_date_time",
    "description": "Get the current date and time",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}
