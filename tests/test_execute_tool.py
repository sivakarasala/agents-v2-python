from src.agent.execute_tool import execute_tool


def test_unknown_tool():
    result = execute_tool("nonexistent", {})
    assert "Unknown tool" in result


def test_execute_date_time():
    result = execute_tool("get_date_time", {})
    assert len(result) > 0
