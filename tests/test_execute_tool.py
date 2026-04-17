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
    # read_file_execute returns its own error string for missing file,
    # so this exercises a tool that internally swallows the FileNotFoundError.
    out = execute_tool("read_file", {"path": "/definitely/not/a/real/path/xyz"})
    assert out.startswith("Error")


def test_registry_has_expected_tools():
    names = set(TOOL_EXECUTORS.keys())
    assert {"read_file", "write_file", "list_files", "delete_file",
            "run_command", "execute_code", "web_search"} <= names


def test_function_tool_definitions_well_formed():
    # Responses API flat format: {"type": "function", "name": ..., "parameters": ...}
    # plus provider-managed tools like {"type": "web_search"}
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
