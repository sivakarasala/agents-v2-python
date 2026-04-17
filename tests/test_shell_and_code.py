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
