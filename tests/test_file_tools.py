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
