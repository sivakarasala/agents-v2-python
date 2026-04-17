from src.agent.context.token_estimator import (
    estimate_tokens,
    estimate_messages_tokens,
    extract_message_text,
)
from src.agent.context.model_limits import (
    get_model_limits,
    is_over_threshold,
    calculate_usage_percentage,
    DEFAULT_THRESHOLD,
    DEFAULT_LIMITS,
)
from src.agent.system.filter_messages import filter_compatible_messages


def test_estimate_tokens_min_one():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("hi") >= 1


def test_estimate_tokens_scales():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 4)


def test_extract_text_string():
    assert extract_message_text({"role": "user", "content": "hello"}) == "hello"


def test_extract_text_list():
    msg = {"role": "user", "content": [{"text": "a"}, {"text": "b"}]}
    out = extract_message_text(msg)
    assert "a" in out and "b" in out


def test_extract_text_tool_calls():
    msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}
    out = extract_message_text(msg)
    assert "x" in out


def test_estimate_messages_splits_input_output():
    usage = estimate_messages_tokens([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi back"},
    ])
    assert usage.input > 0
    assert usage.output > 0
    assert usage.total == usage.input + usage.output


def test_get_model_limits_known():
    limits = get_model_limits("gpt-5-mini")
    assert limits.context_window == 400_000


def test_get_model_limits_gpt5_prefix():
    assert get_model_limits("gpt-5-foo").context_window == 400_000


def test_get_model_limits_unknown_returns_default():
    assert get_model_limits("mystery").context_window == DEFAULT_LIMITS.context_window


def test_is_over_threshold():
    assert is_over_threshold(900, 1000, 0.8) is True
    assert is_over_threshold(700, 1000, 0.8) is False


def test_calculate_usage_percentage():
    assert calculate_usage_percentage(50, 100) == 50.0


def test_filter_messages_keeps_user_and_typed_items():
    # Responses API shape: role-based messages + typed function_call /
    # function_call_output items.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"type": "function_call", "call_id": "x", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "x", "output": "ok"},
    ]
    assert filter_compatible_messages(msgs) == msgs


def test_filter_messages_drops_empty_assistant():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
    ]
    out = filter_compatible_messages(msgs)
    assert len(out) == 1


def test_filter_messages_keeps_function_call_items():
    msgs = [
        {"type": "function_call", "call_id": "x", "name": "f", "arguments": "{}"},
    ]
    assert len(filter_compatible_messages(msgs)) == 1
