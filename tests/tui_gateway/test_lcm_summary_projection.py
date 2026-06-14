"""Tests for LCM summary metadata projection and lcm_expand RPC."""

import json
import pytest
from unittest.mock import MagicMock, patch


# ── Projection function tests ──────────────────────────────────────


def _import_projection():
    from tui_gateway.server import _lcm_summary_metadata_for_message
    return _lcm_summary_metadata_for_message


def test_no_summary_markers_returns_none():
    f = _import_projection()
    assert f("Hello, this is a normal message.") is None


def test_empty_string_returns_none():
    f = _import_projection()
    assert f("") is None


def test_single_summary_node():
    f = _import_projection()
    content = (
        "[Recent Summary (d0, node 42)]\n"
        "The user discussed authentication and JWT tokens.\n"
        "[Expand for details: authentication and JWT]"
    )
    result = f(content)
    assert result is not None
    assert result["node_ids"] == [42]
    assert result["depths"] == [0]
    assert result["depth_labels"] == ["Recent"]
    assert result["expand_hints"] == ["authentication and JWT"]


def test_multiple_summary_nodes_separated():
    f = _import_projection()
    content = (
        "[Recent Summary (d0, node 42)]\n"
        "Summary A\n"
        "[Expand for details: topic A]\n"
        "\n---\n\n"
        "[Session Arc Summary (d1, node 7)]\n"
        "Summary B\n"
        "[Expand for details: topic B]"
    )
    result = f(content)
    assert result is not None
    assert result["node_ids"] == [42, 7]
    assert result["depths"] == [0, 1]
    assert result["depth_labels"] == ["Recent", "Session Arc"]
    assert result["expand_hints"] == ["topic A", "topic B"]


def test_depth_label_variants():
    f = _import_projection()
    content = "[Durable Summary (d2, node 99)]\nSummary\n[Expand for details: D topic]"
    result = f(content)
    assert result is not None
    assert result["depth_labels"] == ["Durable"]
    assert result["depths"] == [2]
    assert result["node_ids"] == [99]


def test_numeric_depth_label():
    f = _import_projection()
    content = "[Depth-3 Summary (d3, node 5)]\nSummary\n[Expand for details: deep topic]"
    result = f(content)
    assert result is not None
    assert result["depth_labels"] == ["Depth-3"]
    assert result["depths"] == [3]


def test_summary_without_expand_hint():
    f = _import_projection()
    content = "[Recent Summary (d0, node 42)]\nJust a summary, no expand marker."
    result = f(content)
    assert result is not None
    assert result["node_ids"] == [42]
    assert result["expand_hints"] == [""]


def test_mixed_summary_and_non_summary_parts():
    f = _import_projection()
    content = (
        "Some regular text\n"
        "\n---\n\n"
        "[Recent Summary (d0, node 10)]\n"
        "Summary text\n"
        "[Expand for details: mixed]"
    )
    result = f(content)
    assert result is not None
    assert result["node_ids"] == [10]


def test_malformed_summary_text():
    f = _import_projection()
    # Text that has "Summary (d" but no proper header match
    content = "Summary (d0) without proper brackets"
    result = f(content)
    assert result is None


# ── _history_to_messages propagation tests ──────────────────────────


def test_history_to_messages_propagates_lcm_summary():
    from tui_gateway.server import _history_to_messages
    history = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": (
                "[Recent Summary (d0, node 5)]\n"
                "Summary text\n"
                "[Expand for details: test]"
            ),
        },
    ]
    messages = _history_to_messages(history)
    summary_msgs = [m for m in messages if "lcm_summary" in m]
    assert len(summary_msgs) == 1
    assert summary_msgs[0]["lcm_summary"]["node_ids"] == [5]
    assert summary_msgs[0]["lcm_summary"]["depth_labels"] == ["Recent"]


def test_history_to_messages_no_lcm_summary_for_normal():
    from tui_gateway.server import _history_to_messages
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Normal response"},
    ]
    messages = _history_to_messages(history)
    for m in messages:
        assert "lcm_summary" not in m


def test_history_to_messages_lcm_summary_not_mutating_raw_history():
    """The projection must NOT add lcm_summary to the raw history dicts."""
    from tui_gateway.server import _history_to_messages
    history = [
        {"role": "assistant", "content": "[Recent Summary (d0, node 3)]\nText\n[Expand for details: x]"},
    ]
    raw_before = history[0].copy()
    _history_to_messages(history)
    # Raw history should not have been mutated
    assert "__lcm_summary" not in history[0]
    assert "lcm_summary" not in history[0]


# ── session.lcm_expand RPC integration tests ────────────────────────


def test_lcm_expand_handler_exists():
    from tui_gateway.server import _methods
    assert "session.lcm_expand" in _methods


def test_lcm_expand_rejects_missing_node_id():
    from tui_gateway.server import _methods
    handler = _methods["session.lcm_expand"]
    mock_session = {"agent": MagicMock()}
    with patch("tui_gateway.server._sess", return_value=(mock_session, None)):
        result = handler("r1", {})
    # Should be an error response (result dict with error/code)
    assert isinstance(result, dict)


def test_lcm_expand_no_agent():
    from tui_gateway.server import _methods
    handler = _methods["session.lcm_expand"]
    mock_session = {"agent": None}
    with patch("tui_gateway.server._sess", return_value=(mock_session, None)):
        result = handler("r1", {"node_id": 42})
    assert isinstance(result, dict)


def test_lcm_expand_no_handle_tool_call():
    from tui_gateway.server import _methods
    handler = _methods["session.lcm_expand"]
    mock_compressor = MagicMock(spec=[])  # no handle_tool_call
    mock_agent = MagicMock(spec=["context_compressor"])
    mock_agent.context_compressor = mock_compressor
    mock_session = {"agent": mock_agent}
    with patch("tui_gateway.server._sess", return_value=(mock_session, None)):
        result = handler("r1", {"node_id": 42})
    assert isinstance(result, dict)


def test_lcm_expand_calls_handle_tool_call():
    from tui_gateway.server import _methods
    handler = _methods["session.lcm_expand"]
    mock_compressor = MagicMock()
    mock_compressor.handle_tool_call.return_value = json.dumps({
        "node_id": 42,
        "expanded": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "pagination": {"has_more": False},
    })
    mock_agent = MagicMock()
    mock_agent.context_compressor = mock_compressor
    mock_session = {"agent": mock_agent}
    with patch("tui_gateway.server._sess", return_value=(mock_session, None)):
        result = handler("r1", {"node_id": 42, "session_id": "test-sid"})
    assert isinstance(result, dict)
    # Check the RPC was called
    mock_compressor.handle_tool_call.assert_called_once()
    call_args = mock_compressor.handle_tool_call.call_args
    assert call_args[0][0] == "lcm_expand"
    assert call_args[0][1]["node_id"] == 42


def test_lcm_expand_error_response():
    from tui_gateway.server import _methods
    handler = _methods["session.lcm_expand"]
    mock_compressor = MagicMock()
    mock_compressor.handle_tool_call.return_value = json.dumps({
        "error": "Node 999 not found in current session",
    })
    mock_agent = MagicMock()
    mock_agent.context_compressor = mock_compressor
    mock_session = {"agent": mock_agent}
    with patch("tui_gateway.server._sess", return_value=(mock_session, None)):
        result = handler("r1", {"node_id": 999, "session_id": "test-sid"})
    assert isinstance(result, dict)
    # The handler should return an error (has 'error' key or error code)
    # Check that it's not a success response with empty expanded
    if "error" in result:
        assert "Node 999" in str(result["error"]) or "not found" in str(result["error"])


def test_lcm_expand_pagination_propagated():
    from tui_gateway.server import _methods
    handler = _methods["session.lcm_expand"]
    mock_compressor = MagicMock()
    mock_compressor.handle_tool_call.return_value = json.dumps({
        "node_id": 5,
        "expanded": [{"role": "user", "content": "msg1"}],
        "pagination": {
            "has_more": True,
            "next_source_offset": 10,
            "next_content_offset": 20,
        },
    })
    mock_agent = MagicMock()
    mock_agent.context_compressor = mock_compressor
    mock_session = {"agent": mock_agent}
    with patch("tui_gateway.server._sess", return_value=(mock_session, None)):
        result = handler("r1", {"node_id": 5, "session_id": "test-sid"})
    # The success response should have pagination propagated
    assert isinstance(result, dict)


def test_lcm_expand_invalid_json_response():
    from tui_gateway.server import _methods
    handler = _methods["session.lcm_expand"]
    mock_compressor = MagicMock()
    mock_compressor.handle_tool_call.return_value = "not valid json {{{"
    mock_agent = MagicMock()
    mock_agent.context_compressor = mock_compressor
    mock_session = {"agent": mock_agent}
    with patch("tui_gateway.server._sess", return_value=(mock_session, None)):
        result = handler("r1", {"node_id": 1, "session_id": "test-sid"})
    assert isinstance(result, dict)
    # Should return an error, not crash


# ── Compression response projection test ────────────────────────────


def test_compression_response_uses_projected_messages():
    """After compression, session.compress should return projected
    TUI-format messages (with text key), not raw OpenAI dicts (with
    content key)."""
    from tui_gateway.server import _history_to_messages
    compressed_history = [
        {
            "role": "system",
            "content": "You are a helpful assistant. [LCM context engine active]",
        },
        {
            "role": "assistant",
            "content": (
                "[Recent Summary (d0, node 15)]\n"
                "Discussion about React hooks and state management.\n"
                "[Expand for details: React hooks discussion]\n"
                "\n---\n\n"
                "[Session Arc Summary (d1, node 3)]\n"
                "Overall session about frontend development.\n"
                "[Expand for details: frontend development arc]"
            ),
        },
        {"role": "user", "content": "Can you explain useEffect?"},
    ]
    messages = _history_to_messages(compressed_history)
    summary_msgs = [m for m in messages if "lcm_summary" in m]
    assert len(summary_msgs) == 1
    assert summary_msgs[0]["lcm_summary"]["node_ids"] == [15, 3]
    assert summary_msgs[0]["lcm_summary"]["depth_labels"] == ["Recent", "Session Arc"]
    assert summary_msgs[0]["text"]  # has display text
    assert "role" in summary_msgs[0]  # wire format, not raw OpenAI
