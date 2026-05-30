"""Tests for the terminal-failure-recovery plugin.

Verifies:
- Family classification is correct
- Failure detection catches all expected result shapes
- Pre-tool-call blocking thresholds are respected
- Successful call resets family counter
- Non-terminal tools are never blocked
"""

import json
import sys
from unittest.mock import MagicMock, patch

# Add the plugin dir to path
sys.path.insert(0, "/home/jack/.hermes/plugins")

from terminal_failure_recovery import (
    _classify_family,
    _classify_terminal_failure,
    _get_session,
    _on_post_tool_call,
    _on_pre_tool_call,
    _on_session_start,
    SessionState,
)


# ---------------------------------------------------------------------------
# Family classification
# ---------------------------------------------------------------------------

def test_family_simple():
    assert _classify_family("ls -la /tmp") == "ls"


def test_family_with_sudo():
    assert _classify_family("sudo apt update") == "apt"


def test_family_with_time():
    assert _classify_family("time python script.py") == "python"


def test_family_with_npx():
    assert _classify_family("npx create-react-app my-app") == "create-react-app"


def test_family_with_python_path():
    assert _classify_family("/usr/bin/python3 -m pytest") == "python3"


def test_family_with_dot_slash():
    assert _classify_family("./configure --prefix=/usr") == "configure"


def test_family_empty():
    assert _classify_family("") == "unknown"


def test_family_none():
    assert _classify_family(None) == "unknown"


def test_family_nested_prefix():
    assert _classify_family("sudo npx tsc --build") == "tsc"


def test_family_uv():
    # uv strips as prefix, next word "run" is the family
    assert _classify_family("uv run pytest tests/") == "run"


def test_family_poetry():
    # poetry strips as prefix, next word "add" is the family
    assert _classify_family("poetry add requests") == "add"


def test_family_nix():
    assert _classify_family("nix-shell -p python3") == "nix-shell"


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------

def test_failure_nonzero_exit():
    assert _classify_terminal_failure(json.dumps({"exit_code": 1})) is True
    assert _classify_terminal_failure(json.dumps({"exit_code": 0})) is False


def test_failure_status_error():
    assert _classify_terminal_failure(json.dumps({"status": "error"})) is True
    assert _classify_terminal_failure(json.dumps({"status": "failed"})) is True
    assert _classify_terminal_failure(json.dumps({"status": "timeout"})) is True
    assert _classify_terminal_failure(json.dumps({"status": "interrupted"})) is True
    assert _classify_terminal_failure(json.dumps({"status": "completed"})) is False


def test_failure_guardrail_text():
    assert _classify_terminal_failure(
        json.dumps({"output": "Tool loop hard stop: terminal failed"})
    ) is True


def test_failure_null():
    assert _classify_terminal_failure(None) is False


def test_failure_empty():
    assert _classify_terminal_failure("") is False


def test_failure_not_json():
    assert _classify_terminal_failure("not json") is False


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def test_session_state_initial():
    s = SessionState()
    assert s.total_failures == 0
    assert s.family_failures == {}
    assert s.should_block("apt") is False


def test_session_blocks_after_2_family_failures():
    s = SessionState()
    s.record_failure("apt")
    assert s.should_block("apt") is False
    s.record_failure("apt")
    assert s.should_block("apt") is True


def test_session_blocks_after_3_total_failures():
    s = SessionState()
    s.record_failure("ls")
    s.record_failure("pwd")
    assert s.should_block("anything") is False
    s.record_failure("echo")
    assert s.should_block("anything") is True


def test_success_resets_family():
    s = SessionState()
    s.record_failure("apt")
    s.record_failure("apt")
    assert s.should_block("apt") is True
    s.record_success("apt")
    assert s.should_block("apt") is False
    # Total should still be 2
    assert s.total_failures == 2


def test_reset():
    s = SessionState()
    s.record_failure("apt")
    s.record_failure("ls")
    s.reset()
    assert s.total_failures == 0
    assert s.family_failures == {}


# ---------------------------------------------------------------------------
# Hook integration
# ---------------------------------------------------------------------------

SID = "test-session-1"


def setup_function():
    _on_session_start(session_id=SID)


def teardown_function():
    import terminal_failure_recovery as m
    m._sessions.clear()


def test_pre_tool_call_does_not_block_non_terminal():
    for tool in ("read_file", "write_file", "web_search", "search_files"):
        result = _on_pre_tool_call(
            tool_name=tool,
            args={},
            session_id=SID,
        )
        assert result is None, f"Blocked non-terminal tool: {tool}"


def test_pre_tool_call_blocks_after_3_total_terminal_failures():
    # Simulate 3 terminal failures
    for i in range(3):
        _on_post_tool_call(
            tool_name="terminal",
            args={"command": f"cmd{i}"},
            result=json.dumps({"exit_code": 1}),
            session_id=SID,
        )

    result = _on_pre_tool_call(
        tool_name="terminal",
        args={"command": "any-command"},
        session_id=SID,
    )
    assert result is not None
    assert result["action"] == "block"
    assert "Terminal recovery mode" in result["message"]


def test_pre_tool_call_blocks_after_2_same_family():
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "apt update"},
        result=json.dumps({"exit_code": 1}),
        session_id=SID,
    )
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "apt install foo"},
        result=json.dumps({"exit_code": 1}),
        session_id=SID,
    )

    result = _on_pre_tool_call(
        tool_name="terminal",
        args={"command": "apt upgrade"},
        session_id=SID,
    )
    assert result is not None
    assert result["action"] == "block"


def test_successful_terminal_resets_family():
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "apt update"},
        result=json.dumps({"exit_code": 1}),
        session_id=SID,
    )
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "apt install foo"},
        result=json.dumps({"exit_code": 0}),  # success
        session_id=SID,
    )

    # After success, same family should NOT be blocked
    result = _on_pre_tool_call(
        tool_name="terminal",
        args={"command": "apt upgrade"},
        session_id=SID,
    )
    assert result is None, "Should not block after family success"


def test_block_message_format():
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "apt update"},
        result=json.dumps({"exit_code": 1}),
        session_id=SID,
    )
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "apt upgrade"},
        result=json.dumps({"exit_code": 1}),
        session_id=SID,
    )

    result = _on_pre_tool_call(
        tool_name="terminal",
        args={"command": "apt remove"},
        session_id=SID,
    )
    assert result is not None
    msg = result["message"]
    assert "Terminal recovery mode" in msg
    assert "Do not call terminal again" in msg
    assert "Summarize the failed approach" in msg
    assert "identify the blocker" in msg
    assert "give Jack" in msg


def test_session_reset():
    # Start with failures
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "cmd1"},
        result=json.dumps({"exit_code": 1}),
        session_id=SID,
    )
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "cmd1"},
        result=json.dumps({"exit_code": 1}),
        session_id=SID,
    )

    # Reset session
    _on_session_start(session_id=SID)

    # Should no longer block
    result = _on_pre_tool_call(
        tool_name="terminal",
        args={"command": "cmd1"},
        session_id=SID,
    )
    assert result is None, "Should not block after session reset"


def test_blocked_call_not_counted_as_failure():
    """When pre_tool_call blocks, the synthetic result should not be tracked."""
    # First, record enough to trigger the block
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "a"},
        result=json.dumps({"exit_code": 1}),
        session_id=SID,
    )
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "a"},
        result=json.dumps({"exit_code": 1}),
        session_id=SID,
    )

    # Now simulate the blocked call - post_tool_call sees the recovery message
    _on_post_tool_call(
        tool_name="terminal",
        args={"command": "blocked-cmd"},
        result="Block reason: Terminal recovery mode. Do not call terminal again...",
        session_id=SID,
    )

    # Total should still be 2, not 3
    state = _get_session(SID)
    assert state.total_failures == 2


def test_different_families_accumulate_total():
    for cmd in ("ls", "pwd", "echo"):
        _on_post_tool_call(
            tool_name="terminal",
            args={"command": cmd},
            result=json.dumps({"exit_code": 1}),
            session_id=SID,
        )

    result = _on_pre_tool_call(
        tool_name="terminal",
        args={"command": "anything-new"},
        session_id=SID,
    )
    assert result is not None
    assert result["action"] == "block"
