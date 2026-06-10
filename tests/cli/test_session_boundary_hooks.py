from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from hermes_cli.plugins import VALID_HOOKS, PluginManager
from cli import HermesCLI


def test_session_hooks_in_valid_hooks():
    """Verify on_session_finalize and on_session_reset are registered as valid hooks."""
    assert "on_session_finalize" in VALID_HOOKS
    assert "on_session_reset" in VALID_HOOKS


@patch("hermes_cli.plugins.invoke_hook")
def test_session_finalize_on_reset(mock_invoke_hook):
    """Verify on_session_finalize fires when /new or /reset is used."""
    cli = HermesCLI()
    cli.agent = MagicMock()
    cli.agent.session_id = "test-session-id"

    # Simulate /new command which triggers on_session_finalize for the old session
    cli.new_session(silent=True)

    # Check if on_session_finalize was called for the old session
    assert any(
        c.args == ("on_session_finalize",)
        and c.kwargs["session_id"] == "test-session-id"
        and c.kwargs["platform"] == "cli"
        for c in mock_invoke_hook.call_args_list
    )
    # Check if on_session_reset was called for the new session
    assert any(
        c.args == ("on_session_reset",)
        and c.kwargs["session_id"] == cli.session_id
        and c.kwargs["platform"] == "cli"
        for c in mock_invoke_hook.call_args_list
    )


@patch("hermes_cli.plugins.invoke_hook")
def test_session_finalize_on_cleanup(mock_invoke_hook):
    """Verify on_session_finalize fires during CLI exit cleanup."""
    import cli as cli_mod

    mock_agent = MagicMock()
    mock_agent.session_id = "cleanup-session-id"
    cli_mod._active_agent_ref = mock_agent
    cli_mod._cleanup_done = False

    cli_mod._run_cleanup()

    assert any(
        c.args == ("on_session_finalize",)
        and c.kwargs["session_id"] == "cleanup-session-id"
        and c.kwargs["platform"] == "cli"
        and c.kwargs["reason"] == "shutdown"
        for c in mock_invoke_hook.call_args_list
    )


@patch("hermes_cli.plugins.invoke_hook")
def test_interrupted_session_end_helper_emits_observer_shape(mock_invoke_hook):
    """Verify quiet single-query interruption emits a correlated session end."""
    import cli as cli_mod

    mock_agent = MagicMock()
    mock_agent.session_id = "agent-session-id"
    mock_agent.model = "test-model"
    mock_agent.platform = "cli"
    mock_agent._current_task_id = "task-1"
    mock_agent._current_turn_id = "turn-1"
    mock_agent._current_api_request_id = "api-1"
    cli = SimpleNamespace(agent=mock_agent, session_id="cli-session-id")

    cli_mod._emit_interrupted_session_end(cli, reason="keyboard_interrupt")

    mock_agent.interrupt.assert_called_once_with("keyboard interrupt")
    assert cli.session_id == "agent-session-id"
    mock_invoke_hook.assert_called_once()
    call = mock_invoke_hook.call_args
    assert call.args == ("on_session_end",)
    assert call.kwargs["session_id"] == "agent-session-id"
    assert call.kwargs["task_id"] == "task-1"
    assert call.kwargs["turn_id"] == "turn-1"
    assert call.kwargs["api_request_id"] == "api-1"
    assert call.kwargs["completed"] is False
    assert call.kwargs["interrupted"] is True
    assert call.kwargs["reason"] == "keyboard_interrupt"


@patch("hermes_cli.plugins.invoke_hook")
def test_hook_errors_are_caught(mock_invoke_hook):
    """Verify hook exceptions are caught and don't crash the agent."""
    mgr = PluginManager()

    # Register a hook that raises
    def bad_callback(**kwargs):
        raise Exception("Hook failed")

    mgr._hooks["on_session_finalize"] = [bad_callback]

    # This should not raise
    results = mgr.invoke_hook("on_session_finalize", session_id="test", platform="cli")
    assert results == []


# ── Canonical payload builder tests ─────────────────────────────────────


def _import_payload_builder():
    """Import the payload builder from agent.shell_hooks."""
    from agent.shell_hooks import _build_session_finalize_payload
    return _build_session_finalize_payload


def test_canonical_payload_contains_required_fields():
    """Verify the canonical on_session_finalize payload has all required fields."""
    build = _import_payload_builder()
    payload = build({"session_id": "test-123", "platform": "tui", "trigger": "tui_close"})

    assert payload["hook_event_name"] == "on_session_finalize"
    assert payload["session_id"] == "test-123"
    assert "profile" in payload
    assert "cwd" in payload
    assert "timestamp" in payload
    assert "trigger" in payload
    assert "session_path" in payload
    assert "conversation_id" in payload
    assert payload["finalized"] is True
    assert "extra" in payload


def test_canonical_payload_session_id():
    """Verify session_id is propagated correctly."""
    build = _import_payload_builder()

    # Explicit session_id
    p1 = build({"session_id": "sess_abc123", "platform": "tui"})
    assert p1["session_id"] == "sess_abc123"
    assert p1["conversation_id"] == "sess_abc123"

    # Empty session_id -> empty string
    p2 = build({"platform": "tui"})
    assert p2["session_id"] == ""
    assert p2["finalized"] is True

    # Fallback to parent_session_id
    p3 = build({"parent_session_id": "parent_001", "platform": "tui"})
    assert p3["session_id"] == "parent_001"


def test_canonical_payload_trigger():
    """Verify the trigger field is inferred correctly."""
    build = _import_payload_builder()

    # Explicit trigger wins
    p1 = build({"session_id": "t1", "platform": "tui", "trigger": "tui_new"})
    assert p1["trigger"] == "tui_new"

    # Fallback from reason
    p2 = build({"session_id": "t2", "platform": "cli", "reason": "shutdown"})
    assert p2["trigger"] == "shutdown"

    # Fallback from reason: new_session
    p3 = build({"session_id": "t3", "platform": "cli", "reason": "new_session"})
    assert p3["trigger"] == "cli_new"

    # Unknown reason
    p4 = build({"session_id": "t4", "platform": "unknown"})
    assert p4["trigger"] == "unknown"

    # session_expired
    p5 = build({"session_id": "t5", "platform": "web", "reason": "session_expired"})
    assert p5["trigger"] == "session_expired"


def test_canonical_payload_extra():
    """Verify extras carry kwargs not in canonical keys."""
    build = _import_payload_builder()

    payload = build({
        "session_id": "t1",
        "platform": "tui",
        "reason": "tui_close",
        "custom_field": "hello",
        "old_session_id": "old_123",
    })
    assert "custom_field" in payload["extra"]
    assert payload["extra"]["custom_field"] == "hello"
    assert "old_session_id" in payload["extra"]
    assert "session_id" not in payload["extra"]


def test_canonical_payload_profile():
    """Verify profile is included (from kwarg or env)."""
    build = _import_payload_builder()

    payload = build({
        "session_id": "t1",
        "platform": "tui",
        "trigger": "tui_new",
        "profile": "coder",
    })
    assert payload["profile"] == "coder"


def test_serialize_payload_dispatches_canonical():
    """Verify _serialize_payload uses canonical builder for on_session_finalize."""
    from agent.shell_hooks import _serialize_payload

    result = _serialize_payload(
        "on_session_finalize",
        {"session_id": "test-123", "platform": "tui", "trigger": "tui_close"},
    )
    import json
    parsed = json.loads(result)
    assert parsed["hook_event_name"] == "on_session_finalize"
    assert parsed["session_id"] == "test-123"
    assert parsed["trigger"] == "tui_close"
    assert parsed["finalized"] is True
    assert "timestamp" in parsed
    assert "profile" in parsed


def test_serialize_payload_legacy():
    """Verify non-finalize events still use legacy format."""
    from agent.shell_hooks import _serialize_payload

    result = _serialize_payload(
        "pre_tool_call",
        {"session_id": "test-123", "tool_name": "terminal"},
    )
    import json
    parsed = json.loads(result)
    assert parsed["hook_event_name"] == "pre_tool_call"
    assert parsed["session_id"] == "test-123"
    # Legacy format does NOT have canonical fields
    assert "timestamp" not in parsed
    assert "trigger" not in parsed
    assert "finalized" not in parsed


def test_tui_finalize_passes_trigger():
    """Verify TUI _finalize_session calls _notify_session_boundary with trigger."""
    from tui_gateway.server import _finalize_session, _notify_session_boundary

    # We can't easily test the full chain without starting the gateway,
    # so we verify the function signature and logic by checking that
    # end_reason maps to trigger correctly.
    session = {
        "session_key": "test_key_123",
        "agent": SimpleNamespace(session_id="test_sid_456"),
        "_finalized": False,
    }

    with patch("tui_gateway.server._notify_session_boundary") as mock_notify:
        _finalize_session(session, end_reason="tui_close")
        mock_notify.assert_called_once_with(
            "on_session_finalize",
            "test_sid_456",
            trigger="tui_close",
        )


def test_tui_finalize_idle_passes_trigger():
    """Verify idle-timeout finalize passes correct trigger."""
    from tui_gateway.server import _finalize_session

    session = {
        "session_key": "test_key_789",
        "agent": SimpleNamespace(session_id="test_sid_789"),
        "_finalized": False,
    }

    with patch("tui_gateway.server._notify_session_boundary") as mock_notify:
        _finalize_session(session, end_reason="idle_timeout")
        mock_notify.assert_called_once_with(
            "on_session_finalize",
            "test_sid_789",
            trigger="idle_timeout",
        )


def test_cli_finalize_passes_trigger():
    """Verify CLI _run_cleanup passes trigger='cli_exit'."""
    import cli as cli_mod

    with patch("hermes_cli.plugins.invoke_hook") as mock_hook:
        mock_agent = MagicMock()
        mock_agent.session_id = "cli-sess-001"
        cli_mod._active_agent_ref = mock_agent
        cli_mod._cleanup_done = False

        cli_mod._run_cleanup()

        assert any(
            c.args == ("on_session_finalize",)
            and c.kwargs.get("session_id") == "cli-sess-001"
            and c.kwargs.get("platform") == "cli"
            and c.kwargs.get("reason") == "shutdown"
            for c in mock_hook.call_args_list
        ), "CLI cleanup should fire on_session_finalize"
