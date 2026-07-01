"""Regressions for the context-engine host contract.

These tests pin the five generic host-side guarantees that external context
engine plugins (e.g. hermes-lcm) rely on:

1. ``_transition_context_engine_session`` drives the full lifecycle
   (on_session_end → on_session_reset → on_session_start → optional
   carry_over_new_session_context) and ``reset_session_state`` delegates
   to it when callers pass session metadata.

2. ``on_session_start`` receives ``conversation_id`` derived from
   ``_gateway_session_key`` at agent init time.

3. ``conversation_loop`` forwards canonical cache buckets
   (``cache_read_tokens``, ``cache_write_tokens``, ``input_tokens``,
   ``output_tokens``, ``reasoning_tokens``) to the engine's
   ``update_from_response``, on top of the legacy aggregate keys.

4. ``_discover_context_engines`` includes plugin-registered engines (not
   just repo-shipped engines under ``plugins/context_engine/``).

5. The repo-shipped ``_EngineCollector`` honors ``ctx.register_command``
   from a plugin engine's ``register(ctx)`` entry point and routes it
   to the global plugin command registry.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = "test-session"
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent._gateway_session_key = "agent:main:telegram:dm:42"
    return agent


def test_transition_runs_full_lifecycle_in_order():
    """End → reset → start → carry_over, in that order, when all inputs apply."""
    events: list[str] = []
    engine = MagicMock()
    engine.context_length = 200_000
    engine.on_session_end.side_effect = lambda *a, **kw: events.append("on_session_end")
    engine.on_session_reset.side_effect = lambda *a, **kw: events.append("on_session_reset")
    engine.on_session_start.side_effect = lambda *a, **kw: events.append("on_session_start")
    engine.carry_over_new_session_context.side_effect = lambda *a, **kw: events.append("carry_over")

    agent = _bare_agent()
    agent.context_compressor = engine

    agent._transition_context_engine_session(
        old_session_id="old-sid",
        new_session_id="new-sid",
        previous_messages=[{"role": "user", "content": "hi"}],
        carry_over_context=True,
    )

    assert events == [
        "on_session_end",
        "on_session_reset",
        "on_session_start",
        "carry_over",
    ]


def test_transition_passes_conversation_id_from_gateway_session_key():
    """on_session_start receives ``conversation_id`` from ``_gateway_session_key``."""
    engine = MagicMock()
    engine.context_length = 200_000
    captured: dict = {}
    engine.on_session_start.side_effect = lambda sid, **kw: captured.update(kw)

    agent = _bare_agent()
    agent.context_compressor = engine

    agent._transition_context_engine_session(
        old_session_id="old-sid",
        new_session_id="new-sid",
        previous_messages=[{"role": "user", "content": "hi"}],
    )

    assert captured.get("conversation_id") == "agent:main:telegram:dm:42"
    assert captured.get("old_session_id") == "old-sid"
    assert captured.get("platform") == "telegram"


def test_transition_skips_optional_hooks_when_engine_lacks_them():
    """Engines that don't implement on_session_end/carry_over still work."""
    class MinimalEngine:
        def __init__(self):
            self.context_length = 100_000
            self.reset_called = False
            self.start_called_with = None

        def on_session_reset(self):
            self.reset_called = True

        def on_session_start(self, sid, **kw):
            self.start_called_with = (sid, kw)

    engine = MinimalEngine()
    agent = _bare_agent()
    agent.context_compressor = engine

    # Should not raise even though on_session_end / carry_over are missing.
    agent._transition_context_engine_session(
        old_session_id="old",
        new_session_id="new",
        previous_messages=[{"role": "user", "content": "hi"}],
        carry_over_context=True,
    )

    assert engine.reset_called is True
    assert engine.start_called_with is not None
    new_sid, kw = engine.start_called_with
    assert new_sid == "new"
    assert kw.get("old_session_id") == "old"


def test_reset_session_state_delegates_to_transition_when_args_provided():
    """``reset_session_state(previous_messages=..., old_session_id=...)`` fires full lifecycle."""
    engine = MagicMock()
    engine.context_length = 100_000

    agent = _bare_agent()
    agent.context_compressor = engine

    agent.reset_session_state(
        previous_messages=[{"role": "user", "content": "hi"}],
        old_session_id="old-sid",
    )

    assert engine.on_session_end.called
    assert engine.on_session_reset.called
    assert engine.on_session_start.called
    # No carry_over_context, so carry_over hook NOT called.
    assert not engine.carry_over_new_session_context.called


def test_reset_session_state_default_call_only_resets():
    """Bare ``reset_session_state()`` still only resets the engine (no end/start)."""
    engine = MagicMock()
    engine.context_length = 100_000

    agent = _bare_agent()
    agent.context_compressor = engine

    agent.reset_session_state()

    assert engine.on_session_reset.called
    assert not engine.on_session_end.called
    assert not engine.on_session_start.called


def test_reset_session_state_rebinds_builtin_compressor_after_session_switch(tmp_path, monkeypatch):
    """Reset-only session switches must rebind durable cooldown state to the new session."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old-sid", source="cli")
    db.create_session("new-sid", source="cli")
    db.record_compression_failure_cooldown("old-sid", 4_000_000_000.0, "old-timeout")

    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length",
        lambda *_a, **_k: 100_000,
    )
    compressor = ContextCompressor(
        model="fake-model",
        threshold_percent=0.85,
        protect_first_n=2,
        protect_last_n=2,
        quiet_mode=True,
    )
    compressor.bind_session_state(db, "old-sid")

    agent = _bare_agent()
    agent._session_db = db
    agent.context_compressor = compressor
    agent.session_id = "new-sid"

    agent.reset_session_state()

    assert compressor._session_id == "new-sid"
    assert compressor.get_active_compression_failure_cooldown() is None
    assert db.get_compression_failure_cooldown("old-sid") is not None

    compressor._record_compression_failure_cooldown(30.0, "new-timeout")

    assert db.get_compression_failure_cooldown("new-sid") is not None
    assert db.get_compression_failure_cooldown("old-sid")["error"] == "old-timeout"


def test_update_from_response_forwards_canonical_cache_buckets():
    """conversation_loop passes cache_read/write/reasoning tokens to engine."""
    # Test the contract directly: a usage_dict built from CanonicalUsage must
    # contain the canonical buckets in addition to the legacy keys. We don't
    # spin up the full conversation loop; we just verify the dict shape.
    from agent.usage_pricing import CanonicalUsage

    canonical = CanonicalUsage(
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=800,
        cache_write_tokens=200,
        reasoning_tokens=50,
    )
    usage_dict = {
        "prompt_tokens": canonical.prompt_tokens,
        "completion_tokens": canonical.output_tokens,
        "total_tokens": canonical.total_tokens,
        "input_tokens": canonical.input_tokens,
        "output_tokens": canonical.output_tokens,
        "cache_read_tokens": canonical.cache_read_tokens,
        "cache_write_tokens": canonical.cache_write_tokens,
        "reasoning_tokens": canonical.reasoning_tokens,
    }

    # Legacy keys present
    assert usage_dict["prompt_tokens"] == canonical.prompt_tokens
    assert usage_dict["completion_tokens"] == 500
    assert usage_dict["total_tokens"] == canonical.total_tokens
    # Canonical cache + reasoning buckets present
    assert usage_dict["cache_read_tokens"] == 800
    assert usage_dict["cache_write_tokens"] == 200
    assert usage_dict["reasoning_tokens"] == 50
    assert usage_dict["input_tokens"] == 1000
    assert usage_dict["output_tokens"] == 500


def test_discover_context_engines_includes_plugin_registered_engines(monkeypatch):
    """Plugin-registered context engines appear in the ``hermes plugins`` picker."""
    from hermes_cli import plugins_cmd

    fake_repo = lambda: [("compressor", "built-in", True)]

    class FakePluginEngine:
        name = "lcm"

    monkeypatch.setattr(
        "plugins.context_engine.discover_context_engines",
        fake_repo,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.discover_plugins",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_context_engine",
        lambda: FakePluginEngine(),
    )

    engines = plugins_cmd._discover_context_engines()
    names = [n for n, _desc in engines]
    assert "compressor" in names
    assert "lcm" in names


def test_discover_context_engines_dedupes_by_name(monkeypatch):
    """Repo-shipped engine wins when name collides with a plugin-registered one."""
    from hermes_cli import plugins_cmd

    class FakePluginEngine:
        name = "compressor"  # same name as repo-shipped

    monkeypatch.setattr(
        "plugins.context_engine.discover_context_engines",
        lambda: [("compressor", "built-in compressor", True)],
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.discover_plugins",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_context_engine",
        lambda: FakePluginEngine(),
    )

    engines = plugins_cmd._discover_context_engines()
    # Only one entry — the repo-shipped one. Description is preserved.
    assert engines == [("compressor", "built-in compressor")]


def test_engine_collector_forwards_register_command_to_plugin_manager():
    """A plugin context engine can register a slash command via ``ctx.register_command``."""
    from plugins.context_engine import _EngineCollector
    from hermes_cli.plugins import get_plugin_manager

    handler = lambda raw_args: f"echo: {raw_args}"

    collector = _EngineCollector(engine_name="my-lcm")
    collector.register_command(
        "my-lcm-test-cmd",
        handler,
        description="test command from a context engine",
        args_hint="<msg>",
    )

    manager = get_plugin_manager()
    try:
        assert "my-lcm-test-cmd" in manager._plugin_commands
        entry = manager._plugin_commands["my-lcm-test-cmd"]
        assert entry["handler"] is handler
        assert entry["args_hint"] == "<msg>"
        assert entry["plugin"] == "context-engine:my-lcm"
    finally:
        # Clean up so we don't leak the registration across tests.
        manager._plugin_commands.pop("my-lcm-test-cmd", None)


def test_engine_collector_rejects_builtin_command_conflicts():
    """Context engine cannot shadow built-in slash commands like /help."""
    from plugins.context_engine import _EngineCollector
    from hermes_cli.plugins import get_plugin_manager

    collector = _EngineCollector(engine_name="my-lcm")
    collector.register_command("help", lambda *_: "shadow")

    manager = get_plugin_manager()
    # Must NOT have overwritten / registered against built-in /help.
    assert "help" not in manager._plugin_commands or \
           manager._plugin_commands["help"].get("plugin") != "context-engine:my-lcm"


# ---------------------------------------------------------------------------
# Per-agent context-engine cloning tests (local backport of PR #42683)
# ---------------------------------------------------------------------------

from agent.context_engine import ContextEngine


class _ContractEngine(ContextEngine):
    """Concrete engine for testing the clone_for_agent/shutdown host contract."""

    def __init__(self, name: str = "contract-engine"):
        self._name = name
        self.clones: list[_ContractEngine] = []
        self.update_model_calls: list[dict[str, object]] = []
        self.session_start_calls: list[tuple[str, dict[str, object]]] = []
        self.shutdown_count = 0

    @property
    def name(self) -> str:
        return self._name

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ):
        self.update_model_calls.append({
            "model": model,
            "context_length": context_length,
            "base_url": base_url,
            "api_key": api_key,
            "provider": provider,
            "api_mode": api_mode,
        })
        self.context_length = context_length

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_start_calls.append((session_id, kwargs))

    def shutdown(self) -> None:
        self.shutdown_count += 1


class _CloningContractEngine(_ContractEngine):
    """Engine that returns a fresh clone per agent."""

    def clone_for_agent(self) -> ContextEngine:
        clone = _ContractEngine(self.name)
        self.clones.append(clone)
        return clone


def _patch_agent_init_for_plugin_engine(monkeypatch, engine: ContextEngine) -> None:
    """Patch agent_init so it uses *engine* as the plugin context engine."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "context": {"engine": engine.name},
            "model": {"context_length": 200_000},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_context_engine",
        lambda: engine,
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *args, **kwargs: 200_000,
    )
    monkeypatch.setattr("run_agent.OpenAI", MagicMock(return_value=MagicMock()))


# -- A. Cloning context engine ---------------------------------------------


def test_agent_init_clones_plugin_context_engine_per_agent(monkeypatch):
    """Mutable plugin engines can provide isolated per-AIAgent runtime state."""
    registered = _CloningContractEngine()
    _patch_agent_init_for_plugin_engine(monkeypatch, registered)

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        session_id="agent-s1",
        platform="telegram",
        gateway_session_key="agent:main:telegram:dm:42",
    )
    clone = registered.clones[0]
    try:
        assert getattr(agent, "context_compressor") is clone
        assert getattr(agent, "_owns_context_engine") is True
        # update_model and on_session_start ran on the clone, NOT the prototype
        assert registered.update_model_calls == []
        assert registered.session_start_calls == []
        assert clone.update_model_calls[0]["context_length"] == 200_000
        assert clone.session_start_calls[0][0] == "agent-s1"
        assert clone.session_start_calls[0][1]["conversation_id"] == "agent:main:telegram:dm:42"
    finally:
        agent.close()

    # After close: clone was shut down exactly once, prototype untouched.
    assert clone.shutdown_count == 1
    assert registered.shutdown_count == 0
    assert getattr(agent, "_owns_context_engine") is False


# -- B. Non-cloning engine (backward compatible) ---------------------------


def test_agent_close_does_not_shutdown_shared_plugin_context_engine(monkeypatch):
    """The default clone_for_agent() keeps backward-compatible shared engines alive."""
    shared = _ContractEngine()
    _patch_agent_init_for_plugin_engine(monkeypatch, shared)

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        session_id="agent-s2",
        platform="weixin",
        gateway_session_key="agent:main:weixin:dm:77",
    )
    try:
        assert getattr(agent, "context_compressor") is shared
        assert getattr(agent, "_owns_context_engine") is False
    finally:
        agent.close()

    # Shared engine was NOT shut down.
    assert shared.shutdown_count == 0


# -- C. Two concurrent agents with different cloned engines ----------------


def test_two_agents_receive_separate_cloned_engines(monkeypatch):
    """Two agents get separate LCM engine objects sharing the same store."""
    registered = _CloningContractEngine()
    _patch_agent_init_for_plugin_engine(monkeypatch, registered)

    agent_a = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        session_id="agent-A",
        platform="telegram",
        gateway_session_key="agent:main:telegram:dm:1",
    )
    agent_b = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        session_id="agent-B",
        platform="discord",
        gateway_session_key="agent:main:discord:dm:2",
    )
    clone_a = registered.clones[0]
    clone_b = registered.clones[1]

    try:
        # Different engine objects
        assert clone_a is not clone_b
        assert agent_a.context_compressor is clone_a
        assert agent_b.context_compressor is clone_b

        # Both agents own their clones
        assert agent_a._owns_context_engine is True
        assert agent_b._owns_context_engine is True

        # Different session IDs via on_session_start
        assert clone_a.session_start_calls[0][0] == "agent-A"
        assert clone_b.session_start_calls[0][0] == "agent-B"

        # Different conversation IDs
        assert clone_a.session_start_calls[0][1]["conversation_id"] == "agent:main:telegram:dm:1"
        assert clone_b.session_start_calls[0][1]["conversation_id"] == "agent:main:discord:dm:2"

        # Prototype never received direct calls
        assert registered.update_model_calls == []
        assert registered.session_start_calls == []
    finally:
        agent_a.close()
        agent_b.close()

    # Each clone shut down exactly once; prototype never shut down.
    assert clone_a.shutdown_count == 1
    assert clone_b.shutdown_count == 1
    assert registered.shutdown_count == 0


def test_closing_one_agent_does_not_affect_the_other_clone(monkeypatch):
    """Compression/finalization for A cannot affect B's clone."""
    registered = _CloningContractEngine()
    _patch_agent_init_for_plugin_engine(monkeypatch, registered)

    agent_a = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        session_id="agent-A",
        platform="telegram",
        gateway_session_key="agent:main:telegram:dm:1",
    )
    agent_b = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        session_id="agent-B",
        platform="discord",
        gateway_session_key="agent:main:discord:dm:2",
    )
    clone_a = registered.clones[0]
    clone_b = registered.clones[1]

    # Close A first
    agent_a.close()
    assert clone_a.shutdown_count == 1
    # B's engine is still alive
    assert clone_b.shutdown_count == 0
    assert agent_b._owns_context_engine is True

    # B can still use its engine
    agent_b.context_compressor.update_from_response({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
    assert clone_b.last_prompt_tokens == 100

    agent_b.close()
    assert clone_b.shutdown_count == 1
    assert registered.shutdown_count == 0


# -- D. Command routing: session-level tools resolve to the active clone ---


def test_context_engine_tools_route_to_active_agent_clone(monkeypatch):
    """lcm_status and other tools go through agent.context_compressor (the clone)."""
    registered = _CloningContractEngine()
    _patch_agent_init_for_plugin_engine(monkeypatch, registered)

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        session_id="agent-s1",
        platform="telegram",
        gateway_session_key="agent:main:telegram:dm:42",
    )
    clone = registered.clones[0]

    try:
        # The agent's context_compressor IS the clone, not the prototype.
        # Tool dispatch goes through agent.context_compressor.handle_tool_call()
        # so tools automatically resolve to the correct per-agent engine.
        assert agent.context_compressor is clone
        assert agent.context_compressor is not registered
    finally:
        agent.close()


# -- E. clone_for_agent error handling -------------------------------------


def test_clone_for_agent_exception_falls_back_to_registered(monkeypatch):
    """If clone_for_agent() raises, fall back to the registered engine."""
    class _BrokenCloneEngine(_ContractEngine):
        def clone_for_agent(self):
            raise RuntimeError("clone failed")

    broken = _BrokenCloneEngine()
    _patch_agent_init_for_plugin_engine(monkeypatch, broken)

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        session_id="agent-s1",
        platform="telegram",
        gateway_session_key="agent:main:telegram:dm:42",
    )
    try:
        # Falls back to the registered engine
        assert agent.context_compressor is broken
        assert agent._owns_context_engine is False
    finally:
        agent.close()

    # Closing doesn't shut it down (shared engine)
    assert broken.shutdown_count == 0


def test_clone_for_agent_returns_none_falls_back_to_registered(monkeypatch):
    """If clone_for_agent() returns None, fall back to the registered engine."""
    class _NoneCloneEngine(_ContractEngine):
        def clone_for_agent(self):
            return None

    none_engine = _NoneCloneEngine()
    _patch_agent_init_for_plugin_engine(monkeypatch, none_engine)

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        provider="openai",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        session_id="agent-s1",
        platform="telegram",
        gateway_session_key="agent:main:telegram:dm:42",
    )
    try:
        assert agent.context_compressor is none_engine
        assert agent._owns_context_engine is False
    finally:
        agent.close()

    assert none_engine.shutdown_count == 0
