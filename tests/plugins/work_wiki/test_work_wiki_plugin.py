from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path


def _load_with_temp_paths(monkeypatch, tmp_path, *, promote=False, strict=False, mode=""):
    home = tmp_path / "home"
    wiki = tmp_path / "wiki"
    db = tmp_path / "work.sqlite3"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_WORK_WIKI_ROOT", str(wiki))
    monkeypatch.setenv("HERMES_WORK_WIKI_DB", str(db))
    if promote:
        monkeypatch.setenv("HERMES_WORK_WIKI_AUTO_PROMOTE_KNOWLEDGE", "1")
    if strict:
        monkeypatch.setenv("HERMES_WORK_WIKI_STRICT_PERSISTENCE", "1")
    if mode:
        monkeypatch.setenv("HERMES_WORK_WIKI_MODE", mode)
    import hermes_cli.plugins as plugins_mod

    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins(force=True)
    return plugins_mod, wiki, db


def test_work_wiki_bundled_plugin_auto_loads(monkeypatch, tmp_path):
    plugins_mod, _wiki, _db = _load_with_temp_paths(monkeypatch, tmp_path)

    manager = plugins_mod.get_plugin_manager()
    loaded = manager._plugins.get("work-wiki")

    assert loaded is not None
    assert loaded.enabled is True
    assert "wiki" in manager._plugin_commands
    assert manager.has_hook("pre_llm_call")
    assert manager.has_hook("transform_llm_output")
    assert manager.has_hook("post_llm_call")
    assert manager.has_hook("subagent_start")
    assert manager.has_hook("subagent_stop")
    assert manager.has_hook("on_session_reset")


def test_work_wiki_disabled_mode_skips_automatic_capture(monkeypatch, tmp_path):
    plugins_mod, wiki, db = _load_with_temp_paths(monkeypatch, tmp_path, mode="disabled")

    result = plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="disabled-s",
        turn_id="t1",
        user_message="Implement disabled mode behavior for Work Wiki.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    assert result == []
    assert not (wiki / "mission-control.md").exists()
    conn = sqlite3.connect(db)
    try:
        exists = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='events'"
        ).fetchone()[0]
        if exists:
            assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM work_items").fetchone()[0] == 0
    finally:
        conn.close()


def test_work_wiki_observe_only_records_classification_without_pages(monkeypatch, tmp_path):
    plugins_mod, wiki, db = _load_with_temp_paths(monkeypatch, tmp_path, mode="observe-only")

    result = plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="observe-s",
        turn_id="t1",
        user_message="Implement observe-only mode behavior for Work Wiki.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    assert result == []
    assert not (wiki / "mission-control.md").exists()
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM work_items").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM events WHERE event_type='work_classified'").fetchone()[0] == 1
    finally:
        conn.close()


def test_work_wiki_manual_checkpoint_mode_skips_automatic_checkpoints(monkeypatch, tmp_path):
    plugins_mod, wiki, db = _load_with_temp_paths(monkeypatch, tmp_path, mode="manual-checkpoint")

    result = plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="manual-mode-s",
        turn_id="t1",
        user_message="Implement manual checkpoint mode behavior for Work Wiki.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )
    plugins_mod.invoke_hook(
        "post_llm_call",
        session_id="manual-mode-s",
        turn_id="t1",
        assistant_response=(
            "Implemented manual checkpoint mode behavior.\n\n"
            "Verification:\n- Mode test executed.\n\n"
            "Next Actions:\n- Review checkpoint count."
        ),
        conversation_history=[],
        model="test-model",
        platform="cli",
    )

    assert result and "ACTIVE MISSION" in result[0]["context"]
    assert (wiki / "mission-control.md").exists()
    assert list((wiki / "work" / "missions").glob("*.md"))
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM work_items WHERE work_kind='mission'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM events WHERE event_type='work_classified'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events WHERE event_type='assistant_progress'").fetchone()[0] == 0
    finally:
        conn.close()


def test_work_wiki_creates_mission_records_events_and_renders(monkeypatch, tmp_path):
    plugins_mod, wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)

    context = plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s1",
        turn_id="t1",
        user_message="Implement the Hermes automatic mission wiki feature with SQLite and Markdown dashboards.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )
    plugins_mod.invoke_hook(
        "pre_tool_call",
        session_id="s1",
        turn_id="t1",
        tool_name="write_file",
        args={"path": "/tmp/demo.py", "content": "x"},
        tool_call_id="tc1",
    )
    plugins_mod.invoke_hook(
        "post_tool_call",
        session_id="s1",
        turn_id="t1",
        tool_name="write_file",
        args={"path": "/tmp/demo.py"},
        result='{"success": true}',
        tool_call_id="tc1",
    )
    plugins_mod.invoke_hook(
        "post_llm_call",
        session_id="s1",
        turn_id="t1",
        user_message="Implement...",
        assistant_response=(
            "Implemented the SQLite event ledger and Markdown renderer.\n\n"
            "Verification:\n- Python compile passed.\n\n"
            "Next Actions:\n- Run focused tests."
        ),
        conversation_history=[],
        model="test-model",
        platform="cli",
    )

    assert context and "ACTIVE MISSION" in context[0]["context"]
    assert (wiki / "mission-control.md").exists()
    assert (wiki / "recovery.md").exists()
    mission_pages = list((wiki / "work" / "missions").glob("*.md"))
    assert len(mission_pages) == 1
    mission_text = mission_pages[0].read_text(encoding="utf-8")
    assert "work_kind: mission" in mission_text
    assert "Evidence and Verification" in mission_text
    assert "Python compile passed" in mission_text
    assert "<!-- work-wiki:checkpoint:" in mission_text

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM work_items WHERE work_kind='mission'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] >= 5
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] >= 2
        assert conn.execute("SELECT COUNT(*) FROM render_jobs WHERE state='completed'").fetchone()[0] >= 2
    finally:
        conn.close()


def test_wiki_command_status_and_repair(monkeypatch, tmp_path):
    plugins_mod, wiki, _db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s2",
        turn_id="t1",
        user_message="Build a durable local mission dashboard.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    handler = plugins_mod.get_plugin_command_handler("wiki")
    status = handler("status")
    repair = handler("repair")

    assert "Recent missions:" in status
    assert "durable local mission dashboard" in status.lower()
    assert "Rebuilt mission pages" in repair
    assert (wiki / "mission-control.md").exists()
    assert (wiki / "work" / "_indexes" / "active-missions.md").exists()
    assert (wiki / "work" / "_indexes" / "projects.md").exists()
    assert (wiki / "work" / "_indexes" / "workstreams.md").exists()


def test_wiki_detach_clears_session_focus(monkeypatch, tmp_path):
    plugins_mod, _wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="detach-session",
        turn_id="t1",
        user_message="Implement detach command persistence.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    handler = plugins_mod.get_plugin_command_handler("wiki")
    result = handler("detach detach-session")

    assert "Detached session detach-session" in result
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT focus, relationship, deactivated_at FROM session_links WHERE session_id='detach-session'"
        ).fetchone()
        assert row[0] == 0
        assert row[1] == "detached"
        assert row[2]
        assert conn.execute("SELECT COUNT(*) FROM events WHERE event_type='work_focus_detached'").fetchone()[0] == 1
    finally:
        conn.close()


def test_wiki_exclude_session_detaches_events_without_deleting(monkeypatch, tmp_path):
    plugins_mod, _wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="exclude-session",
        turn_id="t1",
        user_message="Implement exclude session command.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )
    plugins_mod.invoke_hook(
        "post_tool_call",
        session_id="exclude-session",
        turn_id="t1",
        tool_name="write_file",
        args={"path": "/tmp/exclude.txt"},
        result='{"success": true}',
        tool_call_id="tc-exclude",
    )

    conn = sqlite3.connect(db)
    try:
        work_id = conn.execute("SELECT work_id FROM work_items WHERE work_kind='mission'").fetchone()[0]
        before_events = conn.execute("SELECT COUNT(*) FROM events WHERE work_id=?", (work_id,)).fetchone()[0]
        assert before_events > 0
    finally:
        conn.close()

    handler = plugins_mod.get_plugin_command_handler("wiki")
    result = handler(f"exclude-session {work_id} exclude-session")

    assert "Excluded session exclude-session" in result
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE session_id='exclude-session' AND work_id IS NULL").fetchone()[0] >= before_events
        assert conn.execute("SELECT COUNT(*) FROM events WHERE event_type='session_excluded' AND work_id=?", (work_id,)).fetchone()[0] == 1
        link = conn.execute(
            "SELECT relationship, focus, deactivated_at FROM session_links WHERE session_id='exclude-session' AND work_id=?",
            (work_id,),
        ).fetchone()
        assert link[0] == "excluded"
        assert link[1] == 0
        assert link[2]
    finally:
        conn.close()


def test_work_wiki_off_disables_automatic_capture(monkeypatch, tmp_path):
    plugins_mod, _wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    handler = plugins_mod.get_plugin_command_handler("wiki")
    assert "disabled" in handler("off")

    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s3",
        turn_id="t1",
        user_message="Implement a new durable feature.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    conn = sqlite3.connect(db)
    try:
        exists = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='work_items'"
        ).fetchone()[0]
        if exists:
            row = conn.execute("SELECT COUNT(*) FROM work_items").fetchone()
            assert row[0] == 0
    finally:
        conn.close()


def test_casual_turn_does_not_create_unassigned_recovery_noise(monkeypatch, tmp_path):
    plugins_mod, _wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)

    result = plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s4",
        turn_id="t1",
        user_message="thanks",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    assert result == []
    conn = sqlite3.connect(db)
    try:
        exists = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='events'"
        ).fetchone()[0]
        if exists:
            assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM work_items").fetchone()[0] == 0
    finally:
        conn.close()


def test_work_wiki_tracks_delegate_lifecycle(monkeypatch, tmp_path):
    plugins_mod, wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="parent-s",
        turn_id="turn-1",
        user_message="Implement the delegate-aware mission recovery dashboard.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    plugins_mod.invoke_hook(
        "subagent_start",
        parent_session_id="parent-s",
        parent_turn_id="turn-1",
        child_session_id="child-s",
        child_role="analyst",
        child_goal="Inspect recovery dashboard gaps.",
    )

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM delegations").fetchone()
        assert row is not None
        assert row["state"] == "running"
        assert row["child_session_id"] == "child-s"
        assert conn.execute(
            "SELECT COUNT(*) FROM session_links WHERE session_id='child-s' AND relationship='delegate'"
        ).fetchone()[0] == 1
    finally:
        conn.close()

    mission_control = (wiki / "mission-control.md").read_text(encoding="utf-8")
    assert "Running Delegates" in mission_control
    assert "delegate-aware mission recovery dashboard" in mission_control.lower()

    plugins_mod.invoke_hook(
        "subagent_stop",
        parent_session_id="parent-s",
        parent_turn_id="turn-1",
        child_session_id="child-s",
        child_role="analyst",
        child_status="completed",
        child_summary="Found the recovery gaps.",
        duration_ms=1200,
    )

    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT state, result_summary FROM delegations").fetchone()
        assert row == ("completed", "Found the recovery gaps.")
        assert conn.execute("SELECT COUNT(*) FROM events WHERE event_type='delegate_completed'").fetchone()[0] == 1
    finally:
        conn.close()

    mission_pages = list((wiki / "work" / "missions").glob("*.md"))
    assert "Found the recovery gaps" in mission_pages[0].read_text(encoding="utf-8")


def test_project_pages_aggregate_mission_state_and_dashboard_delegates(monkeypatch, tmp_path):
    plugins_mod, wiki, _db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="project-s",
        turn_id="turn-1",
        user_message="Implement project-level rendering for the automatic work wiki.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    plugin = plugins_mod.get_plugin_manager()._plugins["work-wiki"].module
    mission = plugin._store.recent_missions(limit=1)[0]
    project = plugin._store.get_work(mission.parent_work_id)
    assert project is not None

    plugin._store.update_work_metadata(
        project.work_id,
        {
            "objective": "Keep project mission state visible from one page.",
            "current_state": "Project renderer aggregation is under test.",
            "next_actions": ["Review project dashboard output."],
        },
    )
    plugin._store.update_work_metadata(
        mission.work_id,
        {
            "current_state": "Rendering project aggregates.",
            "next_actions": ["Verify project dashboard sections."],
            "related_knowledge": ["concepts/hermes-mission-memory.md"],
        },
    )
    plugin._store.add_decision(
        mission.work_id,
        "Aggregate project decisions from child missions.",
        rationale="Project pages should not depend on duplicated metadata.",
    )
    plugin._store.add_artifact(
        mission.work_id,
        "/tmp/project-renderer.md",
        description="Renderer aggregation fixture.",
        verified=True,
    )
    plugin._store.start_delegation(
        parent_session_id="project-s",
        child_session_id="project-child",
        work_id=mission.work_id,
        role="reviewer",
        goal="Review project dashboard output.",
    )

    plugin._renderer.render_project(plugin._store.get_work(project.work_id))
    plugin._renderer.render_dashboards()

    project_page = wiki / project.wiki_path
    project_text = project_page.read_text(encoding="utf-8")
    assert "## Project Objective" in project_text
    assert "Keep project mission state visible from one page." in project_text
    assert "## Current Overall State" in project_text
    assert "Project renderer aggregation is under test." in project_text
    assert "## Important Decisions" in project_text
    assert "Aggregate project decisions from child missions." in project_text
    assert "## Project Artifacts" in project_text
    assert "/tmp/project-renderer.md" in project_text
    assert "## Relevant Entities and Concepts" in project_text
    assert "concepts/hermes-mission-memory.md" in project_text
    assert "## Stale Missions" in project_text
    assert "## Unresolved Review Items" in project_text

    mission_control = (wiki / "mission-control.md").read_text(encoding="utf-8")
    assert "| Project | Mission | Status | Current State | Last Checkpoint | Latest Session | Running Delegates | Next Action | Blockers | Last Activity |" in mission_control
    assert "reviewer `project-child`: Review project dashboard output." in mission_control


def test_project_render_preserves_manual_text_and_surfaces_conflicts(monkeypatch, tmp_path):
    plugins_mod, wiki, _db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugin = plugins_mod.get_plugin_manager()._plugins["work-wiki"].module
    project = plugin._store.ensure_project(
        title="Project Manual Preservation",
        project_root="/tmp/project-manual",
        metadata={"current_state": "Initial project render state."},
    )

    plugin._renderer.render_project(project)
    project_path = wiki / project.wiki_path
    project_path.write_text(project_path.read_text(encoding="utf-8") + "\nManual project note.\n", encoding="utf-8")

    plugin._store.update_work_metadata(project.work_id, {"current_state": "Manual text should survive render."})
    plugin._renderer.render_project(plugin._store.get_work(project.work_id))

    text = project_path.read_text(encoding="utf-8")
    assert "Manual project note." in text
    assert "Manual text should survive render." in text

    broken = text.replace("<!-- work-wiki:generated-current-overall-state:end -->", "", 1)
    project_path.write_text(broken, encoding="utf-8")

    try:
        plugin._renderer.render_project(plugin._store.get_work(project.work_id))
    except RuntimeError as exc:
        assert "Malformed managed block" in str(exc)
    else:
        raise AssertionError("expected malformed project managed block to raise")

    plugin._renderer.render_dashboards()
    recovery = (wiki / "recovery.md").read_text(encoding="utf-8")
    assert "Malformed Pages" in recovery
    assert "work/projects" in recovery
    assert list((wiki / "work" / "projects").glob("*.conflict-*"))


def test_wiki_close_requires_evidence_and_split_preserves_lineage(monkeypatch, tmp_path):
    plugins_mod, _wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s5",
        turn_id="t1",
        user_message="Build the evidence gate for mission completion.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    conn = sqlite3.connect(db)
    try:
        work_id = conn.execute("SELECT work_id FROM work_items WHERE work_kind='mission'").fetchone()[0]
    finally:
        conn.close()

    handler = plugins_mod.get_plugin_command_handler("wiki")
    close_result = handler(f"close {work_id}")
    split_result = handler(f"split {work_id} Follow-up verification mission")

    assert "needs review before completion" in close_result
    assert "Created split mission" in split_result

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT status FROM work_items WHERE work_id=?", (work_id,)).fetchone()[0] == "needs_review"
        assert conn.execute("SELECT COUNT(*) FROM work_items WHERE work_kind='mission'").fetchone()[0] == 2
        child_meta = conn.execute(
            "SELECT metadata FROM work_items WHERE title='Follow-up verification mission'"
        ).fetchone()[0]
        assert work_id in child_meta
    finally:
        conn.close()


def test_work_wiki_promotes_high_confidence_knowledge_when_enabled(monkeypatch, tmp_path):
    plugins_mod, wiki, db = _load_with_temp_paths(monkeypatch, tmp_path, promote=True)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s6",
        turn_id="t1",
        user_message="Implement durable Hermes mission memory knowledge promotion.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )
    plugins_mod.invoke_hook(
        "post_llm_call",
        session_id="s6",
        turn_id="t1",
        assistant_response=(
            "Implemented the promotion pass.\n\n"
            "Findings:\n"
            "- SQLite operational ledger requires checkpoint coverage before completion.\n\n"
            "Verification:\n"
            "- Promotion test passed.\n"
        ),
        conversation_history=[],
        model="test-model",
        platform="cli",
    )

    promoted = wiki / "concepts" / "sqlite-operational-ledger.md"
    assert promoted.exists()
    text = promoted.read_text(encoding="utf-8")
    assert "SQLite operational ledger requires checkpoint coverage before completion" in text
    assert "checkpoint `" in text

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE event_type='knowledge_promoted'").fetchone()[0] == 1
        mission_meta = conn.execute("SELECT metadata FROM work_items WHERE work_kind='mission'").fetchone()[0]
        assert "concepts/sqlite-operational-ledger.md" in mission_meta
    finally:
        conn.close()


def test_wiki_close_refuses_running_delegate(monkeypatch, tmp_path):
    plugins_mod, _wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s7",
        turn_id="t1",
        user_message="Implement completion checks for running delegates.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )
    plugins_mod.invoke_hook(
        "post_llm_call",
        session_id="s7",
        turn_id="t1",
        assistant_response=(
            "Implemented completion checks for running delegates.\n\n"
            "Verification:\n- Unit scenario passed."
        ),
        conversation_history=[],
        model="test-model",
        platform="cli",
    )
    plugins_mod.invoke_hook(
        "subagent_start",
        parent_session_id="s7",
        parent_turn_id="t1",
        child_session_id="child-running",
        child_role="reviewer",
        child_goal="Review completion gate.",
    )

    conn = sqlite3.connect(db)
    try:
        work_id = conn.execute("SELECT work_id FROM work_items WHERE work_kind='mission'").fetchone()[0]
    finally:
        conn.close()

    handler = plugins_mod.get_plugin_command_handler("wiki")
    result = handler(f"close {work_id}")

    assert "delegate work is still running" in result
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT status FROM work_items WHERE work_id=?", (work_id,)).fetchone()[0] == "needs_review"
    finally:
        conn.close()


def test_strict_transform_persists_checkpoint_before_post_hook(monkeypatch, tmp_path):
    plugins_mod, wiki, db = _load_with_temp_paths(monkeypatch, tmp_path, strict=True)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s8",
        turn_id="t1",
        user_message="Implement strict persistence before final response delivery.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
        parent_session_id="root-session",
        chat_id="chat-1",
    )
    result = plugins_mod.invoke_hook(
        "transform_llm_output",
        response_text=(
            "Implemented strict persistence.\n\n"
            "Verification:\n- Transform hook persisted the checkpoint."
        ),
        session_id="s8",
        turn_id="t1",
        user_message="Implement strict persistence before final response delivery.",
        model="test-model",
        platform="cli",
        parent_session_id="root-session",
        chat_id="chat-1",
    )
    plugins_mod.invoke_hook(
        "post_llm_call",
        session_id="s8",
        turn_id="t1",
        assistant_response=(
            "Implemented strict persistence.\n\n"
            "Verification:\n- Transform hook persisted the checkpoint."
        ),
        conversation_history=[],
        model="test-model",
        platform="cli",
        parent_session_id="root-session",
        chat_id="chat-1",
    )

    assert result == []
    assert (wiki / "mission-control.md").exists()
    conn = sqlite3.connect(db)
    try:
        sources = [
            row[0]
            for row in conn.execute("SELECT source FROM events WHERE event_type='assistant_claimed_completion'")
        ]
        assert sources == ["hook:transform_llm_output"]
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] >= 2
        link = conn.execute("SELECT parent_session_id, chat_id FROM session_links WHERE session_id='s8'").fetchone()
        assert link == ("root-session", "chat-1")
    finally:
        conn.close()


def test_session_reset_deactivates_focus_and_stops_tool_attachment(monkeypatch, tmp_path):
    plugins_mod, _wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s9",
        turn_id="t1",
        user_message="Implement reset-aware mission focus handling.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    plugins_mod.invoke_hook("on_session_reset", session_id="s9", turn_id="t-reset", reason="manual")
    plugins_mod.invoke_hook(
        "post_tool_call",
        session_id="s9",
        turn_id="t2",
        tool_name="write_file",
        args={"path": "/tmp/after-reset.py"},
        result='{"success": true}',
        tool_call_id="tc-reset",
    )

    conn = sqlite3.connect(db)
    try:
        deactivated = conn.execute(
            "SELECT COUNT(*) FROM session_links WHERE session_id='s9' AND deactivated_at IS NOT NULL"
        ).fetchone()[0]
        assert deactivated >= 1
        unattached = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='file_modified' AND work_id IS NULL"
        ).fetchone()[0]
        assert unattached == 1
    finally:
        conn.close()


def test_strict_transform_failure_creates_fallback_checkpoint(monkeypatch, tmp_path):
    plugins_mod, wiki, db = _load_with_temp_paths(monkeypatch, tmp_path, strict=True)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s10",
        turn_id="t1",
        user_message="Implement fallback checkpoint persistence handling.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    plugin = plugins_mod.get_plugin_manager()._plugins["work-wiki"].module
    def _fail_render(*_args, **_kwargs):
        raise RuntimeError("forced render failure")

    monkeypatch.setattr(plugin._renderer, "process_pending", _fail_render)

    result = plugins_mod.invoke_hook(
        "transform_llm_output",
        response_text=(
            "Implemented fallback checkpoint handling.\n\n"
            "Verification:\n- Forced render failure path exercised."
        ),
        session_id="s10",
        turn_id="t1",
        user_message="Implement fallback checkpoint persistence handling.",
        model="test-model",
        platform="cli",
    )

    assert result and "Work Wiki persistence warning" in result[0]
    conn = sqlite3.connect(db)
    try:
        fallback = conn.execute(
            "SELECT checkpoint_kind, semantic, needs_review FROM checkpoints WHERE checkpoint_kind='fallback'"
        ).fetchone()
        assert fallback == ("fallback", 0, 1)
        assert conn.execute("SELECT status FROM work_items WHERE work_kind='mission'").fetchone()[0] == "needs_review"
        assert conn.execute("SELECT COUNT(*) FROM events WHERE event_type='persistence_failed'").fetchone()[0] == 1
    finally:
        conn.close()

    monkeypatch.undo()
    plugin._renderer.render_all()
    recovery = (wiki / "recovery.md").read_text(encoding="utf-8")
    assert "Fallback Checkpoints Needing Review" in recovery
    assert "Persistence Failures" in recovery
    assert "fallback checkpoint" in recovery


def test_path_deny_list_redacts_payloads_and_skips_artifacts(monkeypatch, tmp_path):
    plugins_mod, _wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="s11",
        turn_id="t1",
        user_message="Implement secret path hygiene for mission memory.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )
    plugins_mod.invoke_hook(
        "post_tool_call",
        session_id="s11",
        turn_id="t1",
        tool_name="write_file",
        args={
            "path": "/home/jack/.ssh/id_rsa",
            "content": "API_KEY=abc123456789 SECRET_TOKEN=deadbeefdeadbeef",
        },
        result="wrote /home/jack/.ssh/id_rsa and /tmp/allowed-output.txt",
        tool_call_id="tc-secret",
    )
    plugins_mod.invoke_hook(
        "post_tool_call",
        session_id="s11",
        turn_id="t1",
        tool_name="write_file",
        args={"path": "/tmp/allowed-output.txt"},
        result='{"success": true}',
        tool_call_id="tc-allowed",
    )

    conn = sqlite3.connect(db)
    try:
        payloads = "\n".join(row[0] for row in conn.execute("SELECT payload FROM events WHERE tool_name='write_file'"))
        assert "/home/jack/.ssh/id_rsa" not in payloads
        assert "[REDACTED_PATH]" in payloads
        assert "abc123456789" not in payloads
        artifacts = [row[0] for row in conn.execute("SELECT path_or_reference FROM artifacts")]
        assert "/home/jack/.ssh/id_rsa" not in artifacts
        assert "/tmp/allowed-output.txt" in artifacts
    finally:
        conn.close()


def test_parent_session_focus_is_inherited_without_duplicate_mission(monkeypatch, tmp_path):
    plugins_mod, _wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="parent-lineage",
        turn_id="t1",
        user_message="Implement parent lineage mission resume behavior.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="child-lineage",
        turn_id="t2",
        user_message="Continue this work in the child branch.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
        parent_session_id="parent-lineage",
        chat_id="chat-lineage",
    )

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM work_items WHERE work_kind='mission'").fetchone()[0] == 1
        row = conn.execute(
            "SELECT relationship, parent_session_id, chat_id FROM session_links WHERE session_id='child-lineage'"
        ).fetchone()
        assert row == ("continuation", "parent-lineage", "chat-lineage")
    finally:
        conn.close()


def test_branch_conflicts_surface_in_recovery_and_resume_context(monkeypatch, tmp_path):
    plugins_mod, wiki, db = _load_with_temp_paths(monkeypatch, tmp_path)
    plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="branch-a-session",
        turn_id="t1",
        branch_id="branch-a",
        lineage_root_id="lineage-1",
        user_message="Implement branch conflict tracking.",
        conversation_history=[],
        is_first_turn=True,
        model="test-model",
        platform="cli",
    )

    conn = sqlite3.connect(db)
    try:
        work_id = conn.execute("SELECT work_id FROM work_items WHERE work_kind='mission'").fetchone()[0]
        conn.execute(
            """
            INSERT INTO session_links(
                session_id, work_id, relationship, focus, lineage_root_id,
                parent_session_id, branch_id, platform, chat_id, activated_at, metadata
            ) VALUES (?, ?, 'focus', 1, 'lineage-1', '', 'branch-b', 'cli', '', '2026-06-21T00:10:00Z', '{}')
            """,
            ("branch-b-session", work_id),
        )
        conn.execute(
            """
            INSERT INTO events(
                event_id, work_id, session_id, branch_id, turn_id, sequence,
                event_type, source, tool_name, summary, payload, observed_at, checkpoint_id, redacted
            ) VALUES ('evt_branch_b', ?, 'branch-b-session', 'branch-b', 't2', 999,
                'assistant_progress', 'test', '', 'newer branch activity', '{}',
                '2026-06-21T00:20:00Z', NULL, 0)
            """,
            (work_id,),
        )
        conn.execute(
            "UPDATE session_links SET activated_at='2026-06-21T00:00:00Z' WHERE session_id='branch-a-session'"
        )
        conn.execute(
            "UPDATE events SET observed_at='2026-06-21T00:00:00Z' WHERE session_id='branch-a-session'"
        )
        conn.execute(
            "UPDATE checkpoints SET created_at='2026-06-21T00:00:00Z' WHERE work_id=? AND branch_id='branch-a'",
            (work_id,),
        )
        conn.commit()
    finally:
        conn.close()

    handler = plugins_mod.get_plugin_command_handler("wiki")
    assert "Rebuilt mission pages" in handler("repair")
    recovery = (wiki / "recovery.md").read_text(encoding="utf-8")
    assert "Unresolved Branch Conflicts" in recovery
    assert "branch-a" in recovery
    assert "branch-b" in recovery

    context = plugins_mod.invoke_hook(
        "pre_llm_call",
        session_id="branch-a-session",
        turn_id="t3",
        branch_id="branch-a",
        lineage_root_id="lineage-1",
        user_message="Continue branch conflict tracking.",
        conversation_history=[],
        is_first_turn=False,
        model="test-model",
        platform="cli",
    )

    assert context and "Branch warnings:" in context[0]["context"]
    assert "older than `branch-b`" in context[0]["context"]
