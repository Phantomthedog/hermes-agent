from __future__ import annotations

import importlib
import sqlite3
import sys
import threading
import time
import types


def _modules(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_WORK_WIKI_ROOT", str(tmp_path / "wiki"))
    monkeypatch.setenv("HERMES_WORK_WIKI_DB", str(tmp_path / "work.sqlite3"))
    import hermes_cli.plugins as plugins_mod

    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins(force=True)
    loaded = plugins_mod.get_plugin_manager()._plugins["work-wiki"].module
    return loaded, tmp_path / "wiki", tmp_path / "work.sqlite3"


def test_store_handles_concurrent_event_writers_and_checkpoint_coverage(monkeypatch, tmp_path):
    plugin, _wiki, db = _modules(monkeypatch, tmp_path)
    store = plugin._store

    project = store.ensure_project(title="Reliability Project", project_root="/tmp/reliability")
    mission = store.create_mission(
        title="Concurrent Event Reliability",
        objective="Verify concurrent event capture remains consistent.",
        project_work_id=project.work_id,
        project_root="/tmp/reliability",
        session_id="root-session",
        branch_id="main",
    )

    errors: list[BaseException] = []
    ids: list[str] = []
    ids_lock = threading.Lock()

    def _writer(index: int) -> None:
        try:
            for seq in range(25):
                event_id = store.add_event(
                    work_id=mission.work_id,
                    session_id=f"worker-{index}",
                    branch_id="main",
                    turn_id=f"turn-{seq}",
                    event_type="command_succeeded",
                    source="test",
                    summary=f"event {index}-{seq}",
                    payload={"worker": index, "seq": seq},
                )
                with ids_lock:
                    ids.append(event_id)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(ids) == 200
    assert len(set(ids)) == 200

    uncovered = store.uncovered_events(mission.work_id, branch_id="main", limit=500)
    assert len(uncovered) == 200
    checkpoint_id = store.create_checkpoint(
        work_id=mission.work_id,
        session_id="root-session",
        branch_id="main",
        checkpoint_kind="automatic",
        summary="Covered concurrent events.",
        status_after="active",
        event_ids=[row["event_id"] for row in uncovered],
        semantic=True,
        needs_review=False,
    )

    conn = sqlite3.connect(db)
    try:
        total, distinct_sequences = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sequence) FROM events WHERE work_id=?",
            (mission.work_id,),
        ).fetchone()
        assert total == 200
        assert distinct_sequences == 200
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE work_id=? AND checkpoint_id=?",
            (mission.work_id, checkpoint_id),
        ).fetchone()[0] == 200
        assert conn.execute(
            "SELECT COUNT(*) FROM checkpoint_events WHERE checkpoint_id=?",
            (checkpoint_id,),
        ).fetchone()[0] == 200
    finally:
        conn.close()


def test_store_sequences_events_across_independent_store_instances(monkeypatch, tmp_path):
    plugin, _wiki, db = _modules(monkeypatch, tmp_path)
    primary = plugin._store
    store_cls = type(primary)

    project = primary.ensure_project(title="Cross Instance Reliability", project_root="/tmp/cross-instance")
    mission = primary.create_mission(
        title="Cross Instance Event Sequencing",
        objective="Verify independent store instances do not duplicate event sequence numbers.",
        project_work_id=project.work_id,
        project_root="/tmp/cross-instance",
        session_id="root-session",
        branch_id="main",
    )

    errors: list[BaseException] = []

    def _writer(index: int) -> None:
        store = store_cls(primary.config)
        try:
            for seq in range(20):
                store.add_event(
                    work_id=mission.work_id,
                    session_id=f"instance-{index}",
                    branch_id="main",
                    turn_id=f"turn-{seq}",
                    event_type="command_succeeded",
                    source="test",
                    summary=f"cross-instance event {index}-{seq}",
                    payload={"instance": index, "seq": seq},
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []

    conn = sqlite3.connect(db)
    try:
        total, distinct_sequences = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sequence) FROM events WHERE work_id=?",
            (mission.work_id,),
        ).fetchone()
        assert total == 120
        assert distinct_sequences == 120
    finally:
        conn.close()


def test_renderer_handles_large_mission_set_with_dashboard_indexes(monkeypatch, tmp_path):
    plugin, wiki, _db = _modules(monkeypatch, tmp_path)
    store = plugin._store
    renderer = plugin._renderer

    project = store.ensure_project(title="Scale Project", project_root="/tmp/scale")
    for index in range(180):
        mission = store.create_mission(
            title=f"Scale Mission {index:03d}",
            objective=f"Mission {index}",
            project_work_id=project.work_id,
            project_root="/tmp/scale",
            session_id=f"scale-{index}",
            branch_id="main",
        )
        status = "blocked" if index % 17 == 0 else "waiting" if index % 13 == 0 else "active"
        store.update_work_metadata(
            mission.work_id,
            {"current_state": f"State {index}", "next_actions": [f"Next {index}"]},
            status=status,
        )
        if index % 25 == 0:
            store.add_event(
                work_id=mission.work_id,
                session_id=f"scale-{index}",
                branch_id="main",
                event_type="command_succeeded",
                source="test",
                summary=f"activity {index}",
            )

    started = time.perf_counter()
    renderer.render_all()
    elapsed = time.perf_counter() - started

    assert elapsed < 10
    assert (wiki / "mission-control.md").exists()
    assert (wiki / "recovery.md").exists()
    assert (wiki / "work" / "_indexes" / "active-missions.md").exists()
    assert (wiki / "work" / "_indexes" / "blocked-missions.md").exists()
    assert (wiki / "work" / "_indexes" / "waiting-missions.md").exists()

    mission_control = (wiki / "mission-control.md").read_text(encoding="utf-8")
    assert "Scale Mission" in mission_control
    assert "Blocked" in mission_control
    mission_pages = list((wiki / "work" / "missions").glob("*.md"))
    assert len(mission_pages) == 180


def test_store_lists_missions_since_cutoff(monkeypatch, tmp_path):
    plugin, _wiki, db = _modules(monkeypatch, tmp_path)
    store = plugin._store

    project = store.ensure_project(title="Since Filter Project", project_root="/tmp/since-filter")
    old = store.create_mission(
        title="Old Durable Mission",
        objective="Durable but too old.",
        project_work_id=project.work_id,
        project_root="/tmp/since-filter",
        session_id="old-session",
        branch_id="main",
    )
    recent = store.create_mission(
        title="Recent Durable Mission",
        objective="Durable and recent.",
        project_work_id=project.work_id,
        project_root="/tmp/since-filter",
        session_id="recent-session",
        branch_id="main",
    )

    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE work_items SET updated_at='2026-06-10T00:00:00Z' WHERE work_id=?", (old.work_id,))
        conn.execute("UPDATE work_items SET updated_at='2026-06-22T00:00:00Z' WHERE work_id=?", (recent.work_id,))
        conn.commit()
    finally:
        conn.close()

    missions = store.missions_since("2026-06-19T00:00:00Z", limit=10)

    assert [mission.work_id for mission in missions] == [recent.work_id]


def test_curator_candidate_missions_supports_since_days(monkeypatch, tmp_path):
    plugin, _wiki, db = _modules(monkeypatch, tmp_path)
    store = plugin._store

    project = store.ensure_project(title="Curator Since Project", project_root="/tmp/curator-since")
    old = store.create_mission(
        title="Old Curator Mission",
        objective="Old mission with durable wiki evidence.",
        project_work_id=project.work_id,
        project_root="/tmp/curator-since",
        session_id="old-curator",
        branch_id="main",
    )
    recent = store.create_mission(
        title="Recent Curator Mission",
        objective="Recent mission with durable wiki evidence.",
        project_work_id=project.work_id,
        project_root="/tmp/curator-since",
        session_id="recent-curator",
        branch_id="main",
    )
    store.update_work_metadata(
        old.work_id,
        {"findings": ["Durable old wiki workflow decision."], "evidence": ["Old evidence."]},
    )
    store.update_work_metadata(
        recent.work_id,
        {"findings": ["Durable recent wiki workflow decision."], "evidence": ["Recent evidence."]},
    )

    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE work_items SET updated_at='2000-01-01T00:00:00Z' WHERE work_id=?", (old.work_id,))
        conn.commit()
    finally:
        conn.close()

    prompt = plugin._command_handler("curate --since-days 3 --limit 10 --prompt-only")

    assert recent.work_id in prompt
    assert "Recent Curator Mission" in prompt
    assert old.work_id not in prompt
    assert "Old Curator Mission" not in prompt


def test_curator_defaults_to_hermes_curator_task_auto_route(monkeypatch, tmp_path):
    monkeypatch.delenv("WORK_WIKI_CURATOR_PROVIDER", raising=False)
    monkeypatch.delenv("WORK_WIKI_CURATOR_MODEL", raising=False)
    plugin, _wiki, _db = _modules(monkeypatch, tmp_path)
    store = plugin._store

    project = store.ensure_project(title="Curator Route Project", project_root="/tmp/curator-route")
    mission = store.create_mission(
        title="Curator Route Mission",
        objective="Verify durable curator model routing uses the Hermes selected model.",
        project_work_id=project.work_id,
        project_root="/tmp/curator-route",
        session_id="curator-route",
        branch_id="main",
    )
    store.update_work_metadata(
        mission.work_id,
        {
            "findings": ["Durable wiki curation should use the active Hermes cloud model route."],
            "evidence": ["The curator command was invoked without provider or model overrides."],
        },
    )

    captured: dict[str, object] = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        message = types.SimpleNamespace(content='{"updates": []}')
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    agent_module = sys.modules.get("agent") or types.ModuleType("agent")
    agent_module.__path__ = getattr(agent_module, "__path__", [])
    aux_module = types.ModuleType("agent.auxiliary_client")
    aux_module.call_llm = fake_call_llm
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", aux_module)

    output = plugin._command_handler(f"curate --work-id {mission.work_id} --max-updates 1")

    assert output.startswith("Dry run: 0 curated durable fact(s).")
    assert captured["task"] == "curator"
    assert captured["provider"] is None
    assert captured["model"] is None
    assert captured["max_tokens"] == 3000
