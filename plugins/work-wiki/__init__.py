"""Hermes Work Wiki plugin.

Automatic mission memory built around Hermes' existing plugin hooks:

* pre_llm_call resolves/creates mission focus and injects compact resume context.
* pre/post_tool_call records material operational events in SQLite.
* post_llm_call creates semantic checkpoints and renders Markdown.
* /wiki exposes manual correction, repair, and inspection commands.
"""

from __future__ import annotations

import json
import logging
import os
import re
import fnmatch
import threading
from pathlib import Path
from typing import Any

from .classifier import classify_user_message, detect_project_root, resolve_or_create_mission
from .commands import CommandHandler
from .config import load_config
from .promoter import KnowledgePromoter
from .renderer import MarkdownRenderer
from .store import WorkItem, WorkWikiStore, stable_hash

logger = logging.getLogger(__name__)

_config = load_config()
_store = WorkWikiStore(_config)
_renderer = MarkdownRenderer(_config, _store)
_promoter = KnowledgePromoter(_config, _store)
_command_handler = CommandHandler(_store, _renderer)
_lock = threading.RLock()
_turn_focus: dict[str, str] = {}
_pre_tool_seen: set[str] = set()
_strict_persisted_turns: set[str] = set()
_last_persistence_errors: dict[str, str] = {}

MATERIAL_TOOLS = {
    "write_file",
    "patch",
    "apply_patch",
    "edit_file",
    "replace_file",
    "delete_file",
    "execute_code",
    "run_terminal",
    "terminal",
    "shell",
    "bash",
    "delegate_task",
    "send_message",
    "browser_click",
    "browser_type",
    "browser_navigate",
    "mcp_call",
}

READ_ONLY_TOOLS = {
    "read_file",
    "list_files",
    "search_files",
    "grep",
    "web_search",
    "web_extract",
    "vision_analyze",
}

SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{8,})"
)

PATH_RE = re.compile(r"(?<![\w`])/(?:mnt|home|tmp|var|etc)/[^\s'\"`<>]+")
NOISY_PATH_TOKENS = ("\\n", "\n", "\r", "$", "@@", "+++", "---", "```", "REMOVED", "FOUND", "PASS", "===")
STRUCTURED_PATH_KEYS = {
    "path",
    "file_path",
    "filepath",
    "filename",
    "target_path",
    "source_path",
}
STRUCTURED_PATH_LIST_KEYS = {
    "files",
    "paths",
    "files_modified",
    "files_created",
    "files_deleted",
    "modified_files",
    "created_files",
    "deleted_files",
}


def register(ctx) -> None:
    global _config, _store, _renderer, _promoter, _command_handler
    _config = load_config()
    _store = WorkWikiStore(_config)
    _renderer = MarkdownRenderer(_config, _store)
    _promoter = KnowledgePromoter(_config, _store)
    _command_handler = CommandHandler(_store, _renderer)
    with _lock:
        _turn_focus.clear()
        _pre_tool_seen.clear()
        _strict_persisted_turns.clear()
        _last_persistence_errors.clear()
    _command_handler.enabled = True
    ctx.register_command(
        "wiki",
        _command_handler,
        description="Inspect, correct, reconcile, and repair Hermes Work Wiki mission memory.",
        args_hint="[status|list|switch|attach|reconcile|repair|on|off]",
    )
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_reset)


def on_session_start(**kwargs: Any) -> None:
    if not _active():
        return
    session_id = _session_id(kwargs)
    if not session_id:
        return
    focus = _store.focus_for_session(session_id, branch_id=_branch_id(kwargs))
    if not focus:
        return
    _store.add_event(
        work_id=focus.work_id,
        session_id=session_id,
        branch_id=_branch_id(kwargs),
        event_type="session_resumed",
        source="hook:on_session_start",
        summary="Hermes session started with Work Wiki active.",
        payload=_safe_payload(kwargs),
    )
    _process_pending(limit=20)


def on_pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    if not _active() or not _config.auto_detect_work:
        return None
    session_id = _session_id(kwargs)
    turn_id = str(kwargs.get("turn_id") or "")
    branch_id = _branch_id(kwargs)
    user_message = kwargs.get("user_message", "")
    classification = classify_user_message(user_message)
    mission = None
    parent_session_id = _parent_session_id(kwargs)
    if parent_session_id:
        mission = _store.focus_for_session(session_id, branch_id=branch_id)
        if not mission:
            mission = _store.focus_for_parent_session(parent_session_id, branch_id=branch_id)
        if not mission:
            mission = _store.focus_for_parent_session(parent_session_id, branch_id="")
    if mission:
        _store.link_session(
            session_id=session_id,
            work_id=mission.work_id,
            branch_id=branch_id,
            relationship="continuation",
            focus=True,
            lineage_root_id=_lineage_root_id(kwargs),
            parent_session_id=parent_session_id,
            platform=str(kwargs.get("platform") or ""),
            chat_id=_chat_id(kwargs),
            metadata={"inferred_from_parent_session": parent_session_id},
        )
    else:
        mission = resolve_or_create_mission(
            _store,
            user_message=user_message,
            session_id=session_id,
            branch_id=branch_id,
            platform=str(kwargs.get("platform") or ""),
            project_root=detect_project_root(),
            classification=classification,
            allow_create=_can_auto_create_work(),
        )
    branch_conflicts = _store.branch_conflicts_for_branch(mission.work_id, branch_id, limit=5) if mission else []
    if mission:
        _store.link_session(
            session_id=session_id,
            work_id=mission.work_id,
            branch_id=branch_id,
            relationship="focus",
            focus=True,
            lineage_root_id=_lineage_root_id(kwargs),
            parent_session_id=_parent_session_id(kwargs),
            platform=str(kwargs.get("platform") or ""),
            chat_id=_chat_id(kwargs),
            metadata={"sender_id": str(kwargs.get("sender_id") or "")},
        )
    logger.info(
        "work-wiki classification: material=%s confidence=%.2f reasons=%s session=%s",
        classification.material,
        classification.confidence,
        ",".join(classification.reason_codes),
        session_id,
    )
    classified_event_id = ""
    if classification.material or mission:
        classified_event_id = _store.add_event(
            work_id=mission.work_id if mission else None,
            session_id=session_id,
            branch_id=branch_id,
            turn_id=turn_id,
            event_type="work_classified",
            source="hook:pre_llm_call",
            summary=(
                f"Classified user turn as material={classification.material} "
                f"confidence={classification.confidence:.2f}"
            ),
            payload={"classification": classification.as_dict(), "message_hash": stable_hash(_text(user_message))},
        )
    if mission:
        _set_focus(session_id, turn_id, mission.work_id)
        initial_checkpoint_created = False
        if _can_auto_checkpoint() and classification.material and not _store.latest_checkpoint(mission.work_id):
            event_ids = [row["event_id"] for row in _store.uncovered_events(mission.work_id, branch_id=branch_id, limit=100)]
            if classified_event_id and classified_event_id not in event_ids:
                event_ids.append(classified_event_id)
            _store.create_checkpoint(
                work_id=mission.work_id,
                session_id=session_id,
                branch_id=branch_id,
                checkpoint_kind="initial",
                summary=f"Mission initialized: {mission.title}",
                status_after=mission.status,
                metadata={
                    "current_state": mission.metadata.get("current_state"),
                    "next_actions": mission.metadata.get("next_actions"),
                    "verification_state": "initial",
                },
                event_ids=event_ids,
                semantic=True,
                needs_review=False,
                confidence=max(0.5, classification.confidence),
            )
            initial_checkpoint_created = True
        if not initial_checkpoint_created and _can_render():
            _store.enqueue_render("mission", work_id=mission.work_id)
            _store.enqueue_render("dashboards")
        if initial_checkpoint_created or _can_render():
            _process_pending(limit=20)
        if _can_resume_context():
            return {"context": _resume_context(mission, branch_id=branch_id, branch_conflicts=branch_conflicts)}
    return None


def on_transform_llm_output(**kwargs: Any) -> str | None:
    if not _active() or not _can_strict_checkpoint():
        return None
    response = str(kwargs.get("response_text") or "")
    if not response.strip():
        return None
    work_id = _focus_work_id(kwargs)
    if not work_id:
        return None
    try:
        persisted = _persist_response_checkpoint(kwargs, response, source_hook="hook:transform_llm_output", force=True)
        if persisted:
            _mark_strict_persisted(kwargs, response)
            _clear_persistence_error(kwargs)
        return None
    except Exception as exc:
        message = _record_persistence_failure(kwargs, exc)
        warning = (
            "\n\nWork Wiki persistence warning: Mission state was not fully persisted before this response. "
            f"{message} Check `recovery.md` or run `/wiki repair`."
        )
        if warning.strip() in response:
            return response
        return response.rstrip() + warning


def on_pre_tool_call(**kwargs: Any) -> None:
    if not _active() or not _config.auto_capture_events:
        return None
    tool_name = str(kwargs.get("tool_name") or "")
    if tool_name in READ_ONLY_TOOLS:
        return None
    call_id = str(kwargs.get("tool_call_id") or f"{_session_id(kwargs)}:{kwargs.get('turn_id')}:{tool_name}")
    if call_id in _pre_tool_seen:
        return None
    _pre_tool_seen.add(call_id)
    work_id = _focus_work_id(kwargs)
    if tool_name in MATERIAL_TOOLS or _looks_material_tool(tool_name, kwargs.get("args")):
        _store.add_event(
            work_id=work_id,
            session_id=_session_id(kwargs),
            branch_id=_branch_id(kwargs),
            turn_id=str(kwargs.get("turn_id") or ""),
            event_type="command_executed" if "terminal" in tool_name or "shell" in tool_name else "tool_started",
            source="hook:pre_tool_call",
            tool_name=tool_name,
            summary=f"Started tool call: {tool_name}",
            payload={"args": _redact(kwargs.get("args") or {})},
        )
    return None


def on_post_tool_call(**kwargs: Any) -> None:
    if not _active() or not _config.auto_capture_events:
        return
    tool_name = str(kwargs.get("tool_name") or "")
    if tool_name in READ_ONLY_TOOLS:
        return
    result = kwargs.get("result")
    work_id = _focus_work_id(kwargs)
    event_type = _event_type_for_tool(tool_name, result)
    summary = _tool_summary(tool_name, kwargs.get("args"), result)
    _store.add_event(
        work_id=work_id,
        session_id=_session_id(kwargs),
        branch_id=_branch_id(kwargs),
        turn_id=str(kwargs.get("turn_id") or ""),
        event_type=event_type,
        source="hook:post_tool_call",
        tool_name=tool_name,
        summary=summary,
        payload={
            "args": _redact(kwargs.get("args") or {}),
            "result": _redact(_compact_result(result)),
            "duration_ms": kwargs.get("duration_ms"),
            "status": kwargs.get("status"),
            "error_type": kwargs.get("error_type"),
        },
    )
    if work_id:
        for path in _artifact_paths(tool_name, kwargs.get("args"), result):
            _store.add_artifact(work_id, path, description=f"Observed via {tool_name}")


def on_subagent_start(**kwargs: Any) -> None:
    if not _active() or not _config.auto_capture_events:
        return
    parent_session_id = str(kwargs.get("parent_session_id") or "")
    child_session_id = str(kwargs.get("child_session_id") or "")
    parent_turn_id = str(kwargs.get("parent_turn_id") or "")
    role = str(kwargs.get("child_role") or "")
    goal = str(kwargs.get("child_goal") or "")
    branch_id = _branch_id({"session_id": parent_session_id, "turn_id": parent_turn_id})
    work_id = _focus_work_id({"session_id": parent_session_id, "turn_id": parent_turn_id, "branch_id": branch_id})
    delegation_id = _store.start_delegation(
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        work_id=work_id or "",
        branch_id=branch_id,
        role=role,
        goal=goal,
        metadata=_safe_payload(kwargs),
    )
    if work_id:
        if child_session_id:
            _store.link_session(
                session_id=child_session_id,
                work_id=work_id,
                relationship="delegate",
                focus=True,
                lineage_root_id=parent_session_id,
                parent_session_id=parent_session_id,
                branch_id="default",
                metadata={"delegation_id": delegation_id, "role": role},
            )
            _set_focus(child_session_id, "", work_id)
        _store.add_event(
            work_id=work_id,
            session_id=parent_session_id,
            branch_id=branch_id,
            turn_id=parent_turn_id,
            event_type="delegate_started",
            source="hook:subagent_start",
            summary=f"Delegate started: {role or 'subagent'}" + (f" - {goal[:160]}" if goal else ""),
            payload={"delegation_id": delegation_id, "child_session_id": child_session_id, "role": role, "goal": goal},
        )
        _store.enqueue_render("mission", work_id=work_id)
    _store.enqueue_render("dashboards")
    _process_pending(limit=20)


def on_subagent_stop(**kwargs: Any) -> None:
    if not _active() or not _config.auto_capture_events:
        return
    parent_session_id = str(kwargs.get("parent_session_id") or "")
    child_session_id = str(kwargs.get("child_session_id") or "")
    parent_turn_id = str(kwargs.get("parent_turn_id") or "")
    role = str(kwargs.get("child_role") or "")
    status = str(kwargs.get("child_status") or "completed")
    summary = str(kwargs.get("child_summary") or "")
    branch_id = _branch_id({"session_id": parent_session_id, "turn_id": parent_turn_id})
    work_id = _focus_work_id({"session_id": parent_session_id, "turn_id": parent_turn_id, "branch_id": branch_id})
    duration_ms = kwargs.get("duration_ms")
    try:
        duration = int(duration_ms) if duration_ms is not None else None
    except (TypeError, ValueError):
        duration = None
    delegation_id = _store.finish_delegation(
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        work_id=work_id or "",
        role=role,
        status=status,
        summary=summary,
        duration_ms=duration,
        metadata=_safe_payload(kwargs),
    )
    if work_id:
        event_type = "delegate_completed" if status in {"completed", "success", "ok"} else "delegate_failed"
        _store.add_event(
            work_id=work_id,
            session_id=parent_session_id,
            branch_id=branch_id,
            turn_id=parent_turn_id,
            event_type=event_type,
            source="hook:subagent_stop",
            summary=f"Delegate {status}: {summary[:180] or role or child_session_id or delegation_id}",
            payload={"delegation_id": delegation_id, "child_session_id": child_session_id, "role": role, "status": status},
        )
        _store.enqueue_render("mission", work_id=work_id)
    _store.enqueue_render("dashboards")
    _process_pending(limit=20)


def on_post_llm_call(**kwargs: Any) -> None:
    if not _active() or not _can_auto_checkpoint():
        return
    response = str(kwargs.get("assistant_response") or "")
    if _was_strict_persisted(kwargs, response):
        return
    try:
        _persist_response_checkpoint(kwargs, response, source_hook="hook:post_llm_call", force=False)
        _clear_persistence_error(kwargs)
    except Exception as exc:
        _record_persistence_failure(kwargs, exc)


def _persist_response_checkpoint(
    kwargs: dict[str, Any],
    response: str,
    *,
    source_hook: str,
    force: bool,
) -> bool:
    session_id = _session_id(kwargs)
    turn_id = str(kwargs.get("turn_id") or "")
    work_id = _focus_work_id(kwargs)
    if not work_id:
        return False
    mission = _store.get_work(work_id)
    if not mission:
        return False
    if not response.strip():
        return False
    uncovered = _store.uncovered_events(work_id, branch_id=_branch_id(kwargs), limit=200)
    if not force and not uncovered and len(response) < 80:
        return False

    summary = _summarize_response(response)
    meta_updates = _metadata_from_response(mission, response)
    status_after = _infer_status(mission, response, meta_updates)
    _store.update_work_metadata(work_id, meta_updates, status=status_after if status_after != mission.status else None)
    event_id = _store.add_event(
        work_id=work_id,
        session_id=session_id,
        branch_id=_branch_id(kwargs),
        turn_id=turn_id,
        event_type="assistant_claimed_completion" if _completion_claim(response) else "assistant_progress",
        source=source_hook,
        summary=summary,
        payload={"response_hash": stable_hash(response), "response_excerpt": response[:1000]},
    )
    event_ids = [row["event_id"] for row in uncovered] + [event_id]
    checkpoint_id = _store.create_checkpoint(
        work_id=work_id,
        session_id=session_id,
        branch_id=_branch_id(kwargs),
        checkpoint_kind="completion" if status_after == "completed" else "automatic",
        summary=summary,
        status_after=status_after,
        metadata={
            "current_state": meta_updates.get("current_state"),
            "next_actions": meta_updates.get("next_actions"),
            "evidence": meta_updates.get("evidence"),
            "completion_claim": _completion_claim(response),
            "verification_state": "observed" if meta_updates.get("evidence") else "unverified",
        },
        event_ids=event_ids,
        semantic=True,
        needs_review=status_after == "needs_review",
        confidence=0.72,
    )
    for decision in meta_updates.get("decisions", [])[:5]:
        _store.add_decision(work_id, str(decision), checkpoint_id=checkpoint_id)
    for artifact in meta_updates.get("artifacts", [])[:20]:
        _store.add_artifact(work_id, str(artifact), checkpoint_id=checkpoint_id)
    promoted = _promoter.promote(work=mission, checkpoint_id=checkpoint_id, updates=meta_updates, summary=summary)
    if promoted:
        _store.enqueue_render("mission", work_id=work_id, checkpoint_id=checkpoint_id)
    _process_pending(limit=30)
    if _strict_mode():
        failures = [
            row for row in _store.render_failures(limit=50)
            if row["checkpoint_id"] == checkpoint_id or row["work_id"] == work_id
        ]
        if failures:
            raise RuntimeError(f"Markdown render failed for checkpoint {checkpoint_id}: {failures[0]['last_error']}")
    return True


def on_session_end(**kwargs: Any) -> None:
    if not _active():
        return
    work_id = _focus_work_id(kwargs)
    _store.add_event(
        work_id=work_id,
        session_id=_session_id(kwargs),
        branch_id=_branch_id(kwargs),
        turn_id=str(kwargs.get("turn_id") or ""),
        event_type="session_turn_ended",
        source="hook:on_session_end",
        summary=f"Turn ended: completed={bool(kwargs.get('completed'))} interrupted={bool(kwargs.get('interrupted'))}",
        payload=_safe_payload(kwargs),
    )
    _process_pending(limit=20)


def on_session_finalize(**kwargs: Any) -> None:
    if not _active():
        return
    work_id = _focus_work_id(kwargs)
    _store.add_event(
        work_id=work_id,
        session_id=_session_id(kwargs),
        branch_id=_branch_id(kwargs),
        event_type="session_interrupted" if str(kwargs.get("reason") or "").lower() in {"keyboard_interrupt", "shutdown", "expired"} else "session_finalized",
        source="hook:on_session_finalize",
        summary=f"Session finalized: {kwargs.get('reason') or 'unknown'}",
        payload=_safe_payload(kwargs),
    )
    if work_id:
        _store.enqueue_render("mission", work_id=work_id)
    _store.enqueue_render("dashboards")
    _process_pending(limit=30)


def on_session_reset(**kwargs: Any) -> None:
    if not _active():
        return
    session_id = _session_id(kwargs)
    branch_id = _branch_id(kwargs)
    work_id = _focus_work_id(kwargs)
    if work_id:
        _store.add_event(
            work_id=work_id,
            session_id=session_id,
            branch_id=branch_id,
            turn_id=str(kwargs.get("turn_id") or ""),
            event_type="session_reset",
            source="hook:on_session_reset",
            summary="Session reset; mission focus deactivated for this session.",
            payload=_safe_payload(kwargs),
        )
        _store.enqueue_render("mission", work_id=work_id)
    _store.deactivate_session(session_id, branch_id=branch_id)
    _clear_focus(session_id)
    _store.enqueue_render("dashboards")
    _process_pending(limit=30)


def _resume_context(mission: WorkItem, branch_id: str = "", branch_conflicts: list[Any] | None = None) -> str:
    latest = _store.latest_checkpoint(mission.work_id)
    debt = len(_store.uncovered_events(mission.work_id, limit=500))
    artifacts = _store.artifacts_for_work(mission.work_id, limit=5)
    running_delegates = [row for row in _store.active_delegations(limit=50) if row["work_id"] == mission.work_id]
    branch_conflicts = branch_conflicts if branch_conflicts is not None else _store.branch_conflicts_for_branch(mission.work_id, branch_id, limit=5)
    meta = mission.metadata
    project = _store.get_work(mission.parent_work_id) if mission.parent_work_id else None
    lines = [
        "ACTIVE MISSION",
        "",
        f"Project: {project.title if project else 'Unassigned'}",
        f"Mission: {mission.title}",
        f"Work ID: {mission.work_id}",
        f"Status: {mission.status}",
        f"Branch: {branch_id or 'default'}",
        "",
        "Objective:",
        f"- {meta.get('objective') or mission.title}",
        "",
        "Current state:",
        f"- {meta.get('current_state') or 'No current state recorded.'}",
        "",
        "Last verified checkpoint:",
        f"- {latest.summary if latest else 'No checkpoint yet.'}",
        "",
        "Blockers:",
        _bullet(meta.get("blockers"), "None recorded."),
        "",
        "Latest artifacts:",
        _bullet([row["path_or_reference"] for row in artifacts], "None recorded."),
        "",
        "Running delegates:",
        _bullet([f"{row['role'] or 'subagent'}: {row['goal'] or row['child_session_id']}" for row in running_delegates], "None recorded."),
        "",
        "Branch warnings:",
        _bullet(
            [
                f"This branch is older than `{row['newer_branch_id']}` in lineage `{row['lineage_key']}`; do not treat newer sibling work as completed in this branch without verification."
                for row in branch_conflicts
            ],
            "None recorded.",
        ),
        "",
        "Next action:",
        f"- {_first(meta.get('next_actions')) or 'Continue the mission and record the next concrete step.'}",
        "",
        f"Checkpoint debt: {debt} uncovered material event(s).",
    ]
    return "\n".join(lines)


def _active() -> bool:
    return bool(_config.enabled and getattr(_command_handler, "enabled", True))


def _mode() -> str:
    return (_config.mode or "automatic").replace("_", "-").strip().lower()


def _observe_only() -> bool:
    return _mode() in {"observe", "observe-only", "observe_only"}


def _manual_checkpoint_mode() -> bool:
    return _mode() in {"manual", "manual-checkpoint", "manual_checkpoint"}


def _strict_mode() -> bool:
    return bool(_config.strict_persistence or _mode() == "strict")


def _can_auto_create_work() -> bool:
    return bool(_config.auto_create_projects and _config.auto_create_missions and not _observe_only())


def _can_auto_checkpoint() -> bool:
    return bool(_config.auto_generate_checkpoints and not _observe_only() and not _manual_checkpoint_mode())


def _can_resume_context() -> bool:
    return bool(_config.auto_resume_context and not _observe_only())


def _can_render() -> bool:
    return bool(_config.auto_render_wiki and not _observe_only())


def _can_strict_checkpoint() -> bool:
    return bool(_strict_mode() and _can_auto_checkpoint())


def _process_pending(limit: int) -> None:
    if _can_render():
        _renderer.process_pending(limit=limit)


def _set_focus(session_id: str, turn_id: str, work_id: str) -> None:
    with _lock:
        _turn_focus[f"{session_id}:{turn_id}"] = work_id
        _turn_focus[session_id] = work_id


def _clear_focus(session_id: str) -> None:
    if not session_id:
        return
    with _lock:
        for key in list(_turn_focus):
            if key == session_id or key.startswith(f"{session_id}:"):
                _turn_focus.pop(key, None)


def _focus_work_id(kwargs: dict[str, Any]) -> str | None:
    session_id = _session_id(kwargs)
    turn_id = str(kwargs.get("turn_id") or "")
    with _lock:
        work_id = _turn_focus.get(f"{session_id}:{turn_id}") or _turn_focus.get(session_id)
    if work_id:
        return work_id
    focus = _store.focus_for_session(session_id, branch_id=_branch_id(kwargs))
    if not focus:
        focus = _store.focus_for_session(session_id, branch_id="")
    if not focus:
        parent_session_id = _parent_session_id(kwargs)
        focus = _store.focus_for_parent_session(parent_session_id, branch_id=_branch_id(kwargs))
        if focus:
            _store.link_session(
                session_id=session_id,
                work_id=focus.work_id,
                branch_id=_branch_id(kwargs),
                relationship="continuation",
                focus=True,
                lineage_root_id=_lineage_root_id(kwargs),
                parent_session_id=parent_session_id,
                platform=str(kwargs.get("platform") or ""),
                chat_id=_chat_id(kwargs),
                metadata={"inferred_from_parent_session": parent_session_id},
            )
            _set_focus(session_id, str(kwargs.get("turn_id") or ""), focus.work_id)
    return focus.work_id if focus else None


def _session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("session_id") or kwargs.get("task_id") or kwargs.get("parent_session_id") or "")


def _branch_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("branch_id") or kwargs.get("lineage_root_id") or kwargs.get("parent_session_id") or "default")


def _parent_session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("parent_session_id") or "")


def _lineage_root_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("lineage_root_id") or kwargs.get("parent_session_id") or _session_id(kwargs) or "")


def _chat_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("chat_id") or os.getenv("HERMES_SESSION_CHAT_ID") or "")


def _turn_key(kwargs: dict[str, Any], response: str = "") -> str:
    session_id = _session_id(kwargs)
    turn_id = str(kwargs.get("turn_id") or "")
    if turn_id:
        return f"{session_id}:{turn_id}"
    return f"{session_id}:response:{stable_hash(response)}"


def _mark_strict_persisted(kwargs: dict[str, Any], response: str) -> None:
    with _lock:
        _strict_persisted_turns.add(_turn_key(kwargs, response))


def _was_strict_persisted(kwargs: dict[str, Any], response: str) -> bool:
    with _lock:
        return _turn_key(kwargs, response) in _strict_persisted_turns


def _clear_persistence_error(kwargs: dict[str, Any]) -> None:
    with _lock:
        _last_persistence_errors.pop(_turn_key(kwargs), None)


def _record_persistence_failure(kwargs: dict[str, Any], exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    key = _turn_key(kwargs, str(kwargs.get("response_text") or kwargs.get("assistant_response") or ""))
    with _lock:
        _last_persistence_errors[key] = message
    work_id = _focus_work_id(kwargs)
    if work_id:
        try:
            failure_event_id = _store.add_event(
                work_id=work_id,
                session_id=_session_id(kwargs),
                branch_id=_branch_id(kwargs),
                turn_id=str(kwargs.get("turn_id") or ""),
                event_type="persistence_failed",
                source="work-wiki",
                summary=f"Work Wiki persistence failed: {message[:180]}",
                payload={"error": message},
                redacted=True,
            )
            uncovered = _store.uncovered_events(work_id, branch_id=_branch_id(kwargs), limit=200)
            event_ids = [row["event_id"] for row in uncovered]
            if failure_event_id not in event_ids:
                event_ids.append(failure_event_id)
            if event_ids:
                _store.create_checkpoint(
                    work_id=work_id,
                    session_id=_session_id(kwargs),
                    branch_id=_branch_id(kwargs),
                    checkpoint_kind="fallback",
                    summary=f"Fallback checkpoint after persistence failure: {message[:180]}",
                    status_after="needs_review",
                    metadata={
                        "verification_state": "unverified",
                        "observed_failure": message,
                        "fallback_reason": "persistence_failed",
                    },
                    event_ids=event_ids,
                    semantic=False,
                    needs_review=True,
                    confidence=0.35,
                )
            _store.enqueue_render("dashboards")
            _process_pending(limit=10)
        except Exception:
            logger.warning("work-wiki failed to record persistence failure", exc_info=True)
    logger.warning("work-wiki persistence failed: %s", message)
    return message


def _looks_material_tool(tool_name: str, args: Any) -> bool:
    lowered = tool_name.lower()
    if any(token in lowered for token in ("write", "patch", "delete", "terminal", "exec", "delegate", "deploy", "install")):
        return True
    text = json.dumps(args, default=str, ensure_ascii=False).lower() if args is not None else ""
    return any(token in text for token in ("apply_patch", "git commit", "npm install", "pytest", "cargo test", "deploy"))


def _event_type_for_tool(tool_name: str, result: Any) -> str:
    if _tool_result_failed(result):
        return "command_failed" if "terminal" in tool_name or "shell" in tool_name else "tool_failed"
    if tool_name in {"write_file", "patch", "apply_patch", "edit_file", "replace_file"}:
        return "file_modified"
    if "delete" in tool_name:
        return "file_deleted"
    if "delegate" in tool_name:
        return "delegate_started"
    if "terminal" in tool_name or "shell" in tool_name or "exec" in tool_name:
        return "command_succeeded"
    return "tool_completed"


def _tool_summary(tool_name: str, args: Any, result: Any) -> str:
    paths = _artifact_paths(tool_name, args, result)
    if paths:
        return f"{tool_name} touched {', '.join(paths[:3])}"
    if isinstance(args, dict):
        command = args.get("cmd") or args.get("command")
        if command:
            return f"{tool_name}: {str(command)[:160]}"
    return f"Completed tool call: {tool_name}"


def _extract_paths(*values: Any) -> list[str]:
    paths: list[str] = []
    for path in _candidate_paths(*values):
        clean = _clean_path(path)
        if not clean:
            continue
        if _path_denied(clean):
            continue
        if clean not in paths:
            paths.append(clean)
    return paths[:50]


def _candidate_paths(*values: Any, allow_plain_text: bool = False) -> list[str]:
    paths: list[str] = []
    for value in values:
        if isinstance(value, dict):
            for key in STRUCTURED_PATH_KEYS:
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.startswith("/"):
                    paths.append(candidate)
            for key in STRUCTURED_PATH_LIST_KEYS:
                candidate = value.get(key)
                if isinstance(candidate, (str, list, tuple, set)):
                    paths.extend(_candidate_paths(candidate, allow_plain_text=allow_plain_text))
            for nested in value.values():
                if isinstance(nested, (dict, list, tuple, set)):
                    paths.extend(_candidate_paths(nested, allow_plain_text=allow_plain_text))
        elif isinstance(value, list):
            for item in value:
                paths.extend(_candidate_paths(item, allow_plain_text=allow_plain_text))
        elif isinstance(value, str):
            if allow_plain_text or _looks_like_command_or_patch_output(value):
                paths.extend(PATH_RE.findall(value[:2000]))
    seen = set()
    out = []
    for path in paths:
        clean = _clean_path(path)
        if not clean:
            continue
        if clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out[:50]


def _artifact_paths(tool_name: str, args: Any, result: Any) -> list[str]:
    values: list[Any] = []
    if isinstance(args, dict):
        values.append(args)
        command = str(args.get("cmd") or args.get("command") or "")
        if _command_can_touch_files(command):
            values.append(command[:2000])
    elif isinstance(args, str) and _command_can_touch_files(args):
        values.append(args[:2000])
    if isinstance(result, dict):
        values.append(result)
    return _extract_paths(*values)


def _clean_path(path: str) -> str:
    clean = str(path or "").strip().rstrip(".,);]\"'")
    if not clean.startswith("/"):
        return ""
    if any(token in clean for token in NOISY_PATH_TOKENS):
        return ""
    if "*" in clean or "?" in clean:
        return ""
    if len(clean) > 260:
        return ""
    return clean


def _looks_like_command_or_patch_output(value: str) -> bool:
    text = str(value or "")
    if not text.strip():
        return False
    return _command_can_touch_files(text) or any(marker in text for marker in ("*** Begin Patch", "diff --git", "\n+++ ", "\n--- "))


def _command_can_touch_files(command: str) -> bool:
    lowered = str(command or "").lower()
    return any(
        token in lowered
        for token in (
            "apply_patch",
            "cat >",
            "tee ",
            "python",
            "node ",
            "npm ",
            "pytest",
            "touch ",
            "mv ",
            "cp ",
            "install ",
            "git ",
            "sqlite3 ",
        )
    )


def _tool_result_failed(result: Any) -> bool:
    if isinstance(result, dict):
        for key in ("success", "ok"):
            if key in result:
                return not bool(result.get(key))
        for key in ("failed", "error", "is_error"):
            if key in result and result.get(key):
                return True
        for key in ("exit_code", "returncode", "status_code", "code", "rc"):
            if key in result:
                try:
                    return int(result.get(key)) != 0
                except (TypeError, ValueError):
                    pass
        status = str(result.get("status") or result.get("state") or "").lower()
        if status in {"failed", "error", "errored", "timeout", "timed_out", "cancelled"}:
            return True
        if status in {"ok", "success", "succeeded", "completed", "done"}:
            return False
    return False


def _metadata_from_response(mission: WorkItem, response: str) -> dict[str, Any]:
    meta = dict(mission.metadata)
    updates: dict[str, Any] = {}
    updates["current_state"] = _summarize_response(response)
    next_actions = _extract_section_items(response, ("next", "next action", "next steps", "todo"))
    if next_actions:
        updates["next_actions"] = next_actions[:8]
    evidence = _extract_section_items(response, ("verified", "verification", "tests", "tested"))
    if evidence:
        updates["evidence"] = list(dict.fromkeys(list(meta.get("evidence", [])) + evidence))[:30]
    blockers = _extract_section_items(response, ("blocker", "blockers", "blocked by", "unable to proceed"))
    if blockers and any(phrase in response.lower() for phrase in ("blocked by", "blocker", "unable to proceed", "cannot proceed")):
        updates["blockers"] = list(dict.fromkeys(list(meta.get("blockers", [])) + blockers))[:20]
    paths = _extract_paths(response) if _looks_like_command_or_patch_output(response) else []
    if paths:
        updates["artifacts"] = list(dict.fromkeys(list(meta.get("artifacts", [])) + paths))[:50]
        updates["changed_files"] = list(dict.fromkeys(list(meta.get("changed_files", [])) + paths))[:50]
    decisions = _extract_section_items(response, ("decision", "decisions", "decided"))
    if decisions:
        updates["decisions"] = list(dict.fromkeys(list(meta.get("decisions", [])) + decisions))[:30]
    findings = _extract_section_items(response, ("finding", "findings", "learned", "root cause"))
    if findings:
        updates["findings"] = list(dict.fromkeys(list(meta.get("findings", [])) + findings))[:30]
    return updates


def _path_denied(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = os.path.basename(normalized)
    for pattern in _config.path_deny_patterns:
        pat = str(pattern).replace("\\", "/")
        if fnmatch.fnmatch(normalized, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


def _infer_status(mission: WorkItem, response: str, updates: dict[str, Any]) -> str:
    existing_blockers = [str(item).strip() for item in mission.metadata.get("blockers", []) if str(item).strip()]
    if updates.get("blockers"):
        return "blocked"
    if existing_blockers and _completion_claim(response):
        return "needs_review"
    if existing_blockers:
        return "blocked"
    if _completion_claim(response) and _has_running_delegate(mission.work_id):
        return "needs_review"
    if _completion_claim(response) and updates.get("evidence"):
        return "completed"
    if _completion_claim(response):
        return "needs_review"
    return mission.status if mission.status not in {"completed", "merged"} else mission.status


def _has_running_delegate(work_id: str) -> bool:
    try:
        return any(row["work_id"] == work_id for row in _store.active_delegations(limit=200))
    except Exception:
        return False


def _completion_claim(response: str) -> bool:
    lowered = response.lower()
    return any(phrase in lowered for phrase in ("implemented", "completed", "done", "fixed", "shipped")) and not any(
        phrase in lowered for phrase in ("not completed", "not done", "unable to complete", "couldn't complete")
    )


def _extract_section_items(text: str, names: tuple[str, ...]) -> list[str]:
    lines = text.splitlines()
    items: list[str] = []
    capture = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("```") or line in {"`", "```text", "```bash"}:
            continue
        if not line:
            if capture:
                capture = False
            continue
        if _is_section_heading(line, names):
            capture = True
            continue
        if capture and (line.startswith("-") or line.startswith("*")):
            items.append(line.lstrip("-* ").strip())
        elif capture and re.match(r"^\d+[.)]\s+", line):
            items.append(re.sub(r"^\d+[.)]\s+", "", line).strip())
    return [item for item in items if _valid_metadata_item(item)][:20]


def _is_section_heading(line: str, names: tuple[str, ...]) -> bool:
    stripped = line.strip().strip("#").strip()
    lower = stripped.lower().strip(":")
    if len(stripped) > 80:
        return False
    return any(lower == name or lower.startswith(f"{name}:") for name in names)


def _valid_metadata_item(item: str) -> bool:
    stripped = str(item or "").strip()
    if not stripped or stripped.startswith("```"):
        return False
    if stripped in {"`", "```text", "```bash", "evidence", "summary text"}:
        return False
    if stripped in {"@@", "+++", "---"}:
        return False
    return len(stripped) >= 4


def _summarize_response(response: str) -> str:
    for line in response.splitlines():
        clean = line.strip().strip("-* ")
        if clean and not clean.startswith("```") and len(clean) > 12:
            return clean[:240]
    return response.strip()[:240] or "Progress recorded."


def _compact_result(result: Any) -> Any:
    if isinstance(result, str):
        return result[:2000]
    return result


def _safe_payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "session_id",
        "task_id",
        "turn_id",
        "completed",
        "interrupted",
        "model",
        "platform",
        "reason",
        "status",
        "parent_session_id",
        "lineage_root_id",
        "branch_id",
        "chat_id",
        "sender_id",
        "parent_turn_id",
        "child_session_id",
        "child_role",
        "child_status",
        "duration_ms",
    }
    return _redact({key: value for key, value in kwargs.items() if key in allowed})


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if re.search(r"(?i)(api[_-]?key|token|secret|password|authorization)", str(key)):
                out[str(key)] = "[REDACTED]"
            elif re.search(r"(?i)(env|environment|dotenv|private[_-]?key|credentials?)", str(key)):
                out[str(key)] = "[REDACTED]"
            else:
                out[str(key)] = _redact(item)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value[:50]]
    if isinstance(value, str):
        text = value[:4000]
        for path in _candidate_paths(text, allow_plain_text=True):
            if _path_denied(path):
                text = text.replace(path, "[REDACTED_PATH]")
        text = SECRET_RE.sub(r"\1=[REDACTED]", text)
        text = re.sub(r"(?m)^([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY)[A-Z0-9_]*)=.*$", r"\1=[REDACTED]", text)
        return text
    return value


def _bullet(values: Any, empty: str) -> str:
    if isinstance(values, str) and values.strip():
        return f"- {values.strip()}"
    if isinstance(values, list) and values:
        return "\n".join(f"- {item}" for item in values)
    return f"- {empty}"


def _first(values: Any) -> str:
    if isinstance(values, str):
        return values
    if isinstance(values, list) and values:
        return str(values[0])
    return ""


def _text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for item in message:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(message)
