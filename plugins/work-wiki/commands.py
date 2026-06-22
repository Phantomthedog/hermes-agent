from __future__ import annotations

import argparse
import shlex
from typing import Any

from .classifier import classify_user_message, detect_project_root, resolve_or_create_mission
from .curator import KnowledgeCurator
from .renderer import MarkdownRenderer
from .store import WorkItem, WorkWikiStore


HELP = """Work Wiki commands:
  /wiki status [work-id]
  /wiki list [active|blocked|waiting|paused|completed|needs_review]
  /wiki switch <work-id-or-query>
  /wiki attach <work-id>
  /wiki detach [session-id]
  /wiki exclude-session <work-id> [session-id]
  /wiki reassign <source-work-id> <project-work-id>
  /wiki merge <source> <canonical>
  /wiki split <source-work-id> <new mission title>
  /wiki reopen <work-id>
  /wiki close <work-id>
  /wiki pause <work-id>
  /wiki block <work-id> [reason]
  /wiki inbox
  /wiki conflicts
  /wiki curate [--apply] [--limit N] [--since-days N] [--work-id ID] [--provider P] [--model M]
  /wiki review
  /wiki reconcile
  /wiki repair
  /wiki on|off
"""


class CommandHandler:
    def __init__(self, store: WorkWikiStore, renderer: MarkdownRenderer):
        self.store = store
        self.renderer = renderer
        self.enabled = True

    def __call__(self, raw_args: str = "") -> str:
        try:
            return self.handle(raw_args)
        except Exception as exc:
            return f"Work Wiki command failed: {exc}"

    def handle(self, raw_args: str = "") -> str:
        args = shlex.split(raw_args or "")
        if not args:
            return self.status("")
        cmd = args[0].lower()
        rest = args[1:]
        if cmd in {"help", "-h", "--help"}:
            return HELP.rstrip()
        if cmd == "status":
            return self.status(rest[0] if rest else "")
        if cmd == "list":
            return self.list_status(rest[0] if rest else "")
        if cmd == "switch":
            return self.switch(" ".join(rest))
        if cmd == "attach":
            return self.attach(rest[0] if rest else "")
        if cmd == "detach":
            return self.detach(rest)
        if cmd in {"exclude-session", "exclude"}:
            return self.exclude_session(rest)
        if cmd == "reassign":
            return self.reassign(rest)
        if cmd == "merge":
            return self.merge(rest)
        if cmd == "split":
            return self.split(rest)
        if cmd == "reopen":
            return self.set_status(rest, "active", "Mission reopened by manual command.")
        if cmd == "close":
            return self.close(rest)
        if cmd == "pause":
            return self.set_status(rest, "paused", "Mission paused by manual command.")
        if cmd == "block":
            return self.block(rest)
        if cmd == "inbox":
            return self.inbox()
        if cmd == "curate":
            return self.curate(rest)
        if cmd in {"conflicts", "review"}:
            return self.review()
        if cmd == "reconcile":
            return self.reconcile()
        if cmd == "repair":
            return self.repair()
        if cmd == "on":
            self.enabled = True
            return "Work Wiki automatic capture enabled for this process."
        if cmd == "off":
            self.enabled = False
            return "Work Wiki automatic capture disabled for this process."
        return HELP.rstrip()

    def status(self, work_id: str = "") -> str:
        work = self.store.get_work(work_id) if work_id else None
        if work is None and work_id:
            matches = self.store.find_missions(query=work_id, limit=1)
            work = matches[0] if matches else None
        if work is None:
            missions = self.store.recent_missions(limit=10)
            if not missions:
                return "Work Wiki is active. No missions recorded yet."
            return "Recent missions:\n" + "\n".join(self._line(m) for m in missions)
        debt = len(self.store.uncovered_events(work.work_id, limit=1000))
        latest = self.store.latest_checkpoint(work.work_id)
        meta = work.metadata
        return "\n".join(
            [
                f"{work.title}",
                f"  work_id: {work.work_id}",
                f"  status: {work.status}",
                f"  wiki: {work.wiki_path}",
                f"  current_state: {meta.get('current_state', 'not recorded')}",
                f"  next_action: {self._first(meta.get('next_actions')) or 'not recorded'}",
                f"  blockers: {', '.join(map(str, meta.get('blockers', []))) or 'none'}",
                f"  last_checkpoint: {latest.checkpoint_id if latest else 'none'}",
                f"  checkpoint_debt: {debt}",
            ]
        )

    def list_status(self, status: str = "") -> str:
        statuses = (status,) if status else ("active", "blocked", "waiting", "paused", "needs_review")
        missions = self.store.find_missions(statuses=statuses, limit=50)
        if not missions:
            return "No matching missions."
        return "\n".join(self._line(m) for m in missions)

    def switch(self, query: str) -> str:
        if not query:
            return "Usage: /wiki switch <work-id-or-query>"
        work = self.store.get_work(query)
        if work is None:
            matches = self.store.find_missions(query=query, project_root=detect_project_root(), limit=5)
            if not matches:
                return f"No mission matched {query!r}."
            work = matches[0]
        self.store.link_session(session_id=_session_id(), work_id=work.work_id, relationship="focus", focus=True)
        self.store.add_event(
            work_id=work.work_id,
            session_id=_session_id(),
            event_type="work_focus_switched",
            source="manual",
            summary=f"Focus switched to {work.title}",
            payload={"command": "switch"},
        )
        return f"Switched focus to {work.title} ({work.work_id})."

    def attach(self, work_id: str) -> str:
        if not work_id:
            return "Usage: /wiki attach <work-id>"
        work = self.store.get_work(work_id)
        if not work:
            return f"Unknown work id: {work_id}"
        count = self.store.attach_unassigned(work.work_id)
        self.store.link_session(session_id=_session_id(), work_id=work.work_id, relationship="focus", focus=True)
        self.store.enqueue_render("mission", work_id=work.work_id)
        self.store.enqueue_render("dashboards")
        self.renderer.process_pending()
        return f"Attached {count} unassigned event(s) to {work.title}."

    def detach(self, args: list[str]) -> str:
        session_id = args[0] if args else _session_id()
        work_ids = self.store.detach_session_focus(session_id)
        for work_id in work_ids:
            self.store.add_event(
                work_id=work_id,
                session_id=session_id,
                event_type="work_focus_detached",
                source="manual",
                summary=f"Session {session_id} detached from mission focus.",
                payload={"command": "detach"},
            )
            self.store.enqueue_render("mission", work_id=work_id)
        self.store.enqueue_render("dashboards")
        self.renderer.process_pending()
        if not work_ids:
            return f"No active Work Wiki focus found for session {session_id}."
        return f"Detached session {session_id} from {len(work_ids)} focused mission(s)."

    def exclude_session(self, args: list[str]) -> str:
        if not args:
            return "Usage: /wiki exclude-session <work-id> [session-id]"
        work = self.store.get_work(args[0])
        if not work:
            return f"Unknown work id: {args[0]}"
        session_id = args[1] if len(args) > 1 else _session_id()
        changed = self.store.exclude_session_from_work(session_id, work.work_id)
        self.store.add_event(
            work_id=work.work_id,
            session_id=_session_id(),
            event_type="session_excluded",
            source="manual",
            summary=f"Excluded session {session_id} from {work.title}.",
            payload={"excluded_session_id": session_id, "changed": changed},
        )
        self.store.enqueue_render("mission", work_id=work.work_id)
        self.store.enqueue_render("dashboards")
        self.renderer.process_pending()
        return f"Excluded session {session_id} from {work.title}; {changed} ledger row(s) updated."

    def reassign(self, args: list[str]) -> str:
        if len(args) < 2:
            return "Usage: /wiki reassign <mission-work-id> <project-work-id>"
        mission = self.store.get_work(args[0])
        project = self.store.get_work(args[1])
        if not mission or not project:
            return "Unknown mission or project id."
        self.store.update_work_metadata(mission.work_id, {"reassigned_at": _now(), "previous_project_work_id": mission.parent_work_id})
        with self.store.connect() as conn:
            conn.execute("UPDATE work_items SET parent_work_id=?, updated_at=? WHERE work_id=?", (project.work_id, _now(), mission.work_id))
        self.store.enqueue_render("mission", work_id=mission.work_id)
        self.store.enqueue_render("project", work_id=project.work_id)
        self.store.enqueue_render("dashboards")
        self.renderer.process_pending()
        return f"Reassigned {mission.title} to project {project.title}."

    def merge(self, args: list[str]) -> str:
        if len(args) < 2:
            return "Usage: /wiki merge <source> <canonical>"
        source = self.store.get_work(args[0])
        canonical = self.store.get_work(args[1])
        if not source or not canonical:
            return "Unknown source or canonical work id."
        with self.store.connect() as conn:
            source_meta = dict(source.metadata)
            source_meta["merged_into"] = canonical.work_id
            conn.execute("UPDATE events SET work_id=? WHERE work_id=?", (canonical.work_id, source.work_id))
            conn.execute("UPDATE checkpoints SET work_id=? WHERE work_id=?", (canonical.work_id, source.work_id))
            conn.execute("UPDATE artifacts SET work_id=? WHERE work_id=?", (canonical.work_id, source.work_id))
            conn.execute("UPDATE decisions SET work_id=? WHERE work_id=?", (canonical.work_id, source.work_id))
            conn.execute(
                "UPDATE work_items SET status='merged', closed_at=?, updated_at=?, metadata=? WHERE work_id=?",
                (_now(), _now(), self.store_json(source_meta), source.work_id),
            )
        self.store.enqueue_render("mission", work_id=canonical.work_id)
        self.store.enqueue_render("dashboards")
        self.renderer.process_pending()
        return f"Merged {source.work_id} into {canonical.work_id}; source retained as merged."

    def split(self, args: list[str]) -> str:
        if len(args) < 2:
            return "Usage: /wiki split <source-work-id> <new mission title>"
        source = self.store.get_work(args[0])
        if not source:
            return f"Unknown source work id: {args[0]}"
        title = " ".join(args[1:]).strip()
        child = self.store.create_mission(
            title=title,
            objective=title,
            project_work_id=source.parent_work_id or "",
            project_root=source.project_root,
            session_id=_session_id(),
            confidence=max(0.5, source.confidence - 0.1),
            metadata={
                "split_from": source.work_id,
                "current_state": f"Split from {source.title}.",
                "related_missions": [source.work_id],
            },
        )
        related = list(source.metadata.get("related_missions", []))
        if child.work_id not in related:
            related.append(child.work_id)
        self.store.update_work_metadata(source.work_id, {"related_missions": related})
        self.store.add_event(
            work_id=source.work_id,
            session_id=_session_id(),
            event_type="mission_split",
            source="manual",
            summary=f"Split follow-up mission {child.work_id}: {title}",
            payload={"child_work_id": child.work_id},
        )
        self.store.add_event(
            work_id=child.work_id,
            session_id=_session_id(),
            event_type="mission_created_from_split",
            source="manual",
            summary=f"Created from split of {source.work_id}",
            payload={"source_work_id": source.work_id},
        )
        self.store.enqueue_render("mission", work_id=source.work_id)
        self.store.enqueue_render("mission", work_id=child.work_id)
        self.store.enqueue_render("dashboards")
        self.renderer.process_pending()
        return f"Created split mission {child.title} ({child.work_id}) from {source.work_id}."

    def close(self, args: list[str]) -> str:
        if not args:
            return "Usage: /wiki close <work-id>"
        work = self.store.get_work(args[0])
        if not work:
            return f"Unknown work id: {args[0]}"
        latest = self.store.latest_checkpoint(work.work_id)
        artifacts = self.store.artifacts_for_work(work.work_id, limit=20)
        blockers = [str(item).strip() for item in work.metadata.get("blockers", []) if str(item).strip()]
        active_delegates = [row for row in self.store.active_delegations(limit=200) if row["work_id"] == work.work_id]
        has_evidence = bool(work.metadata.get("evidence")) or any(bool(row["verified"]) for row in artifacts)
        has_completion_checkpoint = bool(latest and latest.checkpoint_kind == "completion" and not latest.needs_review)
        if blockers:
            note = "Manual close requested with unresolved blockers; mission moved to needs_review."
            self.store.set_status(work.work_id, "needs_review", note)
            self.store.add_event(
                work_id=work.work_id,
                session_id=_session_id(),
                event_type="completion_needs_review",
                source="manual",
                summary=note,
                payload={"command": "close", "blockers": blockers},
            )
            self.store.enqueue_render("mission", work_id=work.work_id)
            self.store.enqueue_render("dashboards")
            self.renderer.process_pending()
            return f"{work.title} needs review before completion; unresolved blockers remain."
        if active_delegates:
            note = "Manual close requested while delegates are still running; mission moved to needs_review."
            self.store.set_status(work.work_id, "needs_review", note)
            self.store.add_event(
                work_id=work.work_id,
                session_id=_session_id(),
                event_type="completion_needs_review",
                source="manual",
                summary=note,
                payload={"command": "close", "active_delegations": [row["delegation_id"] for row in active_delegates]},
            )
            self.store.enqueue_render("mission", work_id=work.work_id)
            self.store.enqueue_render("dashboards")
            self.renderer.process_pending()
            return f"{work.title} needs review before completion; delegate work is still running."
        if not (has_evidence or has_completion_checkpoint):
            note = "Manual close requested without verification evidence; mission moved to needs_review."
            self.store.set_status(work.work_id, "needs_review", note)
            self.store.add_event(
                work_id=work.work_id,
                session_id=_session_id(),
                event_type="completion_needs_review",
                source="manual",
                summary=note,
                payload={"command": "close"},
            )
            self.store.enqueue_render("mission", work_id=work.work_id)
            self.store.enqueue_render("dashboards")
            self.renderer.process_pending()
            return f"{work.title} needs review before completion; add verification evidence or inspect the latest checkpoint."
        uncovered_ids = [row["event_id"] for row in self.store.uncovered_events(work.work_id, limit=1000)]
        self.store.set_status(work.work_id, "completed", "Mission manually closed with verification evidence.")
        event_id = self.store.add_event(
            work_id=work.work_id,
            session_id=_session_id(),
            event_type="mission_completed",
            source="manual",
            summary="Mission manually closed with verification evidence.",
            payload={"command": "close"},
        )
        self.store.create_checkpoint(
            work_id=work.work_id,
            session_id=_session_id(),
            checkpoint_kind="completion",
            summary="Mission manually closed with verification evidence.",
            status_after="completed",
            metadata={"verification_state": "manual", "evidence": work.metadata.get("evidence", [])},
            event_ids=uncovered_ids + [event_id],
            semantic=True,
            needs_review=False,
            confidence=0.7,
        )
        self.renderer.process_pending()
        return f"Closed {work.title} as completed."

    def set_status(self, args: list[str], status: str, note: str) -> str:
        if not args:
            return f"Usage: /wiki {status} <work-id>"
        work = self.store.get_work(args[0])
        if not work:
            return f"Unknown work id: {args[0]}"
        self.store.set_status(work.work_id, status, note)
        self.store.add_event(
            work_id=work.work_id,
            session_id=_session_id(),
            event_type=f"mission_{status}",
            source="manual",
            summary=note,
            payload={"command_status": status},
        )
        self.store.enqueue_render("mission", work_id=work.work_id)
        self.store.enqueue_render("dashboards")
        self.renderer.process_pending()
        return f"Set {work.title} to {status}."

    def block(self, args: list[str]) -> str:
        if not args:
            return "Usage: /wiki block <work-id> [reason]"
        work = self.store.get_work(args[0])
        if not work:
            return f"Unknown work id: {args[0]}"
        reason = " ".join(args[1:]) or "Blocked by manual command."
        blockers = list(work.metadata.get("blockers", []))
        if reason not in blockers:
            blockers.append(reason)
        self.store.update_work_metadata(work.work_id, {"blockers": blockers, "current_state": reason}, status="blocked")
        self.store.add_event(
            work_id=work.work_id,
            session_id=_session_id(),
            event_type="blocker_discovered",
            source="manual",
            summary=reason,
            payload={"command": "block"},
        )
        self.store.enqueue_render("mission", work_id=work.work_id)
        self.store.enqueue_render("dashboards")
        self.renderer.process_pending()
        return f"Blocked {work.title}: {reason}"

    def inbox(self) -> str:
        events = self.store.unassigned_events(limit=50)
        if not events:
            return "No unassigned material events."
        return "\n".join(f"{row['event_id']} {row['event_type']}: {row['summary']}" for row in events)

    def curate(self, args: list[str]) -> str:
        parser = argparse.ArgumentParser(prog="/wiki curate", add_help=False)
        parser.add_argument("--apply", action="store_true")
        parser.add_argument("--limit", type=int, default=8)
        parser.add_argument("--since-days", type=int, default=0)
        parser.add_argument("--work-id", default="")
        parser.add_argument("--provider", default="")
        parser.add_argument("--model", default="")
        parser.add_argument("--max-updates", type=int, default=8)
        parser.add_argument("--prompt-only", action="store_true")
        parser.add_argument("-h", "--help", action="store_true")
        try:
            parsed = parser.parse_args(args)
        except SystemExit:
            return "Usage: /wiki curate [--apply] [--limit N] [--since-days N] [--work-id ID] [--provider P] [--model M] [--max-updates N] [--prompt-only]"
        if parsed.help:
            return "Usage: /wiki curate [--apply] [--limit N] [--since-days N] [--work-id ID] [--provider P] [--model M] [--max-updates N] [--prompt-only]"
        result = KnowledgeCurator(self.store.config, self.store).run(
            apply=bool(parsed.apply),
            limit=max(1, parsed.limit),
            work_id=parsed.work_id,
            since_days=max(0, parsed.since_days),
            provider=parsed.provider,
            model=parsed.model,
            max_updates=max(1, parsed.max_updates),
            prompt_only=bool(parsed.prompt_only),
        )
        if result.applied:
            self.renderer.render_all()
        return result.to_text()

    def review(self) -> str:
        failures = self.store.render_failures(limit=20)
        unassigned = self.store.unassigned_events(limit=20)
        missions = [m for m in self.store.recent_missions(limit=100) if m.status == "needs_review"]
        parts = []
        if missions:
            parts.append("Needs review:\n" + "\n".join(self._line(m) for m in missions))
        if unassigned:
            parts.append("Unassigned events:\n" + "\n".join(f"- {row['event_id']}: {row['summary']}" for row in unassigned))
        if failures:
            parts.append("Render failures:\n" + "\n".join(f"- {row['render_job_id']}: {row['last_error']}" for row in failures))
        return "\n\n".join(parts) if parts else "No review items."

    def reconcile(self) -> str:
        self.store.enqueue_render("dashboards")
        for mission in self.store.recent_missions(limit=1000):
            self.store.enqueue_render("mission", work_id=mission.work_id)
        self.renderer.process_pending(limit=1000)
        return "Reconciled pending ledger state into Markdown."

    def repair(self) -> str:
        self.renderer.render_all()
        return "Rebuilt mission pages, project pages, dashboards, indexes, and monthly log."

    def _line(self, work: WorkItem) -> str:
        return f"- {work.work_id} [{work.status}] {work.title} -> {work.wiki_path}"

    def store_json(self, value: Any) -> str:
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    def _first(self, values: Any) -> str:
        if isinstance(values, str):
            return values
        if isinstance(values, list) and values:
            return str(values[0])
        return ""


def _session_id() -> str:
    import os

    return os.getenv("HERMES_SESSION_KEY") or os.getenv("HERMES_SESSION_ID") or "manual"


def _now() -> str:
    from .store import utc_now

    return utc_now()
