from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import WorkWikiConfig


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return {} if default is None else default
    try:
        return json.loads(value)
    except Exception:
        return {} if default is None else default


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def slugify(text: str, fallback: str = "work") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:90].strip("-") or fallback


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class WorkItem:
    work_id: str
    title: str
    slug: str
    work_kind: str
    status: str
    parent_work_id: str | None
    project_root: str
    wiki_path: str
    confidence: float
    created_at: str
    updated_at: str
    closed_at: str | None
    metadata: dict[str, Any]


@dataclass
class Checkpoint:
    checkpoint_id: str
    work_id: str
    session_id: str
    branch_id: str
    checkpoint_kind: str
    summary: str
    status_after: str
    semantic: bool
    needs_review: bool
    confidence: float
    created_at: str
    render_status: str
    render_error: str
    metadata: dict[str, Any]


class WorkWikiStore:
    def __init__(self, config: WorkWikiConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._init_lock = threading.Lock()
        self._event_lock = threading.Lock()

    @contextmanager
    def connect(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            if not self._initialized:
                with self._init_lock:
                    if not self._initialized:
                        self._migrate(conn)
                        conn.commit()
                        self._initialized = True
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS work_items (
                work_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                slug TEXT NOT NULL,
                work_kind TEXT NOT NULL,
                workstream TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                parent_work_id TEXT,
                project_root TEXT NOT NULL DEFAULT '',
                wiki_path TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                closed_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_work_items_kind_status
                ON work_items(work_kind, status, updated_at);
            CREATE INDEX IF NOT EXISTS idx_work_items_parent
                ON work_items(parent_work_id);
            CREATE INDEX IF NOT EXISTS idx_work_items_root
                ON work_items(project_root);

            CREATE TABLE IF NOT EXISTS session_links (
                session_id TEXT NOT NULL,
                work_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                focus INTEGER NOT NULL DEFAULT 0,
                lineage_root_id TEXT NOT NULL DEFAULT '',
                parent_session_id TEXT NOT NULL DEFAULT '',
                branch_id TEXT NOT NULL DEFAULT '',
                platform TEXT NOT NULL DEFAULT '',
                chat_id TEXT NOT NULL DEFAULT '',
                activated_at TEXT NOT NULL,
                deactivated_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(session_id, work_id, branch_id)
            );

            CREATE INDEX IF NOT EXISTS idx_session_links_session_focus
                ON session_links(session_id, focus, activated_at);
            CREATE INDEX IF NOT EXISTS idx_session_links_work
                ON session_links(work_id, activated_at);

            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                work_id TEXT,
                session_id TEXT NOT NULL DEFAULT '',
                branch_id TEXT NOT NULL DEFAULT '',
                turn_id TEXT NOT NULL DEFAULT '',
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                source TEXT NOT NULL,
                tool_name TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                observed_at TEXT NOT NULL,
                checkpoint_id TEXT,
                redacted INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_events_work_debt
                ON events(work_id, checkpoint_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_events_session
                ON events(session_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_events_unassigned
                ON events(work_id, event_type, observed_at);

            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                work_id TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                branch_id TEXT NOT NULL DEFAULT '',
                checkpoint_kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                status_after TEXT NOT NULL,
                semantic INTEGER NOT NULL DEFAULT 1,
                needs_review INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                render_status TEXT NOT NULL DEFAULT 'pending',
                render_error TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_work_created
                ON checkpoints(work_id, created_at);

            CREATE TABLE IF NOT EXISTS checkpoint_events (
                checkpoint_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(checkpoint_id, event_id)
            );

            CREATE TABLE IF NOT EXISTS delegations (
                delegation_id TEXT PRIMARY KEY,
                parent_session_id TEXT NOT NULL DEFAULT '',
                child_session_id TEXT NOT NULL DEFAULT '',
                work_id TEXT NOT NULL DEFAULT '',
                branch_id TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT '',
                goal TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                result_summary TEXT NOT NULL DEFAULT '',
                artifacts TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                work_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL DEFAULT '',
                artifact_type TEXT NOT NULL,
                path_or_reference TEXT NOT NULL,
                content_hash TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                work_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL DEFAULT '',
                decision TEXT NOT NULL,
                rationale TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                superseded_by TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS render_jobs (
                render_job_id TEXT PRIMARY KEY,
                work_id TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL DEFAULT '',
                job_type TEXT NOT NULL,
                target_path TEXT NOT NULL DEFAULT '',
                payload TEXT NOT NULL DEFAULT '{}',
                state TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("schema_version", "1"),
        )

    def ensure_project(
        self,
        *,
        title: str,
        project_root: str = "",
        confidence: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> WorkItem:
        normalized = title.lower().strip()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM work_items
                WHERE work_kind='project'
                  AND (normalized_title=? OR (? != '' AND project_root=?))
                ORDER BY CASE WHEN normalized_title=? THEN 0 ELSE 1 END, updated_at DESC
                LIMIT 1
                """,
                (normalized, project_root, project_root, normalized),
            ).fetchone()
            if row:
                return self._work_from_row(row)

            now = utc_now()
            work_id = new_id("prj")
            slug = slugify(title, "project")
            wiki_path = f"work/projects/{slug}.md"
            conn.execute(
                """
                INSERT INTO work_items(
                    work_id, title, normalized_title, slug, work_kind, status,
                    parent_work_id, project_root, wiki_path, confidence,
                    created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, 'project', 'active', NULL, ?, ?, ?, ?, ?, ?)
                """,
                (
                    work_id,
                    title,
                    normalized,
                    slug,
                    project_root,
                    wiki_path,
                    confidence,
                    now,
                    now,
                    _json_dumps(metadata or {}),
                ),
            )
            return self.get_work(work_id, conn=conn)

    def create_mission(
        self,
        *,
        title: str,
        objective: str,
        project_work_id: str,
        project_root: str = "",
        session_id: str = "",
        branch_id: str = "",
        confidence: float = 0.75,
        metadata: dict[str, Any] | None = None,
    ) -> WorkItem:
        now = utc_now()
        base_slug = slugify(title, "mission")
        work_id = new_id("wrk")
        wiki_path = f"work/missions/{base_slug}-{work_id[-6:]}.md"
        meta = dict(metadata or {})
        meta.setdefault("objective", objective)
        meta.setdefault("definition_of_done", [])
        meta.setdefault("current_state", "Mission created automatically; work has started.")
        meta.setdefault("next_actions", [])
        meta.setdefault("blockers", [])
        meta.setdefault("artifacts", [])
        meta.setdefault("findings", [])
        meta.setdefault("decisions", [])
        meta.setdefault("evidence", [])
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO work_items(
                    work_id, title, normalized_title, slug, work_kind, status,
                    parent_work_id, project_root, wiki_path, confidence,
                    created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, 'mission', 'active', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    work_id,
                    title,
                    title.lower().strip(),
                    base_slug,
                    project_work_id,
                    project_root,
                    wiki_path,
                    confidence,
                    now,
                    now,
                    _json_dumps(meta),
                ),
            )
            if session_id:
                self.link_session(
                    session_id=session_id,
                    work_id=work_id,
                    branch_id=branch_id,
                    relationship="focus",
                    focus=True,
                    conn=conn,
                )
            return self.get_work(work_id, conn=conn)

    def get_work(self, work_id: str, conn: sqlite3.Connection | None = None) -> WorkItem | None:
        close = False
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            close = True
        try:
            row = conn.execute("SELECT * FROM work_items WHERE work_id=?", (work_id,)).fetchone()
            return self._work_from_row(row) if row else None
        finally:
            if close:
                conn.close()

    def update_work_metadata(self, work_id: str, updates: dict[str, Any], *, status: str | None = None) -> None:
        with self.connect() as conn:
            row = conn.execute("SELECT metadata FROM work_items WHERE work_id=?", (work_id,)).fetchone()
            if not row:
                return
            meta = _json_loads(row["metadata"])
            meta.update({k: v for k, v in updates.items() if v is not None})
            now = utc_now()
            if status:
                closed_at = now if status == "completed" else None
                conn.execute(
                    "UPDATE work_items SET metadata=?, status=?, updated_at=?, closed_at=? WHERE work_id=?",
                    (_json_dumps(meta), status, now, closed_at, work_id),
                )
            else:
                conn.execute(
                    "UPDATE work_items SET metadata=?, updated_at=? WHERE work_id=?",
                    (_json_dumps(meta), now, work_id),
                )

    def set_status(self, work_id: str, status: str, note: str = "") -> None:
        updates: dict[str, Any] = {}
        if note:
            updates["current_state"] = note
        self.update_work_metadata(work_id, updates, status=status)

    def link_session(
        self,
        *,
        session_id: str,
        work_id: str,
        branch_id: str = "",
        relationship: str = "related",
        focus: bool = False,
        lineage_root_id: str = "",
        parent_session_id: str = "",
        platform: str = "",
        chat_id: str = "",
        metadata: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not session_id or not work_id:
            return
        close = False
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            close = True
        try:
            now = utc_now()
            if focus:
                conn.execute(
                    "UPDATE session_links SET focus=0 WHERE session_id=? AND branch_id=?",
                    (session_id, branch_id),
                )
            conn.execute(
                """
                INSERT INTO session_links(
                    session_id, work_id, relationship, focus, lineage_root_id,
                    parent_session_id, branch_id, platform, chat_id, activated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, work_id, branch_id) DO UPDATE SET
                    relationship=CASE
                        WHEN session_links.relationship IN ('continuation', 'delegate')
                         AND excluded.relationship='focus'
                        THEN session_links.relationship
                        ELSE excluded.relationship
                    END,
                    focus=excluded.focus,
                    lineage_root_id=COALESCE(NULLIF(excluded.lineage_root_id, ''), session_links.lineage_root_id),
                    parent_session_id=COALESCE(NULLIF(excluded.parent_session_id, ''), session_links.parent_session_id),
                    platform=COALESCE(NULLIF(excluded.platform, ''), session_links.platform),
                    chat_id=COALESCE(NULLIF(excluded.chat_id, ''), session_links.chat_id),
                    activated_at=excluded.activated_at,
                    deactivated_at=NULL,
                    metadata=excluded.metadata
                """,
                (
                    session_id,
                    work_id,
                    relationship,
                    1 if focus else 0,
                    lineage_root_id,
                    parent_session_id,
                    branch_id,
                    platform,
                    chat_id,
                    now,
                    _json_dumps(metadata or {}),
                ),
            )
            if close:
                conn.commit()
        finally:
            if close:
                conn.close()

    def focus_for_session(self, session_id: str, branch_id: str = "") -> WorkItem | None:
        if not session_id:
            return None
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT w.* FROM session_links s
                JOIN work_items w ON w.work_id=s.work_id
                WHERE s.session_id=? AND (?='' OR s.branch_id=?) AND s.focus=1
                  AND s.deactivated_at IS NULL
                  AND w.work_kind='mission'
                ORDER BY s.activated_at DESC
                LIMIT 1
                """,
                (session_id, branch_id, branch_id),
            ).fetchone()
            return self._work_from_row(row) if row else None

    def focus_for_parent_session(self, parent_session_id: str, branch_id: str = "") -> WorkItem | None:
        if not parent_session_id:
            return None
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT w.* FROM session_links s
                JOIN work_items w ON w.work_id=s.work_id
                WHERE s.session_id=? AND (?='' OR s.branch_id=?) AND s.focus=1
                  AND s.deactivated_at IS NULL
                  AND w.work_kind='mission'
                ORDER BY s.activated_at DESC
                LIMIT 1
                """,
                (parent_session_id, branch_id, branch_id),
            ).fetchone()
            if row:
                return self._work_from_row(row)
            row = conn.execute(
                """
                SELECT w.* FROM session_links s
                JOIN work_items w ON w.work_id=s.work_id
                WHERE s.parent_session_id=? AND (?='' OR s.branch_id=?)
                  AND s.deactivated_at IS NULL
                  AND w.work_kind='mission'
                ORDER BY s.activated_at DESC
                LIMIT 1
                """,
                (parent_session_id, branch_id, branch_id),
            ).fetchone()
            return self._work_from_row(row) if row else None

    def deactivate_session(self, session_id: str, branch_id: str = "", relationship: str = "") -> int:
        if not session_id:
            return 0
        with self.connect() as conn:
            now = utc_now()
            clauses = ["session_id=?", "deactivated_at IS NULL"]
            params: list[Any] = [session_id]
            if branch_id:
                clauses.append("branch_id=?")
                params.append(branch_id)
            if relationship:
                clauses.append("relationship=?")
                params.append(relationship)
            conn.execute(
                f"UPDATE session_links SET focus=0, deactivated_at=? WHERE {' AND '.join(clauses)}",
                (now, *params),
            )
            return conn.total_changes

    def detach_session_focus(self, session_id: str, branch_id: str = "") -> list[str]:
        if not session_id:
            return []
        with self.connect() as conn:
            now = utc_now()
            params: list[Any] = [session_id]
            branch_clause = ""
            if branch_id:
                branch_clause = "AND branch_id=?"
                params.append(branch_id)
            rows = conn.execute(
                f"""
                SELECT DISTINCT work_id FROM session_links
                WHERE session_id=? {branch_clause}
                  AND focus=1
                  AND deactivated_at IS NULL
                """,
                params,
            ).fetchall()
            work_ids = [row["work_id"] for row in rows]
            conn.execute(
                f"""
                UPDATE session_links
                SET focus=0, deactivated_at=?, relationship='detached'
                WHERE session_id=? {branch_clause}
                  AND focus=1
                  AND deactivated_at IS NULL
                """,
                (now, *params),
            )
            return work_ids

    def exclude_session_from_work(self, session_id: str, work_id: str, branch_id: str = "") -> int:
        if not session_id or not work_id:
            return 0
        with self.connect() as conn:
            now = utc_now()
            branch_clause = "AND branch_id=?" if branch_id else ""
            params: list[Any] = [session_id, work_id]
            if branch_id:
                params.append(branch_id)
            conn.execute(
                f"""
                UPDATE session_links
                SET focus=0, deactivated_at=?, relationship='excluded'
                WHERE session_id=? AND work_id=? {branch_clause}
                """,
                (now, *params),
            )
            event_params: list[Any] = [work_id, session_id]
            event_branch_clause = "AND branch_id=?" if branch_id else ""
            if branch_id:
                event_params.append(branch_id)
            conn.execute(
                f"""
                UPDATE events
                SET work_id=NULL, checkpoint_id=NULL
                WHERE work_id=? AND session_id=? {event_branch_clause}
                """,
                event_params,
            )
            conn.execute(
                """
                DELETE FROM checkpoint_events
                WHERE event_id IN (
                    SELECT event_id FROM events
                    WHERE work_id IS NULL AND session_id=?
                )
                """,
                (session_id,),
            )
            conn.execute(
                "UPDATE work_items SET updated_at=? WHERE work_id=?",
                (now, work_id),
            )
            return conn.total_changes

    def find_missions(
        self,
        *,
        query: str = "",
        project_root: str = "",
        statuses: tuple[str, ...] = ("active", "blocked", "waiting", "paused", "needs_review"),
        limit: int = 20,
    ) -> list[WorkItem]:
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        with self.connect() as conn:
            clauses = ["work_kind='mission'"]
            params: list[Any] = []
            if statuses:
                clauses.append("status IN ({})".format(",".join("?" for _ in statuses)))
                params.extend(statuses)
            if project_root:
                clauses.append("(project_root=? OR project_root='')")
                params.append(project_root)
            rows = conn.execute(
                f"SELECT * FROM work_items WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?",
                (*params, max(limit * 4, limit)),
            ).fetchall()
        items = [self._work_from_row(row) for row in rows]
        if not terms:
            return items[:limit]

        def score(item: WorkItem) -> int:
            hay = " ".join(
                [
                    item.title.lower(),
                    item.project_root.lower(),
                    str(item.metadata.get("objective", "")).lower(),
                    " ".join(map(str, item.metadata.get("aliases", []))),
                ]
            )
            return sum(2 if term in item.title.lower() else 1 for term in terms if term in hay)

        ranked = sorted(((score(item), item) for item in items), key=lambda pair: pair[0], reverse=True)
        return [item for s, item in ranked if s > 0][:limit]

    def recent_missions(self, limit: int = 20) -> list[WorkItem]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM work_items
                WHERE work_kind='mission'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._work_from_row(row) for row in rows]

    def add_event(
        self,
        *,
        work_id: str | None,
        session_id: str = "",
        branch_id: str = "",
        turn_id: str = "",
        event_type: str,
        source: str,
        tool_name: str = "",
        summary: str,
        payload: dict[str, Any] | None = None,
        redacted: bool = False,
        checkpoint_id: str | None = None,
    ) -> str:
        payload_text = _json_dumps(payload or {})
        if len(payload_text) > self.config.max_event_payload_chars:
            payload_text = payload_text[: self.config.max_event_payload_chars] + "...[truncated]"
            redacted = True
        event_id = new_id("evt")
        with self._event_lock:
            with self.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                seq = conn.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM events").fetchone()[0]
                now = utc_now()
                conn.execute(
                    """
                    INSERT INTO events(
                        event_id, work_id, session_id, branch_id, turn_id, sequence,
                        event_type, source, tool_name, summary, payload, observed_at,
                        checkpoint_id, redacted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        work_id,
                        session_id,
                        branch_id,
                        turn_id,
                        seq,
                        event_type,
                        source,
                        tool_name,
                        summary,
                        payload_text,
                        now,
                        checkpoint_id,
                        1 if redacted else 0,
                    ),
                )
                if checkpoint_id:
                    conn.execute(
                        "INSERT OR IGNORE INTO checkpoint_events(checkpoint_id, event_id) VALUES (?, ?)",
                        (checkpoint_id, event_id),
                    )
                if work_id:
                    conn.execute(
                        "UPDATE work_items SET updated_at=? WHERE work_id=?",
                        (now, work_id),
                    )
        return event_id

    def uncovered_events(self, work_id: str, branch_id: str = "", limit: int = 200) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM events
                WHERE work_id=? AND checkpoint_id IS NULL AND (?='' OR branch_id=?)
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (work_id, branch_id, branch_id, limit),
            ).fetchall()

    def unassigned_events(self, limit: int = 50) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM events
                WHERE work_id IS NULL
                ORDER BY sequence DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def create_checkpoint(
        self,
        *,
        work_id: str,
        session_id: str = "",
        branch_id: str = "",
        checkpoint_kind: str,
        summary: str,
        status_after: str,
        metadata: dict[str, Any] | None = None,
        event_ids: list[str] | None = None,
        semantic: bool = True,
        needs_review: bool = False,
        confidence: float = 0.8,
    ) -> str:
        checkpoint_id = new_id("chk")
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints(
                    checkpoint_id, work_id, session_id, branch_id, checkpoint_kind,
                    summary, status_after, semantic, needs_review, confidence,
                    created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    work_id,
                    session_id,
                    branch_id,
                    checkpoint_kind,
                    summary,
                    status_after,
                    1 if semantic else 0,
                    1 if needs_review else 0,
                    confidence,
                    now,
                    _json_dumps(metadata or {}),
                ),
            )
            ids = list(event_ids or [])
            if ids:
                conn.executemany(
                    "INSERT OR IGNORE INTO checkpoint_events(checkpoint_id, event_id) VALUES (?, ?)",
                    [(checkpoint_id, event_id) for event_id in ids],
                )
                conn.execute(
                    "UPDATE events SET checkpoint_id=? WHERE event_id IN ({})".format(
                        ",".join("?" for _ in ids)
                    ),
                    (checkpoint_id, *ids),
                )
            row = conn.execute("SELECT metadata FROM work_items WHERE work_id=?", (work_id,)).fetchone()
            work_meta = _json_loads(row["metadata"]) if row else {}
            work_meta["last_checkpoint_id"] = checkpoint_id
            conn.execute(
                "UPDATE work_items SET status=?, updated_at=?, metadata=? WHERE work_id=?",
                (status_after, now, _json_dumps(work_meta), work_id),
            )
            self.enqueue_render("mission", work_id=work_id, checkpoint_id=checkpoint_id, conn=conn)
            self.enqueue_render("dashboards", conn=conn)
        return checkpoint_id

    def checkpoints_for_work(self, work_id: str, limit: int = 20) -> list[Checkpoint]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE work_id=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (work_id, limit),
            ).fetchall()
            return [self._checkpoint_from_row(row) for row in rows]

    def latest_checkpoint(self, work_id: str) -> Checkpoint | None:
        cps = self.checkpoints_for_work(work_id, limit=1)
        return cps[0] if cps else None

    def add_artifact(self, work_id: str, path_or_reference: str, *, checkpoint_id: str = "", description: str = "", verified: bool = False) -> None:
        if not path_or_reference:
            return
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT artifact_id FROM artifacts WHERE work_id=? AND path_or_reference=?",
                (work_id, path_or_reference),
            ).fetchone()
            if existing:
                return
            conn.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, work_id, checkpoint_id, artifact_type, path_or_reference,
                    content_hash, description, created_at, verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("art"),
                    work_id,
                    checkpoint_id,
                    "file" if path_or_reference.startswith("/") else "reference",
                    path_or_reference,
                    stable_hash(path_or_reference),
                    description,
                    utc_now(),
                    1 if verified else 0,
                ),
            )

    def artifacts_for_work(self, work_id: str, limit: int = 50) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM artifacts WHERE work_id=? ORDER BY created_at DESC LIMIT ?",
                (work_id, limit),
            ).fetchall()

    def add_decision(self, work_id: str, decision: str, rationale: str = "", checkpoint_id: str = "") -> None:
        if not decision:
            return
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT decision_id FROM decisions WHERE work_id=? AND decision=?",
                (work_id, decision),
            ).fetchone()
            if existing:
                return
            conn.execute(
                """
                INSERT INTO decisions(decision_id, work_id, checkpoint_id, decision, rationale, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (new_id("dec"), work_id, checkpoint_id, decision, rationale, utc_now()),
            )

    def decisions_for_work(self, work_id: str, limit: int = 50) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM decisions WHERE work_id=? ORDER BY created_at DESC LIMIT ?",
                (work_id, limit),
            ).fetchall()

    def start_delegation(
        self,
        *,
        parent_session_id: str,
        child_session_id: str = "",
        work_id: str = "",
        branch_id: str = "",
        role: str = "",
        goal: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = utc_now()
        delegation_id = new_id("dlg")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO delegations(
                    delegation_id, parent_session_id, child_session_id, work_id,
                    branch_id, role, goal, state, started_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (
                    delegation_id,
                    parent_session_id,
                    child_session_id,
                    work_id,
                    branch_id,
                    role,
                    goal,
                    now,
                    _json_dumps(metadata or {}),
                ),
            )
            if work_id:
                conn.execute("UPDATE work_items SET updated_at=? WHERE work_id=?", (now, work_id))
        return delegation_id

    def finish_delegation(
        self,
        *,
        parent_session_id: str,
        child_session_id: str = "",
        work_id: str = "",
        role: str = "",
        status: str = "",
        summary: str = "",
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        state = status or "completed"
        now = utc_now()
        metadata = dict(metadata or {})
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
        with self.connect() as conn:
            row = None
            if child_session_id:
                row = conn.execute(
                    """
                    SELECT * FROM delegations
                    WHERE parent_session_id=? AND child_session_id=? AND state='running'
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (parent_session_id, child_session_id),
                ).fetchone()
            if row is None:
                row = conn.execute(
                    """
                    SELECT * FROM delegations
                    WHERE parent_session_id=? AND state='running'
                      AND (?='' OR role=?)
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (parent_session_id, role, role),
                ).fetchone()
            if row is None:
                delegation_id = new_id("dlg")
                conn.execute(
                    """
                    INSERT INTO delegations(
                        delegation_id, parent_session_id, child_session_id, work_id,
                        role, goal, state, started_at, finished_at,
                        result_summary, metadata
                    ) VALUES (?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?)
                    """,
                    (
                        delegation_id,
                        parent_session_id,
                        child_session_id,
                        work_id,
                        role,
                        state,
                        now,
                        now,
                        summary,
                        _json_dumps(metadata),
                    ),
                )
            else:
                delegation_id = row["delegation_id"]
                existing_meta = _json_loads(row["metadata"])
                existing_meta.update(metadata)
                resolved_work_id = work_id or row["work_id"]
                conn.execute(
                    """
                    UPDATE delegations
                    SET state=?, finished_at=?, result_summary=?, metadata=?,
                        child_session_id=COALESCE(NULLIF(?, ''), child_session_id),
                        work_id=COALESCE(NULLIF(?, ''), work_id),
                        role=COALESCE(NULLIF(?, ''), role)
                    WHERE delegation_id=?
                    """,
                    (
                        state,
                        now,
                        summary,
                        _json_dumps(existing_meta),
                        child_session_id,
                        resolved_work_id,
                        role,
                        delegation_id,
                    ),
                )
            updated_work_id = work_id or (row["work_id"] if row else "")
            if updated_work_id:
                conn.execute("UPDATE work_items SET updated_at=? WHERE work_id=?", (now, updated_work_id))
        return delegation_id

    def active_delegations(self, limit: int = 50) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM delegations
                WHERE state='running'
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def duplicate_mission_candidates(self, limit: int = 25) -> list[tuple[WorkItem, WorkItem, str]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT a.*,
                       b.work_id AS b_work_id,
                       b.title AS b_title,
                       b.slug AS b_slug,
                       b.work_kind AS b_work_kind,
                       b.status AS b_status,
                       b.parent_work_id AS b_parent_work_id,
                       b.project_root AS b_project_root,
                       b.wiki_path AS b_wiki_path,
                       b.confidence AS b_confidence,
                       b.created_at AS b_created_at,
                       b.updated_at AS b_updated_at,
                       b.closed_at AS b_closed_at,
                       b.metadata AS b_metadata
                FROM work_items a
                JOIN work_items b
                  ON a.work_kind='mission'
                 AND b.work_kind='mission'
                 AND a.work_id < b.work_id
                 AND a.normalized_title=b.normalized_title
                 AND COALESCE(a.project_root, '')=COALESCE(b.project_root, '')
                 AND a.status NOT IN ('merged', 'completed')
                 AND b.status NOT IN ('merged', 'completed')
                ORDER BY a.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        candidates: list[tuple[WorkItem, WorkItem, str]] = []
        for row in rows:
            left = self._work_from_row(row)
            right_data = {
                "work_id": row["b_work_id"],
                "title": row["b_title"],
                "slug": row["b_slug"],
                "work_kind": row["b_work_kind"],
                "status": row["b_status"],
                "parent_work_id": row["b_parent_work_id"],
                "project_root": row["b_project_root"],
                "wiki_path": row["b_wiki_path"],
                "confidence": row["b_confidence"],
                "created_at": row["b_created_at"],
                "updated_at": row["b_updated_at"],
                "closed_at": row["b_closed_at"],
                "metadata": row["b_metadata"],
            }
            right = WorkItem(
                work_id=str(right_data["work_id"]),
                title=str(right_data["title"]),
                slug=str(right_data["slug"]),
                work_kind=str(right_data["work_kind"]),
                status=str(right_data["status"]),
                parent_work_id=right_data["parent_work_id"],
                project_root=str(right_data["project_root"]),
                wiki_path=str(right_data["wiki_path"]),
                confidence=float(right_data["confidence"] or 0),
                created_at=str(right_data["created_at"]),
                updated_at=str(right_data["updated_at"]),
                closed_at=right_data["closed_at"],
                metadata=_json_loads(str(right_data["metadata"])),
            )
            if left and right:
                candidates.append((left, right, "same normalized title and project root"))
        return candidates

    def branch_conflicts(self, work_id: str = "", *, limit: int = 50) -> list[sqlite3.Row]:
        with self.connect() as conn:
            work_clause = "AND s.work_id=?" if work_id else ""
            params: list[Any] = []
            if work_id:
                params.append(work_id)
            params.append(limit)
            return conn.execute(
                f"""
                WITH branch_activity AS (
                    SELECT
                        s.work_id,
                        COALESCE(NULLIF(s.lineage_root_id, ''), NULLIF(s.parent_session_id, ''), s.session_id) AS lineage_key,
                        COALESCE(NULLIF(s.branch_id, ''), 'default') AS branch_id,
                        MIN(s.activated_at) AS first_seen,
                        MAX(COALESCE(e.observed_at, '')) AS latest_event,
                        MAX(COALESCE(c.created_at, '')) AS latest_checkpoint,
                        COUNT(DISTINCT s.session_id) AS session_count,
                        SUM(CASE WHEN e.checkpoint_id IS NULL AND e.event_id IS NOT NULL THEN 1 ELSE 0 END) AS uncovered_events
                    FROM session_links s
                    LEFT JOIN events e
                      ON e.work_id=s.work_id
                     AND COALESCE(NULLIF(e.branch_id, ''), 'default')=COALESCE(NULLIF(s.branch_id, ''), 'default')
                    LEFT JOIN checkpoints c
                      ON c.work_id=s.work_id
                     AND COALESCE(NULLIF(c.branch_id, ''), 'default')=COALESCE(NULLIF(s.branch_id, ''), 'default')
                    WHERE s.deactivated_at IS NULL
                      AND COALESCE(NULLIF(s.branch_id, ''), 'default') != 'default'
                      {work_clause}
                    GROUP BY s.work_id, lineage_key, COALESCE(NULLIF(s.branch_id, ''), 'default')
                )
                SELECT
                    a.work_id,
                    a.lineage_key,
                    a.branch_id AS older_branch_id,
                    b.branch_id AS newer_branch_id,
                    MAX(a.latest_event, a.latest_checkpoint, a.first_seen) AS older_latest_activity,
                    MAX(b.latest_event, b.latest_checkpoint, b.first_seen) AS newer_latest_activity,
                    a.uncovered_events AS older_uncovered_events,
                    b.uncovered_events AS newer_uncovered_events,
                    a.session_count AS older_session_count,
                    b.session_count AS newer_session_count
                FROM branch_activity a
                JOIN branch_activity b
                  ON a.work_id=b.work_id
                 AND a.lineage_key=b.lineage_key
                 AND a.branch_id<>b.branch_id
                 AND MAX(a.latest_event, a.latest_checkpoint, a.first_seen) < MAX(b.latest_event, b.latest_checkpoint, b.first_seen)
                ORDER BY newer_latest_activity DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

    def branch_conflicts_for_branch(self, work_id: str, branch_id: str, *, limit: int = 10) -> list[sqlite3.Row]:
        if not work_id or not branch_id or branch_id == "default":
            return []
        return [
            row for row in self.branch_conflicts(work_id, limit=limit * 4)
            if row["older_branch_id"] == branch_id
        ][:limit]

    def enqueue_render(
        self,
        job_type: str,
        *,
        work_id: str = "",
        checkpoint_id: str = "",
        target_path: str = "",
        payload: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        close = False
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            close = True
        try:
            now = utc_now()
            conn.execute(
                """
                INSERT INTO render_jobs(
                    render_job_id, work_id, checkpoint_id, job_type, target_path,
                    payload, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    new_id("rj"),
                    work_id,
                    checkpoint_id,
                    job_type,
                    target_path,
                    _json_dumps(payload or {}),
                    now,
                    now,
                ),
            )
            if close:
                conn.commit()
        finally:
            if close:
                conn.close()

    def pending_render_jobs(self, limit: int = 50) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM render_jobs
                WHERE state IN ('pending', 'failed')
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def mark_render_job(self, render_job_id: str, state: str, error: str = "") -> None:
        with self.connect() as conn:
            now = utc_now()
            completed_at = now if state == "completed" else None
            conn.execute(
                """
                UPDATE render_jobs
                SET state=?, attempt_count=attempt_count+1, last_error=?,
                    updated_at=?, completed_at=?
                WHERE render_job_id=?
                """,
                (state, error, now, completed_at, render_job_id),
            )

    def render_failures(self, limit: int = 50) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM render_jobs
                WHERE state='failed'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def attach_unassigned(self, work_id: str, event_ids: list[str] | None = None) -> int:
        with self.connect() as conn:
            if event_ids:
                conn.execute(
                    "UPDATE events SET work_id=? WHERE event_id IN ({})".format(",".join("?" for _ in event_ids)),
                    (work_id, *event_ids),
                )
                return conn.total_changes
            conn.execute("UPDATE events SET work_id=? WHERE work_id IS NULL", (work_id,))
            return conn.total_changes

    def _work_from_row(self, row: sqlite3.Row | None) -> WorkItem | None:
        if row is None:
            return None
        return WorkItem(
            work_id=row["work_id"],
            title=row["title"],
            slug=row["slug"],
            work_kind=row["work_kind"],
            status=row["status"],
            parent_work_id=row["parent_work_id"],
            project_root=row["project_root"],
            wiki_path=row["wiki_path"],
            confidence=float(row["confidence"] or 0),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            closed_at=row["closed_at"],
            metadata=_json_loads(row["metadata"]),
        )

    def _checkpoint_from_row(self, row: sqlite3.Row) -> Checkpoint:
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            work_id=row["work_id"],
            session_id=row["session_id"],
            branch_id=row["branch_id"],
            checkpoint_kind=row["checkpoint_kind"],
            summary=row["summary"],
            status_after=row["status_after"],
            semantic=bool(row["semantic"]),
            needs_review=bool(row["needs_review"]),
            confidence=float(row["confidence"] or 0),
            created_at=row["created_at"],
            render_status=row["render_status"],
            render_error=row["render_error"],
            metadata=_json_loads(row["metadata"]),
        )
