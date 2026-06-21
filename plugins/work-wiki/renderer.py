from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import WorkWikiConfig
from .store import Checkpoint, WorkItem, WorkWikiStore, utc_now


SECTION_NAMES = [
    "Objective",
    "Definition of Done",
    "Status",
    "Current State",
    "Active Tasks",
    "Decisions",
    "Findings",
    "Evidence and Verification",
    "Artifacts",
    "Changed Systems and Files",
    "Blockers",
    "Delegates",
    "Next Actions",
    "Related Knowledge",
    "Sessions and Branches",
    "Checkpoints",
]


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    if not text or any(c in text for c in ":#[]{}&*!|>'\"%@`"):
        return f'"{text}"'
    return text


def _yaml_list(values: Any) -> list[str]:
    if not values:
        return ["[]"]
    if isinstance(values, str):
        values = [values]
    return [""] + [f"  - {_yaml_scalar(v)}" for v in values]


def _frontmatter(data: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in data.items():
        if isinstance(value, (list, tuple, set)):
            item_lines = _yaml_list(list(value))
            if item_lines == ["[]"]:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                lines.extend(item_lines[1:])
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _bullet(values: Any, empty: str = "None recorded.") -> str:
    if values is None:
        return empty
    if isinstance(values, str):
        values = [line.strip() for line in values.splitlines() if line.strip()] or [values] if values.strip() else []
    if not values:
        return empty
    return "\n".join(f"- {str(v).strip()}" for v in values if str(v).strip())


def _relative_link(root: Path, path: str) -> str:
    return path.replace("\\", "/")


def _managed_block(name: str, content: str) -> str:
    return (
        f"<!-- work-wiki:generated-{name}:start -->\n"
        f"{content.rstrip()}\n"
        f"<!-- work-wiki:generated-{name}:end -->"
    )


def _replace_block(existing: str, name: str, content: str) -> tuple[str, bool]:
    start = f"<!-- work-wiki:generated-{name}:start -->"
    end = f"<!-- work-wiki:generated-{name}:end -->"
    if existing.count(start) != existing.count(end):
        return existing, False
    block = _managed_block(name, content)
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(existing):
        return pattern.sub(block, existing), True
    if existing and not existing.endswith("\n"):
        existing += "\n"
    return existing + "\n" + block + "\n", True


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


class MarkdownRenderer:
    def __init__(self, config: WorkWikiConfig, store: WorkWikiStore):
        self.config = config
        self.store = store
        self.root = Path(config.wiki_root)

    def ensure_layout(self) -> None:
        for rel in (
            "work/projects",
            "work/missions",
            "work/tasks",
            "work/inbox",
            "work/_indexes",
            "work/_archive",
            "logs",
            "entities",
            "concepts",
            "sources",
            "syntheses",
        ):
            (self.root / rel).mkdir(parents=True, exist_ok=True)
        schema = self.root / "SCHEMA.md"
        if not schema.exists():
            _atomic_write(schema, self._schema_text())

    def process_pending(self, limit: int = 50) -> None:
        if not self.config.auto_render_wiki:
            return
        self.ensure_layout()
        for job in self.store.pending_render_jobs(limit=limit):
            try:
                kind = job["job_type"]
                if kind == "mission":
                    work_id = job["work_id"]
                    work = self.store.get_work(work_id)
                    if work:
                        self.render_mission(work)
                        if work.parent_work_id:
                            project = self.store.get_work(work.parent_work_id)
                            if project:
                                self.render_project(project)
                elif kind == "project":
                    work = self.store.get_work(job["work_id"])
                    if work:
                        self.render_project(work)
                elif kind == "dashboards":
                    self.render_dashboards()
                    self.render_indexes()
                    self.render_monthly_log()
                else:
                    self.render_dashboards()
                self.store.mark_render_job(job["render_job_id"], "completed")
            except Exception as exc:
                self.store.mark_render_job(job["render_job_id"], "failed", str(exc))

    def render_all(self) -> None:
        self.ensure_layout()
        for mission in self.store.recent_missions(limit=10000):
            self.render_mission(mission)
            if mission.parent_work_id:
                project = self.store.get_work(mission.parent_work_id)
                if project:
                    self.render_project(project)
        self.render_dashboards()
        self.render_indexes()
        self.render_monthly_log()

    def render_mission(self, work: WorkItem) -> Path:
        path = self.root / work.wiki_path
        project = self.store.get_work(work.parent_work_id) if work.parent_work_id else None
        checkpoints = list(reversed(self.store.checkpoints_for_work(work.work_id, limit=50)))
        artifacts = self.store.artifacts_for_work(work.work_id, limit=50)
        decisions = self.store.decisions_for_work(work.work_id, limit=50)
        events = self.store.uncovered_events(work.work_id, limit=25)
        meta = work.metadata

        front = _frontmatter(
            {
                "title": work.title,
                "type": "work",
                "work_id": work.work_id,
                "work_kind": "mission",
                "project_work_id": work.parent_work_id or "",
                "workstream": meta.get("workstream", ""),
                "status": work.status,
                "confidence": work.confidence,
                "created_at": work.created_at,
                "updated_at": work.updated_at,
                "last_checkpoint_id": meta.get("last_checkpoint_id", ""),
                "project_roots": [work.project_root] if work.project_root else [],
                "aliases": meta.get("aliases", []),
                "tags": meta.get("tags", ["mission-memory"]),
            }
        )
        body = f"# {work.title}\n\n"
        sections = {
            "objective": str(meta.get("objective") or work.title),
            "definition-of-done": _bullet(meta.get("definition_of_done"), "Not yet specified."),
            "status": "\n".join(
                [
                    f"- Status: {work.status}",
                    f"- Project: {project.title if project else 'Unassigned'}",
                    f"- Work ID: `{work.work_id}`",
                    f"- Updated: {work.updated_at}",
                    f"- Confidence: {work.confidence:.2f}",
                ]
            ),
            "current-state": str(meta.get("current_state") or "No current state recorded."),
            "active-tasks": _bullet(meta.get("active_tasks") or meta.get("tasks"), "No active tasks recorded."),
            "decisions": _bullet(
                [f"{row['decision']}" + (f" - {row['rationale']}" if row["rationale"] else "") for row in decisions]
                or meta.get("decisions"),
                "No decisions recorded.",
            ),
            "findings": _bullet(meta.get("findings"), "No findings recorded."),
            "evidence-and-verification": _bullet(meta.get("evidence"), "No verification evidence recorded."),
            "artifacts": _bullet(
                [f"`{row['path_or_reference']}`" + (f" - {row['description']}" if row["description"] else "") for row in artifacts]
                or meta.get("artifacts"),
                "No artifacts recorded.",
            ),
            "changed-systems-and-files": _bullet(meta.get("changed_files") or meta.get("changed_systems"), "No changed files recorded."),
            "blockers": _bullet(meta.get("blockers"), "No blockers recorded."),
            "delegates": self._delegates_for_work(work.work_id),
            "next-actions": _bullet(meta.get("next_actions"), "No next action recorded."),
            "related-knowledge": _bullet(meta.get("related_knowledge"), "No related knowledge promoted yet."),
            "sessions-and-branches": self._sessions_for_work(work.work_id),
            "checkpoints": self._checkpoint_list(checkpoints, events),
        }
        for section in SECTION_NAMES:
            key = slugify_heading(section)
            body += f"## {section}\n\n{_managed_block(key, sections[key])}\n\n"

        content = front + body.rstrip() + "\n"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if existing:
            content = self._merge_frontmatter(existing, front)
            if not re.search(r"^#\s+", content, flags=re.MULTILINE):
                content += f"# {work.title}\n\n"
            for section in SECTION_NAMES:
                key = slugify_heading(section)
                if f"## {section}" not in content:
                    content += f"\n## {section}\n\n{_managed_block(key, sections[key])}\n"
                else:
                    content, ok = _replace_block(content, key, sections[key])
                    if not ok:
                        self._write_conflict(path, content, f"Malformed managed block for {section}")
                        raise RuntimeError(f"Malformed managed block in {path}")
        _atomic_write(path, content)
        return path

    def render_project(self, project: WorkItem) -> Path:
        path = self.root / project.wiki_path
        missions = self._missions_for_project(project.work_id)
        active = [m for m in missions if m.status in {"active", "needs_review"}]
        blocked = [m for m in missions if m.status == "blocked"]
        completed = [m for m in missions if m.status == "completed"]
        stale = self._stale_missions(missions)
        meta = project.metadata
        front = _frontmatter(
            {
                "title": project.title,
                "type": "work",
                "work_id": project.work_id,
                "work_kind": "project",
                "status": project.status,
                "confidence": project.confidence,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "project_roots": [project.project_root] if project.project_root else [],
                "tags": meta.get("tags", ["mission-memory"]),
            }
        )
        body = f"# {project.title}\n\n"
        sections = {
            "project-objective": str(meta.get("objective") or meta.get("summary") or f"Coordinate work for {project.title}."),
            "current-overall-state": self._project_overall_state(project, missions),
            "active-missions": self._mission_links(active, "No active missions."),
            "blocked-missions": self._mission_links(blocked, "No blocked missions."),
            "recently-completed-missions": self._mission_links(completed[:10], "No completed missions."),
            "important-decisions": self._project_decisions(missions),
            "project-artifacts": self._project_artifacts(missions),
            "relevant-entities-and-concepts": self._project_related_knowledge(missions),
            "project-next-actions": _bullet(meta.get("next_actions"), "No project-level next action recorded."),
            "recent-activity": self._mission_links(missions[:15], "No recent activity."),
            "stale-missions": self._mission_links(stale, "No stale missions."),
            "unresolved-review-items": self._mission_links([m for m in missions if m.status == "needs_review"], "No unresolved review items."),
        }
        project_sections = (
            ("Project Objective", "project-objective"),
            ("Current Overall State", "current-overall-state"),
            ("Active Missions", "active-missions"),
            ("Blocked Missions", "blocked-missions"),
            ("Recently Completed Missions", "recently-completed-missions"),
            ("Important Decisions", "important-decisions"),
            ("Project Artifacts", "project-artifacts"),
            ("Relevant Entities and Concepts", "relevant-entities-and-concepts"),
            ("Project Next Actions", "project-next-actions"),
            ("Recent Activity", "recent-activity"),
            ("Stale Missions", "stale-missions"),
            ("Unresolved Review Items", "unresolved-review-items"),
        )
        for heading, key in project_sections:
            body += f"## {heading}\n\n{_managed_block(key, sections[key])}\n\n"
        content = front + body.rstrip() + "\n"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if existing:
            content = self._merge_frontmatter(existing, front)
            if not re.search(r"^#\s+", content, flags=re.MULTILINE):
                content += f"# {project.title}\n\n"
            for heading, key in project_sections:
                if f"## {heading}" not in content:
                    content += f"\n## {heading}\n\n{_managed_block(key, sections[key])}\n"
                else:
                    content, ok = _replace_block(content, key, sections[key])
                    if not ok:
                        self._write_conflict(path, content, f"Malformed managed block for {heading}")
                        raise RuntimeError(f"Malformed managed block in {path}")
        _atomic_write(path, content)
        return path

    def render_dashboards(self) -> None:
        missions = self.store.recent_missions(limit=1000)
        self._write_dashboard("mission-control.md", self._mission_control(missions))
        self._write_dashboard("recovery.md", self._recovery_dashboard(missions))

    def render_indexes(self) -> None:
        missions = self.store.recent_missions(limit=10000)
        projects = self._recent_projects(limit=10000)
        index_dir = self.root / "work" / "_indexes"
        groups = {
            "active-missions.md": [m for m in missions if m.status == "active"],
            "blocked-missions.md": [m for m in missions if m.status == "blocked"],
            "completed-missions.md": [m for m in missions if m.status == "completed"],
            "needs-review.md": [m for m in missions if m.status == "needs_review"],
            "waiting-missions.md": [m for m in missions if m.status == "waiting"],
            "stale-missions.md": self._stale_missions(missions),
            "running-delegates.md": self._running_delegate_missions(missions),
        }
        for name, items in groups.items():
            title = name[:-3].replace("-", " ").title()
            _atomic_write(index_dir / name, f"# {title}\n\n{self._mission_links(items, 'No entries.')}\n")
        _atomic_write(index_dir / "projects.md", f"# Projects\n\n{self._project_links(projects)}\n")
        _atomic_write(index_dir / "workstreams.md", f"# Workstreams\n\n{self._workstream_index(missions)}\n")

    def render_monthly_log(self) -> None:
        now = utc_now()
        month = now[:7]
        missions = self.store.recent_missions(limit=50)
        lines = [f"# {month} Work Log", "", f"Last generated: {now}", ""]
        lines.append("## Recent Missions")
        lines.append("")
        lines.append(self._mission_links(missions, "No recent missions."))
        _atomic_write(self.root / "logs" / f"{month}.md", "\n".join(lines).rstrip() + "\n")

    def _mission_control(self, missions: list[WorkItem]) -> str:
        running_work_ids = {row["work_id"] for row in self.store.active_delegations(limit=100) if row["work_id"]}
        sections = {
            "Active Now": [m for m in missions if m.status == "active"],
            "Blocked": [m for m in missions if m.status == "blocked"],
            "Waiting for External Input": [m for m in missions if m.status == "waiting"],
            "Running Delegates": [m for m in missions if m.work_id in running_work_ids],
            "Needs Review": [m for m in missions if m.status == "needs_review"],
            "Stale Missions": self._stale_missions(missions),
            "Recently Completed": [m for m in missions if m.status == "completed"][:20],
        }
        lines = ["# Mission Control", "", f"Last generated: {utc_now()}", ""]
        for heading, items in sections.items():
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(self._mission_table(items))
            lines.append("")
        lines.extend(
            [
                "## Recent Decisions",
                "",
                self._recent_decisions(),
                "",
                "## Recent Artifacts",
                "",
                self._recent_artifacts(),
                "",
                "## Unassigned Activity",
                "",
                self._unassigned_activity(),
                "",
                "## Wiki or Persistence Failures",
                "",
                self._render_failures(),
                "",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    def _recovery_dashboard(self, missions: list[WorkItem]) -> str:
        unassigned = self.store.unassigned_events(limit=50)
        failures = self.store.render_failures(limit=50)
        needs_review = [m for m in missions if m.status == "needs_review"]
        active_delegates = self.store.active_delegations(limit=50)
        duplicates = self.store.duplicate_mission_candidates(limit=25)
        debt = [(m, len(self.store.uncovered_events(m.work_id, limit=500))) for m in missions[:200]]
        debt = [(m, count) for m, count in debt if count]
        lines = ["# Recovery", "", f"Last generated: {utc_now()}", ""]
        sections = {
            "Interrupted Sessions with Uncovered Events": [
                f"- [{m.title}]({_relative_link(self.root, m.wiki_path)}) has {count} uncovered event(s). Suggested action: `/wiki reconcile {m.work_id}`."
                for m, count in debt
            ],
            "Fallback Checkpoints Needing Review": [
                f"- [{m.title}]({_relative_link(self.root, m.wiki_path)}) has fallback checkpoint `{cp.checkpoint_id}`: {cp.summary}. Suggested action: inspect latest checkpoint, then `/wiki status {m.work_id}`."
                for m, cp in self._fallback_checkpoints_needing_review(missions)
            ] or [
                f"- [{m.title}]({_relative_link(self.root, m.wiki_path)}) needs review. Suggested action: inspect latest checkpoint, then `/wiki status {m.work_id}`."
                for m in needs_review
            ],
            "Failed Markdown Render Jobs": [
                f"- `{row['job_type']}` for `{row['work_id'] or 'global'}` failed: {row['last_error']}. Suggested action: `/wiki repair`."
                for row in failures
            ],
            "Persistence Failures": self._persistence_failures(),
            "SQLite/Markdown Inconsistencies": self._markdown_inconsistencies(missions),
            "Unassigned Material Events": [
                f"- `{row['event_id']}` {row['event_type']}: {row['summary']}. Suggested action: `/wiki attach <work-id>`."
                for row in unassigned
            ],
            "Duplicate Mission Candidates": [
                f"- [{left.title}]({_relative_link(self.root, left.wiki_path)}) and [{right.title}]({_relative_link(self.root, right.wiki_path)}) overlap: {reason}. Suggested action: `/wiki merge {right.work_id} {left.work_id}` or `/wiki split <work-id> <title>`."
                for left, right, reason in duplicates
            ],
            "Unresolved Branch Conflicts": self._branch_conflicts(missions),
            "Running or Orphaned Delegates": [
                f"- `{row['delegation_id']}` role `{row['role'] or 'subagent'}` child `{row['child_session_id'] or 'unknown'}` work `{row['work_id'] or 'unassigned'}` started {row['started_at']}. Suggested action: inspect child session or reconcile parent mission."
                for row in active_delegates
            ],
            "Stale Locks": [],
            "Malformed Pages": self._malformed_pages(),
            "Missions Reported Complete Without Sufficient Evidence": [
                f"- [{m.title}]({_relative_link(self.root, m.wiki_path)}) is completed but lacks verification evidence. Suggested action: reopen or add evidence."
                for m in missions
                if m.status == "completed" and not m.metadata.get("evidence")
            ],
        }
        for heading, items in sections.items():
            lines.append(f"## {heading}")
            lines.append("")
            lines.append("\n".join(items) if items else "No items.")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _write_dashboard(self, rel: str, content: str) -> None:
        _atomic_write(self.root / rel, content)

    def _mission_table(self, missions: list[WorkItem]) -> str:
        if not missions:
            return "No entries."
        rows = [
            "| Project | Mission | Status | Current State | Last Checkpoint | Latest Session | Running Delegates | Next Action | Blockers | Last Activity |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for mission in missions[:50]:
            project = self.store.get_work(mission.parent_work_id) if mission.parent_work_id else None
            latest = self.store.latest_checkpoint(mission.work_id)
            meta = mission.metadata
            rows.append(
                "| {project} | [{title}]({link}) | {status} | {state} | {checkpoint} | {session} | {delegates} | {next_action} | {blockers} | {last_activity} |".format(
                    project=(project.title if project else "Unassigned").replace("|", "\\|"),
                    title=mission.title.replace("|", "\\|"),
                    link=_relative_link(self.root, mission.wiki_path),
                    status=mission.status,
                    state=self._cell(str(meta.get("current_state", "")) or "No state."),
                    checkpoint=latest.created_at if latest else "None",
                    session=self._latest_session(mission.work_id),
                    delegates=self._cell(self._running_delegate_summary(mission.work_id), max_len=70),
                    next_action=self._cell(self._first(meta.get("next_actions"))),
                    blockers=self._cell(", ".join(map(str, meta.get("blockers", []))) or "None"),
                    last_activity=self._activity_age(mission.updated_at),
                )
            )
        return "\n".join(rows)

    def _mission_links(self, missions: list[WorkItem], empty: str) -> str:
        if not missions:
            return empty
        return "\n".join(
            f"- [{m.title}]({_relative_link(self.root, m.wiki_path)}) - {m.status}; next: {self._first(m.metadata.get('next_actions')) or 'not recorded'}"
            for m in missions
        )

    def _missions_for_project(self, project_work_id: str) -> list[WorkItem]:
        return [m for m in self.store.recent_missions(limit=10000) if m.parent_work_id == project_work_id]

    def _recent_projects(self, limit: int = 1000) -> list[WorkItem]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM work_items
                WHERE work_kind='project'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self.store._work_from_row(row) for row in rows if row]

    def _project_links(self, projects: list[WorkItem]) -> str:
        if not projects:
            return "No projects."
        lines = []
        for project in projects:
            mission_count = len(self._missions_for_project(project.work_id))
            lines.append(f"- [{project.title}]({_relative_link(self.root, project.wiki_path)}) - {project.status}; missions: {mission_count}")
        return "\n".join(lines)

    def _project_overall_state(self, project: WorkItem, missions: list[WorkItem]) -> str:
        meta = project.metadata
        explicit = meta.get("current_state")
        if explicit:
            return str(explicit)
        counts: dict[str, int] = {}
        for mission in missions:
            counts[mission.status] = counts.get(mission.status, 0) + 1
        if not counts:
            return "Project page created automatically; no missions recorded yet."
        ordered = ", ".join(f"{status}: {counts[status]}" for status in sorted(counts))
        return f"Project has {len(missions)} mission(s): {ordered}."

    def _project_decisions(self, missions: list[WorkItem]) -> str:
        mission_ids = {mission.work_id for mission in missions}
        if not mission_ids:
            return "No important decisions recorded."
        lines: list[str] = []
        for mission in missions[:100]:
            for row in self.store.decisions_for_work(mission.work_id, limit=10):
                text = str(row["decision"])
                if row["rationale"]:
                    text = f"{text} - {row['rationale']}"
                lines.append(f"- {row['created_at']}: {text} ([{mission.title}]({_relative_link(self.root, mission.wiki_path)}))")
                if len(lines) >= 20:
                    return "\n".join(lines)
        if lines:
            return "\n".join(lines)

        metadata_decisions: list[str] = []
        seen: set[str] = set()
        for mission in missions:
            for decision in mission.metadata.get("decisions", []) or []:
                text = str(decision).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                metadata_decisions.append(f"- {text} ([{mission.title}]({_relative_link(self.root, mission.wiki_path)}))")
                if len(metadata_decisions) >= 20:
                    return "\n".join(metadata_decisions)
        return "\n".join(metadata_decisions) if metadata_decisions else "No important decisions recorded."

    def _project_artifacts(self, missions: list[WorkItem]) -> str:
        if not missions:
            return "No project artifacts recorded."
        lines: list[str] = []
        seen: set[str] = set()
        for mission in missions[:100]:
            for row in self.store.artifacts_for_work(mission.work_id, limit=10):
                ref = str(row["path_or_reference"])
                if ref in seen:
                    continue
                seen.add(ref)
                detail = row["description"] or row["artifact_type"]
                lines.append(f"- `{ref}` - {detail} ([{mission.title}]({_relative_link(self.root, mission.wiki_path)}))")
                if len(lines) >= 30:
                    return "\n".join(lines)
            for artifact in mission.metadata.get("artifacts", []) or []:
                ref = str(artifact).strip()
                if not ref or ref in seen:
                    continue
                seen.add(ref)
                lines.append(f"- `{ref}` ([{mission.title}]({_relative_link(self.root, mission.wiki_path)}))")
                if len(lines) >= 30:
                    return "\n".join(lines)
        return "\n".join(lines) if lines else "No project artifacts recorded."

    def _project_related_knowledge(self, missions: list[WorkItem]) -> str:
        if not missions:
            return "No relevant entities or concepts promoted yet."
        lines: list[str] = []
        seen: set[str] = set()
        for mission in missions[:100]:
            for rel in mission.metadata.get("related_knowledge", []) or []:
                target = str(rel).strip()
                if not target or target in seen:
                    continue
                seen.add(target)
                title = Path(target).stem.replace("-", " ").title()
                lines.append(f"- [{title}]({_relative_link(self.root, target)}) - from [{mission.title}]({_relative_link(self.root, mission.wiki_path)})")
                if len(lines) >= 30:
                    return "\n".join(lines)
        return "\n".join(lines) if lines else "No relevant entities or concepts promoted yet."

    def _workstream_index(self, missions: list[WorkItem]) -> str:
        groups: dict[str, list[WorkItem]] = {}
        for mission in missions:
            workstream = str(mission.metadata.get("workstream") or "default")
            groups.setdefault(workstream, []).append(mission)
        if not groups:
            return "No workstreams."
        lines: list[str] = []
        for workstream in sorted(groups):
            lines.append(f"## {workstream}")
            lines.append("")
            lines.append(self._mission_links(groups[workstream], "No entries."))
            lines.append("")
        return "\n".join(lines).rstrip()

    def _running_delegate_missions(self, missions: list[WorkItem]) -> list[WorkItem]:
        running_work_ids = {row["work_id"] for row in self.store.active_delegations(limit=500) if row["work_id"]}
        return [mission for mission in missions if mission.work_id in running_work_ids]

    def _running_delegate_summary(self, work_id: str) -> str:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, child_session_id, goal
                FROM delegations
                WHERE work_id=? AND state='running'
                ORDER BY started_at DESC
                LIMIT 6
                """,
                (work_id,),
            ).fetchall()
        if not rows:
            return "None"
        labels = []
        shown = rows[:5]
        for row in shown:
            role = row["role"] or "subagent"
            child = row["child_session_id"] or "unknown"
            goal = row["goal"] or ""
            label = f"{role} `{child}`"
            if goal:
                label = f"{label}: {goal}"
            labels.append(label)
        suffix = "+" if len(rows) > len(shown) else ""
        return f"{len(shown)}{suffix}: " + "; ".join(labels)

    def _sessions_for_work(self, work_id: str) -> str:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM session_links
                WHERE work_id=?
                ORDER BY activated_at DESC
                LIMIT 30
                """,
                (work_id,),
            ).fetchall()
        if not rows:
            return "No sessions linked."
        return "\n".join(
            f"- `{row['session_id']}` branch `{row['branch_id'] or 'default'}` relationship `{row['relationship']}` focus `{bool(row['focus'])}` activated {row['activated_at']}"
            for row in rows
        )

    def _delegates_for_work(self, work_id: str) -> str:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM delegations
                WHERE work_id=?
                ORDER BY started_at DESC
                LIMIT 30
                """,
                (work_id,),
            ).fetchall()
        if not rows:
            return "No delegates recorded."
        lines = []
        for row in rows:
            detail = row["goal"] or "No goal recorded."
            if row["result_summary"]:
                detail = f"{detail} Result: {row['result_summary']}"
            lines.append(f"- `{row['delegation_id']}` {row['role'] or 'subagent'} {row['state']}: {detail}")
        return "\n".join(lines)

    def _checkpoint_list(self, checkpoints: list[Checkpoint], uncovered: list[sqlite3.Row]) -> str:
        lines: list[str] = []
        for cp in checkpoints:
            marker = f"<!-- work-wiki:checkpoint:{cp.checkpoint_id} -->"
            review = " needs_review" if cp.needs_review else ""
            lines.append(f"{marker}\n- {cp.created_at} `{cp.checkpoint_kind}`{review}: {cp.summary}")
        if uncovered:
            lines.append("")
            lines.append(f"Uncovered event debt: {len(uncovered)} recent event(s).")
        return "\n".join(lines) if lines else "No checkpoints recorded."

    def _latest_session(self, work_id: str) -> str:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT session_id FROM session_links WHERE work_id=? ORDER BY activated_at DESC LIMIT 1",
                (work_id,),
            ).fetchone()
        return row["session_id"] if row else ""

    def _recent_decisions(self) -> str:
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        if not rows:
            return "No recent decisions."
        return "\n".join(f"- {row['created_at']}: {row['decision']}" for row in rows)

    def _recent_artifacts(self) -> str:
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        if not rows:
            return "No recent artifacts."
        return "\n".join(f"- `{row['path_or_reference']}` - {row['description'] or row['artifact_type']}" for row in rows)

    def _unassigned_activity(self) -> str:
        rows = self.store.unassigned_events(limit=20)
        if not rows:
            return "No unassigned material events."
        return "\n".join(f"- `{row['event_id']}` {row['event_type']}: {row['summary']}" for row in rows)

    def _render_failures(self) -> str:
        rows = self.store.render_failures(limit=20)
        if not rows:
            return "No render failures."
        return "\n".join(f"- `{row['render_job_id']}` {row['job_type']}: {row['last_error']}" for row in rows)

    def _persistence_failures(self) -> list[str]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM events
                WHERE event_type='persistence_failed'
                ORDER BY observed_at DESC
                LIMIT 25
                """
            ).fetchall()
        return [
            f"- `{row['observed_at']}` work `{row['work_id'] or 'unassigned'}`: {row['summary']}. Suggested action: inspect Recovery, then `/wiki repair`."
            for row in rows
        ]

    def _fallback_checkpoints_needing_review(self, missions: list[WorkItem]) -> list[tuple[WorkItem, Checkpoint]]:
        by_id = {mission.work_id: mission for mission in missions}
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE semantic=0 AND needs_review=1
                ORDER BY created_at DESC
                LIMIT 50
                """
            ).fetchall()
        out: list[tuple[WorkItem, Checkpoint]] = []
        for row in rows:
            mission = by_id.get(row["work_id"]) or self.store.get_work(row["work_id"])
            if mission:
                out.append((mission, self.store._checkpoint_from_row(row)))
        return out

    def _branch_conflicts(self, missions: list[WorkItem]) -> list[str]:
        by_id = {mission.work_id: mission for mission in missions}
        items = []
        for row in self.store.branch_conflicts(limit=50):
            mission = by_id.get(row["work_id"]) or self.store.get_work(row["work_id"])
            if not mission:
                continue
            items.append(
                f"- [{mission.title}]({_relative_link(self.root, mission.wiki_path)}) branch `{row['older_branch_id']}` is older than `{row['newer_branch_id']}` in lineage `{row['lineage_key']}`. Suggested action: inspect branch checkpoints, then merge, reassign, or continue the authoritative branch."
            )
        return items

    def _stale_missions(self, missions: list[WorkItem]) -> list[WorkItem]:
        return [m for m in missions if m.status in {"active", "waiting"} and m.updated_at[:10] < utc_now()[:10]][:25]

    def _markdown_inconsistencies(self, missions: list[WorkItem]) -> list[str]:
        items: list[str] = []
        for mission in missions[:500]:
            if mission.wiki_path and not (self.root / mission.wiki_path).exists():
                items.append(
                    f"- `{mission.work_id}` has no Markdown page at `{mission.wiki_path}`. Suggested action: `/wiki repair`."
                )
        return items[:50]

    def _malformed_pages(self) -> list[str]:
        try:
            conflicts = sorted(
                list((self.root / "work" / "missions").glob("*.conflict-*"))
                + list((self.root / "work" / "projects").glob("*.conflict-*")),
                key=lambda p: str(p.relative_to(self.root)),
            )
        except OSError:
            return []
        return [
            f"- `{path.relative_to(self.root)}` was preserved after a managed-block conflict. Suggested action: inspect the conflict file, then `/wiki repair`."
            for path in conflicts[:50]
        ]

    def _first(self, values: Any) -> str:
        if isinstance(values, str):
            return values
        if isinstance(values, list) and values:
            return str(values[0])
        return ""

    def _cell(self, value: str, max_len: int = 90) -> str:
        text = " ".join(str(value).replace("|", "\\|").split())
        return text[: max_len - 3] + "..." if len(text) > max_len else text

    def _activity_age(self, timestamp: str) -> str:
        try:
            observed = time.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
            now = time.strptime(utc_now(), "%Y-%m-%dT%H:%M:%SZ")
            seconds = max(0, int(time.mktime(now) - time.mktime(observed)))
        except (TypeError, ValueError, OverflowError):
            return timestamp[:10] if timestamp else "unknown"
        if seconds < 3600:
            return f"{max(1, seconds // 60)}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        return f"{seconds // 86400}d ago"

    def _merge_frontmatter(self, existing: str, frontmatter: str) -> str:
        if existing.startswith("---\n"):
            end = existing.find("\n---\n", 4)
            if end != -1:
                return frontmatter + existing[end + 5 :].lstrip("\n")
        return frontmatter + existing.lstrip("\n")

    def _write_conflict(self, path: Path, content: str, reason: str) -> None:
        conflict = path.with_suffix(path.suffix + f".conflict-{utc_now().replace(':', '')}")
        _atomic_write(conflict, f"Conflict reason: {reason}\n\n{content}")

    def _schema_text(self) -> str:
        return """# Work Wiki Schema

This wiki is maintained by Hermes Work Wiki.

Generated content is bounded by `<!-- work-wiki:generated-...:start -->`
and `<!-- work-wiki:generated-...:end -->` markers. Manual notes outside
managed blocks are preserved by the renderer.
"""


def slugify_heading(section: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
