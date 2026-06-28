from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .config import WorkWikiConfig
from .store import WorkItem, WorkWikiStore, slugify, stable_hash, utc_now

FIELD_WEIGHTS = {
    "decisions": 3,
    "findings": 2,
    "evidence": 1,
}

DURABLE_TERMS = (
    "architecture",
    "configuration",
    "decision",
    "durable",
    "evidence",
    "hook",
    "ledger",
    "lineage",
    "mission control",
    "plugin",
    "recovery",
    "renderer",
    "requires",
    "root cause",
    "sqlite",
    "workflow",
)

NOISE_PREFIXES = (
    "done.",
    "finished",
    "i created",
    "i edited",
    "i removed",
    "i updated",
    "no.",
    "pass ",
    "revised.",
    "updated the markdown",
    "yes.",
)

NOISE_TERMS = (
    "question",
    "transcript",
    "video",
    "pdf",
    "docx",
    "mp4",
    "folder slug",
)

REQUIRED_CONTEXT_TERMS = (
    "codex",
    "hermes",
    "mission",
    "work wiki",
    "wiki",
    "sqlite",
    "ledger",
    "renderer",
    "plugin",
    "hook",
    "delegate",
    "subagent",
)


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


def _needs_review(item: str) -> bool:
    lowered = item.lower()
    return any(term in lowered for term in ("maybe", "unclear", "not sure", "possibly"))


class KnowledgePromoter:
    def __init__(self, config: WorkWikiConfig, store: WorkWikiStore):
        self.config = config
        self.store = store
        self.root = Path(config.wiki_root)

    def promote(
        self,
        *,
        work: WorkItem,
        checkpoint_id: str,
        updates: dict[str, Any],
        summary: str = "",
    ) -> list[str]:
        if not self.config.auto_promote_knowledge:
            return []
        items = self._candidate_items(updates, summary)
        if not items:
            return []

        target_rel, target_title = self._target_for(work, items)
        target_path = self.root / target_rel
        existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        content = existing or self._new_page(target_title)
        if "## Mission-Sourced Findings" not in content:
            content = content.rstrip() + "\n\n## Mission-Sourced Findings\n\n"

        added: list[str] = []
        for item in items:
            item_hash = stable_hash(item.lower(), length=16)
            marker = f"<!-- work-wiki:promotion:{item_hash} -->"
            if marker in content:
                continue
            evidence = self._evidence_for(item, updates, summary)
            content = content.rstrip() + "\n\n" + "\n".join(
                [
                    marker,
                    f"- {item}",
                    f"  Source mission: [{work.title}](../{work.wiki_path}).",
                    f"  checkpoint `{checkpoint_id}`.",
                    f"  Evidence: {evidence}",
                    f"  Target page: `{target_rel}`.",
                ]
            ) + "\n"
            added.append(item)

        if not added:
            return []

        content = self._touch_updated(content)
        _atomic_write(target_path, content)
        related = list(work.metadata.get("related_knowledge", []))
        if target_rel not in related:
            related.append(target_rel)
            self.store.update_work_metadata(work.work_id, {"related_knowledge": related[:50]})
        self.store.add_event(
            work_id=work.work_id,
            event_type="knowledge_promoted",
            source="work-wiki",
            summary=f"Promoted {len(added)} finding(s) to {target_rel}",
            payload={
                "target": target_rel,
                "checkpoint_id": checkpoint_id,
                "source_mission": work.wiki_path,
                "items": added,
                "evidence": [self._evidence_for(item, updates, summary) for item in added],
            },
            checkpoint_id=checkpoint_id,
        )
        return [target_rel]

    def _candidate_items(self, updates: dict[str, Any], summary: str) -> list[str]:
        weighted: list[tuple[int, str]] = []
        for key in ("decisions", "findings", "evidence"):
            raw = updates.get(key)
            if isinstance(raw, str):
                weighted.append((FIELD_WEIGHTS[key], raw))
            elif isinstance(raw, list):
                weighted.extend((FIELD_WEIGHTS[key], str(item)) for item in raw)

        candidates = []
        for weight, value in weighted:
            item = " ".join(str(value).strip().split())
            if not self._is_durable_item(item, weight):
                continue
            if _needs_review(item):
                item = f"Review required: {item}"
            candidates.append(item)
        return list(dict.fromkeys(candidates))[:8]

    def _is_durable_item(self, item: str, weight: int) -> bool:
        lowered = item.lower()
        if len(item) < 32 or len(item) > 260:
            return False
        if lowered.startswith(NOISE_PREFIXES):
            return False
        context_hits = sum(1 for term in REQUIRED_CONTEXT_TERMS if term in lowered)
        if any(term in lowered for term in NOISE_TERMS) and context_hits == 0:
            return False
        durable_hits = sum(1 for term in DURABLE_TERMS if term in lowered)
        if durable_hits == 0:
            return False
        # Evidence is often verification chatter. Only promote it when it carries
        # multiple durable signals and a Mission Memory context. Findings and
        # decisions can also pass when they carry multiple durable signals.
        if weight < 2:
            return context_hits > 0 and durable_hits >= 2
        return context_hits > 0 or durable_hits >= 2

    def _evidence_for(self, item: str, updates: dict[str, Any], summary: str) -> str:
        evidence_values: list[str] = []
        raw = updates.get("evidence")
        if isinstance(raw, str):
            evidence_values.append(raw)
        elif isinstance(raw, list):
            evidence_values.extend(str(value) for value in raw)
        if summary:
            evidence_values.append(summary)
        for value in evidence_values:
            evidence = " ".join(str(value).strip().split())
            if evidence and evidence != item:
                return evidence[:220]
        return "Promoted from checkpoint summary and extracted mission metadata."

    def _target_for(self, work: WorkItem, items: list[str]) -> tuple[str, str]:
        text = " ".join([work.title, str(work.metadata.get("objective", "")), *items]).lower()
        if "session" in text and ("lineage" in text or "branch" in text):
            return "concepts/hermes-session-lineage.md", "Hermes Session Lineage"
        if "delegate" in text or "subagent" in text:
            return "concepts/hermes-delegation.md", "Hermes Delegation"
        if "sqlite" in text or "ledger" in text:
            return "concepts/sqlite-operational-ledger.md", "SQLite Operational Ledger"
        if "markdown" in text or "wiki" in text or "mission" in text:
            return "concepts/hermes-mission-memory.md", "Hermes Mission Memory"
        title = " ".join(re.split(r"\W+", work.title)[:5]) or "Mission Findings"
        title = title.strip().title()
        return f"concepts/{slugify(title, 'mission-findings')}.md", title

    def _new_page(self, title: str) -> str:
        today = utc_now()[:10]
        return "\n".join(
            [
                "---",
                f"title: {title}",
                f"created: {today}",
                f"updated: {today}",
                "type: concept",
                "tags: [mission-memory, promoted-knowledge]",
                "sources: [mission-control.md]",
                "confidence: medium",
                "---",
                "",
                f"# {title}",
                "",
            ]
        )

    def _touch_updated(self, content: str) -> str:
        today = utc_now()[:10]
        if re.search(r"^updated:\s*.*$", content, flags=re.MULTILINE):
            return re.sub(r"^updated:\s*.*$", f"updated: {today}", content, count=1, flags=re.MULTILINE)
        if re.search(r"^updated_at:\s*.*$", content, flags=re.MULTILINE):
            return re.sub(r"^updated_at:\s*.*$", f"updated: {today}", content, count=1, flags=re.MULTILINE)
        if content.startswith("---\n"):
            end = content.find("\n---\n", 4)
            if end != -1:
                return content[:end] + f"\nupdated: {today}" + content[end:]
        return content
