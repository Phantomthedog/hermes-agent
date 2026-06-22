from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import WorkWikiConfig
from .store import WorkItem, WorkWikiStore, slugify, stable_hash, utc_now


ALLOWED_TARGET_DIRS = {
    "concepts",
    "entities",
    "operations",
    "tools",
    "comparisons",
    "queries",
}

TYPE_BY_DIR = {
    "concepts": "concept",
    "entities": "entity",
    "operations": "operations",
    "tools": "tool",
    "comparisons": "comparison",
    "queries": "query",
}

INDEX_SECTION_BY_TYPE = {
    "concept": "Concepts",
    "entity": "Entities",
    "operations": "Operations",
    "tool": "Tools",
    "comparison": "Comparisons",
    "query": "Queries",
}

DEFAULT_TAGS_BY_TYPE = {
    "concept": ["wiki", "memory", "knowledge-base"],
    "entity": ["tool", "workflow"],
    "operations": ["workflow", "documentation", "automation"],
    "tool": ["tool", "cli"],
    "comparison": ["comparison"],
    "query": ["wiki"],
}

LOW_CONFIDENCE_TERMS = (
    "maybe",
    "possibly",
    "probably",
    "unclear",
    "not sure",
    "might",
    "could be",
    "appears to",
)

DURABLE_SIGNAL_TERMS = (
    "architecture",
    "backup",
    "because",
    "checkpoint",
    "configuration",
    "decision",
    "durable",
    "evidence",
    "fix",
    "hook",
    "ledger",
    "plugin",
    "promote",
    "provenance",
    "recovery",
    "requires",
    "root cause",
    "runbook",
    "verified",
    "wiki",
    "workflow",
)

NOISE_SIGNAL_TERMS = (
    "tool_ok",
    "speed test",
    "sha256",
    "os.getcwd",
    "platform.platform",
    "python3 - <<",
)


@dataclass
class CuratedFact:
    target: str
    page_type: str
    title: str
    summary: str
    fact: str
    evidence: str
    confidence: str
    source_work_id: str
    checkpoint_id: str


@dataclass
class CuratorRunResult:
    applied: bool
    prompt_only: bool
    prompt: str
    facts: list[CuratedFact]
    changed_pages: list[str]
    skipped: list[str]

    def to_text(self) -> str:
        if self.prompt_only:
            return self.prompt
        mode = "Applied" if self.applied else "Dry run"
        lines = [
            f"{mode}: {len(self.facts)} curated durable fact(s).",
        ]
        if self.changed_pages:
            lines.append("Pages:")
            lines.extend(f"- {page}" for page in self.changed_pages)
        if self.facts:
            lines.append("Facts:")
            for fact in self.facts:
                lines.append(f"- {fact.target}: {fact.fact}")
        if self.skipped:
            lines.append("Skipped:")
            lines.extend(f"- {item}" for item in self.skipped[:20])
        if not self.applied and self.facts:
            lines.append("Run again with --apply to write these wiki updates.")
        return "\n".join(lines)


class KnowledgeCurator:
    """Cloud-LLM-assisted durable-fact curator for the LLM Wiki.

    The hook path should stay deterministic and low-latency. This curator is a
    second-stage review pass: it reads already-captured mission evidence,
    asks the configured Hermes auxiliary LLM for strict JSON, validates the
    plan locally, then optionally writes provenance-marked facts into
    human-authored wiki pages.
    """

    def __init__(self, config: WorkWikiConfig, store: WorkWikiStore):
        self.config = config
        self.store = store
        self.root = Path(config.wiki_root)

    def run(
        self,
        *,
        apply: bool = False,
        limit: int = 8,
        work_id: str = "",
        since_days: int = 0,
        provider: str = "",
        model: str = "",
        max_updates: int = 8,
        prompt_only: bool = False,
    ) -> CuratorRunResult:
        missions = self._candidate_missions(limit=limit, work_id=work_id, since_days=since_days)
        prompt = self._build_prompt(missions, max_updates=max_updates)
        if prompt_only:
            return CuratorRunResult(False, True, prompt, [], [], [])
        if not missions:
            return CuratorRunResult(False, False, prompt, [], [], ["No candidate missions with durable evidence."])

        raw = self._call_llm(prompt, provider=provider, model=model)
        plan = self._parse_plan(raw)
        facts, skipped = self._normalize_plan(plan, missions, max_updates=max_updates)
        if not apply or not facts:
            return CuratorRunResult(False, False, prompt, facts, sorted({f.target for f in facts}), skipped)

        changed_pages = self._apply_facts(facts)
        return CuratorRunResult(True, False, prompt, facts, changed_pages, skipped)

    def _candidate_missions(self, *, limit: int, work_id: str = "", since_days: int = 0) -> list[dict[str, Any]]:
        works: list[WorkItem] = []
        if work_id:
            work = self.store.get_work(work_id)
            if work:
                works = [work]
        elif since_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
            since_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
            works = self.store.missions_since(since_iso, limit=max(1, min(limit, 100)))
        else:
            works = self.store.recent_missions(limit=max(1, min(limit, 25)))

        candidates: list[dict[str, Any]] = []
        explicit_work = bool(work_id)
        for work in works:
            meta = work.metadata or {}
            latest = self.store.latest_checkpoint(work.work_id)
            checkpoints = self.store.checkpoints_for_work(work.work_id, limit=3)
            evidence = _as_text_list(meta.get("evidence"))
            findings = _as_text_list(meta.get("findings"))
            decisions = _as_text_list(meta.get("decisions"))
            candidate = {
                "work_id": work.work_id,
                "title": work.title,
                "status": work.status,
                "wiki_path": work.wiki_path,
                "objective": str(meta.get("objective") or work.title)[:500],
                "current_state": str(meta.get("current_state") or "")[:500],
                "findings": findings[:8],
                "decisions": decisions[:8],
                "evidence": evidence[:8],
                "artifacts": _as_text_list(meta.get("artifacts"))[:8],
                "latest_checkpoint": _checkpoint_view(latest),
                "recent_checkpoints": [_checkpoint_view(cp) for cp in checkpoints],
            }
            if explicit_work or _has_durable_signal(candidate):
                candidates.append(candidate)
        return candidates

    def _build_prompt(self, missions: list[dict[str, Any]], *, max_updates: int) -> str:
        schema = _read_excerpt(self.root / "SCHEMA.md", 3500)
        index = _read_excerpt(self.root / "index.md", 9000)
        recent_log = _tail_excerpt(self.root / "log.md", 3000)
        payload = {
            "wiki_root": str(self.root),
            "allowed_target_dirs": sorted(ALLOWED_TARGET_DIRS),
            "max_updates": max_updates,
            "missions": missions,
        }
        return "\n".join(
            [
                "You are the LLM Wiki durable-fact curator for Jack's Hermes/Codex Mission Memory.",
                "",
                "Task: review captured mission evidence and propose only durable wiki updates.",
                "",
                "Rules:",
                "- Return JSON only, no markdown fences.",
                "- If there are no reusable durable facts, return {\"updates\": []}.",
                "- Do not summarize temporary chatter, raw command output, guesses, secrets, or one-off progress.",
                "- Do not target generated files: mission-control.md, recovery.md, logs/**, or work/**.",
                "- Prefer existing pages from index.md when a page already covers the topic.",
                "- Create a new page only when the fact is central and reusable.",
                "- Every fact must be supported by the provided mission evidence.",
                "- Use target paths only under concepts/, entities/, operations/, tools/, comparisons/, or queries/.",
                "- Keep each fact <= 240 characters and each evidence string <= 240 characters.",
                "",
                "Required JSON shape:",
                "{",
                "  \"updates\": [",
                "    {",
                "      \"target\": \"concepts/example.md\",",
                "      \"title\": \"Example Page Title\",",
                "      \"type\": \"concept\",",
                "      \"summary\": \"One-line index summary if a new page is needed.\",",
                "      \"facts\": [",
                "        {",
                "          \"fact\": \"Durable fact to add.\",",
                "          \"evidence\": \"Short evidence from mission records.\",",
                "          \"confidence\": \"high|medium|low\",",
                "          \"source_work_id\": \"wrk_...\",",
                "          \"checkpoint_id\": \"chk_...\"",
                "        }",
                "      ]",
                "    }",
                "  ]",
                "}",
                "",
                "SCHEMA EXCERPT:",
                schema,
                "",
                "INDEX EXCERPT:",
                index,
                "",
                "RECENT LOG EXCERPT:",
                recent_log,
                "",
                "MISSION EVIDENCE JSON:",
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            ]
        )

    def _call_llm(self, prompt: str, *, provider: str = "", model: str = "") -> str:
        provider = (provider or os.getenv("WORK_WIKI_CURATOR_PROVIDER") or "").strip()
        model = (model or os.getenv("WORK_WIKI_CURATOR_MODEL") or "").strip()
        if provider.lower() in {"sidecar", "ollama", "ollama-sidecar", "local", "local-ollama", "ollama-local"}:
            raise RuntimeError("Local curator providers are disabled; use the active Hermes cloud model or an explicit cloud override.")

        try:
            from agent.auxiliary_client import call_llm
        except Exception as exc:
            raise RuntimeError(f"Cannot import Hermes auxiliary LLM client: {exc}") from exc

        response = call_llm(
            task="curator",
            provider=provider or None,
            model=model or None,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0,
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else str(content or "")

    def _parse_plan(self, raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end <= start:
                raise ValueError("LLM curator returned no JSON object.")
            data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("LLM curator JSON must be an object.")
        return data

    def _normalize_plan(
        self,
        plan: dict[str, Any],
        missions: list[dict[str, Any]],
        *,
        max_updates: int,
    ) -> tuple[list[CuratedFact], list[str]]:
        valid_work_ids = {str(item.get("work_id")) for item in missions}
        valid_checkpoint_ids = {
            str(cp.get("checkpoint_id"))
            for item in missions
            for cp in [item.get("latest_checkpoint"), *(item.get("recent_checkpoints") or [])]
            if isinstance(cp, dict) and cp.get("checkpoint_id")
        }
        updates = plan.get("updates") or []
        if not isinstance(updates, list):
            return [], ["updates is not a list"]

        facts: list[CuratedFact] = []
        skipped: list[str] = []
        for update in updates[: max_updates * 2]:
            if not isinstance(update, dict):
                skipped.append("update is not an object")
                continue
            target = self._sanitize_target(update.get("target"))
            if not target:
                skipped.append(f"invalid target: {update.get('target')!r}")
                continue
            page_type = str(update.get("type") or TYPE_BY_DIR.get(target.split("/", 1)[0], "concept")).strip()
            if page_type not in set(INDEX_SECTION_BY_TYPE):
                page_type = TYPE_BY_DIR.get(target.split("/", 1)[0], "concept")
            title = _clean_line(update.get("title")) or _title_from_target(target)
            summary = _clean_line(update.get("summary")) or f"Durable facts curated from Mission Memory for {title}."
            raw_facts = update.get("facts") or []
            if isinstance(raw_facts, str):
                raw_facts = [{"fact": raw_facts}]
            if not isinstance(raw_facts, list):
                skipped.append(f"{target}: facts is not a list")
                continue
            for raw_fact in raw_facts:
                item = raw_fact if isinstance(raw_fact, dict) else {"fact": raw_fact}
                fact = _clean_line(item.get("fact"))
                evidence = _clean_line(item.get("evidence"))
                confidence = str(item.get("confidence") or "medium").strip().lower()
                source_work_id = str(item.get("source_work_id") or "").strip()
                checkpoint_id = str(item.get("checkpoint_id") or "").strip()
                if len(facts) >= max_updates:
                    break
                if len(fact) < 30 or len(fact) > 260:
                    skipped.append(f"{target}: fact length out of range")
                    continue
                if any(term in fact.lower() for term in LOW_CONFIDENCE_TERMS) and confidence == "high":
                    confidence = "medium"
                if confidence not in {"high", "medium", "low"}:
                    confidence = "medium"
                if not evidence or len(evidence) > 280:
                    skipped.append(f"{target}: missing or overlong evidence")
                    continue
                if source_work_id not in valid_work_ids:
                    skipped.append(f"{target}: unknown source_work_id {source_work_id!r}")
                    continue
                if checkpoint_id and checkpoint_id not in valid_checkpoint_ids:
                    skipped.append(f"{target}: unknown checkpoint_id {checkpoint_id!r}")
                    continue
                if _looks_secret(fact) or _looks_secret(evidence):
                    skipped.append(f"{target}: rejected possible secret")
                    continue
                facts.append(
                    CuratedFact(
                        target=target,
                        page_type=page_type,
                        title=title,
                        summary=summary,
                        fact=fact,
                        evidence=evidence,
                        confidence=confidence,
                        source_work_id=source_work_id,
                        checkpoint_id=checkpoint_id,
                    )
                )
            if len(facts) >= max_updates:
                break
        return facts, skipped

    def _sanitize_target(self, raw: Any) -> str:
        value = str(raw or "").strip().replace("\\", "/")
        value = value.lstrip("/")
        if not value.endswith(".md"):
            value += ".md"
        if ".." in value or value.startswith(("_", ".")):
            return ""
        parts = [part for part in value.split("/") if part]
        if len(parts) != 2 or parts[0] not in ALLOWED_TARGET_DIRS:
            return ""
        slug = slugify(parts[1][:-3], "curated-facts")
        return f"{parts[0]}/{slug}.md"

    def _apply_facts(self, facts: list[CuratedFact]) -> list[str]:
        changed: list[str] = []
        new_pages: dict[str, CuratedFact] = {}
        facts_by_target: dict[str, list[CuratedFact]] = {}
        for fact in facts:
            facts_by_target.setdefault(fact.target, []).append(fact)

        for target, target_facts in facts_by_target.items():
            path = self.root / target
            existed = path.exists()
            content = path.read_text(encoding="utf-8") if existed else self._new_page(target_facts[0])
            content, added = self._append_facts(content, target_facts)
            if not added:
                continue
            content = self._touch_updated(content)
            _atomic_write(path, content)
            changed.append(target)
            if not existed:
                new_pages[target] = target_facts[0]
            self._record_events(target, added)

        if new_pages:
            self._update_index(new_pages)
            changed.append("index.md")
        if changed:
            self._append_log(facts, changed)
            changed.append("log.md")
        return changed

    def _new_page(self, fact: CuratedFact) -> str:
        today = _today()
        tags = DEFAULT_TAGS_BY_TYPE.get(fact.page_type, ["wiki"])
        source = self._source_path_for_work(fact.source_work_id)
        return "\n".join(
            [
                "---",
                f"title: {fact.title}",
                f"created: {today}",
                f"updated: {today}",
                f"type: {fact.page_type}",
                f"tags: [{', '.join(tags)}]",
                "sources:",
                f"  - {source}" if source else "  - mission-control.md",
                "confidence: medium",
                "---",
                "",
                f"# {fact.title}",
                "",
                fact.summary,
                "",
            ]
        )

    def _append_facts(self, content: str, facts: list[CuratedFact]) -> tuple[str, list[CuratedFact]]:
        if "## LLM-Curated Mission Facts" not in content:
            content = content.rstrip() + "\n\n## LLM-Curated Mission Facts\n"
        added: list[CuratedFact] = []
        for fact in facts:
            marker = f"<!-- work-wiki:llm-curated:{stable_hash(fact.target + '|' + fact.fact.lower(), 16)} -->"
            if marker in content or fact.fact.lower() in content.lower():
                continue
            mission = self.store.get_work(fact.source_work_id)
            source_title = mission.title if mission else fact.source_work_id
            source_path = mission.wiki_path if mission else ""
            source_line = (
                f"  Source mission: [{source_title}](../{source_path})."
                if source_path
                else f"  Source mission: `{fact.source_work_id}`."
            )
            block = [
                "",
                marker,
                f"- {fact.fact}",
                source_line,
                f"  Checkpoint: `{fact.checkpoint_id or 'not recorded'}`.",
                f"  Evidence: {fact.evidence}",
                f"  Confidence: {fact.confidence}.",
            ]
            content = content.rstrip() + "\n" + "\n".join(block) + "\n"
            added.append(fact)
        return content, added

    def _touch_updated(self, content: str) -> str:
        today = _today()
        now = utc_now()
        if re.search(r"^updated:\s*.*$", content, flags=re.MULTILINE):
            return re.sub(r"^updated:\s*.*$", f"updated: {today}", content, count=1, flags=re.MULTILINE)
        if re.search(r"^updated_at:\s*.*$", content, flags=re.MULTILINE):
            return re.sub(r"^updated_at:\s*.*$", f"updated_at: {now}", content, count=1, flags=re.MULTILINE)
        if content.startswith("---\n"):
            end = content.find("\n---\n", 4)
            if end != -1:
                return content[:end] + f"\nupdated: {today}" + content[end:]
        return content

    def _source_path_for_work(self, work_id: str) -> str:
        work = self.store.get_work(work_id)
        return work.wiki_path if work else ""

    def _record_events(self, target: str, facts: list[CuratedFact]) -> None:
        by_work: dict[str, list[CuratedFact]] = {}
        for fact in facts:
            by_work.setdefault(fact.source_work_id, []).append(fact)
        for work_id, work_facts in by_work.items():
            work = self.store.get_work(work_id)
            if work:
                related = list(work.metadata.get("related_knowledge", []))
                if target not in related:
                    related.append(target)
                    self.store.update_work_metadata(work_id, {"related_knowledge": related[:50]})
            self.store.add_event(
                work_id=work_id,
                event_type="knowledge_curated",
                source="work-wiki-llm-curator",
                summary=f"LLM curator added {len(work_facts)} durable fact(s) to {target}",
                payload={
                    "target": target,
                    "items": [fact.fact for fact in work_facts],
                    "checkpoint_ids": [fact.checkpoint_id for fact in work_facts],
                },
                checkpoint_id=work_facts[0].checkpoint_id or None,
            )

    def _update_index(self, pages: dict[str, CuratedFact]) -> None:
        path = self.root / "index.md"
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        existing = set(re.findall(r"\[\[([^|\]#]+)", content))
        for target, fact in pages.items():
            stem = Path(target).stem
            if stem in existing:
                continue
            section = INDEX_SECTION_BY_TYPE.get(fact.page_type)
            if not section:
                continue
            entry = f"- [[{stem}|{fact.title}]] - {fact.summary} `[{', '.join(DEFAULT_TAGS_BY_TYPE.get(fact.page_type, ['wiki']))}]`"
            lines = _insert_index_entry(lines, section, entry)
            existing.add(stem)
        entry_count = sum(1 for line in lines if line.startswith("- [["))
        text = "\n".join(lines).rstrip() + "\n"
        text = re.sub(
            r"> Last updated: .*?\| Total pages: \d+",
            f"> Last updated: {_today()} | Total pages: {entry_count}",
            text,
            count=1,
        )
        _atomic_write(path, text)

    def _append_log(self, facts: list[CuratedFact], changed: list[str]) -> None:
        path = self.root / "log.md"
        targets = sorted({fact.target for fact in facts})
        source_ids = sorted({fact.source_work_id for fact in facts})
        block = [
            "",
            f"## [{_today()}] update | LLM-curated durable mission facts",
            f"- Reviewed Mission Memory evidence and added {len(facts)} durable fact(s).",
            f"- Updated pages: {', '.join(f'`{page}`' for page in targets)}.",
            f"- Source missions: {', '.join(f'`{work_id}`' for work_id in source_ids)}.",
            f"- Changed files: {', '.join(f'`{page}`' for page in changed)}.",
        ]
        existing = path.read_text(encoding="utf-8") if path.exists() else "# Wiki Log\n"
        _atomic_write(path, existing.rstrip() + "\n" + "\n".join(block).rstrip() + "\n")


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


def _checkpoint_view(checkpoint: Any) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "kind": checkpoint.checkpoint_kind,
        "summary": checkpoint.summary,
        "status_after": checkpoint.status_after,
        "needs_review": bool(checkpoint.needs_review),
        "created_at": checkpoint.created_at,
        "metadata": checkpoint.metadata,
    }


def _has_durable_signal(candidate: dict[str, Any]) -> bool:
    chunks: list[str] = []
    for key in ("title", "objective", "current_state"):
        chunks.append(str(candidate.get(key) or ""))
    chunks.extend(candidate.get("findings") or [])
    chunks.extend(candidate.get("decisions") or [])
    chunks.extend(candidate.get("evidence") or [])
    latest = candidate.get("latest_checkpoint") or {}
    if isinstance(latest, dict):
        chunks.append(str(latest.get("summary") or ""))
        chunks.append(json.dumps(latest.get("metadata") or {}, ensure_ascii=False, default=str))
    text = " ".join(chunks).lower()
    if any(term in text for term in NOISE_SIGNAL_TERMS) and not any(candidate.get(key) for key in ("findings", "decisions")):
        return False
    return any(candidate.get(key) for key in ("findings", "decisions", "evidence")) or any(
        term in text for term in DURABLE_SIGNAL_TERMS
    )


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_clean_line(value)] if value.strip() else []
    if isinstance(value, list):
        return [_clean_line(item) for item in value if _clean_line(item)]
    return []


def _clean_line(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _read_excerpt(path: Path, limit: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _tail_excerpt(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _today() -> str:
    return utc_now()[:10]


def _title_from_target(target: str) -> str:
    return Path(target).stem.replace("-", " ").title()


def _looks_secret(value: str) -> bool:
    lowered = value.lower()
    if any(term in lowered for term in ("api key", "token=", "password", "bearer ", "secret")):
        return True
    return bool(re.search(r"[A-Za-z0-9_\-]{32,}", value))


def _insert_index_entry(lines: list[str], section: str, entry: str) -> list[str]:
    header = f"## {section}"
    try:
        start = lines.index(header)
    except ValueError:
        return lines + ["", header, entry]
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    section_lines = lines[start + 1 : end]
    prefix = [line for line in section_lines if not line.startswith("- [[")]
    entries = [line for line in section_lines if line.startswith("- [[")]
    entries.append(entry)
    entries = sorted(dict.fromkeys(entries), key=lambda line: line.lower())
    return lines[: start + 1] + prefix + entries + lines[end:]
