from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .store import WorkItem, WorkWikiStore, slugify, stable_hash


MATERIAL_SIGNALS = {
    "implement",
    "build",
    "fix",
    "debug",
    "diagnose",
    "deploy",
    "migrate",
    "create",
    "write",
    "refactor",
    "audit",
    "investigate",
    "integrate",
    "continue",
    "resume",
    "repair",
    "test",
    "verify",
    "configure",
    "install",
}

TRIVIAL_PATTERNS = [
    re.compile(r"^\s*(hi|hello|thanks|thank you|ok|okay)\W*$", re.I),
    re.compile(r"^\s*(translate|rewrite|summari[sz]e) .{1,160}$", re.I),
    re.compile(r"^\s*(what is|who is|when is|where is) .{1,160}\??\s*$", re.I),
    re.compile(r"^\s*\d+\s*[-+*/]\s*\d+\s*$"),
]


@dataclass
class Classification:
    material: bool
    recommended_work_kind: str = "mission"
    project_candidates: list[str] = field(default_factory=list)
    mission_candidates: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    should_create: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "material": self.material,
            "recommended_work_kind": self.recommended_work_kind,
            "project_candidates": self.project_candidates,
            "mission_candidates": self.mission_candidates,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
            "should_create": self.should_create,
        }


def classify_user_message(message: Any, *, tool_activity: bool = False) -> Classification:
    text = _text(message)
    lowered = text.lower()
    if not text.strip():
        return Classification(material=False, confidence=0.0, reason_codes=["empty"])
    if any(pattern.match(text) for pattern in TRIVIAL_PATTERNS):
        return Classification(material=False, confidence=0.15, reason_codes=["trivial_pattern"])

    reason_codes: list[str] = []
    score = 0.0
    if len(text) > 240:
        score += 0.20
        reason_codes.append("long_request")
    hits = sorted(signal for signal in MATERIAL_SIGNALS if re.search(rf"\b{re.escape(signal)}\b", lowered))
    if hits:
        score += min(0.45, 0.18 * len(hits))
        reason_codes.extend(f"verb:{hit}" for hit in hits[:5])
    if any(
        token in lowered
        for token in (
            "file",
            "code",
            "repo",
            "test",
            "sqlite",
            "markdown",
            "dashboard",
            "wiki",
            "plugin",
            "feature",
            "mission",
            "completion",
            "persistence",
            "workflow",
            "branch",
            "lineage",
            "conflict",
            "session",
            "exclude",
            "detach",
        )
    ):
        score += 0.25
        reason_codes.append("durable_artifact_or_code")
    if any(token in lowered for token in ("continue", "resume", "session", "handoff", "checkpoint")):
        score += 0.15
        reason_codes.append("resume_or_session")
    if tool_activity:
        score += 0.35
        reason_codes.append("tool_activity")
    material = score >= 0.35
    return Classification(
        material=material,
        confidence=min(0.99, max(0.0, score)),
        reason_codes=reason_codes or ["no_material_signal"],
        should_create=material,
    )


def resolve_or_create_mission(
    store: WorkWikiStore,
    *,
    user_message: Any,
    session_id: str,
    branch_id: str = "",
    platform: str = "",
    project_root: str = "",
    classification: Classification | None = None,
    allow_create: bool = True,
) -> WorkItem | None:
    classification = classification or classify_user_message(user_message)
    existing = store.focus_for_session(session_id, branch_id=branch_id)
    if existing:
        return existing
    text = _text(user_message)
    if not classification.material:
        return None

    root = project_root or detect_project_root()
    candidates = store.find_missions(query=text, project_root=root, limit=5)
    if candidates:
        best = candidates[0]
        if _match_confidence(text, best) >= 0.60:
            store.link_session(
                session_id=session_id,
                work_id=best.work_id,
                branch_id=branch_id,
                relationship="focus",
                focus=True,
                platform=platform,
            )
            return best

    if not allow_create:
        return None

    title = infer_title(text)
    project_title = infer_project_title(root, title)
    project = store.ensure_project(
        title=project_title,
        project_root=root,
        confidence=0.75,
        metadata={"source": "automatic", "project_root": root},
    )
    mission = store.create_mission(
        title=title,
        objective=infer_objective(text, title),
        project_work_id=project.work_id,
        project_root=root,
        session_id=session_id,
        branch_id=branch_id,
        confidence=classification.confidence or 0.65,
        metadata={
            "source": "automatic",
            "classification": classification.as_dict(),
            "next_actions": ["Continue the requested work and record material progress."],
            "definition_of_done": infer_definition_of_done(text),
            "project_root": root,
        },
    )
    store.add_event(
        work_id=mission.work_id,
        session_id=session_id,
        branch_id=branch_id,
        event_type="mission_created",
        source="work-wiki",
        summary=f"Created mission: {mission.title}",
        payload={"classification": classification.as_dict(), "user_message_hash": stable_hash(text)},
    )
    return mission


def infer_title(text: str) -> str:
    stripped = " ".join(text.strip().split())
    stripped = re.sub(r"^(please\s+)?(can you\s+|could you\s+)?", "", stripped, flags=re.I)
    stripped = re.sub(r"^(continue|resume)\s+", "Continue ", stripped, flags=re.I)
    if len(stripped) <= 78:
        return stripped[:1].upper() + stripped[1:]
    # Prefer a filename/quoted subject if present.
    quoted = re.findall(r'"([^"]{8,90})"', stripped)
    if quoted:
        return quoted[0]
    words = stripped.split()
    return " ".join(words[:10]).rstrip(".,:;")[:90]


def infer_objective(text: str, title: str) -> str:
    clean = text.strip()
    if len(clean) > 20:
        return clean
    return title


def infer_definition_of_done(text: str) -> list[str]:
    lowered = text.lower()
    items = ["Requested work is implemented or completed."]
    if any(word in lowered for word in ("test", "verify", "bug", "fix", "implement", "code")):
        items.append("Relevant verification has been run or the missing verification is explicitly recorded.")
    if any(word in lowered for word in ("wiki", "dashboard", "markdown", "memory")):
        items.append("Mission state, artifacts, decisions, blockers, and next actions are visible in the wiki.")
    return items


def infer_project_title(project_root: str, mission_title: str) -> str:
    if project_root:
        name = Path(project_root).name
        if name:
            return _pretty_name(name)
    lowered = mission_title.lower()
    if "hermes" in lowered:
        return "Hermes Memory Ecosystem"
    return "General Work"


def detect_project_root() -> str:
    cwd = os.getenv("TERMINAL_CWD") or os.getcwd()
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
            timeout=1.5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return str(Path(cwd).resolve())


def _match_confidence(text: str, mission: WorkItem) -> float:
    text_terms = {t for t in re.split(r"\W+", text.lower()) if len(t) > 3}
    hay = " ".join(
        [
            mission.title.lower(),
            str(mission.metadata.get("objective", "")).lower(),
            " ".join(map(str, mission.metadata.get("aliases", []))).lower(),
        ]
    )
    hay_terms = {t for t in re.split(r"\W+", hay) if len(t) > 3}
    if not text_terms or not hay_terms:
        return 0.0
    overlap = len(text_terms & hay_terms) / max(1, len(text_terms))
    if slugify(mission.title) in slugify(text):
        overlap += 0.35
    return min(1.0, overlap)


def _pretty_name(value: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[-_]+", value) if part)


def _text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for part in message:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif "text" in part:
                    parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(message)
