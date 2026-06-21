from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


def _truthy(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "auto"}


def _bool_setting(env_name: str, *cfg_keys: str, default: bool = True) -> bool:
    env_value = os.getenv(env_name)
    if env_value is not None:
        return _truthy(env_value, default)
    return _truthy(_cfg_get(*cfg_keys), default)


def _list_setting(env_name: str, *cfg_keys: str, default: list[str] | None = None) -> list[str]:
    env_value = os.getenv(env_name)
    if env_value is not None:
        return [item.strip() for item in env_value.split(",") if item.strip()]
    value = _cfg_get(*cfg_keys, default=default or [])
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;\n]", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(default or [])


def _cfg_get(*keys: str, default: Any = None) -> Any:
    try:
        from hermes_cli.config import load_config

        cur: Any = load_config()
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur
    except Exception:
        return default


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser()


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        try:
            if path.exists():
                return path
        except OSError:
            continue
    return None


@dataclass(frozen=True)
class WorkWikiConfig:
    enabled: bool
    mode: str
    strict_persistence: bool
    wiki_root: Path
    db_path: Path
    auto_detect_work: bool
    auto_create_projects: bool
    auto_create_missions: bool
    auto_capture_events: bool
    auto_generate_checkpoints: bool
    auto_render_wiki: bool
    auto_resume_context: bool
    auto_promote_knowledge: bool
    classification_confidence_attach: float
    classification_confidence_provisional: float
    max_event_payload_chars: int
    path_deny_patterns: tuple[str, ...]


def load_config() -> WorkWikiConfig:
    root_from_env = os.getenv("HERMES_WORK_WIKI_ROOT") or os.getenv("HERMES_LLM_WIKI_ROOT")
    root_from_cfg = _cfg_get("work_wiki", "wiki_root") or _cfg_get("work_wiki", "root")
    if root_from_env:
        wiki_root = _expand(root_from_env)
    elif root_from_cfg:
        wiki_root = _expand(root_from_cfg)
    else:
        candidates = [
            Path("/mnt/c/AI_WORKSPACE/10_MEMORY/wiki"),
            Path("/mnt/c/AI_WORKSPACE/20_PROJECTS/llm-wiki"),
            get_hermes_home() / "work-wiki" / "wiki",
        ]
        wiki_root = _first_existing(candidates) or candidates[-1]

    db_from_env = os.getenv("HERMES_WORK_WIKI_DB")
    db_from_cfg = _cfg_get("work_wiki", "db_path")
    db_path = (
        _expand(db_from_env)
        if db_from_env
        else _expand(db_from_cfg)
        if db_from_cfg
        else get_hermes_home() / "work-wiki" / "work-wiki.sqlite3"
    )

    mode = str(os.getenv("HERMES_WORK_WIKI_MODE") or _cfg_get("work_wiki", "mode", default="automatic")).lower()
    enabled = _truthy(os.getenv("HERMES_WORK_WIKI_ENABLED"), default=_truthy(_cfg_get("work_wiki", "enabled"), True))

    return WorkWikiConfig(
        enabled=enabled and mode != "disabled",
        mode=mode,
        strict_persistence=_bool_setting("HERMES_WORK_WIKI_STRICT_PERSISTENCE", "work_wiki", "strict_persistence", default=False),
        wiki_root=wiki_root,
        db_path=db_path,
        auto_detect_work=_bool_setting("HERMES_WORK_WIKI_AUTO_DETECT_WORK", "work_wiki", "auto_detect_work"),
        auto_create_projects=_bool_setting("HERMES_WORK_WIKI_AUTO_CREATE_PROJECTS", "work_wiki", "auto_create_projects"),
        auto_create_missions=_bool_setting("HERMES_WORK_WIKI_AUTO_CREATE_MISSIONS", "work_wiki", "auto_create_missions"),
        auto_capture_events=_bool_setting("HERMES_WORK_WIKI_AUTO_CAPTURE_EVENTS", "work_wiki", "auto_capture_events"),
        auto_generate_checkpoints=_bool_setting("HERMES_WORK_WIKI_AUTO_GENERATE_CHECKPOINTS", "work_wiki", "auto_generate_checkpoints"),
        auto_render_wiki=_bool_setting("HERMES_WORK_WIKI_AUTO_RENDER_WIKI", "work_wiki", "auto_render_wiki"),
        auto_resume_context=_bool_setting("HERMES_WORK_WIKI_AUTO_RESUME_CONTEXT", "work_wiki", "auto_resume_context"),
        auto_promote_knowledge=_bool_setting("HERMES_WORK_WIKI_AUTO_PROMOTE_KNOWLEDGE", "work_wiki", "auto_promote_knowledge", default=False),
        classification_confidence_attach=float(_cfg_get("work_wiki", "classification_confidence_attach", default=0.85) or 0.85),
        classification_confidence_provisional=float(_cfg_get("work_wiki", "classification_confidence_provisional", default=0.60) or 0.60),
        max_event_payload_chars=int(_cfg_get("work_wiki", "max_event_payload_chars", default=12000) or 12000),
        path_deny_patterns=tuple(
            _list_setting(
                "HERMES_WORK_WIKI_PATH_DENY",
                "work_wiki",
                "path_deny_patterns",
                default=[
                    "/home/*/.ssh/*",
                    "/home/*/.gnupg/*",
                    "/home/*/.aws/*",
                    "/home/*/.config/gh/*",
                    "*/.env",
                    "*/.env.*",
                    "*/id_rsa",
                    "*/id_ed25519",
                    "*/credentials",
                    "*/credentials.json",
                    "*/secrets/*",
                ],
            )
        ),
    )
