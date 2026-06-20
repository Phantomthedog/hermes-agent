"""Auto-generate short session titles from the first user/assistant exchange.

Runs synchronously after the first response is delivered (no background thread).
Handles reasoning models (DeepSeek, etc.) that may consume all max_tokens on
thinking and return empty content with finish_reason="length".
"""

import logging
from typing import Callable, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# Callback signature: (task_name, exception) -> None. Used to surface
# auxiliary failures to the user through AIAgent._emit_auxiliary_failure
# so silent-drops (e.g. OpenRouter 402 exhausting the fallback chain)
# become visible instead of piling up as NULL session titles.
FailureCallback = Callable[[str, BaseException], None]
TitleCallback = Callable[[str], None]

# ── Prompts ──────────────────────────────────────────────────────────────────

# Default prompt (generic, works for most models)
_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Write the title in the same language the user is writing in. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)

# Pinned-language prompt: when auxiliary.title_generation.language is set,
# generate titles in that language instead of matching the user's language.
_TITLE_PROMPT_PINNED_LANGUAGE = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Write the title in {language}. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)

# Ultra-compact prompt for reasoning models that need less cognitive load
_COMPACT_TITLE_PROMPT = (
    "You generate session titles. Return only the title text. "
    "No markdown. No quotes. No explanation. Maximum 8 words."
)


def _title_language() -> str:
    """Return configured title language, or empty string to match the user."""
    try:
        from hermes_cli.config import load_config

        return str(
            ((load_config() or {}).get("auxiliary") or {})
            .get("title_generation", {})
            .get("language", "")
        ).strip()
    except Exception:
        return ""

# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_kwargs(
    messages: list,
    max_tokens: int = 500,
    temperature: float = 0.3,
    timeout: float = 30.0,
    extra_body: Optional[dict] = None,
    main_runtime: Optional[dict] = None,
) -> dict:
    """Build kwargs for call_llm with task=title_generation."""
    kwargs = {
        "task": "title_generation",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
        "main_runtime": main_runtime,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


# ── Core function ────────────────────────────────────────────────────────────


def generate_title(
    user_message: str,
    assistant_response: str,
    timeout: float = 30.0,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
) -> Optional[str]:
    """Generate a session title from the first exchange.

    Uses the main runtime's model when available, falling back to the
    auxiliary LLM client (cheapest/fastest available model).

    For reasoning models (DeepSeek, etc.), automatically retries with:
    - thinking/reasoning disabled (``{"thinking": {"type": "disabled"}}``)
    - larger max_tokens budget
    - compact prompt

    Returns the title string or None on failure.
    """
    # Truncate long messages to keep the request small
    user_snippet = user_message[:500] if user_message else ""
    assistant_snippet = assistant_response[:500] if assistant_response else ""

    language = _title_language()
    prompt = _TITLE_PROMPT_PINNED_LANGUAGE.format(language=language) if language else _TITLE_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"User: {user_snippet}\n\nAssistant: {assistant_snippet}"},
    ]

    # ── First attempt ────────────────────────────────────────────────────
    finish = None
    retry_finish = None
    compact_messages = None
    try:
        kwargs = _build_kwargs(
            messages, max_tokens=500, temperature=0.3,
            timeout=timeout, main_runtime=main_runtime,
        )
        response = call_llm(**kwargs)
        title, finish = _extract_title(response)
        if title:
            return title
        logger.debug(
            "Title generation first attempt returned empty (finish=%s, model=%s)",
            finish, getattr(response, "model", "?"),
        )
    except Exception as e:
        logger.debug("Title generation first attempt failed: %s", e)
        response = None
        finish = None

    # ── Retry with thinking disabled (reasoning model workaround) ────────
    # If the first attempt had empty content (likely reasoning model ate the
    # token budget), retry with thinking disabled + higher max_tokens.
    try:
        compact_messages = [
            {"role": "system", "content": _COMPACT_TITLE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Create a concise title for this conversation.\n\n"
                    f"User message:\n{user_snippet}\n\n"
                    f"Assistant response:\n{assistant_snippet}\n\n"
                    f"Return only the title."
                ),
            },
        ]
        retry_kwargs = _build_kwargs(
            compact_messages,
            max_tokens=1024,
            temperature=0.0,
            timeout=timeout,
            extra_body={"thinking": {"type": "disabled"}},
            main_runtime=main_runtime,
        )
        retry_response = call_llm(**retry_kwargs)
        title, retry_finish = _extract_title(retry_response)
        if title:
            return title
        logger.warning(
            "Title generation retry with thinking disabled also empty "
            "(finish=%s, model=%s)",
            retry_finish, getattr(retry_response, "model", "?"),
        )
    except Exception as e:
        logger.warning("Title generation retry failed: %s", e)
        if failure_callback is not None:
            try:
                failure_callback("title generation (retry)", e)
            except Exception:
                logger.debug("Title generation failure_callback raised", exc_info=True)

    # ── Final attempt: even larger budget ────────────────────────────────
    if finish == "length" or retry_finish == "length":
        try:
            mega_kwargs = _build_kwargs(
                compact_messages,
                max_tokens=4096,
                temperature=0.0,
                timeout=timeout,
                extra_body={"thinking": {"type": "disabled"}},
                main_runtime=main_runtime,
            )
            mega_response = call_llm(**mega_kwargs)
            title, mega_finish = _extract_title(mega_response)
            if title:
                return title
            logger.warning(
                "Title generation mega-retry empty (finish=%s, model=%s)",
                mega_finish, getattr(mega_response, "model", "?"),
            )
        except Exception as e:
            logger.warning("Title generation mega-retry failed: %s", e)

    return None


def _extract_title(response) -> tuple[Optional[str], Optional[str]]:
    """Extract and clean a title from an LLM response.

    Returns (title, finish_reason). title is None if content is empty.
    """
    if not response or not response.choices:
        return None, None
    choice = response.choices[0]
    finish = getattr(choice, "finish_reason", None)
    raw = (choice.message.content or "").strip()
    if not raw:
        return None, finish
    # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
    title = raw.strip('"\'')
    if title.lower().startswith("title:"):
        title = title[6:].strip()
    # Enforce reasonable length
    if len(title) > 80:
        title = title[:77] + "..."
    return title, finish


# ── Public entry points ──────────────────────────────────────────────────────


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
) -> None:
    """Generate and set a session title if one doesn't already exist.

    Called synchronously after the first exchange completes.
    Silently skips if:
    - session_db is None
    - session already has a title (user-set or previously auto-generated)
    - title generation fails
    """
    if not session_db or not session_id:
        return

    # Check if title already exists (user may have set one via /title before first response)
    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            return
    except Exception:
        return

    title = generate_title(
        user_message, assistant_response, failure_callback=failure_callback, main_runtime=main_runtime
    )
    if not title:
        return

    try:
        session_db.set_session_title(session_id, title)
        logger.debug("Auto-generated session title: %s", title)
        if title_callback is not None:
            try:
                title_callback(title)
            except Exception:
                logger.debug("Auto-title callback failed", exc_info=True)
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)
