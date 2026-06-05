"""
Hermes Terminal Activity Status

Uses Windows Terminal OSC sequences to show activity progress in the terminal
tab/taskbar. Supports direct Windows Terminal and tmux passthrough modes.

Emits to /dev/tty when available (direct terminal access), falling back to
sys.stderr. Never emits to sys.stdout directly — this prevents ESC sequence
pollution in redirected/piped stdout, while still working inside prompt_toolkit
or other stdout-wrapping environments.

Environment variable: HERMES_TERMINAL_STATUS=auto|on|off

OSC 9;4 sequences (Windows Terminal taskbar progress):
  state:  0=clear, 1=success, 2=error, 3=indeterminate
  progress: 0-100 (unused for clear/indeterminate)

OSC 2 sequence: set terminal tab title.

Title display modes:
  - No session title:  tab shows status text (e.g. "✦ Working")
  - Session title set: tab shows icon + title  (e.g. "✦ Fix DB migration")
    The status word (Working/Done/Failed/Ask) is hidden; only icon remains.
"""

import atexit
import os
import sys
import time
import logging

logger = logging.getLogger(__name__)

# --- Output stream ---

_TTY_FD: int | None = None  # fd for /dev/tty (3 or higher), None if unavailable


def _open_tty() -> int | None:
    """Try to open /dev/tty for direct terminal output.  Returns fd or None."""
    try:
        fd = os.open("/dev/tty", os.O_WRONLY)
        return fd
    except OSError:
        return None


def _write(data: bytes) -> None:
    """Write raw bytes to terminal.  Prefers /dev/tty, falls back to stderr."""
    global _TTY_FD
    if _TTY_FD is None:
        _TTY_FD = _open_tty()
    if _TTY_FD is not None:
        try:
            os.write(_TTY_FD, data)
            return
        except OSError:
            _TTY_FD = None  # stale fd — retry next time
    # Fallback: stderr (visible, won't pollute piped stdout)
    try:
        sys.stderr.buffer.write(data)
        sys.stderr.flush()
    except Exception:
        pass


# --- Detection helpers ---

def _is_tmux() -> bool:
    return "TMUX" in os.environ


def _in_windows_terminal() -> bool:
    return "WT_SESSION" in os.environ


def _has_tty_device() -> bool:
    """Check if /dev/tty is accessible (we have a real terminal)."""
    try:
        fd = os.open("/dev/tty", os.O_RDONLY)
        os.close(fd)
        return True
    except OSError:
        return False


# --- Configuration ---

_ENABLED_CACHE: bool | None = None


def is_enabled() -> bool:
    """
    Check whether terminal-status sequences should be emitted.

    ``HERMES_TERMINAL_STATUS`` env var:
      ``off`` — never emit.
      ``on`` — emit if /dev/tty is accessible.
      ``auto`` (default) — emit if /dev/tty is accessible AND the outer
      environment looks like Windows Terminal (or tmux under it).
    """
    global _ENABLED_CACHE
    if _ENABLED_CACHE is not None:
        return _ENABLED_CACHE

    mode = os.environ.get("HERMES_TERMINAL_STATUS", "auto").strip().lower()

    if mode == "off":
        _ENABLED_CACHE = False
        return False

    has_tty = _has_tty_device()

    if mode == "on":
        _ENABLED_CACHE = has_tty
        return has_tty

    # auto mode
    if not has_tty:
        _ENABLED_CACHE = False
        return False

    # auto + has tty: enable if WT_SESSION is set or inside tmux
    if _in_windows_terminal():
        _ENABLED_CACHE = True
        return True
    if _is_tmux():
        _ENABLED_CACHE = True
        return True

    _ENABLED_CACHE = False
    return False


def invalidate_cache() -> None:
    """Force re-check on next call. Useful after env var changes mid-session."""
    global _ENABLED_CACHE
    _ENABLED_CACHE = None


# --- Low-level emission ---

def _emit(raw_sequence: str) -> None:
    """Write a raw escape sequence to /dev/tty, wrapped for tmux if needed."""
    if not is_enabled():
        return
    try:
        data = raw_sequence.encode("utf-8")
        if _is_tmux():
            # tmux passthrough: ESC P tmux ; <data> ESC \
            _write(b"\033Ptmux;" + data + b"\033\\")
        else:
            _write(data)
    except Exception:
        # Never crash Hermes because of terminal decoration
        pass


def set_title(title: str) -> None:
    """Set terminal tab title via OSC 2."""
    _emit(f"\033]2;{title}\a")


def emit_progress(state: int, progress: int = 0) -> None:
    """
    Emit Windows Terminal taskbar progress via OSC 9;4.

    State: 0=clear, 1=success, 2=error, 3=indeterminate.
    Progress: 0-100 (used for success/error; ignored for clear/indeterminate).
    """
    _emit(f"\033]9;4;{state};{progress}\a")


def _bell() -> None:
    """Ring the terminal bell once."""
    try:
        _write(b"\a")
    except Exception:
        pass


# --- Session title state (simple mode) ---

_session_title: str | None = None
"""Global session title set by title_generator. Once set, all tab titles
use the format ``<icon> <title>`` instead of the full status text."""

_current_icon: str = ""
"""Icon for the current state: ✦, ✓, ✗, ❓, or empty."""


def set_session_title(title: str | None) -> None:
    """Store a session title and immediately refresh the tab.

    Once set, all subsequent status updates show ``<icon> <title>``
    instead of the full status word. Pass ``None`` to clear.
    """
    global _session_title
    _session_title = title
    # Immediately refresh the tab with the current icon + new title
    if _session_title:
        _apply_title(_current_icon, _session_title)
    else:
        # Cleared — show nothing (caller should call start_working etc.)
        pass


def get_session_title() -> str | None:
    """Return the current session title, or None if not yet generated."""
    return _session_title


# --- Title display helpers ---

def _apply_title(icon: str, title_text: str) -> None:
    """Build and emit the tab title."""
    if icon and title_text:
        set_title(f"{icon} {title_text}")
    elif icon:
        set_title(icon)
    elif title_text:
        set_title(title_text)
    else:
        set_title("")


def _effective_title(status_text: str, icon: str) -> str:
    """Determine the actual tab title to show.

    No session title: return ``status_text`` (e.g. "✦ Working").
    Session title exists: return ``<icon> <title>`` (e.g. "✦ Fix DB migration").
    """
    global _session_title, _current_icon
    _current_icon = icon
    if _session_title:
        # Show only icon + session title, strip status word
        return f"{icon} {_session_title}" if icon else _session_title
    return status_text


# --- Public high-level API ---

_WORKING: bool = False  # module-level tracking for atexit cleanup


def start_working(label: str = "Working") -> None:
    """Set the terminal to working state (indeterminate progress, title)."""
    global _WORKING
    if not is_enabled():
        return
    try:
        title = _effective_title(f"\u2726 {label}", "\u2726")
        set_title(title)
        emit_progress(3, 0)  # indeterminate
        _WORKING = True
    except Exception:
        pass


def success() -> None:
    """Show success state: progress bar fills, bell, title."""
    global _WORKING
    if not is_enabled():
        return
    try:
        title = _effective_title("\u2713 Done", "\u2713")
        set_title(title)
        emit_progress(1, 100)
        _bell()
    except Exception:
        pass
    finally:
        _WORKING = False


def error(skip_bell: bool = False) -> None:
    """Show error state: progress bar error, bell, title. Does NOT auto-clear."""
    global _WORKING
    if not is_enabled():
        return
    try:
        title = _effective_title("\u2717 Failed", "\u2717")
        set_title(title)
        emit_progress(2, 100)
        if not skip_bell:
            _bell()
    except Exception:
        pass
    finally:
        _WORKING = False


def ask_question() -> None:
    """Show question-waiting state: clear progress, title with question mark.

    Use when the agent uses ``clarify`` and is waiting for user input.
    Call ``start_working()`` again once the user has answered.
    """
    global _WORKING
    if not is_enabled():
        return
    try:
        emit_progress(0, 0)  # clear the spinning indeterminate bar
        title = _effective_title("\u2753 Ask", "\u2753")
        set_title(title)
    except Exception:
        pass
    finally:
        _WORKING = False


def clear() -> None:
    """Clear progress indicator and restore base title."""
    global _WORKING
    if not is_enabled():
        return
    try:
        emit_progress(0, 0)  # clear
        set_title("")
    except Exception:
        pass
    finally:
        _WORKING = False


class active_turn:
    """
    Context manager wrapping one complete assistant turn.

    On entry:         working state (indeterminate progress + title).
    On success exit:  success state (persists until next turn).
    On exception:     error state (persists, not auto-cleared).

    In ``auto`` or ``on`` mode where ``is_enabled()`` is False, this
    becomes a near-zero-overhead no-op (single truthy check).
    """

    def __init__(self, label: str = "Working"):
        self.label = label

    def __enter__(self):
        start_working(self.label)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                error()
                return False  # do NOT suppress the exception

            # Normal success path (persists until next turn)
            success()
            # Brief pause so the success state is visible
            time.sleep(0.4)

        except Exception:
            # Safety net — never let terminal_status crash Hermes
            pass
        return False


# --- atexit cleanup ---

def _cleanup() -> None:
    """On process exit, clear terminal state if we left it dirty."""
    if _WORKING:
        try:
            _write(b"\033]9;4;0;0\a")
            _write(b"\033]2;\a")
        except Exception:
            pass


atexit.register(_cleanup)
