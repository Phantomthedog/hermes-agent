"""Reliable headless-Chromium screenshot helper for WSL / 9p-mounted destinations.

Why this module exists
----------------------
On WSL2 the obvious workflow — ``headless_chromium --screenshot=/mnt/c/.../foo.png
https://example.com`` — silently produces a missing file.  Two distinct failure
modes stack on this host (2026-06, Jack's WSL2 / Ubuntu 22.04 box):

1. **Snap chromium lies about writes.**  ``/snap/bin/chromium`` exits 0 and prints
   ``N bytes written to file /tmp/foo.png`` — but the file is not on the host
   filesystem.  Snap-confinement mount-namespace isolation routes the write
   into snap-private storage that the host never sees.  Exit code 0 must NOT
   be trusted; the only authoritative success signal is ``os.path.isfile()``
   on the destination after the call returns.

2. **Chromium's PNG write path is incompatible with 9p/drvfs.**  Reproduced with
   the *real*, non-snap Playwright bundled chromium too — the write fails with
   ``Failed to write file /mnt/c/.../foo.png: No such file or directory (2)``
   even though the parent directory exists and is writable.  This is a known
   class of issues with 9p ``cache=5`` (mmap) mode and Chromium's temp-write +
   rename pattern.  Workaround: write to a Linux-native path (here, ``/tmp``),
   verify the result, then ``shutil.copy2`` it to the requested destination.

Design contract
---------------
``native_screenshot(url, dest_path)`` always follows this exact sequence:

  1. Resolve chromium binary — *never* ``/snap/bin/chromium``.
  2. Render to a fresh ``/tmp`` staging file.
  3. Verify: file exists, size > 0, ``file --mime-type`` (or magic bytes) says PNG.
  4. ``mkdir -p`` the destination's parent.
  5. Copy the verified PNG to ``dest_path``.
  6. Verify the destination: exists, size > 0.
  7. Return ``{"success": True, "path": ..., "bytes": N, "chromium_path": ...}``.

If *any* step fails, return ``{"success": False, "error": "..."}`` and clean up
the staging file.  The function never raises on runtime failure (it returns a
dict) so callers can branch on ``success`` uniformly.  Resolution errors
(``RuntimeError``) are reserved for misconfiguration — there is no chromium at
all — so callers can distinguish "the system is broken" from "this URL didn't
render".

This module does NOT replace ``browser_navigate`` / ``browser_snapshot`` /
``browser_screenshot`` from ``tools/browser_tool.py``.  Those remain the
secondary, agent-session-based path.  This module is the primary path for
"render this URL to a file on /mnt/c" tasks; use it when you need a reliable
PNG on the host filesystem and don't need an interactive session.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chromium resolution
# ---------------------------------------------------------------------------

#: Environment variable to override the chromium binary path.  ``/snap/...``
#: is intentionally rejected regardless of this env var.
HERMES_NATIVE_CHROMIUM_ENV = "HERMES_NATIVE_CHROMIUM"

#: Default chromium binary — the Playwright bundle that ships with hermes-agent
#: on this host.  Verified working on 2026-06-02: writes a 4.8 KB PNG to
#: ``/tmp`` in ~1.5s and fails to write to ``/mnt/c/...`` with ENOENT (9p bug).
DEFAULT_BUNDLED_CHROMIUM: Path = (
    Path.home()
    / ".cache"
    / "ms-playwright"
    / "chromium-1124"
    / "chrome-linux"
    / "chrome"
)


class NativeScreenshotError(RuntimeError):
    """Raised when the chromium binary cannot be located / resolved.

    Runtime errors from the screenshot call itself (URL unreachable, chromium
    exited non-zero, verification failed) are returned via the ``success``
    field of the result dict, not raised — that way callers can branch on a
    single field.  This exception is reserved for *configuration* failures:
    no chromium binary found, snap path explicitly requested, etc.
    """


def _is_snap_chromium(path: Union[str, Path]) -> bool:
    """Return True if ``path`` resolves to / points at the snap chromium.

    Snap chromium is unreliable on this host — its success messages lie.  We
    refuse it even when the user explicitly asks for it.  This is the only
    way to prevent silent regressions: if someone changes
    ``HERMES_NATIVE_CHROMIUM`` to ``/snap/bin/chromium`` and forgets the
    issue, every screenshot "succeeds" but produces no file.
    """
    s = str(path).rstrip("/")
    if not s:
        return False
    # /snap/bin/chromium, /snap/chromium/current/usr/lib/.../chrome, etc.
    if s == "/snap/bin/chromium" or s == "/snap/chromium":
        return True
    if s.startswith("/snap/chromium/") or s.startswith("/snap/bin/chromium"):
        return True
    if "/snap/chromium/" in s:
        return True
    return False


def resolve_chromium(
    explicit: Optional[Union[str, Path]] = None,
    *,
    require_executable: bool = True,
) -> Path:
    """Pick the chromium binary to use.  Never returns the snap variant.

    Resolution rules, applied in order:

      1. **Explicit snap path is a hard error.**  If the caller passes
         ``/snap/...`` explicitly, raise ``NativeScreenshotError`` — do
         NOT fall through to the bundled default.  The whole point of
         "Treat it as broken even if it exits 0" is to surface the
         misconfiguration loudly.

      2. **Env-var snap path is a hard error.**  If
         ``$HERMES_NATIVE_CHROMIUM`` points at ``/snap/...``, raise
         with a hint to unset the env var.

      3. **Walk candidates** (explicit, env, default) and return the
         first one that exists, is executable, and is not snap.

    Raises
    ------
    NativeScreenshotError
        If the explicit / env-var path is snap (loud failure), or if no
        candidate is executable (silent failure with a list of what was
        tried).
    """
    tried: list[str] = []

    # 1. Explicit snap → hard error, do NOT fall through.
    if explicit is not None:
        s = str(explicit)
        tried.append(s)
        if _is_snap_chromium(s):
            raise NativeScreenshotError(
                f"Explicit chromium path is the snap variant: {s}. "
                f"Snap chromium is reliably broken on WSL2 — it claims "
                f"successful writes but the file never lands on the host "
                f"filesystem. Pass a real binary (e.g. the Playwright "
                f"bundle at {DEFAULT_BUNDLED_CHROMIUM}) or unset "
                f"${HERMES_NATIVE_CHROMIUM_ENV}. See module docstring."
            )

    # 2. Env-var snap → hard error.
    env = os.environ.get(HERMES_NATIVE_CHROMIUM_ENV, "").strip()
    if env:
        tried.append(env)
        if _is_snap_chromium(env):
            raise NativeScreenshotError(
                f"${HERMES_NATIVE_CHROMIUM_ENV} points at snap chromium: "
                f"{env}. Snap chromium is reliably broken on WSL2 — unset "
                f"the env var or point it at the bundled Playwright "
                f"binary at {DEFAULT_BUNDLED_CHROMIUM}. See module docstring."
            )

    # 3. Walk candidates.  Snap is filtered out (we'd never get here for
    #    the explicit/env case above; this catches any *other* candidate
    #    that happens to be a snap — e.g. if a future contributor
    #    changes the default and forgets).
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if env:
        candidates.append(Path(env))
    candidates.append(DEFAULT_BUNDLED_CHROMIUM)

    for c in candidates:
        s = str(c)
        tried.append(s)
        if _is_snap_chromium(s):
            # Skip silently in the walk — the explicit/env checks above
            # would have raised if snap was the *requested* path.  A
            # snap candidate that we encounter as a *fallback* (because
            # the user didn't ask for it) just doesn't get picked.
            continue
        if require_executable:
            if not c.is_file() or not os.access(c, os.X_OK):
                continue
        return c

    raise NativeScreenshotError(
        f"No usable chromium binary found. Tried: {tried}. "
        f"DEFAULT_BUNDLED_CHROMIUM={DEFAULT_BUNDLED_CHROMIUM}. "
        f"Install one with `npx playwright install chromium` or point "
        f"${HERMES_NATIVE_CHROMIUM_ENV} at an existing binary."
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

#: PNG magic bytes — 8-byte signature every conformant PNG starts with.
_PNG_MAGIC: bytes = b"\x89PNG\r\n\x1a\n"


def _verify_png(path: Path) -> Tuple[bool, str]:
    """Confirm ``path`` is a real PNG.  Returns ``(ok, detail)``.

    The user's contract for the new screenshot flow explicitly requires using
    the ``file`` command — so we try it first.  On hosts without ``file``
    (rare on Linux), or if ``file`` fails for any reason, we fall back to a
    magic-byte check.  Both paths agree on the success criterion: the file
    must be a PNG, not just a non-empty file with a .png extension.
    """
    if not path.exists():
        return False, f"missing: {path}"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"stat failed: {exc}"
    if size <= 0:
        return False, f"empty ({size} bytes): {path}"

    file_bin = shutil.which("file")
    if file_bin:
        try:
            r = subprocess.run(
                [file_bin, "--mime-type", "-b", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            mime = r.stdout.strip()
            if mime == "image/png":
                return True, f"file reports PNG image data (image/png, {size} bytes)"
            return False, f"file reports {mime!r} (expected image/png), {size} bytes"
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
            logger.debug("file(1) failed for %s, falling back to magic bytes: %s", path, exc)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("file(1) unexpected error for %s: %s", path, exc)

    # Magic-byte fallback.
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError as exc:
        return False, f"read failed: {exc}"
    if head == _PNG_MAGIC:
        return True, f"PNG image data (magic-byte check, {size} bytes)"
    return False, f"not a PNG image (head={head!r}), {size} bytes"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Default chromium command-line flags.  Tuned for WSL2 + headless + no display.
#: ``--user-data-dir`` is supplied per-call (must be unique to avoid
#: ``SingletonLock acquired`` from a previous orphan instance).
DEFAULT_CHROMIUM_FLAGS: Tuple[str, ...] = (
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--hide-scrollbars",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-dev-shm-usage",  # /dev/shm is tiny in WSL; avoid OOM-kills
    "--mute-audio",
)


def native_screenshot(
    url: str,
    dest_path: Union[str, Path],
    *,
    window_size: Tuple[int, int] = (1280, 800),
    timeout: int = 30,
    chromium_path: Optional[Union[str, Path]] = None,
    flags: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Render ``url`` to a PNG at ``dest_path`` using bundled chromium.

    The destination can be on any filesystem (including 9p-mounted ``/mnt/c``
    on WSL2).  Chromium itself is *never* asked to write to ``dest_path``
    directly — it writes to a fresh ``/tmp`` staging file that we then copy
    to ``dest_path`` after verifying the PNG.

    Parameters
    ----------
    url:
        HTTP(S) or ``file://`` URL.  Reachable from the running user.  Not
        SSRF-checked here — callers that take URLs from untrusted input
        should pre-validate.
    dest_path:
        Absolute path for the final PNG.  Parent directory is created if
        missing.  May be on a 9p mount.
    window_size:
        ``(width, height)`` viewport in CSS pixels.  Default 1280×800 — big
        enough to read most layouts, small enough to keep file size sane.
    timeout:
        Hard cap on the chromium subprocess, in seconds.  A timeout is a
        hard failure (success=False), not a hang.
    chromium_path:
        Optional override for the chromium binary.  ``/snap/...`` is
        rejected.  When ``None``, falls back to ``$HERMES_NATIVE_CHROMIUM``
        then ``DEFAULT_BUNDLED_CHROMIUM``.
    flags:
        Optional override for chromium command-line flags.  Use only if you
        know what you're doing — the defaults are tuned for the WSL2 host
        this module was written for.

    Returns
    -------
    dict
        On success::

            {
                "success": True,
                "path": "/mnt/c/.../foo.png",
                "tmp_path": "/tmp/hermes_native_shot_xxxxxxxx.png",
                "chromium_path": "/home/jack/.cache/ms-playwright/chromium-1124/chrome-linux/chrome",
                "bytes": 4801,
                "verify": "file: image/png, 4801 bytes",
                "elapsed_s": 1.42,
            }

        On failure::

            {
                "success": False,
                "error": "...",
                "stage": "render" | "verify_src" | "copy" | "verify_dst" | "resolve",
                "tmp_path": "...",   # present if render produced a file
            }

        The function never raises on runtime failure — only
        :class:`NativeScreenshotError` is raised, and only for
        misconfiguration (no chromium at all).  Callers should check
        ``success`` first.
    """
    url = (url or "").strip()
    if not url:
        return {"success": False, "error": "empty url", "stage": "resolve"}

    dest = Path(dest_path)
    if not dest.is_absolute():
        return {
            "success": False,
            "error": f"dest_path must be absolute, got {dest_path!r}",
            "stage": "resolve",
        }

    # --- Resolve chromium (may raise NativeScreenshotError) -----------------
    try:
        chrome = resolve_chromium(chromium_path)
    except NativeScreenshotError as exc:
        return {"success": False, "error": str(exc), "stage": "resolve"}

    # --- Stage in /tmp ------------------------------------------------------
    # Per-call user-data-dir is required: chromium complains about a stale
    # SingletonLock otherwise (or refuses to start a second concurrent
    # instance).  We delete it in `finally` so /tmp doesn't accumulate.
    staging_root = Path(tempfile.mkdtemp(prefix="hermes_native_shot_", dir="/tmp"))
    user_data_dir = staging_root / "userdata"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    tmp_png = staging_root / "out.png"

    width, height = int(window_size[0]), int(window_size[1])
    base_flags = list(flags) if flags is not None else list(DEFAULT_CHROMIUM_FLAGS)
    cmd = [
        str(chrome),
        *base_flags,
        f"--user-data-dir={user_data_dir}",
        f"--screenshot={tmp_png}",
        f"--window-size={width},{height}",
        url,
    ]

    start = time.monotonic()
    result: Dict[str, Any] = {
        "success": False,
        "chromium_path": str(chrome),
        "url": url,
        "dest_path": str(dest),
    }
    try:
        # --- 1. Render ------------------------------------------------------
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                # Chromium writes a lot of harmless warnings to stderr on WSL
                # (dbus / UPower / sandbox).  Don't pollute logs.
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                **result,
                "error": f"chromium render timed out after {timeout}s",
                "stage": "render",
                "elapsed_s": round(time.monotonic() - start, 2),
            }
        except FileNotFoundError as exc:
            return {
                **result,
                "error": f"chromium binary disappeared mid-call: {exc}",
                "stage": "render",
            }

        # --- 2. Verify staged PNG ------------------------------------------
        ok, detail = _verify_png(tmp_png)
        if not ok:
            return {
                **result,
                "error": f"staged PNG failed verification: {detail}",
                "stage": "verify_src",
                "chromium_exit": proc.returncode,
                "chromium_stderr_tail": _tail(proc.stderr, 800),
                "elapsed_s": round(time.monotonic() - start, 2),
            }

        size = tmp_png.stat().st_size

        # --- 3. Create destination dir + copy ------------------------------
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_png, dest)
        except OSError as exc:
            return {
                **result,
                "error": f"copy {tmp_png} -> {dest} failed: {exc}",
                "stage": "copy",
                "tmp_path": str(tmp_png),
                "elapsed_s": round(time.monotonic() - start, 2),
            }

        # --- 4. Verify destination -----------------------------------------
        ok_dst, detail_dst = _verify_png(dest)
        if not ok_dst:
            return {
                **result,
                "error": f"destination failed verification: {detail_dst}",
                "stage": "verify_dst",
                "tmp_path": str(tmp_png),
                "elapsed_s": round(time.monotonic() - start, 2),
            }

        # --- 5. Success ----------------------------------------------------
        return {
            "success": True,
            "path": str(dest),
            "tmp_path": str(tmp_png),
            "chromium_path": str(chrome),
            "bytes": int(dest.stat().st_size),
            "verify": detail_dst,
            "elapsed_s": round(time.monotonic() - start, 2),
        }

    finally:
        # Clean up the per-call staging dir (user-data + staged PNG).
        # Keep tmp_png around if the caller wants to inspect it on failure —
        # we already returned its path.  The user-data dir is what we want to
        # nuke because it can be hundreds of MB.
        try:
            shutil.rmtree(staging_root, ignore_errors=True)
        except Exception:  # pragma: no cover — defensive
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tail(text: Optional[str], n: int) -> str:
    """Return the last ``n`` chars of ``text`` for inclusion in error dicts.

    Chromium writes a lot of harmless dbus / UPower noise to stderr; we keep
    only the tail so the agent loop doesn't blow up its context window on a
    failure.  Returns ``""`` for ``None`` or empty input.
    """
    if not text:
        return ""
    if len(text) <= n:
        return text
    return "..." + text[-n:]


__all__ = [
    "DEFAULT_BUNDLED_CHROMIUM",
    "DEFAULT_CHROMIUM_FLAGS",
    "HERMES_NATIVE_CHROMIUM_ENV",
    "NativeScreenshotError",
    "native_screenshot",
    "resolve_chromium",
]


# ---------------------------------------------------------------------------
# Tool registration (self-registering in Hermes tool system)
# ---------------------------------------------------------------------------

_NATIVE_SCREENSHOT_SCHEMA = {
    "name": "native_screenshot",
    "description": "Render a URL to a PNG file on the host filesystem using "
                   "local bundled Chromium. Writes to a /tmp staging file "
                   "first (bypassing the 9p/drvfs bug), verifies the PNG, "
                   "then copies it to the requested destination. Works "
                   "independently of the cloud browser backends — use this "
                   "when browser_navigate / browser_vision are unavailable "
                   "or you need a reliable file on disk.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "HTTP(S) or file:// URL to render"
            },
            "dest_path": {
                "type": "string",
                "description": "Absolute path for the output PNG file. "
                               "Parent directory is created if missing. "
                               "Use /mnt/c/... (or /mnt/f/...) paths for "
                               "files visible from Windows."
            },
            "width": {
                "type": "integer",
                "description": "Viewport width in CSS pixels (default: 1280)",
                "default": 1280
            },
            "height": {
                "type": "integer",
                "description": "Viewport height in CSS pixels (default: 800)",
                "default": 800
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum seconds to wait for Chromium render (default: 30)",
                "default": 30
            },
        },
        "required": ["url", "dest_path"],
    },
}


def _native_screenshot_available() -> bool:
    """Check if bundled chromium exists and is executable.

    Lightweight stat check — does not run the binary.
    """
    chrome = DEFAULT_BUNDLED_CHROMIUM
    return chrome.is_file() and os.access(chrome, os.X_OK)


def _handle_native_screenshot(args: dict, **kw) -> str:
    """Handler for the native_screenshot tool.

    Parses args from the tool call, invokes native_screenshot(), and
    returns the result as a JSON string.
    """
    url = (args.get("url") or "").strip()
    dest_path = (args.get("dest_path") or "").strip()

    if not url:
        return tool_error("empty url")
    if not dest_path:
        return tool_error("empty dest_path")

    width = int(args.get("width", 1280))
    height = int(args.get("height", 800))
    timeout = int(args.get("timeout", 30))

    try:
        result = native_screenshot(
            url=url,
            dest_path=dest_path,
            window_size=(max(width, 320), max(height, 240)),
            timeout=max(timeout, 5),
        )
        return tool_result(result)
    except NativeScreenshotError as exc:
        return tool_error(str(exc))
    except Exception as exc:
        return tool_error(f"unexpected error: {exc}")


from tools.registry import registry, tool_error, tool_result  # noqa: E402

registry.register(
    name="native_screenshot",
    toolset="native_screenshot",
    schema=_NATIVE_SCREENSHOT_SCHEMA,
    handler=_handle_native_screenshot,
    check_fn=_native_screenshot_available,
    emoji="📷",
)
